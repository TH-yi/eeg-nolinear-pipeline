"""Command-line entry point for the EEG non-linear-feature pipeline.

Example
-------
python -m eeg_pipeline.main process \
        --data-dir   /path/to/mat_files \
        --output-dir /path/to/output
"""
from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────
from pathlib import Path
import tempfile, uuid, os, json
from itertools import count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, Future
import multiprocessing as mp
import logging
from typing import List, Dict

# ── 3rd-party ─────────────────────────────────────────────────────────────
import numpy as np
import shutil
import typer
from tqdm import tqdm

# ── project modules ───────────────────────────────────────────────────────
from config import *
from logger import get_logger
from utils.memory import safe_worker_count, memory_limited_worker_count, compute_max_threads
from utils.fileio import load_subject_mat  # wrapper around scipy.loadmat / mat73
from preprocessing.channel_correction import (
    cz_interpolation,
    reorder_channels,
)
from preprocessing.segmentation import segment_signal
from features.nonlinear_analysis import nonlinear_analysis  # 17-feature API

# ──────────────────────────────────────────────────────────────────────────

app = typer.Typer()
logger = get_logger("main")


# -------------------------------------------------------------------------
# 1. MAT inspection helper (unchanged)
# -------------------------------------------------------------------------
@app.command(help="List variables in every S*.mat file (debug).")
def inspect(
        data_dir: Path = typer.Option(None, help="Folder with S*.mat files."),
) -> None:
    import mat73
    from scipy.io import loadmat

    data_dir = data_dir or INPUT_DIR
    logger.info(f"Scanning {data_dir}")

    for f in sorted(data_dir.glob("S*.mat")):
        logger.info(f"--- {f.name} ---")
        try:
            mat = loadmat(f)
        except NotImplementedError:
            mat = mat73.loadmat(f)

        for k, v in mat.items():
            if k.startswith("__"):
                continue
            logger.info(
                f"{k:<20} type={type(v).__name__:<12} shape={getattr(v, 'shape', '')}"
            )


# -------------------------------------------------------------------------
# 2. MATLAB-equivalent window rule
# -------------------------------------------------------------------------
def _choose_window(n_samples: int) -> tuple[int, int]:
    """
    Emulate Read_Signals.m:

        if len >= 5000:
            w1 = len / 5     # 20 % window
            w2 = len / 10    # 40 % overlap
        else:
            w1 = 500
            w2 = 100
    """
    if n_samples >= 5_000:
        return n_samples // 5, n_samples // 10
    return 500, 100


# -------------------------------------------------------------------------
# 3. segment-level worker
# -------------------------------------------------------------------------
def analyze_segment(
    seg: np.ndarray,
    fs: int,
    tau: int,
    lag: int,
    emb_dim:int,
    subj: str,
    task: str,
    trial: str,
    seg_id: int,
    position: int,
    total: int,
    max_threads_per_channel: int,
    max_workers: int,
    use_cache: bool = False,
    cache_dir=None,
    on_almost_done: callable | None = None,
    channel_threshold: int = 1,
):
    """Extract nonlinear features for a single segment.

    This helper **no longer** spawns a separate subprocess. Instead, it runs in
    the caller's process and simply forwards *max_workers* to
    ``nonlinear_analysis`` so that parallelism (if any) can be handled inside
    that function.

    Parameters
    ----------
    seg : np.ndarray
        The segment (shape: ``channels × samples``) to analyse.
    fs : int
        Sampling rate in Hz.
    subj, task, trial : str
        Identifiers used exclusively for progress‑bar labelling.
    seg_id : int
        Index of the segment within the current trial.
    position : int
        TQDM *position* for the inner progress bar (must differ from the outer
        bar).
    total : int
        Total iterations expected inside *nonlinear_analysis* (e.g. samples or
        windows) so that the inner bar can be sized correctly.
    max_workers : int
        Maximum number of workers that *nonlinear_analysis* may decide to use.

    Returns
    -------
    tuple[bool, str | np.ndarray]
        ``(True, features)`` on success, or ``(False, error_message)`` on
        failure.
    """
    try:
        desc = f"{subj}-{task}-{trial} Segment {int(seg_id)+1} Process"
        with tqdm(
            total=total,
            desc=desc,
            position=position,
            leave=False,
            dynamic_ncols=True,
        ) as bar:
            features = nonlinear_analysis(
                seg,
                fs=fs,
                tau=tau,
                lag=lag,
                emb_dim=emb_dim,
                save_plots=False,
                flatten=True,
                tqdm_progress=bar,
                max_threads_per_channel=max_threads_per_channel,
                max_workers=max_workers,
                use_cache=use_cache,
                cache_dir=cache_dir,
                on_almost_done_channels=on_almost_done,
                channel_threshold=channel_threshold,
            )
        return True, features
    except Exception as exc:  # pragma: no cover ‑‑ diagnostic path
        return False, f"{type(exc).__name__}: {exc}"



# -------------------------------------------------------------------------
# 4. subject-level worker
# -------------------------------------------------------------------------
def _process_subject(
    mat_path: Path,
    fs: int,
    tau: int,
    lag: int,
    emb_dim:int,
    task_map: Dict[str, List[str]],
    output_dir: Path,
    save_mat: bool = True,
) -> str:
    """One *.mat* in → nonlinear features out (JSON + optional MAT).

    A hybrid strategy automatically picks between **in‑memory** and
    **disk‑cached** processing to minimise RAM pressure while preserving speed
    whenever possible.
    """

    import mat73
    from uuid import uuid4
    from scipy.io import loadmat, savemat

    subj = mat_path.stem  # "S1" … "S28"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Start processing %s", subj)

    # ------------------------------------------------------------------
    # 1. Load MAT (once)
    # ------------------------------------------------------------------
    try:
        mat_dict = loadmat(mat_path)
    except NotImplementedError:
        mat_dict = mat73.loadmat(mat_path)

    cpu_cnt = os.cpu_count() or 1
    # Prepare cache dir (always create – might stay empty)
    cache_dir = output_dir / "__sig_cache"
    cache_dir = output_dir / f"__sig_cache_{subj}_{uuid4().hex[:8]}"
    cache_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers for the two strategies
    # ------------------------------------------------------------------
    def _prep_signal_to_cache(task: str, trial_key: str) -> Path | None:
        """Pre‑process signal then save to ``.npy``."""
        if trial_key not in mat_dict:
            logger.error("%s – %s: missing %s", subj, task, trial_key)
            raise KeyError(trial_key)
        sig = mat_dict.pop(trial_key)  # frees RAM
        sig = reorder_channels(sig)
        sig = cz_interpolation(sig)
        cache_p = cache_dir / f"{subj}_{task}_{trial_key}.npy"
        np.save(cache_p, sig, allow_pickle=False)
        del sig
        return cache_p

    def _load_signal_mem(task: str, trial_key: str):
        """Return pre‑processed in‑memory signal."""
        sig = mat_dict[trial_key]
        sig = reorder_channels(sig)
        sig = cz_interpolation(sig)
        return sig

    # Strategy‑agnostic trial processor
    def _process_one_trial(
            task: str,
            trial_key: str,
            sig_source,
            use_cache: bool,
            max_threads_per_channel: int,
            max_workers: int,
            *,
            # ───────────── new optional arguments ──────────────────────────
            preload_next_trial_cb: callable | None = None,
            channel_threshold: int = 1,
    ) -> np.ndarray | None:
        """
        Compute the *mean* 17-dim feature vector for a single *trial*.

        Parameters
        ----------
        task, trial_key : str
            Identifiers used purely for logging / progress bars.
        sig_source :
            Either an in-memory ndarray (channels × samples) **or** a disk-cached
            ``.npy`` path if *use_cache* is True.
        use_cache : bool
            ``True`` → *sig_source* is a path and the signal is mem-mapped.
        max_threads_per_channel :
            Upper bound passed down to `_features_per_channel` (thread fan-out).
        max_workers :
            Process-level fan-out inside `nonlinear_analysis` (per segment).
        preload_next_trial_cb : callable | None, optional
            Callback that *starts loading the next trial*.  It will be handed down
            to `nonlinear_analysis()` **for the last segment only** and will fire
            once that segment has *few enough* workers left alive.
        channel_threshold : int, optional
            “Few enough” is defined as `remaining_workers ≤ channel_threshold`
            inside `nonlinear_analysis`.
        """
        if sig_source is None:
            raise ValueError("sig_source is None!")

        # 1) Load or mem-map the raw EEG signal ----------------------------------
        sig_mm = (
            np.load(sig_source, mmap_mode="r")  # disk-cached (lazy)
            if use_cache else
            sig_source  # already ndarray
        )

        win_len, overlap = _choose_window(sig_mm.shape[1])
        step = win_len - overlap

        if use_cache:
            total_segments = 1 + max(0, (sig_mm.shape[1] - win_len) // step)
        else:
            segments = segment_signal(sig_mm, win_len, step)
            total_segments = len(segments)
            if total_segments == 0:
                logger.warning("%s – %s – %s: no segments", subj, task, trial_key)
                raise ValueError("Total segments == 0!")

        feature_vectors: list[np.ndarray] = []

        with tqdm(
                total=total_segments,
                desc=f"{subj}-{task}-{trial_key} Segments",
                position=0,
                leave=True,
                dynamic_ncols=True,
        ) as bar:

            for seg_idx in range(total_segments):

                # -----------------------------------------------------------------
                # 2) Extract current segment
                # -----------------------------------------------------------------
                if use_cache:
                    start = seg_idx * step
                    end = start + win_len
                    seg = sig_mm[:, start:end]
                else:
                    seg = segments[seg_idx]

                # -----------------------------------------------------------------
                # 3) Decide ONCE whether to attach the “almost-done” callback
                #    • Only the *last* segment carries the callback.
                #    • Earlier segments pass None → will never trigger preload.
                # -----------------------------------------------------------------
                if seg_idx == total_segments - 1:  # ⇐ last segment
                    cb = preload_next_trial_cb
                    chan_thresh = channel_threshold
                else:
                    cb = None
                    chan_thresh = 1  # unused

                ok, payload = analyze_segment(
                    seg,
                    fs,
                    tau,
                    lag,
                    emb_dim,
                    subj,
                    task,
                    trial_key,
                    seg_idx,
                    1,  # tqdm position for inner bar
                    seg.shape[0],  # total rows (channels)
                    max_threads_per_channel,
                    max_workers,
                    use_cache=use_cache,
                    cache_dir=cache_dir,
                    # -------------- propagate callback & threshold --------------
                    on_almost_done=cb,
                    channel_threshold=chan_thresh,
                )

                if ok:
                    feature_vectors.append(payload)
                else:
                    tqdm.write(f"{subj}–{task}-{trial_key} failed: {payload}")

                bar.update()

        # 4) Aggregate results ----------------------------------------------------
        if feature_vectors:
            return np.mean(feature_vectors, axis=0)

        logger.error("%s – %s – %s: no data after QC", subj, task, trial_key)
        return None

    # ------------------------------------------------------------------
    # 2. Task‑level loop (sequential)
    # ------------------------------------------------------------------
    def _process_task(task: str, keys: List[str]) -> List[np.ndarray]:
        """
        Compute a mean-feature vector for every *trial* listed in *keys*.

        Streaming strategy
        ------------------
        • Only **one** upcoming trial is ever kept in memory ahead of the trial
          currently being analysed.
        • Pre-loading of the next trial starts *late*: it is triggered only when
          the running trial enters its last ≤ ``max_workers`` segments.  By that
          time the segment-level ProcessPool is practically idle, avoiding memory
          spikes.
        • If pre-loading is not finished when the current trial completes, the
          main thread blocks until the load is done, guaranteeing strict order.
        """

        trial_vecs: list[np.ndarray] = []

        # ------------------------------------------------------------------ #
        # 0. Helper – load one trial (runs inside a background thread)       #
        # ------------------------------------------------------------------ #
        def _load_trial(k: str):
            """Load / pre-process *k* and decide MEM vs CACHE strategy."""
            #logger.debug("%s – %s: 🔄 START loading %s", subj, task, k)
            if k not in mat_dict:
                logger.error("%s – %s: missing %s", subj, task, k)
                raise ValueError(f"{subj} – {task}: missing {k}")

            sample_size = mat_dict[k].size
            tentative_workers = safe_worker_count(sample_size, cpu_cnt, CPU_UTILIZATION_RATIO)
            use_cache = tentative_workers < (cpu_cnt - 2)  # fall-back to disk
            mem_workers = memory_limited_worker_count(sample_size, MAX_WORKER_MEMORY_LIMIT)

            max_threads_per_ch, tentative_workers = compute_max_threads(
                mem_workers, tentative_workers, PARALLEL_TASK_COUNT
            )

            logger.info(
                "%s – %s – %s → %s (max_threads_per_channel=%d, max_workers=%d)",
                subj, task, k,
                "CACHE" if use_cache else "MEM",
                max_threads_per_ch, tentative_workers,
            )

            # Return either an in-memory ndarray or a .npy path (cache)
            src = (
                _prep_signal_to_cache(task, k) if use_cache
                else _load_signal_mem(task, k)
            )
            #logger.debug("%s – %s: ✅ FINISHED loading %s", subj, task, k)
            return (k, src, use_cache, max_threads_per_ch, tentative_workers)

        # ------------------------------------------------------------------ #
        # 1. Streaming loop                                                  #
        # ------------------------------------------------------------------ #
        key_iter = iter(keys)
        loader = ThreadPoolExecutor(max_workers=1)  # single BG loader
        preload_fut: Future | None = None  # future of next trial

        # ---- (a) synchronously prepare the very first trial ---------------
        try:
            first_key = next(key_iter)
        except StopIteration:
            return trial_vecs  # nothing to process

        k, src, use_cache, max_thr, max_wkr = _load_trial(first_key)

        while True:
            # -------------------------------------------------------------- #
            # Callback to kick off *one* pre-load when almost done           #
            # -------------------------------------------------------------- #
            def _kickoff_preload():
                nonlocal preload_fut
                if preload_fut is None:  # ensure single call
                    try:
                        nxt = next(key_iter)  # may raise StopIteration
                        logger.debug("%s – %s: 🚚 Pre-loading next trial %s", subj, task, nxt)
                        preload_fut = loader.submit(_load_trial, nxt)
                    except StopIteration:
                        logger.debug("%s – %s: 🛑 No further trials", subj, task)
                        pass  # no more trials

            # ---- (b) run the current trial; callback fires late ----------
            vec = _process_one_trial(
                task, k, src, use_cache,
                max_threads_per_channel=max_thr,
                max_workers=max_wkr,
                preload_next_trial_cb=_kickoff_preload,
                channel_threshold=max(1, max_wkr - 1),
            )
            if vec is not None:
                trial_vecs.append(vec)
            else:
                raise RuntimeError(f"{subj} – {task} – {k}: returned None")

            # ---- (c) if a pre-load is in progress, wait for it -----------
            if preload_fut is not None:
                logger.debug("%s – %s: ⏳ Waiting for pre-load to finish", subj, task)
                k, src, use_cache, max_thr, max_wkr = preload_fut.result()
                logger.debug("%s – %s: 📦 Pre-load ready → %s", subj, task, k)
                preload_fut = None  # loop continues
                continue  # begin next trial

            # ---- (d) no further trials scheduled → exit loop ------------
            break

        loader.shutdown(wait=True)
        return trial_vecs

    # ------------------------------------------------------------------
    # 3. Run all tasks sequentially
    # ------------------------------------------------------------------
    results: Dict[str, List[np.ndarray]] = {}
    for task, keys in task_map.items():
        results[task] = _process_task(task, keys)

    # ------------------------------------------------------------------
    # 4. Persist outputs
    # ------------------------------------------------------------------
    jpath = output_dir / f"{subj}_features.json"
    jpath.write_text(
        json.dumps({k: [v.tolist() for v in vecs] for k, vecs in results.items()}, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %s", jpath.name)

    if save_mat:
        mat_out = {f"NL_Features_{k}": np.stack(v, axis=0) for k, v in results.items()}
        savemat(output_dir / f"{subj}_NL_Results.mat", mat_out)
        logger.info("Saved %s_NL_Results.mat", subj)

    # ------------------------------------------------------------------
    # 5. Cleanup
    # ------------------------------------------------------------------
    shutil.rmtree(cache_dir, ignore_errors=True)
    logger.debug("Finish processing %s", subj)
    return subj


# -------------------------------------------------------------------------
# 5. CLI – batch over all subjects
# -------------------------------------------------------------------------
@app.command(help="Extract 5-dim nonlinear features for all S*.mat files.")
def process(
        data_dir: Path = typer.Option(None, help="Input folder with S*.mat files"),
        output_dir: Path = typer.Option(None, help="Output folder for JSON / MAT"),
        fs: int = FS,
        tau: int = TAU,
        lag: int = LAG,
        emb_dim:int = EMB_DIM,
        save_mat: bool = typer.Option(
            True, "--save-mat", help="Also save <subj>_NL_Results.mat files"
        ),
) -> None:
    data_dir = data_dir or INPUT_DIR
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    sub_files = sorted(data_dir.glob("S*.mat"))
    logger.info(f"Found {len(sub_files)} subjects in {data_dir}")

    for f in sub_files:
        _process_subject(f, fs, tau, lag, emb_dim, TASK_MAP, output_dir, save_mat)

    logger.info("✅ All subjects finished.")

    # -------- merge subject JSONs to global JSON / MAT -------------------
    merged: dict[str, list[list[float]]] = {k: [] for k in ["IDG", "IDE", "IDR", "RST1", "RST2"]}

    for sf in sorted(output_dir.glob("S*_features.json")):
        with sf.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        for task in merged:
            merged[task].extend(data.get(task, []))

    (output_dir / "Creativity_NL_Data.json").write_text(
        json.dumps(merged, indent=2), encoding="utf-8"
    )
    logger.info("✅ Saved Creativity_NL_Data.json")

    if save_mat:
        from scipy.io import savemat
        savemat(
            output_dir / "Creativity_NL_Data.mat",
            {f"NL_Features_{k}": np.asarray(v) for k, v in merged.items()},
        )
        logger.info("✅ Saved Creativity_NL_Data.mat")


if __name__ == "__main__":
    app()
