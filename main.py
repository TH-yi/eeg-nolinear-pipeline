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
import concurrent.futures as cf
from concurrent.futures import ProcessPoolExecutor, as_completed  # NEW
import multiprocessing as mp
import logging
from typing import List, Dict

# ── 3rd-party ─────────────────────────────────────────────────────────────
import numpy as np
import shutil
import typer
from tqdm import tqdm

# ── project modules ───────────────────────────────────────────────────────
from config import FS, INPUT_DIR, OUTPUT_DIR, TASK_MAP
from logger import get_logger
from utils.memory import safe_worker_count  # NEW  – memory-aware worker limiter
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
    subj: str,
    task: str,
    trial: str,
    seg_id: int,
    position: int,
    total: int,
    max_workers: int,
    use_cache: bool = False,
    cache_dir=None
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
        desc = f"{subj}-{task} Segment {int(seg_id)+1} Process"
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
                tau=10,
                emb_dim=2,
                save_plots=False,
                flatten=True,
                tqdm_progress=bar,
                max_workers=max_workers,
                use_cache=False,
                cache_dir=cache_dir
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

    # Decide strategy using the *first* key we can find in task_map
    first_trial_key = None
    for keys in task_map.values():
        if keys:
            first_trial_key = keys[0]
            break
    if first_trial_key is None or first_trial_key not in mat_dict:
        raise RuntimeError("Cannot determine sample size for strategy decision")

    sample_size = mat_dict[first_trial_key].size
    tentative_workers = safe_worker_count(sample_size, os.cpu_count(), 0.002)
    use_cache = tentative_workers < (os.cpu_count() - 2)

    logger.info(
        "%s – Strategy: %s (max_workers=%d, cpu=%d)",
        subj,
        "CACHE" if use_cache else "MEM",
        tentative_workers,
        os.cpu_count(),
    )

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
            logger.warning("%s – %s: missing %s", subj, task, trial_key)
            return None
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
        max_workers: int,
    ) -> np.ndarray | None:
        """Compute mean feature vector for *trial_key*."""

        if sig_source is None:
            return None

        # Load signal lazily depending on strategy
        if use_cache:
            sig_mm = np.load(sig_source, mmap_mode="r")  # type: ignore[arg-type]
        else:
            sig_mm = sig_source  # already an ndarray

        win_len, overlap = _choose_window(sig_mm.shape[1])
        step = win_len - overlap

        if use_cache:
            total_segments = 1 + max(0, (sig_mm.shape[1] - win_len) // step)
        else:
            segments = segment_signal(sig_mm, win_len, step)
            total_segments = len(segments)
            if total_segments == 0:
                logger.warning("%s – %s – %s: no segments", subj, task, trial_key)
                return None

        feature_vectors: List[np.ndarray] = []
        with tqdm(
            total=total_segments,
            desc=f"{subj}-{task} Segments",
            position=0,
            leave=True,
            dynamic_ncols=True,
        ) as bar:
            for seg_idx in range(total_segments):
                if use_cache:
                    start = seg_idx * step
                    end = start + win_len
                    seg = sig_mm[:, start:end]
                else:
                    seg = segments[seg_idx]

                ok, payload = analyze_segment(
                    seg,
                    fs,
                    subj,
                    task,
                    trial_key,
                    seg_idx,
                    1,
                    seg.shape[0],
                    max_workers,
                    use_cache=use_cache,
                    cache_dir=cache_dir
                )
                if ok:
                    feature_vectors.append(payload)
                else:
                    tqdm.write(f"{subj}–{task} failed: {payload}")
                bar.update()

        if feature_vectors:
            return np.mean(feature_vectors, axis=0)

        logger.error("%s – %s – %s: no data after QC", subj, task, trial_key)
        return None

    # ------------------------------------------------------------------
    # 2. Task‑level loop (sequential)
    # ------------------------------------------------------------------
    def _process_task(task: str, keys: List[str]) -> List[np.ndarray]:
        """Return list of mean feature vectors, one per trial."""

        trial_sources = {}
        # Prepare signals according to chosen strategy
        for k in keys:
            if use_cache:
                p = _prep_signal_to_cache(task, k)
                if p is not None:
                    trial_sources[k] = p
            else:
                if k not in mat_dict:
                    logger.warning("%s – %s: missing %s", subj, task, k)
                    continue
                trial_sources[k] = _load_signal_mem(task, k)

        # Use the tentative_workers already computed earlier
        max_workers = tentative_workers

        trial_vecs: List[np.ndarray] = []
        for k in keys:
            if k in trial_sources:
                vec = _process_one_trial(task, k, trial_sources[k], max_workers)
                if vec is not None:
                    trial_vecs.append(vec)
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
    if use_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)
    logger.debug("Finish processing %s", subj)
    return subj


# -------------------------------------------------------------------------
# 5. CLI – batch over all subjects
# -------------------------------------------------------------------------
@app.command(help="Extract 17-dim nonlinear features for all S*.mat files.")
def process(
        data_dir: Path = typer.Option(None, help="Input folder with S*.mat files"),
        output_dir: Path = typer.Option(None, help="Output folder for JSON / MAT"),
        fs: int = FS,
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
        _process_subject(f, fs, TASK_MAP, output_dir, save_mat)

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
