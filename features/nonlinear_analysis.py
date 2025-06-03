"""nonlinear_analysis.py
================================
Python re‑implementation of MATLAB `Nonlinear_Analysis.m` **with full feature parity**
(17 non‑linear features per EEG channel) + optional diagnostic plots.

The module exposes a single public function:

    nonlinear_analysis(signal2, fs=500, tau=10, emb_dim=2,
                       save_plots=True, plot_dir="plots", channel_names=None)

and an `__all__` list so that `from nonlinear_analysis import *` only exports
that API.

Dependencies (install with pip):
    numpy, scipy, nolds, antropy, matplotlib, pyts, pywavelets (optional), pyrqa (optional)
"""

from __future__ import annotations

import warnings
import os, shutil, tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis

# Non‑linear dynamics & entropy libraries
import nolds        # corr_dim, lyap_r, etc.
import antropy as ant  # sample_entropy, permutation_entropy
from features.exact_entropy import sample_entropy, permutation_entropy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# from tqdm import tqdm
# Optional: wavelet entropy (PyWavelets)
try:
    import pywt  # nosec B402 – only used for entropy calculation
except ImportError:  # pragma: no cover
    pywt = None
    warnings.warn(
        "PyWavelets not installed — Wavelet entropy will be returned as NaN. Run `pip install PyWavelets`.")

# Optional: Recurrence Quantification Analysis (pyrqa)
try:
    from pyrqa.time_series import TimeSeries
    from pyrqa.settings import Settings
    from pyrqa.analysis_type import Classic
    from pyrqa.neighbourhood import FixedRadius
    from pyrqa.computation import RQAComputation
    from pyrqa.metric import EuclideanMetric
    PYRQA_OK = True
except ImportError:  # pragma: no cover
    PYRQA_OK = False
    warnings.warn(
        "pyrqa not installed — RQA features will be NaN. Run `pip install pyrqa` to enable.")

# Optional diagnostic plots (delegated to a helper module)
try:
    from .plots import generate_channel_plots  # when part of a package
except ImportError:  # standalone script fallback
    try:
        from plots import generate_channel_plots  # type: ignore
    except ImportError:  # pragma: no cover
        def generate_channel_plots(*_args, **_kwargs):  # type: ignore
            return  # plotting silently disabled if helper not found

__all__ = ["nonlinear_analysis"]

# -----------------------------------------------------------------------------
# Internal helper functions
# -----------------------------------------------------------------------------

def _wavelet_entropy(sig: np.ndarray, wavelet: str = "db4", level: int | None = None) -> float:
    """Shannon entropy of wavelet energy distribution (as in MATLAB `wentropy`)."""
    if pywt is None:
        return float("nan")
    coeffs = pywt.wavedec(sig, wavelet=wavelet, level=level)
    energies = np.array([np.sum(c ** 2) for c in coeffs])
    if energies.sum() == 0:
        return 0.0
    probs = energies / energies.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _bandpower(sig: np.ndarray, fs: float, fmin: float, fmax: float, nperseg: int = 1024) -> float:
    """Relative bandpower via Welch PSD integration (equivalent to MATLAB `bandpower`)."""
    f, Pxx = welch(sig, fs=fs, nperseg=min(nperseg, len(sig)))
    idx = np.logical_and(f >= fmin, f <= fmax)
    if not np.any(idx):
        return 0.0
    return float(np.trapz(Pxx[idx], f[idx]))


def _power_signal(sig: np.ndarray) -> float:
    """Mean signal power (µV²) — MATLAB `powersignal`."""
    return float(np.mean(sig ** 2))


def _rqa_features(x: np.ndarray, m: int, tau: int):
    """
    Return RR, DET, ENTR, L for a 1-D signal `x`.
    Falls back to (np.nan, …) if PyRQA is absent.
    """
    if not PYRQA_OK:
        return (np.nan,)*4
    ts = TimeSeries(x.reshape(-1, 1), embedding_dimension=m, time_delay=tau)
    settings = Settings(
        ts,
        analysis_type=Classic,
        neighbourhood=FixedRadius(0.1),   # 等价 MATLAB RPplot_FAN(...,10,0)
        similarity_measure=EuclideanMetric(),
    )
    result = RQAComputation.create(settings).run()
    return (
        result.recurrence_rate,
        result.determinism,
        result.entropy_diagonal_lines,
        result.average_diagonal_line,
    )

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def _features_per_channel(
    x: np.ndarray,
    *,
    fs: float,
    tau: int,
    emb_dim: int,
    save_plots: bool,
    plot_dir: Path,
    ch_label: str,
) -> list[float]:
    """Worker 进程执行的函数，必须定义在模块最外层以便 pickle."""
    # ── 1. Correlation Dimension ────────────────────────────────────
    try:
        f1 = nolds.corr_dim(x, emb_dim)
    except Exception as exc:
        warnings.warn(f"corr_dim failed on {ch_label}: {exc}")
        f1 = float("nan")

    # ── 2. Higuchi Fractal Dimension ────────────────────────────────
    try:
        f2 = ant.higuchi_fd(x)
    except Exception as exc:
        warnings.warn(f"higuchi_fd failed on {ch_label}: {exc}")
        f2 = float("nan")

    # ── 3. Largest Lyapunov Exponent ────────────────────────────────
    try:
        f3 = nolds.lyap_r(x, emb_dim=emb_dim, lag=tau)
    except Exception as exc:
        warnings.warn(f"lyap_r failed on {ch_label}: {exc}")
        f3 = float("nan")

    # ── 4. Wavelet Shannon Entropy ──────────────────────────────────
    f4 = _wavelet_entropy(x)

    # ── 5. Kurtosis (Fisher flag off to match MATLAB) ───────────────
    f5 = kurtosis(x, fisher=False, bias=False)

    # ── 6. Mean signal power ────────────────────────────────────────
    f6 = _power_signal(x)

    # 7 ─ Sample Entropy (r = 0.2·σ) ────────────────────────────────
    f7 = sample_entropy(x, m=emb_dim, tau=tau)

    # 8 ─ Permutation Entropy ───────────────────────────────────────
    f8 = permutation_entropy(x, m=emb_dim, tau=tau, normalize=True)

    # ── 9-13. Band powers δ, θ, α, β, γ ─────────────────────────────
    f9  = _bandpower(x, fs, 0.1, 4)
    f10 = _bandpower(x, fs, 4, 8)
    f11 = _bandpower(x, fs, 8, 13)
    f12 = _bandpower(x, fs, 13, 30)
    f13 = _bandpower(x, fs, 30, 100)

    # ── 14-17. Recurrence Quantification Analysis ──────────────────
    f14, f15, f16, f17 = _rqa_features(x, emb_dim, tau)

    # ── Optional plots ─────────────────────────────────────────────
    if save_plots:
        generate_channel_plots(x, fs, tau, emb_dim, plot_dir, ch_label)

    return [
        f1, f2, f3, f4, f5, f6,
        f7, f8, f9, f10, f11, f12, f13,
        f14, f15, f16, f17,
    ]


def nonlinear_analysis(
    signal2: np.ndarray,
    *,
    fs: float = 500.0,
    tau: int = 10,
    emb_dim: int = 2,
    save_plots: bool = True,
    plot_dir: str | Path = "plots",
    channel_names: list[str] | None = None,
    flatten: bool = True,
    tqdm_progress: tqdm | None = None,
    max_workers: int | None = 1,
    use_cache: bool = False,
    cache_dir=None
) -> np.ndarray:

    if signal2.ndim != 2:
        raise ValueError("`signal2` must be 2-D (channels × samples)")

    n_channels, _ = signal2.shape
    channel_names = channel_names or [f"ch{c:02d}" for c in range(n_channels)]
    plot_dir = Path(plot_dir)


    temp_dir = None
    if use_cache:
        #print(f"use cache -- cpu count: {os.cpu_count()}, max workers: {max_workers }")
        import uuid
        if cache_dir:
            temp_dir = cache_dir / "__nolinear_cache" / f"nl_cache_{uuid.uuid4().hex}"
        else:
            temp_dir = Path(tempfile.gettempdir()) / f"nl_cache_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)  # ensure uniqueness
        for ch_idx in range(n_channels):
            np.save(temp_dir / f"ch_{ch_idx:02d}.npy", signal2[ch_idx])
        del signal2

    def load_channel(ch_idx):
        if use_cache:
            return np.load(temp_dir / f"ch_{ch_idx:02d}.npy", mmap_mode="r")
        else:
            return signal2[ch_idx]

    # ─── Main execution ─────────────────────────────────────────────
    feat_rows = [None] * n_channels
    if not max_workers or max_workers <= 1:
        for ch_idx in range(n_channels):
            x = load_channel(ch_idx)
            res = _features_per_channel(
                x,
                fs=fs, tau=tau, emb_dim=emb_dim,
                save_plots=save_plots, plot_dir=plot_dir,
                ch_label=channel_names[ch_idx],
            )
            feat_rows[ch_idx] = res
            if tqdm_progress is not None:
                tqdm_progress.update()
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as exe:
            fut_to_idx = {
                exe.submit(
                    _features_per_channel,
                    load_channel(ch_idx),
                    fs=fs, tau=tau, emb_dim=emb_dim,
                    save_plots=save_plots, plot_dir=plot_dir,
                    ch_label=channel_names[ch_idx],
                ): ch_idx
                for ch_idx in range(n_channels)
            }
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    feat_rows[idx] = fut.result()
                except Exception as exc:
                    warnings.warn(f"Worker on {channel_names[idx]} failed: {exc}")
                    raise
                finally:
                    if tqdm_progress is not None:
                        tqdm_progress.update()

    # ─── Clean up temp cache ────────────────────────────────────────
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir)

    # ─── Return result ──────────────────────────────────────────────
    feat_arr = np.asarray(feat_rows, dtype=float)
    return feat_arr.reshape(1, -1) if flatten else feat_arr

