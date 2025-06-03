import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis

# Non‑linear dynamics & entropy
import nolds  # corr_dim, lyap_r, etc.
import antropy as ant  # sample_entropy, permutation_entropy

# Optional: wavelet entropy
try:
    import pywt  # nosec B402 – used only for entropy calc
except ImportError:  # pragma: no cover
    pywt = None
    warnings.warn(
        "PyWavelets not installed — wavelet entropy will be returned as NaN. Run `pip install PyWavelets`. ")

# Optional: RQA (Recurrence Quantification Analysis)
try:
    from pyrqa.time_series import TimeSeries
    from pyrqa.settings import Settings
    from pyrqa.metric import EuclideanMetric
    from pyrqa.computation import RQAComputation

    HAS_PYRQA = True
except ImportError:  # pragma: no cover
    HAS_PYRQA = False
    warnings.warn(
        "pyrqa not installed — RQA features will be NaN. `pip install pyrqa` to enable.")

# Plot helper (moved to separate module)
try:
    # When nonlinear_analysis.py lives in the same package as plots.py
    from .plots import generate_channel_plots  # type: ignore
except ImportError:
    # Fallback to absolute import if script executed standalone
    from plots import generate_channel_plots  # type: ignore

__all__ = ["nonlinear_analysis"]

# -----------------------------------------------------------------------------
# Helper functions (private)
# -----------------------------------------------------------------------------


def _wavelet_entropy(sig: np.ndarray, wavelet: str = "db4", level: int | None = None) -> float:
    if pywt is None:
        return np.nan
    coeffs = pywt.wavedec(sig, wavelet=wavelet, level=level)
    energies = np.array([np.sum(c ** 2) for c in coeffs])
    if energies.sum() == 0:
        return 0.0
    probs = energies / energies.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _bandpower(sig: np.ndarray, fs: float, fmin: float, fmax: float, nperseg: int = 1024) -> float:
    f, Pxx = welch(sig, fs=fs, nperseg=min(nperseg, len(sig)))
    idx = np.logical_and(f >= fmin, f <= fmax)
    if not np.any(idx):
        return 0.0
    return float(np.trapz(Pxx[idx], f[idx]))


def _power_signal(sig: np.ndarray) -> float:
    return float(np.mean(sig ** 2))


def _rqa_features(sig: np.ndarray, dim: int, tau: int):
    if not HAS_PYRQA:
        return np.nan, np.nan, np.nan, np.nan
    ts = TimeSeries(sig.tolist(), embedding_dimension=dim, time_delay=tau)
    settings = Settings(ts, analysis_type='Classic', neighbourhood=EuclideanMetric(), similarity_threshold=0.1)
    comp = RQAComputation.create(settings, verbose=False).run()
    return (
        comp.recurrence_rate,
        comp.determinism,
        comp.entropy_diagonal_lines,
        comp.average_diagonal_line,
    )


# -----------------------------------------------------------------------------
# Main feature extraction API
# -----------------------------------------------------------------------------


def nonlinear_analysis(
    signal2: np.ndarray,
    fs: float = 500.0,
    tau: int = 10,
    emb_dim: int = 2,
    save_plots: bool = False,
    plot_dir: str | Path = "plots",
    channel_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Python equivalent of MATLAB *Nonlinear_Analysis*.

    Replicates all 17 channel‑wise non‑linear features:

    1. Correlation Dimension (Grassberger‑Procaccia)
    2. Higuchi Fractal Dimension
    3. Largest Lyapunov Exponent (Rosenstein)
    4. Wavelet Shannon Entropy
    5. Kurtosis (Fisher‑corrected off)
    6. Mean Signal Power
    7. Sample Entropy (m=emb_dim, r=0.2·std)
    8. Permutation Entropy (m=emb_dim, τ=tau)
    9‑13. Band Power δ/θ/α/β/γ (0.1‑4 / 4‑8 / 8‑13 / 13‑30 / 30‑100 Hz)
    14‑17. RQA – RR, DET, ENTR, L (radius=0.1)

    Parameters are identical to the MATLAB script defaults (fs=500 Hz, tau=10, emb_dim=2).
    """

    if signal2.ndim != 2:
        raise ValueError("`signal2` must be 2‑D (channels × samples)")

    n_channels, _ = signal2.shape
    channel_names = channel_names or [f"ch{c:02d}" for c in range(n_channels)]
    plot_dir = Path(plot_dir)

    feats_per_ch: list[list[float]] = []

    for c in range(n_channels):
        x = np.asarray(signal2[c], dtype=float)
        ch_name = channel_names[c]

        # 1. Correlation dimension
        try:
            f1 = nolds.corr_dim(x, emb_dim)
        except Exception as e:
            warnings.warn(f"corr_dim failed on {ch_name}: {e}")
            f1 = np.nan

        # 2. Higuchi FD
        try:
            f2 = nolds.higuchi_fd(x)
        except Exception as e:
            warnings.warn(f"higuchi_fd failed on {ch_name}: {e}")
            f2 = np.nan

        # 3. Lyapunov exponent
        try:
            f3 = nolds.lyap_r(x, emb_dim=emb_dim, lag=tau)
        except Exception as e:
            warnings.warn(f"lyap_r failed on {ch_name}: {e}")
            f3 = np.nan

        # 4. Wavelet entropy
        f4 = _wavelet_entropy(x)

        # 5. Kurtosis
        f5 = kurtosis(x, fisher=False, bias=False)

        # 6. Power
        f6 = _power_signal(x)

        # 7. Sample entropy (r = 0.2·std as per MATLAB)
        try:
            r_val = 0.2 * np.std(x)
            f7 = ant.sample_entropy(x, order=emb_dim, r=r_val)
        except Exception:
            f7 = np.nan

        # 8. Permutation entropy
        try:
            f8 = ant.permutation_entropy(x, order=emb_dim, delay=tau, normalize=True)
        except Exception:
            f8 = np.nan

        # 9‑13. Band powers
        f9  = _bandpower(x, fs, 0.1, 4)
        f10 = _bandpower(x, fs, 4, 8)
        f11 = _bandpower(x, fs, 8, 13)
        f12 = _bandpower(x, fs, 13, 30)
        f13 = _bandpower(x, fs, 30, 100)

        # 14‑17. RQA metrics
        f14, f15, f16, f17 = _rqa_features(x, emb_dim, tau)

        feats_per_ch.append([
            f1, f2, f3, f4, f5, f6,
            f7, f8, f9, f10, f11, f12, f13,
            f14, f15, f16, f17,
        ])

        # Diagnostic plots
        if save_plots:
            generate_channel_plots(x, fs, tau, emb_dim, plot_dir, ch_name)

    feats_arr = np.asarray(feats_per_ch, dtype=float)  # (ch, 17)
    return feats_arr.reshape(1, -1)
