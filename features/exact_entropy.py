import numpy as np
from math import factorial, log


def _phi(signal: np.ndarray, m: int, tau: int, r: float) -> int:
    """
    Count the number of template vector pairs (length m) whose
    Chebyshev distance ≤ r.  Includes *overlapping* pairs (as in Richman & Moorman, 2000).
    """
    n = signal.size - (m - 1) * tau
    if n <= 1:
        return 0
    # build embedding matrix: shape = (n, m)
    idx = np.arange(m) * tau
    emb = np.lib.stride_tricks.sliding_window_view(signal, n)[idx].T
    count = 0
    for i in range(n - 1):
        dist = np.max(np.abs(emb[i + 1:] - emb[i]), axis=1)
        count += np.sum(dist <= r)
    return count


def sample_entropy(signal: np.ndarray, m: int, tau: int = 1, r: float | None = None) -> float:
    """
    Matlab-compatible SampEn for one (m,tau).  Returns NaN if no matches.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    if r is None:
        r = 0.2 * np.std(signal, ddof=1)
    A = _phi(signal, m + 1, tau, r)
    B = _phi(signal, m,     tau, r)
    return -log(A / B) if (A > 0 and B > 0) else np.nan


def permutation_entropy(
    signal: np.ndarray,
    m: int,
    tau: int = 1,
    normalize: bool = True,
) -> float:
    """
    Matlab-compatible PermEn ('Typex' == 'none').
    """
    signal = np.asarray(signal, dtype=float).ravel()
    n = signal.size - (m - 1) * tau
    if n <= 0 or m < 2:
        return 0.0
    # build embedding vectors (rows)
    idx = np.arange(m) * tau
    emb = np.lib.stride_tricks.sliding_window_view(signal, n)[idx].T
    # ordinal patterns
    patterns = np.argsort(emb, axis=1)
    # histogram
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p))
    if not normalize:
        return H
    return H / log(factorial(m))
