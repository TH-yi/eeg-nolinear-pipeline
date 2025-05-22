import numpy as np

CZ_IDX = 23          # zero‑based index for Cz

def cz_interpolation(sig: np.ndarray, neighbors=None):
    """Insert or replace Cz channel (index 23) with the mean of its neighbors."""
    if neighbors is None:
        neighbors = [19, 25, 20, 28, 37, 43, 38]  # example neighbors
    if sig.shape[0] == 63:                       # Cz missing → insert
        sig = np.insert(sig, 23, 0, axis=0)      # create dummy Cz row
    sig = sig.copy()
    sig[23, :] = sig[neighbors, :].mean(axis=0)
    return sig


def reorder_channels(sig: np.ndarray):
    """Placeholder – return signal unchanged. Extend to real montage if needed."""
    return sig
