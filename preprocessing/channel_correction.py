import numpy as np

CZ_IDX = 23          # zero‑based index for Cz

def cz_interpolation(sig: np.ndarray) -> np.ndarray:
    """
    Insert or replace Cz (index 23) with the average of its true neighbors
    from MATLAB Num_Ch_Corr.m.
    """

    # True neighbors used in MATLAB Num_Ch_Corr:
    neighbors = [6, 38, 27, 39, 56, 11, 52, 22]  # 0-based indexing

    sig = sig.copy()
    if sig.shape[0] == 63:
        # Insert a dummy row for Cz at position 23 (zero-based)
        sig = np.insert(sig, 23, 0.0, axis=0)
    sig[23, :] = sig[neighbors, :].mean(axis=0)
    return sig



def reorder_channels(sig: np.ndarray):
    """Placeholder – return signal unchanged. Extend to real montage if needed."""
    return sig
