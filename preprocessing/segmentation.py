import numpy as np


def segment_signal(sig: np.ndarray, win_len: int, overlap: int):
    """Segment signal (ch × samples) into windows with given overlap."""
    n_samples = sig.shape[1]
    step = win_len - overlap
    starts = np.arange(0, n_samples - win_len + 1, step, dtype=int)
    segments = np.stack([sig[:, s : s + win_len] for s in starts])
    return segments
