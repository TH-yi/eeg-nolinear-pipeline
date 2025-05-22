import numpy as np
from scipy.signal import welch


def bandpower(sig, fs, band):
    f, Pxx = welch(sig, fs, nperseg=1024)
    idx = (f >= band[0]) & (f <= band[1])
    return np.trapz(Pxx[idx], f[idx])
