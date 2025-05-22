import matplotlib.pyplot as plt
import numpy as np


def quick_timeseries(sig, fs, title="Signal"):
    """Plot first channel of signal for sanity check."""
    t = np.arange(sig.shape[1]) / fs
    plt.figure()
    plt.plot(t, sig[0])
    plt.xlabel("Time (s)")
    plt.ylabel("µV")
    plt.title(title)
    plt.show()
