from scipy.signal import butter, filtfilt


def butter_bandpass(data, low, high, fs, order=5):
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    b, a = butter(order, [low_n, high_n], btype="bandpass")
    return filtfilt(b, a, data)
