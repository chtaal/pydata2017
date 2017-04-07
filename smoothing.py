import numpy as np
from scipy import signal as sig


def st_rfft(x, frame_size, hop_size, fft_size=None):
    """
    short-time real fast fourier transform. Will apply the rfft to short-time overlapping segments of the input
    signal.

    Parameters
    ----------
    x : np.array
        input signal
    frame_size : int
        length of the window (samples)
    hop_size : int
        time between two consecutive windows (samples)
    fft_size : int
        size of the fft window (samples)

    Returns
    -------
    narray
        2d np.array where its shape corresponds to (frequency bins, time frames)

    """
    if not fft_size:
        fft_size = frame_size
    idx_starts = np.arange(0, len(x)-frame_size, hop_size, dtype='int')
    xf = np.zeros([int(fft_size/2+1), len(idx_starts)], dtype=np.complex)
    win = np.sqrt(sig.hann(frame_size, False))

    for cnt, idx_start in enumerate(idx_starts):
        idx_stop = idx_start + frame_size
        xtemp = np.fft.rfft(x[idx_start:idx_stop]*win, n=fft_size)
        xf[:, cnt] = xtemp

    return xf


def polyfit_window(x, window_length=5, deg=1, deriv=0, delta=1, pos=None):
    """
    Applies a polynomial fit in a sliding window to an input signal. This has the same functionality as a 
    Savitzky-Golay filter. 
    
    Parameters
    ----------
    x: narray
        input signal
    window_length: int
        number of samples used in the sliding window
    deg : int
        degree of the polynomial
    deriv : int
        order of the derivative used
    delta : float
        sampling period
    pos : int
        evaluation position of the polynomial within the window

    Returns
    -------
    narray
        Filtered signal
    """
    if not pos:
        pos = int(window_length/2)+1
    num_samples = len(x)
    idx = np.arange(window_length)
    x_out = np.zeros(num_samples)

    x_padded = np.concatenate([np.zeros(window_length-1), x])

    for frame_start in np.arange(num_samples):
        x_frame = x_padded[idx + frame_start]
        p = np.polyfit(idx*delta, x_frame, deg=deg)
        p = np.polyder(p, m=deriv)
        x_out[frame_start] = np.polyval(p, idx[pos]*delta)

    return x_out