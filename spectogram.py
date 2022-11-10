import numpy as np

def spectogram(x: list[float], nfft: int, window: list[float], noverlap: int = 0, fs: float = 1) -> None:
    M = len(window)

    if M < 2:
        raise ValueError("Window is expected to be completely defined, size of 2 is unacceptable")

    if noverlap < 0:
        raise ValueError("Noverlap must be non negative")

    if len(x) < M:
        # zero pad to match window size
        x = np.concatenate((x, np.zeros(M-len(x))))

    Modd = M % 2
    Mo2 = (M - Modd) // 2
    nhop = M - noverlap
    nx = len(x)
    nframes = 1 + np.ceil(nx / nhop)

    X = np.zeros((nfft, nframes))
    # zero padding per frame
    zp = np.zeros(nfft - M)
    # input time offset
    xoff = 0 - Mo2

    xframe = np.zeros(M)

    for m in nframes:
        if xoff < 0:
            # partial frame
            xframe[:xoff + M] = x[:xoff + M]
        else:
            if xoff + M > nx:
                xframe = np.concatenate((x[xoff:nx], np.zeros(xoff + M - nx)))
            else:
                xframe = x[xoff:xoff + M]
        # windowing
        xw = np.array(window) * xframe
        # zeropad in zero-phase form
        xwzp = np.concatenate((xw[Mo2:M], zp, xw[:Mo2]))
        X[:,m] = np.fft(xwzp)
        # advance by hop size for the next (overlapping) window
        xoff += nhop

    t = np.arange(nframes) * nhop / fs
    f = 0.001 * np.arange(nfft) * fs / nfft

    return X, (t, f)

def test_spectogram():
    pass

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_spectogram()
