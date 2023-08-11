import numpy as np

def spectogram(x: list[float], nfft: int, window: list[float],
               noverlap: int = 0, fs: float = 1) -> None:
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
    nfft = int(nfft)
    nframes = int(nframes)
    print(nframes)
    print(nfft)

    X = np.zeros((nfft, nframes), dtype=complex)
    print(X.shape)

    # zero padding per frame
    zp = np.zeros(nfft - M)
    # input time offset
    xoff = 0 - Mo2

    xframe = np.zeros(M)

    for m in range(nframes):
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
        X[:,m] = np.fft.fft(xwzp, norm="ortho")
        # advance by hop size for the next (overlapping) window
        xoff += nhop

    t = np.arange(nframes) * nhop / fs
    f = 0.001 * np.arange(nfft) * fs / nfft

    return X, (t, f)

def test_spectogram():
    N = 1024
    M = 55
    Nfft = 256
    x = util.chirp(np.arange(N) / N, 80, 10, N)
    m = np.ones(M) / np.sqrt(M)

    S, (t, f) = spectogram(x, Nfft, m, 0)
    plt.imshow(np.abs(S[:Nfft//2])**2,cmap='gray_r',
               extent=(0, 1, 0, np.pi / 2), origin='lower',
               aspect=1/np.pi)
    plt.savefig("spectogram.svg")
    plt.figure()
    plt.plot(x)
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import util
    import windows

    M = 17
    R = (M - 1) // 2
    win = np.ones(M) / np.sqrt(M)
    #_, win = windows.hamming(M)
    #win[-1] = 0

    N=M*60
    Nfft=128
    sig, comp = util.chirplet(N, 0, 0.5, analytic=False)
    plt.plot(comp[0], alpha=0.4)
    plt.plot(np.real(comp[1]), alpha=0.2)
    plt.plot(np.imag(comp[1]), alpha=0.2)
    plt.plot(np.real(sig))
    plt.figure()

    S, (t, f) = spectogram(np.real(sig), Nfft, win)
    plt.imshow(np.abs(S[:Nfft//2])**2,cmap='gray_r',
               extent=(0, 1, 0, np.pi / 2), origin='lower',
               aspect=1/np.pi)
    plt.savefig("spectogram.svg")
    plt.show()
