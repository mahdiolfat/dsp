'''
Module for FFT based filter bank implementations

The Short-Time Fourier Transform (STFT) implements a "uniform" filter bank.

Summing adjacent filter-bank signals sums the corresponding pass-bands to create a
wider pass-baand. However, we must be able to step thee FFT through time and compute
properly sampled time-domain filter-bank signals.

The wider pass-band created by adjacent-channel summing requires a higher sampling
rate in the time domain to avoid aliasing. As a result, the maximum STFT "hop size" is
limited by the widest pass-band in the filter bank.

'''

import numpy as np

def cconvr(W, X):
    wc = np.fft.fft(W)
    xc = np.fft.fft(X)
    yc = wc * xc
    Y = np.real(np.fft.ifft(yc))

    return Y

def alias(x: list, L: int):
    Nx = len(x)
    # aliasing-partition length
    Np = Nx // L
    y = np.zeros(Np)
    # TODO: optimize with reshape() and sum()
    for i in range(len(L)):
        y += x[i*Np:(i+1)*Np]

    return y

def filter_bank(X, k):
    '''Compute the full-rate time-domain signal corresponding to band k
        - X is the FFT of the current frame of data (band k) in the STFT
    '''

    N = len(X)
    x = np.array((N // 2, N))

    # lower/higher spectral samples for band k
    lsn = N // 4
    hsn = int(np.ceil(N / 2))
    bandK = X[lsn:hsn]
    z1 = np.zeros(lsn-1)
    z2 = np.zeros(N - hsn)
    bandKzp = np.concatenate((z1, bandK, z2))
    # output signal vector (length N) for the kth filter bank channel for the
    # current time-domain STFT window.
    # the full-rate time domain signal 
    x[k, :] = np.fft.ifft(bandKzp)

    # Nk is the number of FFT bins in band k
    Nk = hsn - lsn + 1

if __name__ == "__main__":
    bands = [0.1, 0.12, 0.15, 0.19, 0.25, 0.4]
