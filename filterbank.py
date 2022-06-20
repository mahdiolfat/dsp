'''
Module for FFT based filter bank implementations

The Short-Time Fourier Transform (STFT) implements a "uniform" filter bank.

Summing adjacent filter-bank signals sums the corresponding pass-bands to create a
wider pass-baand. However, we must be able to step thee FFT through time and compute
properly sampled time-domain filter-bank signals.

The wider pass-band created by adjacent-channel summing requires a higher sampling
rate in the time domain to avoid aliasing. AAs a result, the maximum STFT "hop size" is
limited by the widest pass-band in the filter bank.

'''

import numpy as np

def filter_bank(X):
    '''Compute the full-rate time-domain signal corresponding to band k
        - X is the FFT of the current frame of data (band k) in the STFT
    '''

    N = len(X)

    # lower spectral samples for band k
    lsn = N / 2
    hsn = N / 2
    bandK = X[lsn:hsn]
    z1 = np.zeros(lsn-1)
    z2 = np.zeros(N - hsn)
    bandKzp = np.concatenate((z1, bandK, z2))
    # output signal vector (length N) for the kth filter bank channel for the
    # current time-domain STFT window.
    x(k, :) = np.fft.ifft(bandKzp)