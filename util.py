'''Typical function used in (statistical)(digital) signal processing'''

import numpy as np

def gain_to_db(gain):
    return 20 * np.log10(gain)

def asinc(M, w):
    '''The Aliasied Sinc function, defined as the sampled Rectangle Window.
       asinc becomes the sinc function in the limit as sampling rate -> inf'''
    ''' TODO: handle the case where w = 0'''
    denom = w / 2 
    num = denom * M
    return np.sin(num) / (M * np.sin(denom))

def fftshift(signal):
    '''take fs/2 as negative frequency and dc (0) as positive to balance out
      frequency components'''
    shifted = np.empty_like(signal)
    midpoint = int(len(signal) / 2)
    adj = 1 if len(signal) % 2 else 0
    shifted[:midpoint] = signal[midpoint+adj:]
    shifted[midpoint:] = signal[:midpoint+adj]
    return shifted

def dft(signal):
    '''
        - discrete time
        - finit frequency
        - discrete frequency
        - finit time
    '''

    wrange = np.arange(len(signal))

def nextpow2(N) -> int:
    return int(2**np.ceil(np.log2(N)))

def bessel_first(x, terms=None):
    return np.i0(x)

def quad_peak(ym1, y0, yp1):
    '''
        return the extremum location peak, height y, and half-curvature a of a parabolic fit through three points
        parabola is  given by y(x) = a * (x - p)**2 + b
        parabola is given by y(x)=y=m1, y(0)=y0, y(1)=yp1
        p ranges [-1/2 to 1/2]
    '''
    p = (yp1 - ym1) / (2 * (2 * y0 - yp1 - ym1))
    y = y0 - 0.25 * (ym1 - yp1) * p
    a = 0.5 * (ym1 - 2 * y0 + yp1)

    return p, y, a

def quad_peak_freq_estimate(k, ym1, y0, yp1, N, fs=1):
    '''
        - k is the estimated bin with max frequency
        - N is FFT size
        - fs is sampling frequency

        return frequency in fs Hz
    '''
    p, _, _ = quad_peak(ym1, y0, yp1)
    return (k + p) * fs / N

def unwrap_spectral_phase(phase):
    '''
    unwrap the phase to make it continuous
    starting from dc phase
    '''

    N = len(phase)
    unwrapped = np.zeros((N))
    phase1 = phase[0]
    unwrapped[0] = phase1

    phase0 = 0
    pi2 = np.pi * 2

    threshold = np.pi - np.finfo().eps

    for idx in np.arange(1, N):
        phasenext = phase[idx] + phase0
        phasediff = phasenext - phase1
        while phasediff > threshold:
            phase0 -= pi2
            phasediff -= pi2

        while phasediff < -threshold:
            phase0 += pi2
            phasediff += pi2
        phasenext = phase[idx] + phase0
        phase1 = phasenext
        unwrapped[idx] = phasenext

    return unwrapped