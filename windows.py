'''A module containing a set of window filters'''

import numpy as np

import util

def hamming_spectrum(M, alpha, beta):
    omega = 2 * np.pi / M
    samplepoints = 1000
    domain_1 = np.linspace(-np.pi, np.pi, num=samplepoints, endpoint=False)
    domain_2 = np.linspace(-np.pi - omega, np.pi - omega, num=samplepoints, endpoint=False)
    domain_3 = np.linspace(-np.pi + omega, np.pi + omega, num=samplepoints, endpoint=False)
    transform = alpha * util.asinc(count, domain_1) + beta * util.asinc(count, domain_2) + beta * util.asinc(count, domain_3)

    spectrum = M * transform
    return spectrum

class window():
    '''Spectrum analysis window'''

    def __init__(self, count, phase, fs = None, **kwargs) -> None:
        self._count = None
        self._phase = None
        self._spectrum = None
        self._magnitude = None
        self._amplitude = None
        self._side_lob = None
        self._roll_off = None
        self._periodic = None # for overlap+add
        self._endpoint = False #  if count 


    @property
    def phase(self):
        return self._phase

    @property
    def spectrum(self):
        return self._spectrum

    @property
    def amplitude(self):
        return self._spectrum

    @property
    def magnitude(self):
        return self._magnitude

    def is_causal(self):
        return False


def analyzer():
    ''' Time/Frequency domain analysis of arbitrary spectrum windows
        frequency response (amplitude, magniuted, and phase)
    '''

def rectangle(M):
    '''
    The rectangle window.
    Zero-phase: symmetric about 0
    M must be odd
    
    properties:
        * zero crossings at integer multiples of Omega = 2*pi/M (frequency sampling interval for a length M DFT)
        * Main lobe width is 2*Omegaa = 4*pi / M
        * As M increases, main lobe narrows (better frequency resolution)
        * M has no effect on the height of the sidee lobes
        * First side lobe is only 13dB down from the main-lobe peak
        * Side lobes roll off at approximately 6dB per octave
        * A linear phase term arises when we shift the window to make it causal (shift of M-1/2 rad)
    '''

    # calculate positive half only, flip and use for negative half
    wrange = np.arange(-M + 1, M)
    window = (np.abs(wrange) <= (M - 1) / 2.).astype(float)
    _spectrum = M * util.asinc(count, np.linspace(-np.pi, np.pi, num=1000, endpoint=False))
    return wrange, window

def hamming_generalize(M, alpha, beta):
    pass

def hann(M, include_zeros=False, periodic=False, causal=True):
    ''' Also known as Hanning or raised-cosine window
        - Main lobe is 4*Omega wide, Omega = 2*pi / M
        - First side lobe is at -31dB
        - Side lobes roll off approximately at -18 dB per octave
    '''
    count = M
    start = 0
    end = M
    if not periodic:
        if include_zeros:
            count = M - 1
        else:
            count = M + 1
            start = 1
            end = M + 1

    
    wrange = np.arange(start, end)
    sign = -1 if causal else 1

    window = 0.5 * (1 + sign * np.cos(2 * np.pi * wrange / count))
    spectrum = M * util.asinc(M, np.linspace(-np.pi, np.pi, num=1000, endpoint=False))

    return wrange, window

def hamming(M, periodic=False):
    '''
        - Discontinuous "slam to zero" at endpoints
        - First side lob is 41 dB down
        - Roll off asymptotically -6 dB per octave
        - Side lobes are closer to equal ripple
    '''
    wrange = np.arange(M)
    count = M if periodic else M - 1

    #alpha = 25 / 46
    #beta = 1 - alpha
    alpha = 0.54
    beta = 0.46

    window = alpha - beta * np.cos(2 * np.pi * wrange / count)
    return wrange, window

def modulated_lapped_transform(M):
    '''
    Modulated lapped transform
        - Side lobes 24 dB down
        - Assymptotically optimal coding gain
        - Zero-phase-window transform has smallest moment of inertia over all windows
    '''
    wrange = np.arange(-M + 1, M)
    window = np.sin((np.pi / 2 / M) * (wrange + 0.5))
    return wrange, window

def blackman(M, coefficients=None):
    '''The Classic Blackman window. Provide coefficients to make it specifilized'''
    wrange, recwindow = rectangle(M)
    if coefficients is None:
        # classic Blackman window
        coefficients = [0.42, 0.5, 0.08]
    costerm = lambda n: np.cos(n * np.pi * 2 / M * wrange)
    #blackman windows have 3 cosine terms (DOFs)
    costerms = np.array([np.ones(M * 2 - 1), costerm(1), costerm(2)])
    window = recwindow * np.dot(coefficients, costerms)
    return wrange, window

def blackman_harris(M):
    return blackman(M, coefficients=[0.4243801, 0.4973406, 0.00782793])

def barlett(M):
    wrange, recwindow = rectangle(M)
    window = recwindow * ( 1 - 2 * np.abs(wrange) / (M - 1))
    return wrange, window

def poisson(M, alpha=10):
    wrange, recwindow = rectangle(M)
    window = recwindow * np.exp(2 * -alpha * np.abs(wrange) / (M - 1))
    return wrange, window

def hann_poisson(M, alpha=10):
    wrange, recwindow = rectangle(M)
    window = recwindow * (1 + np.cos(2 * np.pi * wrange / (M - 1))) * np.exp(2 * -alpha * np.abs(wrange) / (M - 1))
    return wrange, window

def slepian(M):
    pass

def kaiser(M, beta=10):
    pass

def dolph_chebyshev(M):
    pass

def gaussian(M):
    pass

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    count = 21
    #w = rectangle(count)
    #w = mlt(count)
    #w = hann_poisson(count)
    w = hamming(count)
    #print(w[1])
    plt.bar(w[0], w[1], width=0.1)

    #spectrum = np.fft.fft(w[1])
    #spectrum = count * util.asinc(count, np.linspace(-np.pi, np.pi, num=1000, endpoint=False))

    spectrum = hamming_spectrum(count, 0.54, 0.23)

    # normalized frequency (cycles / sample)
    normfreq = np.linspace(-0.5, 0.5, num=1000, endpoint=False)

    amplitude = spectrum**2
    magnitude = 10 * np.log10(amplitude)
    plt.figure()
    plt.plot(normfreq, spectrum)
    plt.figure()
    plt.plot(normfreq, np.sqrt(amplitude))
    plt.figure()
    plt.plot(normfreq, magnitude - np.nanmax(magnitude))
    #plt.plot(normfreq, magnitude)
    plt.ylim((-60, 0))

    plt.show()