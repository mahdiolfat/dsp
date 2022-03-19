'''A module containing a set of window filters'''

import numpy as np

import util

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
    return wrange, window

def hamming_generalize(M, alpha, beta):
    pass

def hann(M):
    ''' Also known as Hanning or raised-cosine window
        - Main lobe is 4*Omega wide, Omega = 2*pi / M
        - First side lobe is at -31dB
        - Side lobes roll off approximately at -18 dB per octave
    '''
    wrange, recwindow = rectangle(M)
    raised_cosine = np.cos(np.pi / M * wrange)**2
    window = recwindow * raised_cosine
    return wrange, window

def hamming(M):
    '''
        - Discontinuous "slam to zero" at endpoints
        - Firstt side lob is 51 dB down
        - Roll off asymptotically -6 dB per octave
        - Side lobes are closer to equal ripple
    '''
    wrange, recwindow = rectangle(M)
    alpha = 25 / 46
    beta = 1 - alpha
    window = recwindow * (alpha + (beta * np.cos(2 * np.pi / M * wrange)))
    return wrange, window

def modulated_lapped_transform(M):
    '''
    Modulated lapped transform
        - Side lobes 24 dB down
        - Assymptotically optimal coding gain
        - 
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

    count = 11
    w = rectangle(count)
    #w = mlt(count)
    #w = hann_poisson(count)
    #w = hann(3)
    #print(w[1])
    plt.bar(w[0], w[1], width=0.1)

    #spectrum = np.fft.fft(w[1])
    spectrum = count * util.asinc(count, np.linspace(-np.pi, np.pi, num=100, endpoint=False))
    plt.figure()

    normfreq = np.linspace(-np.pi, np.pi, num=100, endpoint=False)
    plt.plot(normfreq, spectrum)
    plt.show()