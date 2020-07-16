'''A module containing a set of window filters'''

import numpy as np

def rectangle(M):
    '''
    The rectangle window.
    Zero-phase: symmetric about 0
    M must be odd
    '''

    # calculate positive half only, flip and use for negative half
    wrange = np.arange(-M + 1, M)
    window = (np.abs(wrange) <= (M - 1) / 2.).astype(int)
    return wrange, window

def hann(M):
    wrange, recwindow = rectangle(M)
    raised_cosine = np.cos(np.pi / M * wrange)**2
    window = recwindow * raised_cosine
    return wrange, window

def hamming(M):
    wrange, recwindow = rectangle(M)
    alpha = 25 / 46
    beta = 1 - alpha
    window = recwindow * (alpha + (beta * np.cos(2 * np.pi / M * wrange)))
    return wrange, window

def mlt(M):
    '''Modulated lapped transform'''
    wrange = np.arange(-M + 1, M)
    window = np.sin((np.pi / 2 / M) * (wrange + 0.5))
    return wrange, window

def blackman(M, coefficients=None):
    wrange, recwindow = rectangle(M)
    if coefficients is None:
        coefficients = [0.42, 0.5, 0.08]
    costerm = lambda n: np.cos(n * np.pi * 2 / M * wrange)
    #blackman windows have 3 cosine terms (DOFs)
    costerms = np.array([np.ones(M * 2 - 1), costerm(1), costerm(2)])
    window = recwindow * np.dot(coefficients, costerms)
    return wrange, window

def blackman_harris(M):
    return blackman(M, coefficients=[0.4243801, 0.4973406, 0.00782793])

def barlett(M):
    pass

def poisson(M):
    pass

def hann_poisson(M):
    pass

def slepian(M):
    pass

def kaiser(M):
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
    w = blackman(count)
    w2 = blackman(count)
    plt.bar(w[0], w[1], width=0.1)
    plt.bar(w2[0], w2[1], width=0.1)
    plt.show()
