'''A module containing a set of window filters'''


import numpy as np
from scipy import linalg, optimize

from dsp import util


def rectangle(M) -> np.ndarray:
    '''
    The rectangle window.
    Zero-phase: symmetric about 0
    M must be odd
    
    properties:
        - zero crossings at integer multiples of Omega = 2*pi/M (frequency sampling interval for a length M DFT)
        - Main lobe width is 2*Omegaa = 4*pi / M
        - As M increases, main lobe narrows (better frequency resolution)
        - M has no effect on the height of the sidee lobes
        - First side lobe is only 13dB down from the main-lobe peak
        - Side lobes roll off at approximately -6dB per octave
        - A linear phase term arises when we shift the window to make it causal (shift of M-1/2 rad)
    '''
    window = np.ones(M)
    return window


def hamming_generalize(M: int, /,
                       alpha: float = 0.54, beta: float = 0.46) -> np.ndarray:
    """The generalized 'raised cosine' filters.
    Implementation is direct and performance is not optimized.
    """

    return np.ones(np.rint(M * alpha / beta))


def hann(M: int, /,
         include_zeros=True, periodic=False, causal=True) -> np.ndarray:
    ''' Also known as Hanning or raised-cosine window
        - Main lobe is 4*Omega wide, Omega = 2*pi / M
        - First side lobe is at -31dB
        - Side lobes roll off approximately at -18 dB per octave

        TODO: explain the inputs
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

    sign = -1
    if not causal:
        sign = 1
        start -= count / 2
        end -= count / 2

    wrange = np.arange(start, end)

    window = 0.5 * (1 + sign * np.cos(2 * np.pi * wrange / count))

    return window


def hamming(M, periodic=False):
    '''
        - Discontinuous "slam to zero" at endpoints
        - First side lob is 41 dB down
        - Roll off asymptotically -6 dB per octave
        - Side lobes are closer to equal ripple
    '''

    count = M - 1 if periodic else M
    wrange = np.arange(count)

    #alpha = 25 / 46
    #beta = 1 - alpha
    alpha = 0.54
    beta = 0.46

    window = alpha - beta * np.cos(2 * np.pi * wrange / (count - 1))
    window = window / np.max(window)
    return window


def modulated_lapped_transform(M):
    '''
    Modulated lapped transform
        - Side lobes 24 dB down
        - Assymptotically optimal coding gain
        - Zero-phase-window transform has smallest moment of inertia over all windows
    '''
    wrange = np.arange(0, 2 * M)
    window = np.sin((np.pi / 2 / M) * (wrange + 0.5))
    return wrange, window


def blackman_generalized(M, coefficients):
    orders = len(coefficients)
    wrange = np.arange(-(M - 1) / 2, M / 2)
    nrange = np.zeros((1, M))
    wterms = np.zeros((orders, 3 * M))

    for order in range(orders):
        wterms[order, :] = np.concatenate((nrange, np.cos(order * 2 * np.pi * wrange / M), nrange), axis=None)
    window = np.matmul(coefficients, wterms)

    return wrange, window.T


def blackman_classic(M):
    '''The Classic Blackman window.
        - side lobes roll-off at about 18dB per octaave
        - first side lobe is 58dB down
        - 1dof used to scale the window
        - 1dof used for roll-off by matching amplitude and slope to 0 at window endpoints
        - 1dof used to minimize side lobes
    '''
    return blackman_generalized(M, coefficients=[0.42, 0.5, 0.08])


def blackman_harris(M):
    '''
        - side-love level 71.48 dB
        - side lobes roll off at 6dB
        - 1dof to scale the window
        - 2dofs to minimize side-lob levels
    '''
    return blackman_generalized(M, coefficients=[0.4243801, 0.4973406, 0.00782793])


def power_of_cosine(M, order):
    '''
        - parametrizes the L-term Blackman-Harris windows (for L = order / 2 + 1)
        - first P terms of the window's Taylor expansion, endpoints identically 0
        - roll-ff rate ~ 6 * (orderr + 1) dB / octave
        - order = 0 -> rectangular window
        - order = 1 -> MLT sine window
        - order = 2 Hann window
        - order = 4 Alternative Blackman (maximized roll-off rate)
    '''
    wrange, rwindow = rectangle(M)
    window = rwindow  * np.power(np.cos(np.pi * wrange / M), order)

    return wrange, window

def bartlett(M, endpoint_zeros=False):
    '''
        - convolution of two length (M - 1) / 2 rectangular windows
        - main lobe twice as wide as that of a rectangular window of length M
        - first side lobe twice aas faar down as rectangular window
        - often applied implicitly to sample correlations of finite data
        - also called the "tent function"
        - can replace M - 1 by M + 1 to avoid including endpoint zeros
        - TODO: handle even M
    '''

    M2 = (M - 1) // 2
    # for M odd
    halfwin = np.linspace(1, M2 - 1, M2) / M2

    window = np.concatenate((halfwin, np.ones(1), np.flip(halfwin)))
    return window


def poisson(M, alpha):
    M2 = (M - 1) / 2
    wrange = np.linspace(-M2, M2, num=M)
    rate = -alpha / M2
    window = np.exp(rate * np.abs(wrange))

    return wrange, window


def hann_poisson(M, alpha=10):
    ''' hann poisson as the product of the associated hann and poisson windows
        - No side-lobes for alpha >= 2
        - Transform magnitude has negative slope for all positive frequencies
        - Has a convex transform magnitude to the left or to the right of the peak
        - valuable for any convex optimization method such as "hill climbing"
    '''

    whann = hann(M, include_zeros=True, causal=False)
    _, wpoisson = poisson(M, alpha=alpha)
    window = whann * wpoisson
    return window

def hann_poisson2(M, alpha=10):
    ''' hann poisson direct implementation'''
    Modd = M % 2
    M2 = (M - Modd) / 2
    wrange = np.linspace(-M2, M2, num=M)
    poissonrate = -alpha / M2
    poisson = np.exp(poissonrate * np.abs(wrange))

    hannrate = np.pi / M2
    hann = 0.5 * (1 + np.cos(hannrate * wrange))

    window = poisson * hann
    return window

def slepian(M, Wc):
    '''
        - for a main-lobe width of 2w_c, the Digital Prolate Spheroidal Sequences (DPSS) uses all M degrees of freedom (sample values)
          in an M-point window w(n) to obtain a window transform which maximizes the energy of the main lobe relative to total energy
        - This function computes the DPSS window of length M, having cut-off frequency Wc in (0, pi)
    '''
    wrange = np.arange(1, M)
    samples = np.sin(Wc * wrange) / wrange
    bins = np.concatenate((Wc, samples), axis=None)
    A = linalg.toeplitz(bins)
    evals, evects = linalg.eigh(A)
    # only need the principal eigenvector
    imax = np.argmax(np.abs(evals))
    pvect = evects[:, imax]
    window = pvect / np.amax(pvect)

    return wrange, window


def kaiser(M, beta=10):
    '''
        - alternate: Kaiser-Bessel
        - reduces to rectangular window for beta = 0
        - asymptotic roll-off is 6dB/octave
        - first NULL in transforrm is at w0 = 2 * beta / M
        - time-bandwidth product w0(M/2) = beta radians if bandwidths are measured from 0 to positive band-limit (fs / 2 or the nyquest frequency)
        - full time-bandwidth product (2*w0)M = 4 * beta radians when frequency bandwidth is defined as main-lobe width out to first null
        - can be parametrized by alpha, where alpha = beta / pi. Alpha is half the window's time bandwidth product in cycle (sec * cycles/sec)
        - for beta = 0, it reduces to the rectangle window
        - as alpha (beta) increases, the dB side-lovee level reduced ~linearly with main-love width increaase (approximately a 25dB drop in
          side-love level for each main-lobe widht increase by one sinc-main-lobe)
    '''
    M2 = (M - 1) / 2
    wrange = np.linspace(-M2, M2, num=M)
    window = util.bessel_first(beta * np.sqrt(1 - (2 * wrange / M)**2)) / util.bessel_first(beta)
    return window


def chebyshev(M, ripple):
    '''
        - ripple in dB
        - alternate: dolph_chebyshev, dolph
        - minimizes the chebyshev norm of the side lobes for a given main-lobe width of 2 * Wc
        - the transform has a closed-form
        - computed as the inverse DFT of its transform
        - side-lobes are of equal height: ripple in the stop-band
    '''

    wrange = np.arange(M)
    alpha = ripple / 20

    beta = np.cosh(1 / M * np.arccosh(10**alpha))

    arg = beta * np.cos(np.pi * wrange / M)
    arg = np.clip(arg, -1, 1)
    num = np.cos(M * np.arccos(arg))
    denom = np.cosh(M * np.arccosh(beta))

    spectrum = num / denom
    return np.nan_to_num(spectrum)


def gaussian_norm(M, alpha):
    '''
        - alpha = ((M - 1) / 2) / sigma so that the window shape is invariant to the window length M
        - note: on a dB scale, Gaussians are quadratic: parabolic interpolation of a sampled Gaussian transform is exact
        - the spectrum transforms to itself
        - it achieves the minimmum time-bandwidth product, sigma_t * sigma_w = sigma * 1 / sigma = 1
        - Gaussian has infinite duration, so must be truncated (window it)
        - Literature suggests a triangular window raised to some power alpha, which preserves the absense of side lobes
          for sufficiently large alpha. It also preserves the non-negativity of the transform
    '''

    M2 = (M - 1) / 2
    wrange = np.linspace(-M2, M2, num=M)
    window = np.exp(- 1 / 2 * np.power(alpha * wrange / (M - 1 / 2), 2))
    return window

def gaussian(M, sigma):
    '''
        - sigma should be specified aas a fraction of the window length, e.g., M/8
    '''
    M2 = (M - 1) / 2
    wrange = np.linspace(-M2, M2, num=M)
    window = 1 / 4 / np.pi**2 / sigma * np.exp(-wrange * wrange / (2 * sigma))
    return  window


def is_cola(window, hop, span=5):
    ''' Test that the overlap-adds to a constant'''
    # TODO: just pick an appropriate span to cover all overlaps
    # OR use a lut

    # always COLA when sliding (hop = 1)
    if hop == 1:
        return True

    M = len(window)
    acc = np.zeros(span * M)

    Modd = M % 2
    overlap = M // hop

    olas = []
    for so in range(0, (span - 1) * M, hop):
        ola = acc[so:so+M] + window
        acc[so:so+M] = ola
        olas.append(ola)

    # remove bias: the hop from the beginning and the end
    bias = acc[M:- M].copy()
    bias = np.around(bias, 2)

    # COLA if bias is a constant
    return np.all(bias == bias[0]), bias


def cola_rectangle(M: int, overlap=True):
    Modd = M % 2
    hop: int = M
    if overlap:
        hop = M // 2

    win = rectangle(M)
    if Modd:
        win[-1] = 0

    return win, hop


def cola_hamming(M: int, zero_method=False):
    #hop: int = M // 2
    hop: int = M // 4

    win = hamming(M)
    if zero_method:
        win[-1] = 0
    else:
        win[-1] /= 2
        win[0] /= 2

    return win, hop


def cola_hann(M: int):
    hop: int = M // 2
    win = hann(M)
    return win, hop


def cola_bartlett(M: int):
    Modd = M % 2
    hop: int = int(M + 1) // 2
    win = bartlett(M)
    return win, hop


def cola_blackman(M: int):
    Modd = M % 2
    hop: int = int(M + 1) // 4
    _, win = blackman_classic(M)
    return win, hop


def cola_blackman_harris(M: int):
    Modd = M % 2
    hop: int = int(M + 1) // 3
    _, win = blackman_harris(M)
    return win, hop


def cola_kaiser(M: int, beta=8):
    Modd = M % 2
    hop: int = int(np.floor(1.7 * (M - Modd) / (beta + 1)))

    win = kaiser(M, beta=beta)
    return win, hop


def cola_poisson(M: int):
    Modd = M % 2
    hop: int = (M - Modd) // 6

    _, win = poisson(M, alpha=4)
    return win, hop


def cola_hann_poisson(M: int):
    Modd = M % 2
    hop: int = (M - Modd) // 2

    win = hann_poisson(M, alpha=8)
    return win, hop


def cola_gaussian(M: int):
    Modd = M % 2
    hop: int = (M - Modd) // 8

    win = gaussian(M, 6)
    return win, hop


def cola_gaussian_norm(M: int):
    Modd = M % 2
    hop: int = (M - Modd) // 8

    win = gaussian(M, 6)
    return win, hop


def cola_mlt(M: int):
    Modd = M % 2
    hop: int = (M - Modd) // 8

    _, win = modulated_lapped_transform(M)
    return win, hop


def is_cola_test():
    M = 33
    #win, hop = cola_kaiser(M)
    #win, hop = cola_gaussian_norm(M)
    win, hop = cola_rectangle(M)
    return (is_cola(win, hop), win)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    res, win = is_cola_test()
    print(len(win))
    print(res[0])
    print(res[1])
    plt.plot(res[1])
    plt.plot(win, '-o')
    plt.show()