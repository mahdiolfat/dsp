'''A module containing a set of window filters'''

from matplotlib.pyplot import axis
import numpy as np
from scipy import linalg, optimize

import util

def frequency_sample(M):
    ''' Normalized Frequency sampling resolution: radians per sample '''
    return 2 * np.pi / M

def mainlobe_width_rectangle(M, fs=1):
    # rectangle window has the tightest main-lobe width
    return 2 * fs / M

def mainlobe_width_kaiser(M, beta, fs=1):
    alpha = beta / np.pi
    return alpha * 2 * fs / M

def mainlobe_width_poisson(M, alpha, fs=1):
    return alpha * 2 * fs / M

def mainlobe_width_poisson(M, alpha, fs=1):
    return alpha * 2 * fs / M

def mainlobe_width_hamming(M, fs=1):
    return 4 * fs / M

def mainlobe_width_hann(M, fs=1):
    return 4 * fs / M

def mainlobe_width_generalized_hamming(M, fs=1):
    return 4 * fs / M

def mainlobe_width_blackman(M, fs=1):
    return 6 * fs / M

def mainlobe_width_l_term_blackman(M, L, fs=1):
    return 2 * L * fs / M

def spectrum_hamming(M, alpha, beta):
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
        self._mainlobe = None
        self._side_lob = None
        self._roll_off = None
        self._periodic = None # for overlap+add
        self._endpoint = False #  if count 
        self._alternative_names = None
        self._timeconstant = None # time domain rate of decay
        self._slope = None
        self._curvature = None
        self._normalized_amplitude = True
        self._normalized_frequency = True
        self._side_lob_width = None
        self._main_lob_width = None

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

    def is_symmetric(self):
        return False

    def is_zerophase(self):
        return not self.is_causal()


def rectangle(M):
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
        - Side lobes roll off at approximately 6dB per octave
        - A linear phase term arises when we shift the window to make it causal (shift of M-1/2 rad)
    '''

    # calculate positive half only, flip and use for negative half
    wrange = np.arange(-M + 1, M)
    window = (np.abs(wrange) <= (M - 1) / 2.).astype(float)
    #_spectrum = M * util.asinc(count, np.linspace(-np.pi, np.pi, num=1000, endpoint=False))
    return wrange, window

def hamming_generalize(M, alpha, beta):
    pass

def hann(M, include_zeros=True, periodic=False, causal=True):
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
    wrange = np.arange(0, 2 * M)
    window = np.sin((np.pi / 2 / M) * (wrange + 0.5))
    return wrange, window

def blackman_generalized(M, coefficients):
    orders = len(coefficients)
    wrange = np.arange(-(M - 1) / 2, M / 2)
    nrange = np.zeros((1, M))
    wterms = np.zeros((orders, 3 * M))

    for order in range(orders):
        wterms[order,:] = np.concatenate((nrange, np.cos(order * 2 * np.pi * wrange / M), nrange), axis=None)
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

def spectrum_barlett(M):
    return ((M - 1) / 2)**2 * util.asinc((M - 1) / 2, np.linspace(-np.pi, np.pi, num=1000, endpoint=False))

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

def barlett(M, endpoint_zeros=False):
    '''
        - convolution of two length (M - 1) / 2 rectangular windows
        - main lobe twice as wide as that of a rectangular window of length M
        - first side lobe twice aas faar down as rectangular window
        - often applied implicitly to sample correlations of finite data
        - also called the "tent function"
        - can replace M - 1 by M + 1 to avoid including endpoint zeros


        - TODO: handle even M
    '''
    #wrange, recwindow = rectangle(M)
    #window = recwindow * ( 1 - 2 * np.abs(wrange) / (M - 1))

    num = M - 1 if endpoint_zeros else M
    # for M odd
    halfrange = np.linspace(0, (M - 1) / 2, num=num)
    if not endpoint_zeros:
        halfrange = halfrange[1:]

    wrange = np.linspace(-(M - 1) / 2, (M - 1) / 2, num=M)

    window = 2 * halfrange / (M - 1)
    window = np.concatenate((window, window[-2::-1]), axis=None)
    return wrange, window

def poisson(M, alpha):
    M2 = (M - 1) / 2
    wrange = np.linspace(-M2, M2, num=M)
    rate = -alpha / M2
    window = np.exp(rate * np.abs(wrange))
    #wrange, recwindow = rectangle(M)
    #window = recwindow * np.exp(2 * -alpha * np.abs(wrange) / (M - 1))
    return wrange, window

def hann_poisson(M, alpha=10):
    ''' hann poisson as the product of the associated hann and poisson windows
        - No side-lobes for alpha >= 2
        - Transform magnitude has negative slope for all positive frequencies
        - Has a convex transform magnitude to the left or to the right of the peak
        - valuable for any convext optimization method such as "hill climbing"
    '''

    wrange, whann = hann(count, include_zeros=True, causal=False)
    _, wpoisson = poisson(count, alpha=alpha)
    window = whann * wpoisson
    return wrange, window

def hann_poisson2(M, alpha=10):
    ''' hann poisson direct implementation'''
    M2 = (M - 1) / 2
    wrange = np.linspace(-M2, M2, num=M)
    poissonrate = -alpha / M2
    poisson = np.exp(poissonrate * np.abs(wrange))

    hannrate = np.pi / M2
    hann = 0.5 * (1 + np.cos(hannrate*wrange))

    window = poisson * hann
    return wrange, window

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
    pvect = evects[:,imax]
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
    return wrange, window

def spectrum_kaiser(M, beta):
    wrange = np.linspace(-M, M, num=2 * M + 1)
    term1 = M / util.bessel_first(beta)
    arg = np.sqrt((M * wrange / 2)**2 - beta**2)
    term2 = np.sin(arg) / arg
    spectrum = term1 * term2
    return wrange, spectrum

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

def spectrum_chebyshev(M, Wc):
    '''
        -
    '''

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
    window = np.exp( - 1 / 2 * np.pow(alpha * wrange / (M - 1 / 2), 2))
    return wrange, window

def gaussian(M, sigma):
    '''
        - sigma should be specified aas a fraction of the window length, e.g., M/8
    '''
    M2 = (M - 1) / 2
    wrange = np.linspace(-M2, M2, num=M)
    window = np.exp(-wrange * wrange / (2 * sigma**2))
    return wrange, window

def spectrum_blackman_harris(M, order=None, coefficients=None):
    '''
        Frequency domain implementation can be performed as a (2 * order -1)-point convolution with the spectrum of the unwindowed data
        Generally, any L-term Blackman-Harris window requires convolution of the critically saampled spectrum with a smoother of length 2L-1
    '''

    # M point rectangular data
    wrange, wr = rectangle(M)
    # M point dft
    dft = np.fft.fftshift(np.fft.fft(wr, 1024))

    smoother = [1/4, 1/2, 1/4]

    # convolve DFT data with the smoother
    window = np.convolve(dft, smoother)

    return window

def spectrum_symmetric(L, w):
    '''for zero-phase symmetric windows'''
    lrange = np.arange(1, L + 1)
    upper = np.array(2 * np.cos(lrange * w)).reshape((1, L))
    lower = np.ones((1, 1))
    return np.concatenate((lower, upper), axis=1)

def optimal_lp(M, Wsb):
    '''
        objectives:
        - symmetric zero-phase
        - positive window samples
        - Transform amplitude to be 1 at DC
        - Transform to be within [-delta, delta] in the stop-band
            - for w_sb <= w <= pi
        - delta be small (minimize)
    '''

    # due to symmetry, impulse response h(n) is equal to window w(n) for n >= 0 
    L = int((M - 1) / 2)
    print(f'L = {L}')

    # positive h(n) includes h(0), so the minimized h(n) has L+1 terms
    # size of objective column vector is (L + 2, 1)

    wcount = 300
    wrange = np.linspace(Wsb, np.pi, num=wcount)

    # minimize
    minimizer = np.concatenate((np.zeros((L + 1)), np.ones((1))), axis=None)
    print(f'minimizer = {minimizer.shape}')

    # subject to
    b_eq = [1]
    Aeq = np.concatenate((spectrum_symmetric(L, 0), np.zeros((1, 1))), axis=1)
    print(f'Aeq = {Aeq.shape}')

    Asb = np.empty((wcount, L + 1))
    for k, w in enumerate(wrange):
        Asb[k,:] = spectrum_symmetric(L, w)
    print(f'Asb = {Asb.shape}')

    Asb = np.concatenate((-Asb, Asb))
    print(f'Asb = {Asb.shape}')
    Asb = np.concatenate((Asb, -np.ones((2 * wcount, 1))), axis=1)
    print(f'Asb = {Asb.shape}')

    b_lt = np.zeros((Asb.shape[0], 1))
    print(f'b_lt = {b_lt.shape}')

    # create [min, max] bound tuples for all decision variables
    # all window values are >= 0, no bounds on the stop-band amplitude, delta
    bounds = [(0, None)] * (L + 1)  # impulse responses
    bounds.append((None, None))  # amplitude in stop band

    # solve LP problem
    result = optimize.linprog(minimizer, A_ub=Asb, b_ub=b_lt, A_eq=Aeq, b_eq=b_eq, bounds=bounds)
    print(result)

    if not result["success"]:
        return False, None, None

    # construct the negative negative values of the 0-phase window
    optimized = result["x"]
    delta = optimized[-1]
    window = optimized[:-1]
    window = np.concatenate((window[:0:-1], window), axis=None)
    return True, delta, window

def optimal_monotonicity(M, Wsb):
    # due to symmetry, impulse response h(n) is equal to window w(n) for n >= 0 
    L = int((M - 1) / 2)
    print(f'L = {L}')

    minimizer = np.concatenate((np.zeros((L + 1)), np.ones((1))), axis=None)

    b_eq = [1]
    Aeq = np.concatenate((spectrum_symmetric(L, 0), np.zeros((1, 1))), axis=1)
    print(f'Aeq = {Aeq.shape}')

    wcount = 300
    wrange = np.linspace(Wsb, np.pi, num=wcount)
    Asb = np.empty((wcount, L + 1))
    for k, w in enumerate(wrange):
        Asb[k,:] = spectrum_symmetric(L, w)
    print(f'Asb = {Asb.shape}')

    Asb = np.concatenate((-Asb, Asb))
    print(f'Asb = {Asb.shape}')
    Asb = np.concatenate((Asb, -np.ones((2 * wcount, 1))), axis=1)
    print(f'Asb = {Asb.shape}')

    # monotonicity constraint:
    mono = np.delete(np.diagflat([1] * L, k=1) - np.identity(L + 1), -1, 0)
    Asb_upper = np.concatenate((mono, np.zeros((mono.shape[0], 1))), axis=1)

    Asb = np.concatenate((Asb_upper, Asb))
    b_lt = np.zeros((Asb.shape[0], 1))
    print(f'b_lt = {b_lt.shape}')

    # create [min, max] bound tuples for all decision variables
    # all window values are >= 0, no bounds on the stop-band amplitude, delta
    bounds = [(0, None)] * (L + 1)  # impulse responses
    bounds.append((None, None))  # amplitude in stop band

    # solve LP problem
    result = optimize.linprog(minimizer, A_ub=Asb, b_ub=b_lt, A_eq=Aeq, b_eq=b_eq, bounds=bounds)
    print(result)

    if not result["success"]:
        return False, None, None

    # construct the negative negative values of the 0-phase window
    optimized = result["x"]
    delta = optimized[-1]
    window = optimized[:-1]
    window = np.concatenate((window[:0:-1], window), axis=None)
    return True, delta, window

def optimal_lone(M, Wsb, weight=1):
    ''' L ones is sensitive to all derivatives, not just the largest'''
    # due to symmetry, impulse response h(n) is equal to window w(n) for n >= 0 
    L = int((M - 1) / 2)
    print(f'L = {L}')

    minimizer = np.concatenate((np.zeros((L + 1)), [1]), axis=None)
    minimizer = np.concatenate((minimizer, [weight] * L), axis=None)
    print(f'minimizer = {minimizer.shape}')

    b_eq = [1]
    Aeq = np.concatenate((spectrum_symmetric(L, 0), np.zeros((1, L + 1))), axis=1)
    print(f'Aeq = {Aeq.shape}')

    wcount = 300
    wrange = np.linspace(Wsb, np.pi, num=wcount)
    Asb = np.empty((wcount, L + 1))
    for k, w in enumerate(wrange):
        Asb[k,:] = spectrum_symmetric(L, w)
    print(f'Asb = {Asb.shape}')

    Asb = np.concatenate((-Asb, Asb))
    print(f'Asb = {Asb.shape}')
    Asb = np.concatenate((Asb, -np.ones((2 * wcount, 1))), axis=1)
    Asb = np.concatenate((Asb, np.zeros((Asb.shape[0], L))), axis=1)
    print(f'Asb = {Asb.shape}')

    # monotonicity constraint:
    lone = np.delete(np.diagflat([1] * L, k=1) - np.identity(L + 1), -1, 0)
    lone = np.concatenate((-lone, lone))

    Asb_upper = np.concatenate((lone, np.zeros((lone.shape[0], 1))), axis=1)
    Asb_upper = np.concatenate((Asb_upper, -np.ones((Asb_upper.shape[0], L))), axis=1)

    Asb = np.concatenate((Asb_upper, Asb))
    b_lt = np.zeros((Asb.shape[0], 1))
    print(f'b_lt = {b_lt.shape}')

    # create [min, max] bound tuples for all decision variables
    # all window values are >= 0, no bounds on the stop-band amplitude, delta
    bounds = [(0, None)] * (L + 1)  # impulse responses
    bounds.append((None, None))  # amplitude in stop band
    bounds = bounds + [(None, None)] * L  # L one constraint

    # solve LP problem
    result = optimize.linprog(minimizer, A_ub=Asb, b_ub=b_lt, A_eq=Aeq, b_eq=b_eq, bounds=bounds)
    print(result)

    if not result["success"]:
        return False, None, None

    # construct the negative part of the 0-phase window
    optimized = result["x"]
    #_sigma = optimized[-1]
    window = optimized[:L+1]
    delta = optimized[L+2]
    window = np.concatenate((window[:0:-1], window), axis=None)
    return True, delta, window

def optimal_linf(M, Wsb, weight=1):
    ''' smoothness objective 
        - l-infinity only cares about maximum derivative 
        - large weight means there is more weight on smoothness vs. side lobe level
    '''
    # due to symmetry, impulse response h(n) is equal to window w(n) for n >= 0 
    L = int((M - 1) / 2)
    print(f'L = {L}')

    minimizer = np.concatenate((np.zeros((L + 1)), [1, weight]), axis=None)

    b_eq = [1]
    Aeq = np.concatenate((spectrum_symmetric(L, 0), np.zeros((1, 2))), axis=1)
    print(f'Aeq = {Aeq.shape}')

    wcount = 300
    wrange = np.linspace(Wsb, np.pi, num=wcount)
    Asb = np.empty((wcount, L + 1))
    for k, w in enumerate(wrange):
        Asb[k,:] = spectrum_symmetric(L, w)
    print(f'Asb = {Asb.shape}')

    Asb = np.concatenate((-Asb, Asb))
    print(f'Asb = {Asb.shape}')
    Asb = np.concatenate((Asb, -np.ones((2 * wcount, 1))), axis=1)
    Asb = np.concatenate((Asb, np.zeros((Asb.shape[0], 1))), axis=1)
    print(f'Asb = {Asb.shape}')

    # monotonicity constraint:
    linf = np.delete(np.diagflat([1] * L, k=1) - np.identity(L + 1), -1, 0)
    print(f'linf = {linf.shape}')
    linf = np.concatenate((-linf, linf))
    print(f'linf = {linf.shape}')

    Asb_upper = np.concatenate((linf, np.zeros((linf.shape[0], 1))), axis=1)
    Asb_upper = np.concatenate((Asb_upper, -np.ones((Asb_upper.shape[0], 1))), axis=1)

    Asb = np.concatenate((Asb_upper, Asb))
    b_lt = np.zeros((Asb.shape[0], 1))
    print(f'b_lt = {b_lt.shape}')

    # create [min, max] bound tuples for all decision variables
    # all window values are >= 0, no bounds on the stop-band amplitude, delta
    bounds = [(0, None)] * (L + 1)  # impulse responses
    bounds.append((None, None))  # amplitude in stop band
    bounds.append((None, None))  # L infinity constraint

    # solve LP problem
    result = optimize.linprog(minimizer, A_ub=Asb, b_ub=b_lt, A_eq=Aeq, b_eq=b_eq, bounds=bounds)
    print(result)

    if not result["success"]:
        return False, None, None

    # construct the negative part of the 0-phase window
    optimized = result["x"]
    _sigma = optimized[-1]
    delta = optimized[-2]
    window = optimized[:-2]
    window = np.concatenate((window[:0:-1], window), axis=None)
    return True, delta, window

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    count = 21
    L = int((count - 1 ) / 2)
    success, delta, window = optimal_lone(count, Wsb = np.pi / 8, weight=6)
    print(f'window={window}')
    print(f'delta={delta}')
    plt.plot(np.linspace(-L, L, num=count), window)
    #a = spectrum_symmetric(10, 0)
    #print(a)
    #print(len(a))
    #w = gaussian(count, count/8)
    #spectrum = chebyshev(count, 60)
    #w = poisson(count, 2)
    #w2 = hann(count, causal=False)
    #w3 = hann_poisson(count, 2)
    #w4 = hann_poisson2(count, 2)
    #w = rectangle(count)
    #w = mlt(count)
    #w = hann_poisson(count)
    #w = hann(count)
    #w = blackman_classic(count)
    #print(w[1])
    #w = modulated_lapped_transform(count)
    #print(w[1])
    #plt.bar(w[0], w[1], width=0.15, color='g')
    #plt.bar(w2[0], w2[1], width=0.1, color='b')
    #plt.plot(w[0], w[1], linestyle = "-.", marker= '.', color='g')
    #plt.plot(w2[0], w2[1], linestyle = "--", marker='o', color='b')
    #plt.plot(w3[0], w3[1], marker='D', color='r')
    #plt.plot(w4[0], w4[1], marker='1', color='m')
    plt.grid()
    plt.show()
    exit(0)

    #spectrum = np.fft.fft(w[1])
    #spectrum = count * util.asinc(count, np.linspace(-np.pi, np.pi, num=1000, endpoint=False))

    #spectrum = spectrum_hamming(count, 1/2, 1/4)

    ## normalized frequency (cycles / sample)
    #normfreq = np.linspace(-0.5, 0.5, num=1000, endpoint=False)

    #spectrum = np.fft.fftshift(spectrum)
    #amplitude = np.abs(spectrum)
    #magnitude = 20 * np.log10(amplitude)
    #plt.figure()
    #plt.plot(spectrum)
    #plt.figure()
    #plt.plot(magnitude - np.nanmax(magnitude))
    #plt.figure()
    #plt.plot(normfreq, magnitude - np.nanmax(magnitude))
    ##plt.plot(normfreq, magnitude)
    #plt.ylim((-60, 10))

    #plt.show()