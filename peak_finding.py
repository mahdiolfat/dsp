''' Module for spectral peak finding algorithms, routines, and helpers '''

import numpy as np
from scipy import linalg
import dsp

# a zero padding of factor 5 is sufficient for all window types
# more generally, there is <0.01 % relative frequency error when the window length is one period
SUFFICIENT_ZEROPADDING = 5

# main lobe width-in-bins
main_lobe_factor = {
    'rectangle': 2,
    'hamming': 4,
    'hann': 4,
    'blackman': 4,
}

# empirical (tight) minimum main lobe width-in-bins
main_lobe_factor_tight = {
    'rectangle': 1.44,
    'hamming': 2.22,
    'hann': 2.36,
    'blackman': 2.02,
}

def peak_resolution_length(f1, f2, window='rectangle', tight=False, fs=1) -> int:
    ''' to resolve the frequencies f1 and f2,
        the window length M mustt span aat least K periods of the difference frequency f2 - f1'''
    factors = main_lobe_factor_tight if not tight else main_lobe_factor

    factor = factors[window]
    return np.ceil(factor * fs / np.abs(f2 - f1))

def minimum_zero_padding(window, window_length, max_bias_freq, max_bias_amp=None):
    '''
        max_bias_freq in hertz.
        max_bias_amp in relative fraction
        window_length in seconds: MUST include >1 cycles of the sinusoid period to be estimated

        returns:
            padding factor: 1 means no zero-padding
    '''

    lut = {
        'rectangle': {'c0': 0.4467, 'r0': -0.3218, 'c1': 0.8560, 'r1': -0.2366},
        'hamming': {'c0': 0.2456, 'r0': -0.3282, 'c1': 0.4381, 'r1': -0.2451},
        'hann': {'c0': 0.2436, 'r0': -0.3288, 'c1': 0.4149, 'r1': -0.2456},
        'blackman': {'c0': 0.1868, 'r0': -0.3307, 'c1': 0.3156, 'r1': -0.2475},
    }

    paddingmin = lut[window]['c0'] * (window_length * max_bias_freq) ** lut[window]['r0']

    if max_bias_amp is not None:
        paddingmin_amp = lut[window]['c1'] * (window_length * max_bias_freq) ** lut[window]['r1']
        paddingmin = max(paddingmin, paddingmin_amp)

    if 'rectangle' in window:
        paddingmin = max(1.7, paddingmin)
    else:
        paddingmin = max(1, paddingmin)

    return paddingmin

def pisarenko(signal, order, autocorrelation=None):
    '''
    Pisarenko's method for frequency estimation
    Process is assumed to consist of order complex exponentials in white noise

    signal: signal with additive white noise

    TODO:
        - use covariance
        - handle 0-meaning input data, or just calculate the covariance
    '''
    if autocorrelation is None:
        # estimate the (order + 1) x (order + 1) autocorrelation matrix
        autocorrelation = dsp.autocorrelation(signal)

    # autocorrelation sequence MUST be of size > order + 1
    # only the first (order + 1) of the autocorrelation sequence is used
    rx = np.array(autocorrelation[:order + 1])

    # (order + 1) x (order + 1) hermitian toeplitz autocorrelation matrix
    autocorrelation_matrix = linalg.toeplitz(rx)
    print(autocorrelation_matrix)

    # since autocorrelation matrix is hermitian toeplitz, eigenvalues are non-negative real
    eval, evec = np.linalg.eigh(autocorrelation_matrix)
    print(f'Eigenvalues: {eval}')
    print("Eigenvectors: ", evec)

    # there is only one noise vector
    # it is the column vector corresponding to the minimum eigenvalue
    # the minimum eigenvalue absolute value is the variance of the additive white noise
    vmin_i = np.argmin(eval)
    variance = eval[vmin_i]
    print(f'Min eigenval (variance of white noise): {variance}')
    # noise vector
    vmin = evec[:,vmin_i]
    eigenfilter = scipy.signal.tf2zpk([1], vmin)
    print("Eigenfilter: ", eigenfilter)

    # estimated frequencies
    freqs = np.arccos(np.angle(eigenfilter[1]) / 2)

    # there are (order) signal vectors
    # TODO: calculate power associated with each signal vector

    return freqs, variance

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    SAMPLE_COUNT = 1024
    space = np.linspace(0, 10*2*np.pi, SAMPLE_COUNT)
    var = 0.01
    noise = np.random.normal(0, np.sqrt(var), SAMPLE_COUNT)
    sig = np.sin(space) + noise
    mean = np.mean(sig)
    print(f'Process Mean: {mean}')
    # zero-mean the process
    sig -= mean

    ac = np.array([6, 1.92705 + 4.58522j, -3.42705 + 3.49541j])
    freqs, variance = pisarenko(sig, 2)

    print(f'Frequencies: {freqs}')
    print(f'Variance: {variance}')

    plt.figure(0)
    plt.plot(space, sig)
    plt.grid()
    plt.show()
