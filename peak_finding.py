''' Module for spectral peak finding algorithms, routines, and helpers '''

import numpy as np

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

f = 1000
print(minimum_zero_padding('blackman', 1/f, 0.001*f))
print(minimum_zero_padding('blackman', 2/f, 0.001*f))
print(minimum_zero_padding('blackman', 4/f, 0.001*f))