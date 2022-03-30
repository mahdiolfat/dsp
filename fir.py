import numpy as np

def normalized_freq(freq):
    return 2 * np.pi * freq

def lowpass_ideal(fc, taps, fs=1):
    return 2 * fc / fs * np.sinc(2 * fc / fs * np.arange(taps))

def spectrum_lowpass_ideal(fc, fs=1, fcount=None):
    if fcount is None:
        # scale such that cutoff frequency (fc) has an integer index
        fcount = int(100 * fs / fc)

    frange = np.linspace(0, fs, num = fcount)
    fc_i = int(fc / fs * fcount)
    filter = np.concatenate((np.ones((fc_i)), np.zeros((fcount - fc_i))), axis=None)
    return frange, filter, fcount

def lowpass_least_squares(fc, taps, fs=1):
    '''
        - due to energy Fourier energy theorem, the optimal least squares lowpass filter
          is just the first L terms (# of taps) of the ideal lowpass filter (sinc function)
    '''
    return lowpass_ideal(fc, taps, fs=fs)

def lowpass(freq_pb, ripple_pb, freq_sb, ripple_sb):
    '''
        - passband ripple is an allowed gain deviation
        - stopband ripple is an allowed leakage level
        - optimal:
            - ripples = 0
            - SBA = inf
            - transition width is 0, 
    '''
    transition_width = freq_sb - freq_pb
    stopband_attenuation = 20*np.log10(ripple_sb)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    taps = 30
    fc = 1/6
    filter = lowpass_least_squares(fc, taps)
    filter_spec = np.fft.rfft(filter, 1024)
    plt.plot(np.abs(filter_spec))
    #frange, filter, _ = spectrum_lowpass_ideal(1/4)
    #plt.plot(frange, filter)
    plt.figure()
    lp = lowpass_ideal(fc, taps)
    print(lp)
    print(len(lp))
    plt.bar(np.arange(taps), lp, width = 0.2)
    #plt.bar(np.arange(taps), filter)
    plt.show()