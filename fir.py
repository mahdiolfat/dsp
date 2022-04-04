import numpy as np
from scipy import signal

import windows as win

'''
TODO:
- add notes about transition widths and main-lobe width
- design filter based on kaiser window method
- design filter based on hilbert transform
- derive and test the bandpass ideal reponse
'''

def hilbert_kernel(t):
    ''' continuous time '''
    return 1 / (np.pi * t)

def smallest_fft(ftransition, fs=1):
    return 10 * fs / ftransition

def kaiser_filter_order(stop_band, transition_width, fs=1) -> int:
    ''' based on stop-band attenuation A'''

    if transition_width >= fs / 2:
        raise ValueError("transition width must be less than half of the sampling frequency")

    if stop_band < 8:
        raise ValueError(f'stop band attenuation must be greater than 8db, given {stop_band}')

    # width rad is between 0 and pi
    width_rad = transition_width / fs * 2 * np.pi
    print(f'width rad: {width_rad}')

    return int((stop_band - 8) / (2.285 * width_rad))

def kaiser_window_design(side_lobe):
    # calculate beta based on side-lob attenuation in dB
    beta = None

    if side_lobe <= 13.26:
        beta = 0
    if side_lobe > 13.26 and side_lobe <= 60:
        beta = 0.76609 * (side_lobe - 13.26)**0.4 + 0.09834 * (side_lobe - 13.26)
    if side_lobe > 60:
        beta = 0.12438 * (side_lobe + 6.3)

    return beta

def kaiser_filter_design(stop_band):
    # calculate beta based on stop-band attenuation in dB
    beta = None  # for A <= 21

    if stop_band <= 21:
        beta = 0
    if stop_band > 21 and stop_band <= 50:
        beta = 0.5842*(stop_band - 21)**0.4 + 0.07886*(stop_band - 21)
    if stop_band > 50:
        beta = 0.1102*(stop_band - 8.7)

    return beta

def normalized_freq(freq):
    return 2 * np.pi * freq

def lowpass_ideal(fc, taps, fs=1):
    return 2 * fc / fs * np.sinc(2 * fc / fs * np.arange(taps))

def bandpass_ideal(fc, taps, fs=1):
    return 2 * fc / fs * np.cos(2 * fc / fs * np.arange(taps)) * np.sinc(2 * fc / fs * np.arange(taps))

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

def bandpass_window():
    '''bandpass filter design based on the window method
        - taps of filter == length of window
        - the passband gain is primarily the area under the main lobe of the window transform
          therefore, the total lowpass bandwidth B = 2fc must be >= W, the mainlobe width
        - the stop-band gain is given by an integral oveer a portion of the side-lobes of the window transform.
        - the best stop-band performancee occurs when the cut-off frequency is set so that the stop-band side-lobe
          integral traverses a whole number of side lobes.
        - the transition bandwidth is equal to the bandwidth of the main lobe of the window transform (provided that the
          main lobe "fits" inside the pass-band)
        - as lowpass bandwidth approaches 0, the lowpass filter approaches the window transform. The stop band gain approaches
          the window side-lobe level, and the transition width approaches half the main-lobe width.
        - For good results, the low-pass cut-off frequency should be set no lower than half the window's main-lobe width
    '''

    ''' Using a Kaiser window:
        1. construct the impulse response of the ideal bandpass filter (a cosine modulated sinc function)
        2. 
        3. compute the kaiser window using the estimated length and beta
    '''
    pass

def bandpass_remez(bands):

    M = 101
    fs = 20000
    bands = [0, 3e3, 4e3, 6e3, 7e3, 0.5*fs]
    filter = signal.remez(M, bands, [0, 1, 0], Hz=fs)
    w, h = signal.freqz(filter, [1], worN=2000, fs=fs)
    amplitude = np.abs(h)
    mag_db = 20 * np.log10(amplitude)
    norm_db = mag_db - np.nanmax(mag_db)
    return w, h, norm_db

def bandpass_hilbert():
    '''

    '''

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

def example_hilbert():
    count = 257
    N = 4096
    fs = 22050
    f1 = 530  # transition bandwidth
    beta = 8

    fn = fs / 2 # nyquest
    f2 = fn - f1  # upper transition bandwidth

    # lower-band edge in bins
    k1 = round(N * f1 / fs)
    if k1 < 2:
        # cannot have dc or fn response
        k1 = 2

    k1 = int(k1)

    # bin index at nyquest limit, N even
    kn = N / 2 + 1

    # high-frequency band edge
    k2 = kn - k1 + 1
    k2 = int(k2)

    print(k1, k2, f1, f2)

    # quantized band-edge frequencies
    f1 = k1 * fs / N
    f2 = k2 * fs / N

    print(k1, k2, f1, f2)

    wrange, impluse = win.kaiser(count, beta=beta)
    plt.plot(wrange, impluse)

    # ideal frequency response
    a = np.concatenate(((np.arange(k1-1) / (k1 - 1))**8, np.ones((k2-k1+1))))

    b = np.concatenate(((np.arange(k1-2, -1, -1) / (k1 - 1))**8, np.zeros((N//2-1))))
    c = np.concatenate((a, b))

    plt.figure()
    plt.plot(c)

    impulse_response = np.fft.ifft(c)
    plt.figure()
    hodd = np.imag(impulse_response[::2])
    #ierr = 
    plt.plot(np.fft.ifftshift(np.real(impulse_response)))

    response = np.fft.fft(impluse, N)
    gain = np.abs(response)
    gain_db = 20*np.log10(gain)
    gain_db = gain_db - np.nanmax(gain_db)
    plt.figure()
    plt.plot(np.fft.fftshift(gain_db))
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    example_hilbert()

    #print(np.min(np.diff(bands)))
    #print(kaiser_filter_order(80, np.min(np.diff(bands)), fs=20e3))
    #print(kaiser_filter_design(80))

    #fc = 1/4
    #filter = lowpass_least_squares(fc, taps)
    #filter_spec = np.fft.rfft(filter, 1024, norm="ortho")
    #mag = np.abs(filter_spec)
    #mag_db = 20 * np.log10(mag)
    #norm_db = mag_db - np.nanmax(mag_db)
    #plt.plot(mag)
    #plt.grid()
    ##frange, filter, _ = spectrum_lowpass_ideal(1/4)
    ##plt.plot(frange, filter)
    #plt.figure()
    ##plt.plot(norm_db)
    ##plt.grid()
    ##plt.figure()
    #plt.bar(np.arange(M), filter)
    ##plt.grid()
    #plt.show()