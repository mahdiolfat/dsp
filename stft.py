''' Short Time Fourier Transform Processing Engines '''

import numpy as np
import scipy.signal as signal

import windows
import util

def spectral_envelope_cepstrum(sig, f0, fs=1):
    # spectral envelope by the Cepstral Windowing Method
    # compute log magnitude spectrum
    # inverse FFT to obtain the real cepstrum
    # lowpass-window the cepstrum
    # perform FFT to obtain the smoothed log-magnitude spectrum

    # 40ms analysis window
    Nframe = util.nextpow2(fs / 25)
    _, w = windows.hamming(Nframe)
    winspeech = w * sig[:Nframe]
    Nfft = 4 * Nframe
    sspec = np.fft.fft(winspeech, Nfft)
    dbsspecfull = 20 * np.log10(np.abs(sspec))

    # real cepstrum
    rcep = np.fft.ifft(dbsspecfull)
    # eliminate round-off noise in imag part
    rcep = np.real(rcep)

    period = round(fs / f0)

    nspec = Nfft // 2 + 1

    # real cepstrum
    aliasing = np.linalg.norm(rcep[nspec-10:nspec+10]) / np.linalg.norm(rcep)
    print(f'aliasing = {aliasing}')

    # almost 1 period left and right
    nw = 2 * period - 4

    # make window count odd
    if np.floor(nw/2) == nw/2:
        nw -= 1

    _, w = windows.rectangle(nw)

    # make it zero phase
    wzp = np.concatenate((w[(nw - 1) // 2:nw], np.zeros(Nfft-nw), w[:(nw-1) // 2]))

    # lowpass filter (lifter) the cepstrum
    wrcep = wzp * rcep
    rcepenv = np.fft.fft(wrcep)
    # should be real
    rcepenvp = np.real(rcepenv[:nspec])
    rcepenvp = rcepenvp - np.mean(rcepenvp)

    return rcep, rcepenv

def spectral_envelope_example():
    # formant resonance for an "ah" vowel
    F = np.array([700, 1220, 2600])
    # formant bandwidths
    B = np.array([130, 70, 160])

    fs = 8192

    # pole radii
    R = np.exp(-np.pi * B / fs)
    # pole angles
    theta = 2 * np.pi * F / fs
    poles = R * np.exp(1j * theta)

    b, a = signal.zpk2tf([0], np.concatenate((poles, np.conj(poles))), 1)

    # fundamental frequency in Hz
    f0 = 200
    w0T = 2 * np.pi * f0 / fs

    nharm = int((fs / 2) // f0)
    # a second's worth of samples
    nsamps = fs
    sig = np.zeros(nsamps)

    # synthesize the bandlimited impulse train
    n = np.arange(nsamps)
    for i in range(1, nharm + 1):
        sig += np.cos(i * w0T * n)

    # normalize
    sig /= np.max(sig)
    _, w = windows.hamming(512)
    # compute the speech vowel
    sigbl = sig[:len(w)] * w

    #plt.plot(10*np.log10(np.abs(np.fft.rfft(sig, fs))**2))
    #plt.xlim(0, 4500)

    speech = signal.lfilter([1], a, sig)
    # hamming windowed
    speechbl = speech[:len(w)] * w

    #plt.plot(speech)
    #plt.xlim(0, 600)
    #plt.plot(10*np.log10(np.abs(np.fft.rfft(speech, fs))**2))
    #plt.xlim(0, 4500)

    #plt.show()

    rcep, rcepenv = spectral_envelope_cepstrum(speech, f0, fs)
    #plt.plot(rcep)
    #plt.xlim(0, 140)
    #plt.show()

    # spectral envelope by linear prediction

    # assume three formants and no noise
    M = 6

    # compute Mth order autocorrelation function
    rx = np.zeros(M+1)

    for i in range(M+1):
        rx[i] = rx[i] + speech[:nsamps-i] @ speech[i:nsamps]

    # prep the M by M Toeplitz covariance matrix
    covmatrix = np.zeros((M, M))
    for i in range(M):
        covmatrix[i, i:] = rx[:M-i]
        covmatrix[i:, i] = rx[:M-i]

    print(covmatrix)

    # solve normal equations for prediction coefficients
    Acoeffs = np.linalg.solve(-covmatrix, rx[1:])

    # linear prediction polynomial
    Alp = np.concatenate(([1], Acoeffs))

    Nframe = util.nextpow2(fs / 25)
    Nfft = 4 * Nframe
    nspec = Nfft // 2 + 1
    _, w = windows.hamming(Nframe)
    winspeech = w * sig[:Nframe]
    Nfft = 4 * Nframe
    sspec = np.fft.fft(winspeech, Nfft)
    dbsspecfull = 20 * np.log10(np.abs(sspec))

    w, h = signal.freqz(1, Alp, nspec, fs=fs)
    dbenvlp = 20 * np.log10(np.abs(h))
    diff = np.max(dbenvlp) - np.max(dbsspecfull)
    dbsspecn = dbsspecfull + np.sum(diff)

    plt.plot(w, dbenvlp)
    plt.plot(w, dbsspecn[:1022:-1])
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    spectral_envelope_cepstrum()