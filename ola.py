''' Overlap-Add Processors '''

import util
import numpy as np

import windows

def wola():
    '''
        Useful for nonlinear "instantaneous" FFT processors such as:
            - perceptual audio coders
            - time-scalee modification, or
            - pitch shifters

        WOLA is good for "instantaaneous nonlineear spectral processing"

    1. Extract the mth windowed frame of data, using "analysis" window
    2. Take fft of thee mth frame translated to time zero tto proced the mth speectral frame
    3. process the mth spectral fram as desired to produce the spectrally modified output frame
    4. inverse FFT the spectral frame
    5. apply a synthesis window to yield a weighted output frame
    6. translate the mth output frame to time mR and add to thee accumulaated output signal
    '''

def ola_example():
    # impulse-train signal, 4KHz sampling-rate
    # Length L = 31 causal lowpass filterr, 600 Hz cut-off
    # Length M = L rectangular window
    # Hop size R = M (no overlap)

    # simulation params
    L = 31
    fc = 600
    fs = 4000

    # signal sample count
    Nsig = 150
    # signal period in samples
    period = int(round(L / 3))

    # FFT params
    M = L
    Nfft = util.nextpow2(M + L - 1)
    print(f'FFT size={Nfft}')
    # efficient window size
    M = Nfft - L + 1
    print(f'Window size={M}')
    R = M
    Nframes = 1 + np.floor((Nsig - M) / R)

    # impulse train
    sig = np.zeros(Nsig)
    sig[::period] = np.ones(len(range(0, Nsig, period)))
    plt.plot(sig, 'o')

    # low pass filter design via window method

    # zero-phase
    Lo2 = (L - 1) / 2

    # avoid 0 / 0
    epsilon = .0001
    nfilt = np.linspace(-Lo2, Lo2, L) + epsilon
    hideal = np.sin(2 * np.pi * fc * nfilt / fs) / (np.pi * nfilt)
    _, w = windows.hamming(L)
    # window the ideal impulse response
    h = w * hideal

    # zero-pad
    hzp = np.concatenate((h, np.zeros(Nfft-L)))
    H = np.fft.fft(hzp)

    # process via overlap-add

    # allocate output (Nfft) + ringing (Nsig) vector
    # pre/post-ringing length = half of filter length
    y = np.zeros(Nsig + Nfft)

    for m in np.arange(Nframes).astype(int):
        # indices for mth frame
        index = range(m*R, int(np.min(m*R+M)))
        xm = sig[index]
        xmzp = np.concatenate((xm, np.zeros(Nfft - len(xm))))
        Xm = np.fft.fft(xmzp)
        Ym = Xm * H
        ym = np.real(np.fft.ifft(Ym))
        outindex = range(m*R, m*R+Nfft)
        # overlap add
        y[outindex] += ym

    plt.figure()
    plt.plot(y)
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ola_example()
