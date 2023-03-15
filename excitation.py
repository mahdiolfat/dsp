'''Module for generating excitation signals'''

import numpy as np

#For robustness in the presence of spectral modifications, the frame rate should be morre than twice the highest main-lob frequency

SAMPLING_FREQUENCY = 1e6
DRIVE_FREQUENCY = 1e6
DAC_CLOCK = 12e6

ADC_CLOCK = 150e6 / 4
ADC_RATE  = ADC_CLOCK / 23

LUT_SIZE = 2**16 - 1

# 1.2us
DRIVE_BANDWIDTH = 20e3
DAC_RATE = 150e6 / (11 + 2) / 12
SAMPLE_PER_CYCLE = DAC_RATE // 20e3

# drive conditions that produce the maximum amplitude signal
BIAS = 0.82
UNITY_AMPLITUDE = 0.1


def random_phase_multisine(N, P=1, gain=1, fs=1, process=None):
    '''
        N: number of "time samples" in one signal period

        P: number of time domain periods to generate

        With a normalized amplitude spectrum, the DFT coefficient magnitudes U(k*fs/N) are scaled by 1/sqrt(N)
        U(k*fs/N) are uniformley bounded with a finite number of discontinuities on [0, fs/2].
        Further, the clock frequency fs is independent of N.

        Phase mean drawn from the random process must = 0
    '''

    # spectral resolution = 2*pi / N

    # F is the number of unique spectral lines between [0, fs/2)
    if process is None:
        process = np.random.default_rng()

    # N could be odd or even
    odd = N % 2
    # results in N2 number of "real" spectral lines, but we later set dc component to 0
    N2 = (N - odd) // 2
    # sample positive frequencies up to fs, skipping dc
    k = process.integers(low=1, high=N, size=N2)

    # signal is real, so its spectrum is hermitian
    # negative frequencies are conjugate symmetric
    # magnitude is even while the phase is odd
    phase = np.pi * k / N
    spectral_lines = np.exp(1j * phase)
    positive_spectrum = [0] + list(spectral_lines[:-1])
    negative_spectrum = np.conj(np.flip(spectral_lines))
    # create oscillator bank with dc value U[0] = 0
    U = np.array(list(positive_spectrum) + list(negative_spectrum), dtype=np.complex128)

    # extract the continuous time domain signal corresponding to the constructed fourier series (hermitian spectrum)
    # norm of "ortho" applies a 1/sqrt(N) scaling, as desired

    U = np.fft.fftshift(U)
    u = np.fft.ifft(U, norm="ortho")
    return u, phase


def periodic_noise(N, P=1, gain=1, fs=1, process=None):
    '''
    Amplitudes U(k*fs/N) and phases are the realization of independent random processes (jointly and over k).

    with normalization, the clock frequency fs is independent of N.
    '''
    if process is None:
        process = np.random.default_rng(123)

    k = process.integers(low=1, high=N, size=N-1)
    U = np.zeros(N * 2, dtype=np.complex128)

    # signal is real, so its spectrum is hermitian (negative frequencies are conjugate symmetric)
    samples = k / N * gain * np.exp(1j * 2 * np.pi * k / N)
    # create oscillator bank with dc value U[0] = 0
    U[1:N] = samples

    # create time domain signal
    u = 2 * np.sqrt(UNITY_AMPLITUDE) * np.real(np.fft.irfft(U))
    return u, U[:N]


def gaussian_noise(N, P=1, gain=1, process=None):
    if process is None:
        process = np.random.default_rng(123)

    k = process.integers(low=1, high=N, size=N-1)
    gshape = np.sqrt(2 * np.pi * k / N) * np.exp(-2 * np.pi**2 * (k / N)**2)
    plt.figure()
    plt.plot(gshape)
    U = np.zeros(N * 2, dtype=np.complex128)

    # signal is real, so its spectrum is hermitian (negative frequencies are conjugate symmetric)
    samples = gshape * gain * np.exp(1j * 2 * np.pi * k / N)
    # create oscillator bank with dc value U[0] = 0
    U[1:N] = samples

    # create time domain signal
    u = 2 * np.sqrt(UNITY_AMPLITUDE) * np.real(np.fft.irfft(U))
    return u, U[:N]


def zippered_multisine():
    ''' Input signals are chosen such that each input contains F excited frequencies
        at an interleaved frequency grid'''
    pass


def orthogonal_multisine():
    '''Does not require the number of inputs to be a power of two'''
    pass


def hadamard_multisine():
    '''Requires number of inputs to be a power of two'''
    pass


def hadamard_matrix():
    ''' Square orthogonal matrix '''
    pass


def frf_uncertainty(G0, varY, varU, sY0Y0, SU0U0, SY0U0, varYU):
    ''' The uncertainty is inverrsely proportional to the total power of the excitation signal and also to the shape of its power spectrum '''
    pass

def crest_factor(sig: list[float]) -> float:
    crest = len(sig)
    return crest


def oscillator_bank(N, frequencies, amplitudes, fs=1)->list[float]:
    freqs = np.array(frequencies, dtype=float)
    amps = np.array(amplitudes, dtype=float)
    if np.any(freqs > fs):
        raise ValueError(f'frequencies must be greater than the sampling rate fs={fs / 2}')

    # number of samples that give a full cycle of the desired freq

    oscillators = [amp * np.sin(2 * np.pi * np.arange(N) / N)
                   for freq, amp in zip(freqs, amps)]
    return np.sum(np.array(oscillators), axis=0)

def swept_sine2(f1: float, f2: float, fs: float=73242.1875) -> None:
    # f0 is the minimum resolution

    N = 504
    T0 = 1 / fs * N
    f0 = 1 / T0
    #assert f1 < fc and f2 < fc
    k1 = int(np.round(f1 / f0))
    k2 = int(np.round(f2 / f0))

    # k1 and k2 must be positive ints and > 1, otherwise the frequency resolution is not enough within the given time window
    # k2 > k1
    assert k1 > 1 and k2 > 1
    assert k2 > k1

    a = np.pi * (k2 - k1) * f0**2
    b = 2 * np.pi * k1 * f0

    n = np.linspace(0, T0, N, endpoint=False)

    f1 = k1 * f0
    f2 = k2 * f0
    return n, np.sin((a * n + b) * n), (f1, f2)

def swept_sine(f1: float, f2: float, Ts: float=1) -> None:
    ''' f1, f2, in Hertz
        TS in seconds
    '''

    # f0 is the minimum resolution
    f0 = 1 / Ts
    #assert f1 < fc and f2 < fc
    k1 = int(np.round(f1 / f0))
    k2 = int(np.round(f2 / f0))

    # k1 and k2 must be positive ints and > 1, otherwise the frequency resolution is not enough within the given time window
    # k2 > k1
    assert k1 > 1 and k2 > 1
    assert k2 > k1

    a = np.pi * (k2 - k1) * f0**2
    b = 2 * np.pi * k1 * f0

    # TODO: how many points are required to avoid freq domain aliasing?
    print(k2 * f0 / np.pi)
    # np.pi / N 
    N = 4096
    n = np.linspace(0, Ts, N, endpoint=False)

    f1 = k1 * f0
    f2 = k2 * f0
    return n, np.sin((a * n + b) * n), (f1, f2)

def shroeder_multisine(f1, f2, Ts, F) -> None:
    # F is the number of distinct spectral lines
    f0 = 1 / Ts
    k1 = int(np.round(f1 / f0))
    k2 = int(np.floor(f2 / f0))

    fs = (k2 - k1) * f0 / F
    print(fs)

    assert fs > 1

    k = np.arange(k1, k2, fs)

    print(k1, k2)
    print(len(k))
    phases = -k * (k - 1)
    freqs =  k * fs
    Fs = int(round(k2 / fs))

    prezp = np.zeros(int(round(k1 / fs)))
    postzp = np.zeros(int(round(k2 / fs)))
    #positive_spectrum = list(prezp) + list(np.exp(1j * np.pi / Fs * (freqs + phases))) + list(postzp)
    positive_spectrum = list(prezp) + list(np.exp(1j * np.pi / Fs * (freqs + phases)))
    negative_spectrum = np.conj(np.flip(positive_spectrum))
    # create oscillator bank with dc value U[0] = 0
    U = np.array([0] + list(positive_spectrum) + list(negative_spectrum), dtype=np.complex128)

    # extract the continuous time domain signal corresponding to the constructed fourier series (hermitian spectrum)
    # norm of "ortho" applies a 1/sqrt(N) scaling, as desired
    u = np.fft.ifft(U, norm="ortho")
    return u, U


def swept_sine_example() -> None:
    t, sig, freqs = swept_sine(20, 100, Ts=1)
    sig = sig * np.pi
    print(freqs)

    smoother = 1
    ffted = np.fft.rfft(sig, len(sig) * smoother, norm='ortho') / np.sqrt(2)
    print(np.amax(np.abs(ffted)))
    print(np.sqrt(np.amax(np.abs(ffted))))
    plt.plot(t, sig)
    plt.figure()
    plt.plot(np.linspace(0, len(ffted) / smoother, len(ffted)), np.abs(ffted))
    plt.xlim(0, 200)


def shroeder_multisine_example() -> None:
    u, U = shroeder_multisine(20, 303, 1, 303-20-1)

    #U = np.fftshift
    plt.plot(np.real(u))
    plt.figure()
    plt.plot(np.abs(U[:len(U)//2]))
    #plt.xlim(0, 350)
    plt.grid()
    plt.figure()
    plt.plot(np.angle(U))


def oscillator_bank_example() -> None:
    freq = 8e3
    k1 = DAC_RATE // freq
    k2 = DAC_RATE // freq / 8
    sig = oscillator_bank(1024, np.array([1/k1, 1/k2]), np.array([1, 0.1]))
    print(sig[-1])
    plt.figure()
    plt.plot(sig)

def discrete_spectral_lines(N: int, fs: float = 1) -> list[float]:
    k = np.arange(N)
    return k * fs / N

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #swept_sine_example()
    #shroeder_multisine_example()
    sig = oscillator_bank(8, [0.5], [1], fs=1)
    plt.plot(sig, marker='o')
    plt.grid()

    S = np.fft.fftshift(np.fft.fft(sig, 64))
    plt.figure()
    plt.plot(np.abs(S), marker='o')
    plt.axvline(len(S)//2)
    plt.figure()
    plt.plot(np.angle(S), marker='o')
    plt.axvline(len(S)//2)

    #plt.figure()
    #signal, Signal = random_phase_multisine(N=65)
    #signal = np.real(signal)
    #fig, ax = plt.subplots(3, 1)
    #ax[0].plot(signal, marker='o')
    #ax[1].plot(np.abs(Signal), marker='o')
    #ax[2].plot(np.angle(Signal), marker='o')

    plt.show()
