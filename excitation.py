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

    # N is even
    # results in N2 number of "distinct" spectral lines, but we later set dc component to 0
    N2 = N // 2
    # sample positive frequencies up to fs, skipping dc
    k = process.integers(low=1, high=N, size=N2-1)
    print(len(k))

    # signal is real, so its spectrum is hermitian
    # negative frequencies are conjugate symmetric
    # magnitude is even while the phase is odd
    positive_spectrum = np.exp(1j * 2 * np.pi * k / N)
    negative_spectrum = np.flip(positive_spectrum)
    # create oscillator bank with dc value U[0] = 0
    print(len(positive_spectrum))
    U = np.array([0] + list(positive_spectrum) + [0] + list(np.conj(negative_spectrum)), dtype=np.complex128)
    print(len(U))

    # extract the continuous time domain signal corresponding to the constructed fourier series (hermitian spectrum)
    # norm of "ortho" applies a 1/sqrt(N) scaling, as desired
    u = np.fft.ifft(U, norm="ortho")
    return u, U

def normalized_periodic_noise(N, normalized=True, fs=1, process=None):
    '''
    Amplitudes U(k*fs/N) and phases are the realization of independent random processes (jointly and over k).

    with normalization, the clock frequency fs is independent of N.
    '''
    pass


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


def oscillator_bank(N, freqs, amps, process=None)->list[float]:
    k = np.arange(N)
    harmonics = len(freqs)
    oscillators = [amp * np.sin(np.pi * freq * k) for freq, amp in zip(freqs, amps)]
    return np.sum(np.array(oscillators), axis=0)


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


def swept_sine_example():
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
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
