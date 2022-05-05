import windows
import noise
import util

import numpy as np
import scipy.signal

def ola_framecount(signal_length, frame_length, hop_size):
    return np.floor((signal_length - frame_length) / hop_size) + 1

def autocorrelation_unbiased(signal):
    count = len(signal)
    bias = windows.bartlett(count + 1)[1] * count
    return np.correlate(signal, signal, mode='full') / bias

def sample_variance(signal):
    signal = signal - np.mean(signal)
    return 1 / len(signal) * np.sum(np.pow(signal, 2))

def acyclic_autocorrelation(signal):
    '''
    '''

def cyclic_autocorrelation(signal):
    '''
        Cyclic autocorrelation can be turned acyclic with zero-padding by a factor of 2 
    '''

def sample_autocorrelation(signal):
    '''
        - the cross correlation of a signal with itself is the biased autocorrelation
          the fourier transform of the biased autocorrelation is simply the squared-magnitude of the Fourier transform of the signal
          the bias is a multiplication of the unbiased sample autocorrelation by a Bartlett (triangular) window
          since the Fourier transform of a bartlett window is asinc**2, the DTFT of thee biased autocorrelation is a smoothed version
          of the unbiased PSD (convolved with asinc**2), the smoothing which is desired for statistical stability when analysing the PSD

        - The area under the PSD is the contribution of variance from the given frequency range

        Practical algorithm for a length M sequence:
            1. Choose the FFT size N  to be a power of 2 providing at least M - 1 samples of zero padding (N > 2 * M - 1)
            2. perform a length N FFT
            3. compute the squared magnitude
            4. compute the inverse FFT
            5. if desired, remove the bias by inverting the implicit bartlett-window weighting
    '''
    pass

def periodogram_direct(signal, window=None):
    '''
        Defined as the squared-magnitude DTFT of the windowed signal divided by number of samples
        The periodogram is equal to thee smoothed sample PSD. In the time domain, the autocorrelation function corresponding
        to the periodogram is the Bartlett windowed sample autocorrelation.

        The division by the number of samples can be interpreted as normalizing the peak of the implicit Bartlett window
        on the autocorrelation function to 1. Or as a normalization of the Fourier transform  itself, converting a power
        spectrum (squared-magnitude FFT) to a power spectral density.

        In the limit as M goes to infinity, the expected value of the periodogram equals the true power spectral density
        From the periodogram we should be able to recover a filter, which when used to filter white noise, creates a noise indistinguishable
        statistically from the observed sequence.

        Window is typically a rectangle
    '''

    return np.abs(np.fft.fft(signal))**2 / len(signal)

def periodogram_sample(signal, window=None):
    '''
        same as the direct method except that zero padding is used for stability
    '''

def welch_periodogram(signal, window_length, window_count):
    '''
        Also called the periodogram method
        COLA(M) implementation with a rectangle window
    '''
    M = window_length
    # zero pad to make acyclic
    Nfft = 2 * M
    # PSD accumulator
    Sv = np.zeros(Nfft)
    # per frame
    start = 0
    end = window_length
    for _ in range(window_count):
        v = signal[start:end]
        V = np.fft.fft(v, Nfft)
        # same as conj(V) .* V since abs is taken first
        Vms = np.abs(V)**2
        Sv = Sv + Vms
        start = end
        end = end + window_length

    # average of all scaled periodograms
    Sv = Sv / window_count
    # average bartlett-windowed sample autocorrelation
    rv = np.fft.ifft(Sv)
    rvup = np.concatenate((np.conj(rv[Nfft-M:]), np.conj(rv[:M+1])))
    # normalize for no bias at lag 0
    rvup = rvup / M
    Sv = Sv / M

    return Sv, rvup

def welch_autocorrelation(signal, window):
    '''
    '''
    pass

def autocorrelation(signal):
    '''
    The cross-correlation of a signal with itself -> biased autocorrelation
    '''
    autocorrelation = [np.mean(signal * np.roll(signal, shift))
                        for shift in range(signal.size)]

    variance = autocorrelation[0]
    print(f'Signal Variance: {variance}, {np.var(signal)}')
    return autocorrelation

def powerspectrum(signal):
    '''
        using the simple (biased) autocorrelation, this is the squared magnitude of the signal spectrum
    '''
    return np.fft.fft(autocorrelation(signal))

def welch_example():
    M = 32
    Ks = [1, 128]
    for K in Ks:
        Sv, rvup = welch_periodogram(noise.white(K * M), M, K)
        plt.figure()
        plt.plot(np.abs(Sv))
        # for zero-meaned white noise, autocorrrelation should be equal variance = 1
        plt.ylim(0, 5)
        plt.figure()
        plt.plot(np.abs(rvup))

def stft(signal, window, window_length, window_count, hop_size):
    '''
    Assuming - M is the analysis window legth, odd sized
             - N is a power of two larger than M.

        1. grab the data frame, time normalize about 0
        2. multiply time normalized frame from 1. by the spectrum analysis window to obtain the mth timee-normalized windowed data frame
        3. zero-pad the frame, 0's on each side for a size N, time-normalized dataset, until a factor of N/M zero-padding is aachieved
        4. take length N FFT to obtain the time-normalized frequency-sampled STFT at time m, with w_k = 2 * pi * k * fs / N, k is the bin number
        5. if needed, remove the time normalization via a linear phase term (phase = -mR, shift right by mR), this yields the sampled STFT

        NOTE: there is no irreversible time-aliasing wheen the STFT frequency axis w is sampled to the points w_k, provided thee FFT size N
        is greeater than or equal to the window length M.

        Since the STFT offers only one integration time (the window length), it implements a uniform baandpass filter bank, i.e., spectral samples
        are uniformly spaced and correspond to equal bandwidths.
    '''

    # assume odd
    M = window_length
    Mo2 = (M - 1) / 2
    # add Mo2 leading zeros to the signal so the last frame doesn't go out of range

    N = util.nextpow2(len(signal))

    # pre-allocate STFT output array
    Xtwz = np.zeros((N, window_count))

    padding = np.zeros(N-M)
    offset = 0
    for m in range(window_count):
        # grab the frame
        xt = signal[offset:offset+M]
        # apply the window
        xtw = window * xt
        # zero-pad
        xtwz = np.concatenate((xtw[Mo2:M], padding, xtw[:Mo2]))
        # fft and accumulate
        Xtwz[:,m] = np.fft.fft(xtwz)
        offset += hop_size

    return Xtwz

def filter_bank_example():
    # number of filters
    N = 10
    fs = 1000
    # duration in seconds
    D = 1
    # signal duration in samples
    L = int(np.ceil(fs * D) + 1)
    # discrete time axis in samples
    n = np.arange(L)
    # discrete time axis in seconds
    t = n / fs
    x = util.chirp(t, 0, D, fs/2, analytic=True)
    plt.plot(np.real(x))
    # rectangular
    #h = np.ones((N))
    # hamming
    _, h = windows.hamming(N)
    X = np.zeros((N, L), dtype=complex)
    for k in range(N):
        wk = 2 * np.pi * k / N
        # modulation by complex exponential
        xk = np.exp(-1j * wk * n) * x
        X[k,:] = scipy.signal.lfilter(h, 1, xk)
        plt.figure()
        plt.ylim(0, 6)
        plt.plot(np.abs(X[k]))

def overlap_add_fft():
    '''
        A summation of separately filtered windows.
        Frames MUST overlap.
    '''

def convolution_acyclic(x, h, N=None):
    '''
        The acyclic convolution of Nx samples with Nh samples produces at most Nx + Nh - 1 nonzero samples

        To embed acyclic convolution within a cyclic convolution (ie fft), zero-paad both operands out to length N,
        where N >= Nx + Nh - 1

        If we don't add enough zeros, some of the convolution terms "wrap around" and add back upon other (due to module indexing).
        This can be called time-domain aliasing!
        Zero-paadding in the time  domain results in more samples (closer spacing over the unit circle) in the frequency domain, ie.,
        a higher sampling  rate in the frequency domain.
    '''

    nx = len(x)
    nh = len(h)
    nfft = N if N is not None else util.nextpow2(nx + nh - 1)
    # zero pad signal and filter
    xzp = np.concatenate((x, np.zeros(nfft-nx)))
    hzp = np.concatenate((h, np.zeros(nfft-nh)))
    X = np.fft.fft(xzp)
    H = np.fft.fft(hzp)

    Y = X * H

    # get zero padded acyclic convolution
    y = np.real(np.fft.ifft(Y))
    # trim zeros
    yt = y[:nx+nh-1]

    return yt

def conv_timing():
    #x = [1, 2, 3, 4]
    #h = [1, 1, 1]
    x = np.arange(12000)
    h = np.arange(2000)

    yt = convolution_acyclic(x, h, N=16384)

def conv_timing_np():
    #x = [1, 2, 3, 4]
    #h = [1, 1, 1]
    x = np.arange(12000)
    h = np.arange(2000)

    yt = np.convolve(x, h)

def conv_timing_spfft():
    #x = [1, 2, 3, 4]
    #h = [1, 1, 1]
    x = np.arange(12000)
    h = np.arange(2000)

    yt = scipy.signal.fftconvolve(x, h)

def convolution_acyclic_example_compare():
    x = [1, 2, 3, 4]
    h = [1, 1, 1]

    yt = convolution_acyclic(x, h)
    print(f'yt={yt}')

    # compare
    yc_conv = np.convolve(x, h)
    print(f'yc_conv={yc_conv}')
    yc_fft = scipy.signal.fftconvolve(x, h)
    print(f'yc_fft={yc_fft}')

    print(timeit.timeit("conv_timing()", number=1000, setup="from __main__ import conv_timing"))
    print(timeit.timeit("conv_timing_np()", number=1000, setup="from __main__ import conv_timing_np"))
    print(timeit.timeit("conv_timing_spfft()", number=1000, setup="from __main__ import conv_timing_spfft"))

def convolution_acyclic_example_filter():
    x = [1, 2, 3, 4]
    h = [1, 1, 1]

    yt = convolution_acyclic(x, h)
    print(f'yt={yt}')

    # compare
    yc_conv = np.convolve(x, h)
    print(f'yc_conv={yc_conv}')
    yc_fft = scipy.signal.fftconvolve(x, h)
    print(f'yc_fft={yc_fft}')

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import timeit

    filter_bank_example()
    #convolution_acyclic_example_compare()
    #welch_example()
    plt.show()