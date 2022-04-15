import numpy as np
import scipy as sp
import windows
import noise

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    welch_example()
    plt.show()