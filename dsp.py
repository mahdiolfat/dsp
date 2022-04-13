import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import windows

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

def welch(signal):
    '''
        Also called the periodogram method
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

if __name__ == "__main__":
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
    #ps = dsp.powerspectrum(sig)
    #peaks, _ = scipy.signal.find_peaks(sig)
    #if len(peaks):
        #print(sig[peaks])

    plt.figure(0)
    plt.plot(space, sig)
    #plt.figure(1)
    #plt.plot(ps)
    plt.grid()
    plt.show()
