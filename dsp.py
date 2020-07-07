import numpy as np
import scipy as sp
import scipy.signal
import scipy.linalg
import matplotlib.pyplot as plt

class dsp:
    @staticmethod
    def autocorrelation(signal):
        '''The cross-correlation of a signal with itself'''
        return [np.mean(data * signal) for data in signal]

    @staticmethod
    def powerspectrum(signal):
        return np.fft.fft(dsp.autocorrelation(signal))

def pisarenko(signal, order, autocorrelation=None):
    '''
    Psarenko's method for frequency estimation
    Process is assumed to consist of order complex exponentials in white noise

    signal: signal with additive white noise

    TODO:
        - use covariance
        - handle 0-meaning iput data, or just calculate the covariance
    '''
    if not autocorrelation:
        # estimate the (order + 1) x (order + 1) autocorrelation matrix
        autocorrelation = dsp.autocorrelation(signal)

    # autocorrelation sequence MUST be of size > order + 1
    # only the first (order + 1) of the autocorrelation sequence is used
    rx = np.array(autocorrelation[:order + 1])

    # (order + 1) x (order + 1) hermitian toeplitz auttocorrelation matrix
    autocorrelation_matrix = scipy.linalg.toeplitz(rx)
    print(autocorrelation_matrix)

    # since autocorrelation matrix is hermitian toeplitz, eigenvalues are non-negative real
    eval, evec = np.linalg.eigh(autocorrelation_matrix)
    print(f'Eigen values: {eval}')
    print("Eigen vectors: ", evec)

    # there is only one noise vector
    # it is the column vector corresponding to the minimum eigenvalue
    # the minimum eigenvalue absolute value is the variance of the additive white noise
    vmin_i = np.argmin(eval)
    variance = eval[vmin_i]
    print(f'Min eigenval (variance of white noise): {variance}')
    # noise vector
    vmin = evec[:,vmin_i]
    eigenfilter = scipy.signal.tf2zpk([1], vmin)

    # estimated frequencies
    freqs = np.angle(eigenfilter[1])

    # there are (order) signal vectors
    # TODO: calculate power associated with each signal vector

    return freqs, variance

if __name__ == "__main__":
    SAMPLE_COUNT = 10000
    space = np.linspace(0, 1, SAMPLE_COUNT)
    noise = np.random.normal(0, 0.1, SAMPLE_COUNT)
    sig = np.sin(space * 2*np.pi) + noise

    freqs, variance = pisarenko(sig, 1)

    print(f'Frequencies: {freqs}')
    print(f'Variance: {variance}')
    #ps = dsp.powerspectrum(sig)
    #peaks, _ = scipy.signal.find_peaks(sig)
    #if len(peaks):
        #print(sig[peaks])

    #plt.figure(0)
    #plt.plot(space, sig)
    #plt.figure(1)
    #plt.plot(ps)
    #plt.grid()
    #plt.show()
