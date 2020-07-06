import numpy as np
import scipy as sp
import scipy.signal
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

    signal with additive white noise
    '''
    if autocorrelation:
        if len(autocorrelation) != order + 1:
            raise ValueError("Pisarenki: autocorrelation sequence MUST be the size of order + 1")
        # TODO: estimate autocorrelation here

    # estimate the (order + 1) x (order + 1) autocorrelation matrix
    autocorrelation = dsp.autocorrelation(signal)
    rx = np.array(autocorrelation[:order + 1])

    # TODO: 
    autocorrelation_matrix = sp.linalg.toeplitz(rx)
    print(autocorrelation_matrix)

    eval, evec = np.linalg.eigh(autocorrelation_matrix)
    print(f'Eigen values: {eval}')
    sigma = {np.min(eval)}
    print(f'Min eigenval: {sigma}')
    print("Eigen vectors: ", evec)

    # there is only one noise vector
    # there are (order) signal vectors
    # TODO: find the minimum eigenvalue and the corresponding (p + 1) x (p + 1) auttocorrelation
    # matrix Rx

if __name__ == "__main__":
    SAMPLE_COUNT = 10000
    space = np.linspace(0, 1, SAMPLE_COUNT)
    noise = np.random.normal(0, 0.1, SAMPLE_COUNT)
    sig = np.sin(space * 2*np.pi) + noise

    pisarenko(sig, 1)
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
