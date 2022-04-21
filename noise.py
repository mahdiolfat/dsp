import numpy as np
from scipy import signal

'''
    - For stationary noise signals, the spectral phase is simply random, and therefore devoid of information!
'''

def white(count, mean=0, variance=1, seed=None):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, np.sqrt(variance), count)

def test_white_noise(signal):
    ''' Test whether or not a set of samples can be well modeled as a white noise:
        - compute its sample autocorrelation and verify that it approaches an impulse in the limit as the number of samples becomes large
        - equivilently, its periodogram should be constant
    '''

def pink(count):
    ''' filtered white noise where the amplitude response is proportional to 1/sqrt(f) and PSD is proportional to 1/f
    also known as "1/f noise" or "equal loudness noise"

    Generate using Gaussian white noise
    '''
    filter_num = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    filter_den = [1, -2.494956002, 2.017265875, -0.522189400]
    # location of the first pole at 60dB, indicating end of transient reponse
    nT60 = int(np.round(np.log10(1000) / (1 - max(np.abs(np.roots(filter_den))))))
    wnoise = white(count+nT60, variance=1)
    pnoise = signal.lfilter(filter_num, filter_den, wnoise)
    # skip transient response
    pnoise = pnoise[int(nT60)+1:]
    return pnoise

def brown(count):
    ''' filtered white noise where the amplitude response is proportional to 1/f and PSD is proportional to 1/f**2
    also known as "brownian motion" or "Wiener process" or "random increments" or "random walk"

    Integrated white noise

    Use a leaky integrator to remain within bounds
    '''
    pass

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import dsp

    pnoise = pink(2**16)
    print(pnoise)
    response = dsp.periodogram_direct(pnoise)
    plt.plot(10*np.log10(response[:len(response)//2]))
    plt.show()

    # unbiased autocorrelation:
    # remove that Bartlett window

    #plt.plot(white(10))
    #plt.show()