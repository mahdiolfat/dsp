import numpy as np

'''
    - For stationary noise signals, the spectral phase is simply random, and therefore devoid of information!
'''

def white(count, mean=0, sd=1, swing=None):
    rng = np.random.default_rng()
    return rng.normal(mean, sd, count)

def test_white_noise(signal):
    ''' Test whether or not a set of samples can be well modeled as a white noise:
        - compute its sample autocorrelation and verify that it approaches an impulse in the limit as the number of samples becomes large
    '''

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a = [1, 1, 1, 1]
    N = len(a)
    print(N - np.arange(N))
    print(N + np.arange(-1, -N, -1))

    np.correlate(a, a, mode='full')
    plt.plot(white(10))
    #plt.show()