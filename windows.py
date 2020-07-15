'''A module containing a set of window filters'''

import numpy as np

def rectangle(M):
    '''
    The rectangle window.
    Zero-phase: symmetric about 0
    M must be odd
    '''

    # calculate positive half only, flip and use for negative half
    wrange = np.arange(-M + 1, M)
    window = (np.abs(wrange) <= (M - 1) / 2.).astype(int)
    print(np.array([wrange, window]).T)
    return (window, wrange)

def asinc(M):
    '''The Aliasied Sinc function, defined as the sampled Rectanlge Window'''
    count = M

class Hamming:

    def __init__(M):
        self._count = M

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    count = 21
    w = rectangle(count)
    plt.bar(w[1], w[0], width=0.1)
    plt.show()
