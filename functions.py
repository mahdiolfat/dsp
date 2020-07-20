'''Typical function used in (statistical)(digital) signal processing'''

import numpy as np

def asinc(M, w):
    '''The Aliasied Sinc function, defined as the sampled Rectanlge Window'''
    frange = np.arange(M)
    return np.sin(frange * w / 2) / (frange * np.sin( w / 2))
