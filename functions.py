'''Typical function used in (statistical)(digital) signal processing'''

import numpy as np

def asinc(M, w):
    '''The Aliasied Sinc function, defined as the sampled Rectanlge Window.
       asinc becomes the sinc function in the limit as sampling rate -> inf'''
    frange = np.arange(M)
    return np.sin(frange * w / 2) / (frange * np.sin( w / 2))

def fftshitf(buf):
    '''take fs/2 as negative frequency and dc as positive to balance out
       frequency components'''