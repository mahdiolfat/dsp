'''Typical function used in (statistical)(digital) signal processing'''

import numpy as np

def asinc(M, w):
    '''The Aliasied Sinc function, defined as the sampled Rectanlge Window.
       asinc becomes the sinc function in the limit as sampling rate -> inf'''
    denom = w / 2 
    num = denom * M
    return np.sin(num) / (M * np.sin(denom))

def fftshitf(signal):
    '''take fs/2 as negative frequency and dc as positive to balance out
       frequency components'''
    shifted = signal