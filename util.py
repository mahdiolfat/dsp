'''Typical function used in (statistical)(digital) signal processing'''

import numpy as np

def asinc(M, w):
    '''The Aliasied Sinc function, defined as the sampled Rectanlge Window.
       asinc becomes the sinc function in the limit as sampling rate -> inf'''
    denom = w / 2 
    num = denom * M
    return np.sin(num) / (M * np.sin(denom))

def fftshift(signal):
   '''take fs/2 as negative frequency and dc (0) as positive to balance out
      frequency components'''
   shifted = np.empty_like(signal)
   midpoint = int(len(signal) / 2)
   adj = 1 if len(signal) % 2 else 0
   shifted[:midpoint] = signal[midpoint+adj:]
   shifted[midpoint:] = signal[:midpoint+adj]
   return shifted

print(fftshift([1, 2, 3]))