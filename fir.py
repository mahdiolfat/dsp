import numpy as np

def normalized_freq(freq):
    return 2 * np.pi * freq

def lowpass_ideal(fc, fs=1, fcount=100):
    np.full(())
    np.ones((100))

def lowpass(freq_pb, ripple_pb, freq_sb, ripple_sb):
    '''
        - passband ripple is an allowed gain deviation
        - stopband ripple is an allowed leakage level
        - optimal:
            - ripples = 0
            - SBA = inf
            - transition width is 0, 
    '''
    transition_width = freq_sb - freq_pb
    stopband_attenuation = 

if __name__ == "__main__":
    import matplotlib.pyplot as plt