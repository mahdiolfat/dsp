import numpy as np

def lorentzian(x, width, x0=0):
    return 0.5 * width / np.pi / ((x - x0)**2 + (0.5 * width)**2)

def msample_variance():
    pass

def allan_variance():
    pass
