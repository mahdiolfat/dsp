import numpy as np

def coherence(Syu, Suu, Syy) -> float:
    ''' measures how much of the output power is coherent (linearly related) with the input power'''

    return np.abs(Syu)**2 / (Suu * Syy)