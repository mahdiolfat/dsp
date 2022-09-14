import numpy as np

def coherence(Syu, Suu, Syy) -> float:
    ''' measures how much of the output power is coherent (linearly related) with the input power'''

    return np.abs(Syu)**2 / (Suu * Syy)

def dispersion(Ni, No, normalized=True):
    pass

# In practice, the variance on the FRF is estimated  from  the coherence using (2-50)

# Hanning window (2-12) combined with 1/2 overlap is a good compromise between leakage error
# suppression and  computational  effort.
# These settings are often the default choice in digital spectrum analyzers.

# the FRF of a MIMO system is described by an ny x nu matrix at each frequency