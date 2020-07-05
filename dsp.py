import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

SAMPLE_COUNT = 100
x = np.linspace(0, 1, SAMPLE_COUNT)
sig = np.sin(x * 2*np.pi) + np.random.normal(0, 1, SAMPLE_COUNT)

autocorrelation = np.dot(sig, sig) / 100
print(autocorrelation)
powerspectrum = np.abs(np.fft.rfft(autocorrelation))
print(powerspectrum)
peaks, _ = scipy.signal.find_peaks(powerspectrum)
if len(peaks):
    print(powerspectrum[peaks])

plt.figure(0)
plt.plot(x, sig)
plt.figure(1)
plt.plot(powerspectrum)
plt.grid()
plt.show()
