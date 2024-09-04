import numpy as np
import sys
import os

sys.path.append(os.path.abspath("."))
from pds_utils import sine, plotter

fs = N = 500
(_, signal1) = sine.wave(
    sampling_freq=fs, samples=N, amplitude=1, dc_level=0, frequency=N / 4 + 0.25
)
(_, signal2) = sine.wave(
    sampling_freq=fs, samples=N, amplitude=1, dc_level=0, frequency=N / 4
)
(_, signal3) = sine.wave(
    sampling_freq=fs, samples=N, amplitude=1, dc_level=0, frequency=N / 4 + 0.5
)

X_fft1 = np.fft.fft(signal1)
X_fft2 = np.fft.fft(signal2)
X_fft3 = np.fft.fft(signal3)
plotter.plot_ffts([X_fft1, X_fft2, X_fft3], fs, True)
