import numpy as np
import sys
import os

sys.path.append(os.path.abspath("."))
from pds_utils import sine

def dft(signal: np.ndarray) -> np.ndarray:
    # X[k] = sum(x[n] * exp(-2j * pi * k * n / N))
    N = len(signal)
    X = np.zeros(N, dtype=np.complex64)
    for k in range(N):
        Xi = []
        for n, x in enumerate(signal):
            tf = np.exp(-2j * np.pi * k * n / N)
            Xi.append(x * tf)
        X[k] = np.sum(Xi)

    return X


(_, signal) = sine.wave(
    sampling_freq=8, samples=8, amplitude=1, dc_level=0, frequency=1
)

print(dft(signal))
