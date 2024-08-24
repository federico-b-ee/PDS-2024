import numpy as np
from matplotlib import pyplot as plt
import sys
import os

sys.path.append(os.path.abspath("."))
from pds_utils import sine


def dft(signal: np.ndarray) -> np.ndarray:
    # X[k] = sum(x[n] * exp(-2j * pi * k * n / N))
    N = len(signal)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        Xi = []
        for n, x in enumerate(signal):
            tf = np.exp(-2j * np.pi * k * n / N)
            Xi.append(x * tf)
        X[k] = np.sum(Xi)
    return X

# Alternative implementation using broadcasting, made by the AI
def dft_ai(signal: np.ndarray) -> np.ndarray:
    N = len(signal)
    X = np.zeros(N, dtype=complex)
    n = np.arange(N)
    k = np.arange(N).reshape((N, 1))  # Reshape for broadcasting
    # Compute the DFT using broadcasting
    X = np.sum(signal * np.exp(-2j * np.pi * k * n / N), axis=1)
    return X


(_, signal) = sine.wave(
    sampling_freq=10, samples=10, amplitude=1, dc_level=0, frequency=1
)

X_fft = np.fft.fft(signal)
X_dft_ai = dft_ai(signal)
X_dft = dft(signal)

fig, axs = plt.subplots(2, 4, figsize=(15, 10))
X_fft = np.round(X_fft, decimals=5)
X_dft = np.round(X_dft, decimals=5)
X_dft_ai = np.round(X_dft_ai, decimals=5)

X_smooth = X_dft.copy()
X_smooth[X_smooth == 0+-0j] = 0+0j

print(X_dft)
print(X_fft)

# DFT
axs[0, 0].stem(np.abs(X_dft) / len(signal))
axs[0, 0].set_xlabel("Frequency")
axs[0, 0].set_ylabel("Magnitude")
axs[0, 0].set_title("DFT Signal")
axs[0, 0].set_xlim([-1, len(signal) + 1])

# DFT AI
axs[0, 1].stem(np.abs(X_dft_ai) / len(signal))
axs[0, 1].set_xlabel("Frequency")
axs[0, 1].set_ylabel("Magnitude")
axs[0, 1].set_title("DFT AI Signal")
axs[0, 1].set_xlim([-1, len(signal) + 1])

# FFT
axs[0, 2].stem(np.abs(X_fft) / len(signal))
axs[0, 2].set_xlabel("Frequency")
axs[0, 2].set_ylabel("Magnitude")
axs[0, 2].set_title("FFT Signal")
axs[0, 2].set_xlim([-1, len(signal) + 1])

# DFT Smooth
axs[0, 3].stem(np.abs(X_smooth) / len(signal))
axs[0, 3].set_xlabel("Frequency")
axs[0, 3].set_ylabel("Magnitude")
axs[0, 3].set_title("DFT smooth Signal")
axs[0, 3].set_xlim([-1, len(signal) + 1])

# DFT Phase
axs[1, 0].stem(np.angle(X_dft)/np.pi)
axs[1, 0].set_xlabel("Frequency")
axs[1, 0].set_ylabel("Phase (π)")
axs[1, 0].set_title("DFT Signal Phase")
axs[1, 0].set_ylim([-1.5, 1.5])

# DFT AI Phase
axs[1, 1].stem(np.angle(X_dft_ai)/np.pi)
axs[1, 1].set_xlabel("Frequency")
axs[1, 1].set_ylabel("Phase (π)")
axs[1, 1].set_title("DFT AI Signal Phase")
axs[1, 1].set_ylim([-1.5, 1.5])

# FFT Phase
axs[1, 2].stem(np.angle(X_fft)/np.pi)
axs[1, 2].set_xlabel("Frequency")
axs[1, 2].set_ylabel("Phase (π)")
axs[1, 2].set_title("FFT Signal Phase")
axs[1, 2].set_ylim([-1.5, 1.5])

# DFT Smooth Phase
axs[1, 3].stem(np.angle(X_smooth)/np.pi)
axs[1, 3].set_xlabel("Frequency")
axs[1, 3].set_ylabel("Phase (π)")
axs[1, 3].set_title("DFT smooth Phase")
axs[1, 3].set_ylim([-1.5, 1.5])

plt.tight_layout()
plt.show()
