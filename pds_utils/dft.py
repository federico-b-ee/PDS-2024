import numpy as np
import matplotlib.pyplot as plt


def dft(signal: np.ndarray) -> np.ndarray:
    """Computes the Discrete Fourier Transform (DFT) of a given signal.

    Args:
        signal (np.ndarray): The input signal. In the time domain.

    Returns:
        np.ndarray: Output of the DFT -> Complex-valued coefficients
    """
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
    """Computes the Discrete Fourier Transform (DFT) of a given signal. Using broadcasting.

    Args:
        signal (np.ndarray): The input signal. In the time domain.

    Returns:
        np.ndarray: Output of the DFT -> Complex-valued coefficients
    """
    N = len(signal)
    X = np.zeros(N, dtype=complex)
    n = np.arange(N)
    k = np.arange(N).reshape((N, 1))  # Reshape for broadcasting
    X = np.sum(signal * np.exp(-2j * np.pi * k * n / N), axis=1)
    return X


def plot_dft(dft: np.ndarray) -> None:
    """Plots the magnitude and phase of the DFT coefficients.

    The phase seems to have some issues with precision of the complex numbers.
    https://dsp.stackexchange.com/questions/10638/phase-of-the-fft-of-a-sine-function
    
    Args:
        dft (np.ndarray): DFT coefficients
    """
    axs = plt.subplots(2, 1, figsize=(15, 10))
    axs[0].stem(np.abs(dft) / len(dft))
    axs[0].set_xlabel("Frequency")
    axs[0].set_ylabel("Magnitude")
    axs[0].set_title("DFT Signal")
    axs[0].set_xlim([-1, len(dft) + 1])

    axs[1].stem(np.angle(dft) / np.pi)
    axs[1].set_xlabel("Frequency")
    axs[1].set_ylabel("Phase (radians)")
    axs[1].set_title("DFT Phase")
    axs[1].set_xlim([-1, len(dft) + 1])
    plt.tight_layout()
    plt.show()
