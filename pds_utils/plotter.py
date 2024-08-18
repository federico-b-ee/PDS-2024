from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np


def single(
    time: np.array,
    samples: np.array,
    frequency: float,
) -> None:
    """Plots an array of samples against time: s(t)

    Args:
        time (np.array): Time array
        samples (np.array): Array of samples
        frequency (float): Frequency of the signal
    """

    sampling_period = 1 / frequency

    plt.plot(time, samples)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Frequency {frequency} [Hz]")
    plt.grid()
    # Limit x-axis to one sampling period
    plt.xlim(0, sampling_period)
    plt.show()


def multiple(
    signals: List[Tuple[np.array, np.array, float]],
) -> None:
    """Plots multiple wave signals on the same graph

    Args:
        signals  List[Tuple[np.array, np.array, float]]:
        List of tuples containing:
            s (np.array): Array of samples
            time (np.array): Time array
            f (float): Frequency of the sine wave
    """
    num_frequencies = len(signals)
    _, axs = plt.subplots(num_frequencies, 1, figsize=(10, 2 * num_frequencies))

    for i, (t, s, f) in enumerate(signals):
        axs[i].scatter(t, s, color="red", s=10, label="Samples")
        axs[i].plot(t, s)
        axs[i].set_title(f"Frequency {f} [Hz]")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Amplitude")
        axs[i].grid()
        period = 1 / f
        # Limit x-axis to 2 periods
        axs[i].set_xlim(0, 2 * period)
        # Limit y-axis to the maximum and minimum values of s
        min_value = np.min(s)
        max_value = np.max(s)

        y_range = max_value - min_value
        padding = y_range * 0.2

        axs[i].set_ylim(min_value - padding, max_value + padding)

    plt.tight_layout()
    plt.show()
