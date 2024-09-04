from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def wave(
    sampling_freq: float,
    samples: int,
    amplitude: float,
    dc_level: float,
    frequency: float,
    phase: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a sine wave signal

    Args:
        sampling_freq (float): Sampling frequency
        samples (int): Number of samples
        amplitude (float): Maximum amplitude of the sine wave
        dc_level (float): DC level of the sine wave
        frequency (float): Frequency of the sine wave
        phase (float): Phase of the sine wave

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            np.ndarray: Sine wave signal
            np.ndarray: Time array

    Notes:
        - The time array is generated from 0 to (samples - 1) * (1 / sampling_freq)
    """
    ts: float = 1 / sampling_freq
    t: np.ndarray = np.linspace(0, (samples - 1) * ts, samples)
    s: np.ndarray = amplitude * np.sin((2 * np.pi * frequency) * t + phase) + dc_level
    return (t, s)


def plot_sine(
    sampling_freq: float,
    samples: int,
    amplitude: float,
    frequency: float,
    phase: float = 0.0,
) -> None:
    """Plots a sine wave signal

    Args:
        sampling_freq (float): Sampling frequency
        samples (int): Number of samples
        amplitude (float): Maximum amplitude of the sine wave
        frequency (float): Frequency of the sine wave
        phase (float): Phase of the sine wave
    """

    # Calculate the time for one sine wave period
    sampling_period = 1 / frequency

    t, s = wave(amplitude, phase, frequency, sampling_freq, samples)
    plt.plot(t, s)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Frequency {frequency} [Hz]")
    plt.grid()
    # Limit x-axis to one sampling period
    plt.xlim(0, sampling_period)
    plt.show()


def plot_multiple_sine(
    sampling_freq: float,
    samples: int,
    amplitude: float,
    frequencies_and_phase: List[Tuple[float, float]],
) -> None:
    """Plots multiple sine wave signals on the same graph

    Args:
        amplitude (float): Maximum amplitude of the sine wave
        frequency (float): Frequency of the sine wave
        phase (float): Phase of the sine wave
        sampling_freq (float): Sampling frequency
        samples (int): Number of samples
    """
    num_frequencies = len(frequencies_and_phase)
    _, axs = plt.subplots(num_frequencies, 1, figsize=(10, 2 * num_frequencies))

    for i, (frequency, phase) in enumerate(frequencies_and_phase):
        t, s = wave(
            sampling_freq,
            samples,
            amplitude,
            0.0,
            frequency,
            phase,
        )
        axs[i].scatter(t, s, color="red", s=10, label="Samples")
        axs[i].plot(t, s)
        axs[i].set_title(f"Frequency {frequency} [Hz]")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Amplitude")
        axs[i].grid()
        period = 1 / frequency
        # Limit x-axis to 2 periods
        axs[i].set_xlim(0, 2 * period)
        # Limit y-axis to 1.2 times the amplitude
        axs[i].set_ylim(-amplitude * 1.2, amplitude * 1.2)

    plt.tight_layout()
    plt.show()
