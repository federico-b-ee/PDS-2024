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
    signals: List[Tuple[np.array, np.array, float]], scatter: bool = False
) -> None:
    """Plots multiple wave signals on the same graph

    Args:
        signals  List[Tuple[np.array, np.array, float]]:
        List of tuples containing:
            s (np.array): Array of samples
            time (np.array): Time array
            f (float): Frequency of the sine wave
            scatter (bool): Scatter plot
    """
    num_frequencies = len(signals)
    _, axs = plt.subplots(num_frequencies, 1, figsize=(10, 2 * num_frequencies))

    for i, (t, s, f) in enumerate(signals):
        if scatter:
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


def multiple_cmp_adc(
    signals: List[Tuple[List[np.array], int]], time: np.array, frequency: float
) -> None:
    """Plots multiple wave signals on the same graph
    Args:
        s (np.array): List of samples
        time (np.array): Time array
        frquency (float): Frequency of the sine wave
        bits (int): Number of bits for the quantization
    """
    num_frequencies = len(signals)
    _, axs = plt.subplots(num_frequencies, 1, figsize=(10, 2 * num_frequencies))

    period = 1 / frequency
    t = time
    for i, list_signals in enumerate(signals):
        signals, bits = list_signals
        for s in signals:
            axs[i].plot(t, s)
            axs[i].set_title(f"Bits: {bits}")
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel("Amplitude")
            axs[i].grid()
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


def plot_fft(fft: np.ndarray, fs: float, in_db: bool) -> None:
    """Plots the magnitude and phase of the FFT coefficients.

    Args:
        fft     (np.ndarray): FFT coefficients
        fs      (float): Sampling Frequency
        in_db   (bool): plot in dB
    """
    freq_bins = np.fft.fftfreq(fft.size, d=1 / fs)
    freq_bins = freq_bins[freq_bins >= 0]
    positive_magnitude = np.abs(fft)[: len(freq_bins)] / fft.size
    positive_phase = np.angle(fft)[: len(freq_bins)] / np.pi
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    if in_db:
        # Convert magnitude to dB
        reference = np.max(positive_magnitude)  # Reference for normalization
        positive_magnitude_db = 20 * np.log10(positive_magnitude / reference)
        axs[0].stem(freq_bins, positive_magnitude_db, use_line_collection=True)
        axs[0].set_ylabel("Magnitude (dB)")
        axs[0].set_title("FFT Signal in dB")
    else:
        axs[0].stem(freq_bins, positive_magnitude, use_line_collection=True)
        axs[0].set_ylabel("Magnitude")
        axs[0].set_title("FFT Signal")
    axs[0].set_xlim([-2, np.max(freq_bins) + 2])

    axs[1].stem(freq_bins, positive_phase)
    axs[1].set_xlabel("Frequency")
    axs[1].set_ylabel("Phase (radians)")
    axs[1].set_title("FFT Phase")
    axs[1].set_xlim([-2, np.max(freq_bins) + 2])
    plt.tight_layout()
    plt.show()


def plot_ffts(ffts: List[np.ndarray], fs: float, in_db: bool) -> None:
    """Plots the magnitude and phase of multiple FFT coefficients.

    Args:
        ffts    (list[np.ndarray]): List of FFT coefficients
        fs      (float): Sampling Frequency
        in_db   (bool): Plot in dB
    """
    num_signals = len(ffts)
    fig, axs = plt.subplots(num_signals, 2, figsize=(15, 5 * num_signals))

    for i, fft in enumerate(ffts):
        freq_bins = np.fft.fftfreq(fft.size, d=1 / fs)
        freq_bins = freq_bins[freq_bins >= 0]

        positive_magnitude = np.abs(fft)[: len(freq_bins)] / fft.size
        positive_phase = np.angle(fft)[: len(freq_bins)] / np.pi

        if in_db:
            reference = np.max(positive_magnitude)
            positive_magnitude_db = 20 * np.log10(positive_magnitude / reference)
            axs[i, 0].stem(freq_bins, positive_magnitude_db, use_line_collection=True)
            axs[i, 0].set_ylabel("Magnitude (dB)")
            axs[i, 0].set_title(f"FFT Signal {i+1} in dB")
        else:
            axs[i, 0].stem(freq_bins, positive_magnitude, use_line_collection=True)
            axs[i, 0].set_ylabel("Magnitude")
            axs[i, 0].set_title(f"FFT Signal {i+1}")

        axs[i, 0].set_xlabel("Frequency (Hz)")
        axs[i, 0].set_xlim([0, np.max(freq_bins)])

        axs[i, 1].stem(freq_bins, positive_phase, use_line_collection=True)
        axs[i, 1].set_xlabel("Frequency (Hz)")
        axs[i, 1].set_ylabel("Phase (radians)")
        axs[i, 1].set_title(f"FFT Phase {i+1}")
        axs[i, 1].set_xlim([0, np.max(freq_bins)])

    plt.tight_layout()
    plt.show()
