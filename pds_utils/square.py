from typing import Tuple

import numpy as np


def wave(
    sampling_freq: float,
    samples: int,
    amplitude: float,
    dc_level: float,
    frequency: float,
    phase: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a square wave signal

    Args:
        sampling_freq (float): Sampling frequency
        samples (int): Number of samples
        amplitude (float): Maximum amplitude of the square wave
        dc_level (float): DC level of the square wave
        frequency (float): Frequency of the square wave
        phase (float): Phase of the square wave

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            np.ndarray: Square wave signal
            np.ndarray: Time array

    Notes:
        - The time array is generated from 0 to (samples - 1) * (1 / sampling_freq)
    """
    ts: float = 1 / sampling_freq
    t: np.ndarray = np.linspace(0, (samples - 1) * ts, samples)
    s: np.ndarray = amplitude * np.sin((2 * np.pi * frequency) * t + phase)
    square_wave = s > 0
    square_wave = square_wave + dc_level
    return (t, square_wave)
