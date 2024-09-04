from typing import Tuple

import numpy as np

def adc(
    signal: np.ndarray,
    bits: int = 8,
    vcc: float = 3.3,
    vee: float = -3.3,
) -> Tuple[np.ndarray, float]:
    """Returns a signal simualating the quantization process of an ADC

    Args:
        signal (np.ndarray): Signal.
        bits (int): Bits for the Quantization.
        vcc (float): Positive supply voltage.
        vee (float): Negative supply voltage.


    Returns:
        np.ndarray: Signal with Quantization process.
        float: Quantization steps
    """
    levels = 2**bits
    q = (vcc - vee) / (levels - 1)

    s = np.clip(signal, vee, vcc)
    # Make it start from 0
    s = (s - vee) / q

    s = np.round(s)

    s = s * q + vee
    return (s, q)
