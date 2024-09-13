from typing import Tuple

import numpy as np

def normal(
    signal: np.ndarray,
    mean: float,
    std_dev: float,
    amount_of_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a signal simualating normal noise given the standard deviation and mean

    Args:
        signal (np.ndarray): Signal.
        mean (float): Mean of the noise.
        std_dev (float): Standard deviation of the noise.
        amount_of_samples (int): Amount of samples of noise to generate.


    Returns:
        np.ndarray: Signal with Quantization process.
        np.ndarray: Noise.
    """
    n = np.random.normal(mean, std_dev, amount_of_samples)
    sn = signal + n
    return (sn, n)
