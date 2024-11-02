import numpy as np

class BandWith:
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper
        self.bandwidth = upper - lower

    def __str__(self):
        return (
            f"Ancho de Banda: {self.bandwidth:.2f} Hz\n"
            f"Límite Inferior: {self.lower:.2f} Hz\n"
            f"Límite Superior: {self.upper:.2f} Hz"
        )


def calculate_bandwidth(
    f: np.ndarray, Pxx: np.ndarray, percentage: float = 0.99
) -> BandWith:
    """
    Calcula el ancho de banda de una señal a partir de su densidad espectral de potencia.

    Args:
        f (np.ndarray): frecuencias del espectro.
        Pxx (np.ndarray): densidad espectral de potencia.
        percentage (float, optional): porcentaje de potencia acumulada. Defaults to 0.99.

    Raises:
        ValueError: ambas matrices de entrada no deben estar vacías.
        ValueError: las matrices de entrada 'f' y 'Pxx' deben tener la misma longitud.
        ValueError: no se encontró ancho de banda en el porcentaje especificado.

    Returns:
        BandWith: objeto con los límites de frecuencia del ancho de banda.
    """
    
    # Verifico si las entradas son válidas
    if len(f) == 0 or len(Pxx) == 0:
        raise ValueError("Las matrices de entrada no deben estar vacías.")

    if len(f) != len(Pxx):
        raise ValueError(
            "Las matrices de entrada 'f' y 'Pxx' deben tener la misma longitud."
        )

    # Calcular la potencia total
    total_power = np.sum(Pxx)

    # Calcular la distribución acumulativa
    cumulative_power = np.cumsum(Pxx)

    # Determinar la potencia umbral
    threshold_power = percentage * total_power

    # Encontrar los índices donde la potencia acumulativa supera el umbral
    lower_indices = np.where(cumulative_power >= (1 - percentage) * total_power)[0]
    upper_indices = np.where(cumulative_power >= threshold_power)[0]

    # Si no se encuentran índices, lanzar un error
    if lower_indices.size == 0 or upper_indices.size == 0:
        raise ValueError("No se encontró ancho de banda en el porcentaje especificado.")

    # Obtener los límites de frecuencia del ancho de banda
    bandwidth_lower = f[lower_indices[0]]  # Límite inferior
    bandwidth_upper = f[upper_indices[0]]  # Límite superior

    return BandWith(bandwidth_lower, bandwidth_upper)
