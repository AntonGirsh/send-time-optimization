import numpy as np

def harmonic_f1(p_bank: np.ndarray, p_user: np.ndarray) -> np.ndarray:
    return 2 * p_bank * p_user / (p_bank + p_user + 1e-12)

def expected_conversion_score(p_bank: np.ndarray, p_user: np.ndarray) -> float:
    return float(harmonic_f1(p_bank, p_user).mean())