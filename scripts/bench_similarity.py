import numpy as np
import random

def vectorized_similarity(a: np.ndarray, b: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Uses NumPy's vectorized operations to perform cosine similarity calculations
    a and b are 2D arrays of shape (n, d) where n is the number of vectors and d is the dimension"""
    if metric == "cosine":
        ans = (a @ b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1, keepdims=True).T)
        return ans
    elif metric == "euclidean":
        ans = np.sqrt(np.sum((a - b) ** 2, axis=1))
        return ans
    else:
        raise ValueError("Metric must be 'cosine' or 'euclidean'")

def naive_similarity(a: np.ndarray, b: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Uses a Python loop to compute cosine similarity naively"""
    if metric == "cosine":
        pass
    elif metric == "euclidean":
        pass
    else:
        raise ValueError("Metric must be 'cosine' or 'euclidean'")

def generate_synthetic_data(n: int, d: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic data for benchmarking"""

    pass

def benchmark(a: np.ndarray, b: np.ndarray, seed: int = 42):
    """ Compares the performance of vectorized and naive similarity calculations 
    for both cosine and euclidean similarity calculations
    """
    rng = random.Random(seed)

    pass