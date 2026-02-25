import numpy as np
import random
import math

def vectorized_similarity(a: np.ndarray, b: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Uses NumPy's vectorized operations to perform cosine similarity calculations
    a and b are 2D arrays of shape (n, d) where n is the number of vectors and d is the dimension"""
    if metric == "cosine":
        ans = (np.sum(a * b, axis=1)) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1, keepdims=True).T)
        return ans.reshape(-1)
    elif metric == "euclidean":
        ans = np.linalg.norm(a - b, axis=1)
        return ans.reshape(-1)
    else:
        raise ValueError("Metric must be 'cosine' or 'euclidean'")

def naive_similarity(a: np.ndarray, b: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Uses a Python loop to compute cosine similarity naively, without NumPy"""
    if metric == "cosine":
        result = []
        for i in range(len(a)):
            norm_a = 0
            norm_b = 0
            dot_product = 0
            for j in range(len(a[0])):
                norm_a += a[i][j] ** 2
                norm_b += b[i][j] ** 2
                dot_product += a[i][j] * b[i][j]
            norm_a = math.sqrt(norm_a)
            norm_b = math.sqrt(norm_b)
            result.append(dot_product / (norm_a * norm_b))
        return np.array(result)
    elif metric == "euclidean":
        result = []
        for i in range(len(a)):
            distance = 0
            for j in range(len(a[0])):
                distance += (a[i][j] - b[i][j]) ** 2
            result.append(math.sqrt(distance))
        return np.array(result)
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