import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bench_similarity  # noqa: E402


class TestBenchSimilarity(unittest.TestCase):
    def test_cosine_similarity_obvious_examples(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        sim = bench_similarity.vectorized_similarity(a, b, metric="cosine")
        np.testing.assert_allclose(sim, np.array([1.0, 1.0], dtype=np.float32), rtol=0, atol=1e-7)

        a2 = np.array([[1.0, 0.0]], dtype=np.float32)
        b2 = np.array([[0.0, 1.0]], dtype=np.float32)
        sim2 = bench_similarity.vectorized_similarity(a2, b2, metric="cosine")
        np.testing.assert_allclose(sim2, np.array([0.0], dtype=np.float32), rtol=0, atol=1e-7)

    def test_naive_and_vectorized_agree(self):
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        b = np.array([[7.0, 8.0, 9.0], [6.0, 5.0, 4.0]], dtype=np.float32)

        vec = bench_similarity.vectorized_similarity(a, b, metric="cosine")
        nai = bench_similarity.naive_similarity(a, b, metric="cosine")
        np.testing.assert_allclose(vec, nai, rtol=0, atol=1e-7)

        vec_e = bench_similarity.vectorized_similarity(a, b, metric="euclidean")
        nai_e = bench_similarity.naive_similarity(a, b, metric="euclidean")
        np.testing.assert_allclose(vec_e, nai_e, rtol=0, atol=1e-6)

    def test_invalid_metric_raises(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0]], dtype=np.float32)
        with self.assertRaises(ValueError):
            bench_similarity.vectorized_similarity(a, b, metric="bad")
        with self.assertRaises(ValueError):
            bench_similarity.naive_similarity(a, b, metric="bad")


if __name__ == "__main__":
    unittest.main()
