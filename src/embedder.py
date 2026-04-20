"""
src/embedder.py — FaceNet embedding stage for CLI and load test.

Uses keras-facenet (wrapping InceptionResnetV1/VGGFace2) to produce 512-dimensional L2-normalised face embeddings, consistent with the FaceNetEmbedder used inside SiameseVerifier in scripts/model.py.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


class FaceEmbedder:
    """Standalone FaceNet embedder for CLI inference and load testing."""

    INPUT_SIZE = (160, 160)

    def __init__(self) -> None:
        from keras_facenet import FaceNet
        self._facenet = FaceNet()

    # Preprocessing

    def preprocess(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load and resize an image to (160, 160, 3) uint8 numpy array."""
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.INPUT_SIZE, Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    # Embedding generation

    def embed(self, image_array: np.ndarray) -> np.ndarray:
        """Run FaceNet and return a (512,) L2-normalised embedding.

        Parameters
        ----------
        image_array : Output of preprocess(), shape (160, 160, 3), uint8.

        Returns
        -------
        np.ndarray of shape (512,), float32, L2-normalised.
        """
        batch = np.expand_dims(image_array, axis=0)
        try:
            emb = self._facenet.embeddings(batch, verbose=0)
        except TypeError:
            emb = self._facenet.embeddings(batch)
        return emb[0].astype(np.float32)

    # Similarity scoring

    @staticmethod
    def similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised embeddings."""
        return float(np.dot(emb_a, emb_b))

    @staticmethod
    def score(cosine_sim: float) -> float:
        """Convert cosine similarity to a sigmoid probability in [0, 1]."""
        return float(1.0 / (1.0 + np.exp(-cosine_sim)))

    # Full pair inference with per-stage timing

    def verify_pair(
        self,
        path_a: Union[str, Path],
        path_b: Union[str, Path],
        threshold: float,
    ) -> dict:
        """Run the full inference pipeline on one pair and return a result dict."""
        t0 = time.perf_counter()

        # Preprocessing
        t1 = time.perf_counter()
        arr_a = self.preprocess(path_a)
        arr_b = self.preprocess(path_b)
        t2 = time.perf_counter()

        # Embedding
        ea = self.embed(arr_a)
        eb = self.embed(arr_b)
        t3 = time.perf_counter()

        # Score decision and confidenc
        cos_sim = self.similarity(ea, eb)
        sig_score = self.score(cos_sim)
        decision = "same" if sig_score >= threshold else "different"
        confidence = self._confidence(sig_score, threshold)
        t4 = time.perf_counter()

        return {
            "image_a":               str(path_a),
            "image_b":               str(path_b),
            "embedding_dim":         512,
            "cosine_similarity":     round(cos_sim, 6),
            "score":                 round(sig_score, 6),
            "threshold":             threshold,
            "decision":              decision,
            "confidence":            round(confidence, 4),
            "latency_preprocess_ms": round((t2 - t1) * 1000, 2),
            "latency_embed_ms":      round((t3 - t2) * 1000, 2),
            "latency_score_ms":      round((t4 - t3) * 1000, 2),
            "latency_total_ms":      round((t4 - t0) * 1000, 2),
        }

    # Confidence

    @staticmethod
    def _confidence(score: float, threshold: float) -> float:
        """Map a sigmoid score to a [0, 1] confidence value."""
        if score >= threshold:
            denom = max(1.0 - threshold, 1e-9)
            return min((score - threshold) / denom, 1.0)
        else:
            denom = max(threshold, 1e-9)
            return min((threshold - score) / denom, 1.0)
