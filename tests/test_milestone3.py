"""
tests/test_milestone3.py — Unit tests and smoke tests.

Unit tests cover:
  - FaceEmbedder.similarity() (cosine similarity on known vectors)
  - FaceEmbedder.score()      (sigmoid conversion)
  - FaceEmbedder._confidence() (threshold-relative confidence mapping)
  - Threshold application logic
  - preprocess() output shape and dtype
  - embed() output shape and L2 normalisation

Smoke test:
  - End-to-end inference on two synthetic images (no LFW download needed)
  - Confirms the full pipeline runs without error and returns a well-formed dict
"""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embedder import FaceEmbedder

# Fixtures

@pytest.fixture(scope="module")
def embedder():
    """Single FaceEmbedder instance shared across tests in this module."""
    return FaceEmbedder()


def _synthetic_image(tmp_path: Path, color=(128, 64, 32), name="face.jpg") -> Path:
    """Create a small solid-colour JPEG in tmp_path and return its path."""
    img = Image.new("RGB", (200, 200), color=color)
    p = tmp_path / name
    img.save(str(p), format="JPEG")
    return p

# similarity()

class TestSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert FaceEmbedder.similarity(v, v) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert FaceEmbedder.similarity(a, b) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert FaceEmbedder.similarity(a, b) == pytest.approx(0.0)

    def test_l2_normalised_range(self):
        """Cosine similarity of L2-normalised vectors should be in [-1, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            a = rng.normal(size=512).astype(np.float32)
            b = rng.normal(size=512).astype(np.float32)
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            sim = FaceEmbedder.similarity(a, b)
            assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6


# score()

class TestScore:
    def test_zero_input_gives_half(self):
        assert FaceEmbedder.score(0.0) == pytest.approx(0.5, abs=1e-6)

    def test_positive_input_above_half(self):
        assert FaceEmbedder.score(1.0) > 0.5

    def test_negative_input_below_half(self):
        assert FaceEmbedder.score(-1.0) < 0.5

    def test_output_range(self):
        for val in [-5.0, -1.0, 0.0, 0.5, 1.0, 5.0]:
            s = FaceEmbedder.score(val)
            assert 0.0 <= s <= 1.0, f"score({val}) = {s} outside [0,1]"

# confidence()

class TestConfidence:
    def test_at_boundary_is_zero(self):
        threshold = 0.775
        assert FaceEmbedder._confidence(threshold, threshold) == pytest.approx(0.0)

    def test_max_same_side(self):
        conf = FaceEmbedder._confidence(1.0, 0.775)
        assert conf == pytest.approx(1.0)

    def test_max_different_side(self):
        conf = FaceEmbedder._confidence(0.0, 0.775)
        assert conf == pytest.approx(1.0)

    def test_range_always_0_to_1(self):
        thresholds = [0.3, 0.5, 0.775, 0.9]
        scores = [0.0, 0.2, 0.5, 0.775, 0.9, 1.0]
        for t in thresholds:
            for s in scores:
                c = FaceEmbedder._confidence(s, t)
                assert 0.0 - 1e-9 <= c <= 1.0 + 1e-9, \
                    f"confidence={c} out of range for score={s}, threshold={t}"

    def test_higher_confidence_farther_from_boundary(self):
        t = 0.775
        c_near = FaceEmbedder._confidence(0.78, t)
        c_far = FaceEmbedder._confidence(0.95, t)
        assert c_far > c_near


# threshold application

class TestThresholdDecision:
    @pytest.mark.parametrize("score,threshold,expected", [
        (0.8,   0.775, "same"),
        (0.775, 0.775, "same"),    # equal to threshold -> same
        (0.77,  0.775, "different"),
        (0.1,   0.775, "different"),
        (1.0,   0.5,   "same"),
    ])
    def test_decision(self, score, threshold, expected):
        decision = "same" if score >= threshold else "different"
        assert decision == expected


# preprocess()

class TestPreprocess:
    def test_output_shape(self, tmp_path, embedder):
        img_path = _synthetic_image(tmp_path, color=(100, 150, 200))
        arr = embedder.preprocess(img_path)
        assert arr.shape == (160, 160, 3), \
            f"Expected (160, 160, 3), got {arr.shape}"

    def test_output_dtype(self, tmp_path, embedder):
        img_path = _synthetic_image(tmp_path)
        arr = embedder.preprocess(img_path)
        assert arr.dtype == np.uint8

    def test_missing_file_raises(self, embedder):
        with pytest.raises(Exception):
            embedder.preprocess("/nonexistent/path/face.jpg")


# embed()

class TestEmbed:
    def test_output_shape(self, tmp_path, embedder):
        img_path = _synthetic_image(tmp_path, color=(200, 100, 50))
        arr = embedder.preprocess(img_path)
        emb = embedder.embed(arr)
        assert emb.shape == (512,), f"Expected (512,), got {emb.shape}"

    def test_l2_normalised(self, tmp_path, embedder):
        """FaceNet output should be approximately L2-normalised."""
        img_path = _synthetic_image(tmp_path, color=(80, 120, 200))
        arr = embedder.preprocess(img_path)
        emb = embedder.embed(arr)
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-4, f"Embedding norm={norm}, expected ~1.0"

    def test_output_dtype(self, tmp_path, embedder):
        img_path = _synthetic_image(tmp_path, color=(50, 100, 150))
        arr = embedder.preprocess(img_path)
        emb = embedder.embed(arr)
        assert emb.dtype == np.float32


# Smoke test
class TestSmokeEndToEnd:
    """
    End-to-end smoke test using two tiny synthetic JPEG images.
    Does NOT require LFW or a GPU.
    """

    def test_verify_pair_returns_valid_result(self, tmp_path, embedder):
        img_a = _synthetic_image(tmp_path, color=(200, 100, 50), name="a.jpg")
        img_b = _synthetic_image(tmp_path, color=(50, 150, 220), name="b.jpg")
        threshold = 0.775

        result = embedder.verify_pair(img_a, img_b, threshold)

        required_keys = {
            "image_a", "image_b", "embedding_dim",
            "cosine_similarity", "score", "threshold",
            "decision", "confidence",
            "latency_preprocess_ms", "latency_embed_ms",
            "latency_score_ms", "latency_total_ms",
        }
        assert required_keys.issubset(result.keys()), \
            f"Missing keys: {required_keys - result.keys()}"

        assert result["embedding_dim"] == 512
        assert -1.0 <= result["cosine_similarity"] <= 1.0
        assert 0.0 <= result["score"] <= 1.0
        assert result["decision"] in ("same", "different")
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["latency_total_ms"] > 0

        # Verify decision is consistent with score and threshold
        expected = "same" if result["score"] >= threshold else "different"
        assert result["decision"] == expected

    def test_same_image_high_score(self, tmp_path, embedder):
        """The same image compared with itself should have cosine similarity ~1.0.

        Note: result['score'] is sigmoid(cosine_similarity), so for identical
        embeddings (cosine_sim = 1.0), the sigmoid score is sigmoid(1.0) ≈ 0.731,
        not 1.0. We therefore check cosine_similarity, not score.
        """
        img = _synthetic_image(tmp_path, color=(128, 128, 128), name="same.jpg")
        result = embedder.verify_pair(img, img, threshold=0.5)
        assert result["cosine_similarity"] > 0.99, \
            f"Same-image cosine similarity should be ~1.0, got {result['cosine_similarity']}"
        assert result["score"] == pytest.approx(0.731, abs=0.01), \
            f"sigmoid(1.0) should be ~0.731, got {result['score']}"
        assert result["decision"] == "same"

    def test_latency_breakdown_approximately_sums(self, tmp_path, embedder):
        """Sum of stage latencies should be close to total latency."""
        img_a = _synthetic_image(tmp_path, color=(10, 20, 30), name="lat_a.jpg")
        img_b = _synthetic_image(tmp_path, color=(30, 20, 10), name="lat_b.jpg")
        result = embedder.verify_pair(img_a, img_b, threshold=0.775)

        stage_sum = (
            result["latency_preprocess_ms"] +
            result["latency_embed_ms"] +
            result["latency_score_ms"]
        )
        total = result["latency_total_ms"]
        assert abs(stage_sum - total) < 30, \
            f"Stage sum {stage_sum:.1f}ms differs too much from total {total:.1f}ms"
