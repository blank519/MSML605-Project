"""
scripts/score_pairs.py — Run model inference on a pairs CSV and write a
scored CSV that the threshold sweep and evaluation scripts can consume.

This is the bridge between train.py (which saves the model) and the
Milestone 2 evaluation pipeline (which needs 'score' and 'label' columns).

Usage
-----
    # Score the validation split with the saved model:
    python scripts/score_pairs.py \
        --config configs/milestone2.yaml \
        --split val \
        --model-path outputs/models/siamese_verifier.keras

    # Score the test split:
    python scripts/score_pairs.py \
        --config configs/milestone2.yaml \
        --split test \
        --model-path outputs/models/siamese_verifier.keras

Outputs
-------
    outputs/scores/{split}_scored.csv
        Columns: left_path, right_path, label, split, score
        'score' is the sigmoid probability from SiameseVerifier.
        Higher score = more likely same person.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from project root whether called from root or scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

from scripts.utils import find_lfw_root, load_config
# Import custom model classes so Keras can locate them during load_model().
# SiameseVerifier and FaceEmbedder must be in scope before load_model() runs,
# otherwise Keras raises 'Could not locate class SiameseVerifier'.
from scripts.model import FaceEmbedder, SiameseVerifier  # noqa: F401

from src.embedder import FaceEmbedder as StandaloneFaceEmbedder


# ---------------------------------------------------------------------------
# Image loading  (reuses the same logic as train.py)
# ---------------------------------------------------------------------------

def _read_image(path: str, image_size: tuple[int, int]) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(
        image_bytes, channels=3, expand_animations=False
    )
    img = tf.image.resize(img, image_size, method=tf.image.ResizeMethod.BILINEAR)
    return tf.cast(img, tf.float32)


def _resolve_pair_path(lfw_root: Path, stored_path: str) -> Path:
    """Resolve a stored left_path/right_path to an absolute file path.

    make_pairs.py stores paths relative to lfw_root (e.g.
    'Colin_Powell/Colin_Powell_0001.jpg'), so the correct join is simply
    lfw_root / stored_path.  This helper centralises that logic so any
    future change to the storage convention only needs to be updated here.
    """
    p = Path(stored_path)
    # Guard: if the path somehow still has a leading 'lfw/' segment, strip it
    # (can happen if pairs were generated with an older version of make_pairs.py).
    parts = list(p.parts)
    if parts and parts[0].lower() == "lfw":
        p = Path(*parts[1:])
    return lfw_root / p


def _validate_image_paths(df: pd.DataFrame, lfw_root: Path, sample_size: int = 20) -> None:
    """Check that image files referenced in the pairs CSV actually exist on disk.

    Samples up to *sample_size* rows (first + last) to catch both the start
    and end of the file list without scanning every row.  Raises a clear
    FileNotFoundError with actionable guidance if any are missing.
    """
    check_rows = pd.concat([df.head(sample_size // 2), df.tail(sample_size // 2)]).drop_duplicates()
    missing: list[str] = []
    for row in check_rows.itertuples(index=False):
        for col in ("left_path", "right_path"):
            full = _resolve_pair_path(lfw_root, getattr(row, col))
            if not full.exists():
                missing.append(str(full))
        if len(missing) >= 5:
            break   # report the first batch; no need to scan further

    if missing:
        examples = "\n  ".join(missing[:5])
        raise FileNotFoundError(
            f"\n\n[score_pairs] {len(missing)} sampled image path(s) do not exist "
            f"under lfw_root:\n  {lfw_root}\n\nMissing examples:\n  {examples}\n\n"
            "Likely causes and fixes:\n"
            "  1. INCOMPLETE EXTRACTION — The TFDS download did not fully extract.\n"
            "     Fix: delete the tfds_cache and re-run:\n"
            "       python scripts/ingest_lfw.py --config configs/milestone2.yaml\n\n"
            "  2. PAIRS CSV FROM A DIFFERENT MACHINE — The pairs were generated on\n"
            "     a machine with a different cache location.\n"
            "     Fix: re-run make_pairs.py so paths match this machine's cache:\n"
            "       python scripts/make_pairs.py --config configs/milestone2.yaml\n\n"
            "  3. WRONG lfw_root — The extracted directory layout changed.\n"
            "     Fix: run scripts/verify_pair_paths.py to diagnose the layout:\n"
            "       python scripts/verify_pair_paths.py --config configs/milestone2.yaml\n"
        )


def _make_score_dataset(
    df: pd.DataFrame,
    lfw_root: Path,
    image_size: tuple[int, int],
    batch_size: int,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset of (img_a, img_b) pairs in the same order as df."""
    left  = df["left_path"].astype(str).apply(
        lambda p: str(_resolve_pair_path(lfw_root, p))
    ).tolist()
    right = df["right_path"].astype(str).apply(
        lambda p: str(_resolve_pair_path(lfw_root, p))
    ).tolist()

    ds = tf.data.Dataset.from_tensor_slices((left, right))

    def _map_fn(lp, rp):
        return _read_image(lp, image_size), _read_image(rp, image_size)

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_pairs_csv(df: pd.DataFrame, path: Path) -> None:
    required = {"left_path", "right_path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Pairs CSV {path} is missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Labels in {path} must be binary (0/1). Found: {unique_labels}"
        )
    if len(df) == 0:
        raise ValueError(f"Pairs CSV {path} is empty.")


# ---------------------------------------------------------------------------
# Main scoring logic
# ---------------------------------------------------------------------------

def score_pairs(
    model_path: Path,
    pairs_csv: Path,
    lfw_root: Path,
    out_csv: Path,
    image_size: tuple[int, int] = (160, 160),
    batch_size: int = 32,
    scoring_mode: str = "model",
) -> pd.DataFrame:
    """Run inference and attach a 'score' column to the pairs DataFrame.

    Parameters
    ----------
    model_path : Path to a saved SiameseVerifier (.keras file).
    pairs_csv  : Path to a pairs CSV with left_path, right_path, label.
    lfw_root   : Root directory of extracted LFW images.
    out_csv    : Where to write the scored CSV.
    image_size : (H, W) — must match what the model was trained on.
    batch_size : Inference batch size.

    Returns
    -------
    DataFrame with all original columns plus 'score'.
    """
    # Load and validate pairs
    df = pd.read_csv(pairs_csv)
    _validate_pairs_csv(df, pairs_csv)
    print(f"  Pairs loaded : {len(df):,} pairs from {pairs_csv}")

    # Check that image files actually exist before loading the (slow) model.
    # Catches incomplete TFDS extractions or stale pairs CSVs immediately.
    print(f"  Validating image paths under: {lfw_root} ...")
    _validate_image_paths(df, lfw_root)
    print(f"  Image path check passed.")

    all_scores: list[float] = []
    scoring_mode = str(scoring_mode).lower()

    if scoring_mode == "model":
        # Load model
        print(f"  Loading model: {model_path}")
        model = tf.keras.models.load_model(str(model_path))

        # Build dataset (preserves row order)
        ds = _make_score_dataset(df, lfw_root, image_size, batch_size)

        # Run inference
        print(f"  Running inference (batch_size={batch_size}) ...")
        for img_a_batch, img_b_batch in ds:
            logits = model((img_a_batch, img_b_batch), training=False)
            probs = tf.sigmoid(logits)  # (batch, 1)
            all_scores.extend(probs.numpy().flatten().tolist())

    elif scoring_mode == "embedding":
        if tuple(image_size) != (160, 160):
            raise ValueError(
                "Embedding scoring mode requires image_size=(160, 160) to match "
                "keras-facenet. Got image_size=%r" % (tuple(image_size),)
            )

        embedder = StandaloneFaceEmbedder()

        def _facenet_embeddings(images: list[np.ndarray]) -> np.ndarray:
            try:
                return embedder._facenet.embeddings(images, verbose=0)
            except TypeError:
                return embedder._facenet.embeddings(images)

        print(f"  Running embedding scoring (batch_size={batch_size}) ...")
        left_paths = df["left_path"].astype(str).tolist()
        right_paths = df["right_path"].astype(str).tolist()

        def _abs_paths(paths: list[str]) -> list[str]:
            return [str(_resolve_pair_path(lfw_root, p)) for p in paths]

        left_abs = _abs_paths(left_paths)
        right_abs = _abs_paths(right_paths)

        for i in range(0, len(df), batch_size):
            batch_left = left_abs[i : i + batch_size]
            batch_right = right_abs[i : i + batch_size]

            imgs_left = [embedder.preprocess(p) for p in batch_left]
            imgs_right = [embedder.preprocess(p) for p in batch_right]

            emb_left = _facenet_embeddings(imgs_left)
            emb_right = _facenet_embeddings(imgs_right)

            cos = np.sum(emb_left * emb_right, axis=1)
            scores = 1.0 / (1.0 + np.exp(-cos))
            all_scores.extend(scores.astype(float).tolist())

    else:
        raise ValueError(
            f"Unknown scoring_mode={scoring_mode!r}. Expected 'model' or 'embedding'."
        )

    # Sanity check
    if len(all_scores) != len(df):
        raise RuntimeError(
            f"Score count ({len(all_scores)}) != pair count ({len(df)}). "
            "Inference result does not align with input pairs."
        )

    # Attach scores and save
    df = df.copy()
    df["score"] = all_scores

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"  Scored CSV saved -> {out_csv}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score LFW pairs with the saved SiameseVerifier model"
    )
    parser.add_argument("--config",     required=True,  help="Path to YAML config")
    parser.add_argument(
        "--split", default="val",
        choices=["train", "val", "test"],
        help="Which pairs split to score (default: val)",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Path to saved .keras model (overrides config)",
    )
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument(
        "--scoring-mode",
        choices=["model", "embedding"],
        default=None,
        help="Scoring definition: 'model' uses saved SiameseVerifier; 'embedding' uses keras-facenet embeddings + sigmoid(cosine_similarity).",
    )
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=None,
        help="Image H W (default: from config or 160 160)",
    )
    parser.add_argument(
        "--extracted-root", default=None,
        help="Path to TFDS extracted root (overrides config)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg  = load_config(args.config)

    outputs_dir = PROJECT_ROOT / cfg.get("paths", {}).get("outputs", "outputs")
    pairs_dir   = outputs_dir / "pairs"
    scores_dir  = outputs_dir / "scores"

    # Resolve image size
    if args.image_size is not None:
        image_size = tuple(args.image_size)
    else:
        image_size = tuple(
            cfg.get("training", {}).get("image_size", [160, 160])
        )

    # Resolve scoring mode
    scoring_mode = args.scoring_mode
    if scoring_mode is None:
        if "milestone3" in cfg:
            scoring_mode = "embedding"
        else:
            scoring_mode = "model"

    # Resolve model path (only required for model-based scoring)
    if args.model_path is not None:
        model_path = Path(args.model_path)
    else:
        model_path = outputs_dir / "models" / "siamese_verifier.keras"

    if scoring_mode == "model" and not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    # Resolve LFW root
    if args.extracted_root is not None:
        extracted_root = Path(args.extracted_root)
    else:
        data_cache = cfg.get("paths", {}).get("data_cache", "data/tfds_cache")
        extracted_root = PROJECT_ROOT / data_cache / "downloads" / "extracted"

    lfw_root = find_lfw_root(extracted_root)

    pairs_csv = pairs_dir / f"{args.split}_pairs.csv"
    out_csv   = scores_dir / f"{args.split}_scored.csv"

    print(f"\n[score_pairs]")
    print(f"  Split        : {args.split}")
    print(f"  Scoring mode : {scoring_mode}")
    if scoring_mode == "model":
        print(f"  Model        : {model_path}")
    print(f"  LFW root     : {lfw_root}")
    print(f"  Image size   : {image_size}")

    score_pairs(
        model_path=model_path,
        pairs_csv=pairs_csv,
        lfw_root=lfw_root,
        out_csv=out_csv,
        image_size=image_size,
        batch_size=args.batch_size,
        scoring_mode=scoring_mode,
    )
    print("[score_pairs] Done.\n")


if __name__ == "__main__":
    main()
