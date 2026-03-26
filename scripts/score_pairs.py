"""
score_pairs.py — Run model inference on a pairs CSV and write a scored CSV that the threshold sweep and evaluation scripts can consume

Bridge between train.py (which saves the model) and the Milestone 2 evaluation pipeline (which needs 'score' and 'label' columns)

Usage:
    # Score the validation split with the saved model:
    python scripts/score_pairs.py
        - config configs/milestone2.yaml
        - split val
        - model-path outputs/models/siamese_verifier.keras

    # Score the test split:
    python scripts/score_pairs.py
        - config configs/milestone2.yaml
        - split test
        - model-path outputs/models/siamese_verifier.keras

Outputs:
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

# Load the image

def _read_image(path: str, image_size: tuple[int, int]) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(
        image_bytes, channels=3, expand_animations=False
    )
    img = tf.image.resize(img, image_size, method=tf.image.ResizeMethod.BILINEAR)
    return tf.cast(img, tf.float32)


def _make_score_dataset(
    df: pd.DataFrame,
    lfw_root: Path,
    image_size: tuple[int, int],
    batch_size: int,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset of (img_a, img_b) pairs in the same order as df."""
    left  = df["left_path"].astype(str).apply(lambda p: str(lfw_root / p)).tolist()
    right = df["right_path"].astype(str).apply(lambda p: str(lfw_root / p)).tolist()

    ds = tf.data.Dataset.from_tensor_slices((left, right))

    def _map_fn(lp, rp):
        return _read_image(lp, image_size), _read_image(rp, image_size)

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# Validation of the pairs CSV format and content before scoring

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


# Scoring logic

def score_pairs(
    model_path: Path,
    pairs_csv: Path,
    lfw_root: Path,
    out_csv: Path,
    image_size: tuple[int, int] = (160, 160),
    batch_size: int = 32,
) -> pd.DataFrame:
    """Run inference and attach a 'score' column to the pairs DataFrame.

    Args:
        model_path: Path to a saved SiameseVerifier (.keras file).
        pairs_csv: Path to a pairs CSV with left_path, right_path, label.
        lfw_root: Root directory of extracted LFW images.
        out_csv: Where to write the scored CSV.
        image_size: (H, W) — must match what the model was trained on.
        batch_size: Inference batch size.

    Retuns:
        DataFrame with all original columns plus 'score'.
    """
    # Load and validate pairs
    df = pd.read_csv(pairs_csv)
    _validate_pairs_csv(df, pairs_csv)
    print(f"  Pairs loaded : {len(df):,} pairs from {pairs_csv}")

    # Load model
    print(f"  Loading model: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    # Build dataset (preserves row order)
    ds = _make_score_dataset(df, lfw_root, image_size, batch_size)

    # Run inference
    print(f"  Running inference (batch_size={batch_size}) ...")
    all_scores: list[float] = []
    for img_a_batch, img_b_batch in ds:
        logits = model((img_a_batch, img_b_batch), training=False)
        probs  = tf.sigmoid(logits)                        # (batch, 1)
        all_scores.extend(probs.numpy().flatten().tolist())

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


#CLI

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

    # Resolve model path
    if args.model_path is not None:
        model_path = Path(args.model_path)
    else:
        model_path = outputs_dir / "models" / "siamese_verifier.keras"

    if not model_path.exists():
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
    )
    print("[score_pairs] Done.\n")


if __name__ == "__main__":
    main()
