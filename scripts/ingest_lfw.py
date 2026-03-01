"""
ingest_lfw.py

Deterministic LFW ingestion script.

Usage:
    python scripts/ingest_lfw.py --config configs/milestone1.yaml

Outputs:
    outputs/manifest.json
    - Dataset manifest with counts, seed, and split policy
"""
# Required imports
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Required third-party imports, included in requirements.txt
import numpy as np
import yaml


# Load yaml config

def load_config(config_path: str) -> dict:
    """Loads the yaml file and returns a dictionary of its contents.

    Args:
        config_path (str): yaml file path

    Returns:
        dict: key-value pairs from the yaml file
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Loading LFW dataset using tensorflow-datasets (TFDS)

def load_lfw_tfds(cache_dir: str):
    """Load LFW from TFDS and returns an ordered list of the image records.
    
    Args:
    cache_dir (str): Local directory where TFDS will cache downloaded data.

    Returns:
        list[dict]: A list of records sorted by (identity, filename), where each
        record contains:
            - identity (str): The person's name
            - filename (str): image filename.
            - image_path (str): image file path
    """
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        print(
            "ERROR: tensorflow-datasets is not installed. Make sure that requirements.txt is installed and that you are using the correct Python environment."
        )
        sys.exit(1)

    print(f"[ingest] Loading LFW via TFDS (cache_dir={cache_dir}) ...")
    ds, info = tfds.load(
        "lfw",
        split = "train",
        data_dir = cache_dir,
        with_info = True,
        # We will handle shuffling and splitting ourselves, so disable TFDS shuffling
        shuffle_files = False,
    )

    records = []
    for example in ds.as_numpy_iterator():
        # TFDS LFW fields: 'image', 'label' (identity index), 'image/filename'
        identity_idx = int(example["label"])
        identity_name = info.features["label"].int2str(identity_idx)
        # Decode filename bytes
        raw_fname = example.get("image/filename", b"")
        if isinstance(raw_fname, bytes):
            raw_fname = raw_fname.decode("utf-8")
        filename = raw_fname if raw_fname else f"{identity_name}_{len(records):06d}.jpg"

        records.append(
            {
                "identity": identity_name,
                "filename": filename,
                # Store a logical path relative to the cache for portability
                "image_path": os.path.join("lfw", identity_name, filename),
            }
        )

    # Deterministic ordering: sort by (identity, filename)
    records.sort(key=lambda r: (r["identity"], r["filename"]))
    print(f"[ingest] Loaded {len(records)} images across "
          f"{len({r['identity'] for r in records})} identities.")
    return records


# Deterministic split by identit

def split_by_identity(records: list, seed: int, val_frac: float, test_frac: float,) -> dict[str, list]:
    """
    Split image records into train, val, and test sets by identity, ensuring no identity appears in more than one split.

    Args:
        records (list): List of image record dicts from load_lfw_tfds.
        seed (int): Random seed for reproducible shuffling.
        val_frac (float): Fraction of identities to assign to the val split.
        test_frac (float): Fraction of identities to assign to the test split.

    Returns:
        dict[str, list]: A dictionary with keys 'train', 'val', and 'test' mapping to a list of image record dicts
    """
    # Sorted unique identities (deterministic order before shuffle)
    all_identities = sorted({r["identity"] for r in records})
    n = len(all_identities)

    # Seed already set in config
    rng = random.Random(seed)
    shuffled = all_identities[:]
    rng.shuffle(shuffled)

    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))

    test_ids = set(shuffled[:n_test])
    val_ids = set(shuffled[n_test: n_test + n_val])
    train_ids = set(shuffled[n_test + n_val:])

    splits = defaultdict(list)
    for r in records:
        if r["identity"] in test_ids:
            splits["test"].append(r)
        elif r["identity"] in val_ids:
            splits["val"].append(r)
        else:
            splits["train"].append(r)

    return dict(splits)

# Manifest writing

def compute_checksum(path: str) -> str:
    """
    Compute the SHA-256 checksum of a file for determinism verification.
    Similar to in-class exercise A5.

    Args:
        path (str): Path to the file to checksum.

    Returns:
        str: Hexadecimal SHA-256 digest of the file contents.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(splits: dict[str, list], seed: int, split_policy: str, val_frac: float, test_frac: float, data_source: str, cache_dir: str,) -> dict:
    """
    Build a manifest dictionary summarising the ingestion run.

    Args:
        splits (dict): Output of split_by_identity.
        seed (int): Random seed used during ingestion.
        split_policy (str): Human-readable description of the split strategy.
        val_frac (float): Fraction of identities assigned to val.
        test_frac (float): Fraction of identities assigned to test.
        data_source (str): Description of how LFW was obtained.
        cache_dir (str): Local path where the dataset is cached.

    Returns:
        dict: Manifest containing seed, split_policy, val_frac, test_frac,counts (images and identities per split), data_source, cache_dir, and a UTC timestamp
    """
    counts = {}
    for split_name, recs in splits.items():
        identities = {r["identity"] for r in recs}
        counts[split_name] = {
            "images": len(recs),
            "identities": len(identities),
        }

    return {
        "seed": seed,
        "split_policy": split_policy,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "counts": counts,
        "data_source": data_source,
        "cache_dir": str(cache_dir),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


# Main

def main():
    """
    Main function

    Reads the config file, loads LFW, applies deterministic splits, writes the manifest to outputs/manifest.json, and saves each split as a CSV file to outputs/splits/.
    """
    parser = argparse.ArgumentParser(description="Ingest LFW and write manifest.")
    # Force user to provide a config file path
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed: int = cfg["seed"]
    val_frac: float = cfg["splits"]["val_frac"]
    test_frac: float = cfg["splits"]["test_frac"]
    cache_dir: str = cfg["paths"]["data_cache"]
    outputs_dir: str = cfg["paths"]["outputs"]
    split_policy: str = cfg["splits"]["policy"]
    data_source: str = cfg.get("data_source", "tensorflow_datasets: lfw (latest)")

    # Seed numpy for reproducibility
    np.random.seed(seed)

    # Ensure output directory exists
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)

    # Load LFW
    records = load_lfw_tfds(cache_dir)

    # Splits
    splits = split_by_identity(records, seed=seed, val_frac=val_frac, test_frac=test_frac)

    for split_name, recs in splits.items():
        n_identities = len({r['identity'] for r in recs})
        print(f"[ingest] {split_name:5s}: {len(recs):6d} images, {n_identities:5d} identities")

    # Build and write manifest
    manifest = build_manifest(
        splits=splits,
        seed=seed,
        split_policy=split_policy,
        val_frac=val_frac,
        test_frac=test_frac,
        data_source=data_source,
        cache_dir=cache_dir,
    )

    manifest_path = os.path.join(outputs_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    checksum = compute_checksum(manifest_path)
    print(f"[ingest] Manifest written to {manifest_path}")
    print(f"[ingest] Manifest SHA-256: {checksum}")

    # Save split record lists for downstream use (make_pairs.py)
    splits_dir = os.path.join(outputs_dir, "splits")
    Path(splits_dir).mkdir(parents=True, exist_ok=True)
    csv_fieldnames = ["identity", "filename", "image_path"]
    for split_name, recs in splits.items():
        split_path = os.path.join(splits_dir, f"{split_name}.csv")
        with open(split_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(recs)
        print(f"[ingest] Split '{split_name}' saved to {split_path}")

    print("[ingest] Done.")


if __name__ == "__main__":
    main()