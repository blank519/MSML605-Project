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
from datetime import datetime
from pathlib import Path

# Required third-party imports, included in requirements.txt
import numpy as np

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
import tensorflow_datasets as tfds

from utils import find_lfw_root, is_image_file, load_config


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
    project_root = Path(__file__).resolve().parents[1]
    cache_path = (project_root / cache_dir).resolve() if not os.path.isabs(cache_dir) else Path(cache_dir).resolve()

    print(f"[ingest] Preparing LFW via TFDS (cache_dir={cache_path}) ...")
    builder = tfds.builder("lfw", data_dir=str(cache_path))
    builder.download_and_prepare()

    extracted_root = cache_path / "downloads" / "extracted"
    if not extracted_root.exists():
        raise FileNotFoundError(
            f"TFDS extracted directory not found: {extracted_root}. "
            "Try re-running ingestion or check your tfds cache."
        )

    lfw_root = find_lfw_root(extracted_root)

    records: list[dict] = []
    for ident_dir in sorted([p for p in lfw_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        identity_name = ident_dir.name
        image_files = sorted(
            [p for p in ident_dir.iterdir() if p.is_file() and is_image_file(p)],
            key=lambda p: p.name,
        )

        for img_path in image_files:
            filename = img_path.name
            records.append(
                {
                    "identity": identity_name,
                    "filename": filename,
                    "image_path": (Path("lfw") / identity_name / filename).as_posix(),
                }
            )

    records.sort(key=lambda r: (r["identity"], r["filename"]))
    print(
        f"[ingest] Loaded {len(records)} images across "
        f"{len({r['identity'] for r in records})} identities from extracted files."
    )
    return records


# Deterministic split by image/file

def _stable_int_from_str(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def _split_counts_per_identity(n: int, val_frac: float, test_frac: float) -> tuple[int, int, int]:
    n_val = max(1, int(round(n * val_frac)))
    n_test = max(1, int(round(n * test_frac)))
    n_train = n - n_val - n_test

    if n_train < 1:
        n_train = 1
        remainder = n - n_train

        n_val = max(1, min(n_val, remainder - 1))
        n_test = remainder - n_val
        if n_test < 1:
            n_test = 1
            n_val = remainder - n_test

    while (n_train + n_val + n_test) > n:
        if n_train >= n_val and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break

    while (n_train + n_val + n_test) < n:
        n_train += 1

    return n_train, n_val, n_test


def split_by_image(records: list, seed: int, val_frac: float, test_frac: float,) -> dict[str, list]:
    rng = random.Random(seed)

    by_identity: dict[str, list[dict]] = {}
    for r in records:
        by_identity.setdefault(str(r["identity"]), []).append(r)

    kept_identities = {k: v for k, v in by_identity.items() if len(v) >= 10}

    train_recs: list[dict] = []
    val_recs: list[dict] = []
    test_recs: list[dict] = []

    for identity in sorted(kept_identities.keys()):
        imgs = list(kept_identities[identity])

        ident_seed = (seed + _stable_int_from_str(identity)) % (2**32)
        ident_rng = random.Random(ident_seed)
        ident_rng.shuffle(imgs)

        n = len(imgs)
        n_train, n_val, n_test = _split_counts_per_identity(n, val_frac=val_frac, test_frac=test_frac)

        train_recs.extend(imgs[:n_train])
        val_recs.extend(imgs[n_train : n_train + n_val])
        test_recs.extend(imgs[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train_recs)
    rng.shuffle(val_recs)
    rng.shuffle(test_recs)

    train_recs.sort(key=lambda r: (r["identity"], r["filename"]))
    val_recs.sort(key=lambda r: (r["identity"], r["filename"]))
    test_recs.sort(key=lambda r: (r["identity"], r["filename"]))

    return {"train": train_recs, "val": val_recs, "test": test_recs}


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
    splits = split_by_image(records, seed=seed, val_frac=val_frac, test_frac=test_frac)

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