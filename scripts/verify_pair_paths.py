"""
scripts/verify_pair_paths.py — Diagnostic tool for Milestone 2.

Checks every image path referenced in the pairs CSVs against the actual
files on disk, reports what is missing, and suggests fixes.

Run this whenever score_pairs.py fails with a file-not-found error.

Usage
-----
    python scripts/verify_pair_paths.py --config configs/milestone2.yaml
    python scripts/verify_pair_paths.py --config configs/milestone2.yaml --split val
    python scripts/verify_pair_paths.py --config configs/milestone2.yaml --full-scan
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.utils import find_lfw_root, load_config


def _resolve(lfw_root: Path, stored: str) -> Path:
    p = Path(stored)
    parts = list(p.parts)
    if parts and parts[0].lower() == "lfw":
        p = Path(*parts[1:])
    return lfw_root / p


def check_split(
    pairs_csv: Path,
    lfw_root: Path,
    full_scan: bool = False,
) -> dict:
    """Check all (or sampled) image paths in a pairs CSV.

    Returns a dict with keys: total_pairs, missing_count, missing_examples.
    """
    if not pairs_csv.exists():
        return {"error": f"Pairs CSV not found: {pairs_csv}"}

    df = pd.read_csv(pairs_csv)
    required = {"left_path", "right_path"}
    if not required.issubset(df.columns):
        return {"error": f"Missing columns in {pairs_csv}: {required - set(df.columns)}"}

    # Collect unique paths to check
    all_paths = pd.concat([
        df["left_path"].astype(str),
        df["right_path"].astype(str),
    ]).unique().tolist()

    if not full_scan:
        # Sample first and last 50 unique paths — fast sanity check
        sample = all_paths[:50] + all_paths[-50:]
        all_paths = list(dict.fromkeys(sample))   # deduplicate, preserve order

    missing = []
    for p in all_paths:
        full = _resolve(lfw_root, p)
        if not full.exists():
            missing.append(str(full))

    return {
        "pairs_csv":       str(pairs_csv),
        "total_pairs":     len(df),
        "unique_paths_checked": len(all_paths),
        "full_scan":       full_scan,
        "missing_count":   len(missing),
        "missing_examples": missing[:10],
    }


def print_report(results: dict, lfw_root: Path) -> None:
    print(f"\n{'='*70}")
    print(f"  LFW root : {lfw_root}")
    print(f"  Exists   : {lfw_root.exists()}")

    if lfw_root.exists():
        subdirs = [d for d in lfw_root.iterdir() if d.is_dir()]
        print(f"  Identity subdirs found: {len(subdirs)}")
        if subdirs:
            sample_dir = subdirs[0]
            imgs = list(sample_dir.iterdir())
            print(f"  Sample identity '{sample_dir.name}': {len(imgs)} file(s)")

    print(f"{'='*70}\n")

    for split, res in results.items():
        print(f"  Split: {split}")
        if "error" in res:
            print(f"    ERROR: {res['error']}")
            continue

        status = "OK" if res["missing_count"] == 0 else "PROBLEMS FOUND"
        print(f"    Status              : {status}")
        print(f"    Total pairs         : {res['total_pairs']:,}")
        print(f"    Unique paths checked: {res['unique_paths_checked']:,}  "
              f"({'full scan' if res['full_scan'] else 'sample'})")
        print(f"    Missing files       : {res['missing_count']}")
        if res["missing_examples"]:
            print(f"    Missing examples:")
            for p in res["missing_examples"]:
                print(f"      {p}")
        print()

    any_missing = any(
        r.get("missing_count", 0) > 0
        for r in results.values()
        if "error" not in r
    )

    if any_missing:
        print("DIAGNOSIS: image files are missing from disk.\n")
        print("Most likely causes:\n")
        print("  1. INCOMPLETE TFDS EXTRACTION")
        print("     The TFDS download did not fully extract all images.")
        print("     Fix:")
        print("       - Delete the tfds_cache directory entirely:")
        print("           rmdir /s /q data\\tfds_cache   (Windows)")
        print("           rm -rf data/tfds_cache        (Linux/Mac)")
        print("       - Re-run ingestion:")
        print("           python scripts/ingest_lfw.py --config configs/milestone2.yaml\n")
        print("  2. PAIRS CSV FROM A DIFFERENT MACHINE / CACHE LOCATION")
        print("     The pairs were generated on a machine with a different cache path.")
        print("     Fix:")
        print("       - Re-run pair generation so paths match this machine's cache:")
        print("           python scripts/make_pairs.py --config configs/milestone2.yaml\n")
        print("  3. PARTIAL RE-EXTRACTION")
        print("     Only some identities extracted successfully.")
        print("     Fix: delete tfds_cache and re-run ingestion (same as #1).\n")
    else:
        print("All checked paths exist on disk.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that pairs CSV image paths exist on disk"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--split", default=None, choices=["train", "val", "test"],
        help="Check only this split (default: check all three)"
    )
    parser.add_argument(
        "--full-scan", action="store_true",
        help="Check every unique path instead of a sample (slower)"
    )
    parser.add_argument(
        "--extracted-root", default=None,
        help="Override the TFDS extracted root path"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg  = load_config(args.config)

    outputs_dir = PROJECT_ROOT / cfg.get("paths", {}).get("outputs", "outputs")
    pairs_dir   = outputs_dir / "pairs"

    if args.extracted_root is not None:
        extracted_root = Path(args.extracted_root)
    else:
        data_cache = cfg.get("paths", {}).get("data_cache", "data/tfds_cache")
        extracted_root = PROJECT_ROOT / data_cache / "downloads" / "extracted"

    print(f"\n[verify_pair_paths]")
    print(f"  Extracted root : {extracted_root}")
    print(f"  Pairs dir      : {pairs_dir}")

    try:
        lfw_root = find_lfw_root(extracted_root)
    except FileNotFoundError as e:
        print(f"\nERROR: Could not locate lfw root.\n  {e}")
        print(
            "\n  The TFDS extraction may not have run yet, or the cache directory\n"
            "  does not exist.  Re-run:\n"
            "    python scripts/ingest_lfw.py --config configs/milestone2.yaml"
        )
        sys.exit(1)

    splits_to_check = (
        [args.split] if args.split else ["train", "val", "test"]
    )

    results = {}
    for split in splits_to_check:
        pairs_csv = pairs_dir / f"{split}_pairs.csv"
        print(f"  Checking {split} pairs ({pairs_csv.name}) ...")
        results[split] = check_split(pairs_csv, lfw_root, full_scan=args.full_scan)

    print_report(results, lfw_root)


if __name__ == "__main__":
    main()
