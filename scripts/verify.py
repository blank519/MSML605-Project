"""
scripts/verify.py — CLI inference interface

Runs pair-level face verification using FaceNet embeddings and prints score, decision, confidence, and latency for each pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.utils import load_config
from src.embedder import FaceEmbedder


# Output formatting

def print_result(result: dict, pair_idx: int | None = None) -> None:
    """Print a single pair result in a readable, grader-friendly format."""
    prefix = f"[Pair {pair_idx}] " if pair_idx is not None else ""
    print(f"\n{prefix}{'='*60}")
    print(f"  Image A      : {result['image_a']}")
    print(f"  Image B      : {result['image_b']}")
    print(f"  {'─'*56}")
    print(f"  Score        : {result['score']:.6f}  "
          f"(sigmoid probability, range [0, 1])")
    if 'cosine_similarity' in result:
        print(f"  Cosine sim   : {result['cosine_similarity']:.6f}  "
              f"(raw embedding similarity, range [-1, 1])")
    print(f"  Threshold    : {result['threshold']:.6f}  "
          f"(TAR@FAR=0.01, selected on val split)")
    print(f"  Decision     : {result['decision'].upper()}")
    print(f"  Confidence   : {result['confidence']:.4f}  "
          f"(0=boundary, 1=max confidence)")
    print(f"  {'─'*56}")
    print(f"  Latency breakdown:")
    print(f"    Preprocess : {result['latency_preprocess_ms']:>8.2f} ms")
    print(f"    Embed      : {result['latency_embed_ms']:>8.2f} ms")
    print(f"    Score+dec  : {result['latency_score_ms']:>8.2f} ms")
    print(f"    Total      : {result['latency_total_ms']:>8.2f} ms")
    print(f"{'='*60}")


# Single-pair inference

def run_single(embedder: FaceEmbedder, threshold: float, args: argparse.Namespace) -> None:
    if not Path(args.image_a).exists():
        print(f"ERROR: image-a not found: {args.image_a}")
        sys.exit(1)
    if not Path(args.image_b).exists():
        print(f"ERROR: image-b not found: {args.image_b}")
        sys.exit(1)

    result = embedder.verify_pair(args.image_a, args.image_b, threshold)
    print_result(result)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Result saved -> {args.output_json}")

# Batch inference

def run_batch(embedder: FaceEmbedder, threshold: float, args: argparse.Namespace) -> None:
    pairs_csv = Path(args.pairs_csv)
    if not pairs_csv.exists():
        print(f"ERROR: pairs CSV not found: {pairs_csv}")
        sys.exit(1)

    df = pd.read_csv(pairs_csv)
    required = {"left_path", "right_path"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: pairs CSV missing columns: {missing}")
        sys.exit(1)

    # Resolve LFW root for relative paths
    lfw_root = Path(args.lfw_root) if args.lfw_root else None

    def resolve(p: str) -> Path:
        path = Path(p)
        parts = list(path.parts)
        if parts and parts[0].lower() == "lfw":
            path = Path(*parts[1:])
        return lfw_root / path if lfw_root else path

    # Limit batch size if requested
    if args.max_pairs:
        df = df.head(args.max_pairs)

    results = []
    n_errors = 0
    print(f"\n[verify] Running batch inference on {len(df)} pairs...")
    print(f"[verify] Threshold : {threshold:.6f}")

    for i, row in enumerate(df.itertuples(index=False)):
        path_a = resolve(row.left_path)
        path_b = resolve(row.right_path)
        try:
            result = embedder.verify_pair(path_a, path_b, threshold)
            if args.verbose:
                print_result(result, pair_idx=i + 1)
            else:
                decision_flag = "SAME" if result["decision"] == "same" else "DIFF"
                print(f"  [{i+1:>5}] {decision_flag}  "
                      f"score={result['score']:+.4f}  "
                      f"conf={result['confidence']:.3f}  "
                      f"latency={result['latency_total_ms']:.1f}ms")
            results.append(result)
        except Exception as e:
            print(f"  [{i+1:>5}] ERROR: {e}")
            n_errors += 1

    if results:
        latencies = [r["latency_total_ms"] for r in results]
        scores = [r["score"] for r in results]
        same_count = sum(1 for r in results if r["decision"] == "same")
        print(f"\n[verify] Batch summary ({len(results)} pairs, {n_errors} errors):")
        print(f"  Decisions  : {same_count} SAME / {len(results)-same_count} DIFFERENT")
        print(f"  Avg score  : {sum(scores)/len(scores):.4f}")
        print(f"  Avg latency: {sum(latencies)/len(latencies):.1f} ms")

    # Save results
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"results": results, "n_errors": n_errors}, f, indent=2)
        print(f"  Results saved -> {args.output_json}")


# CLI

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face verification CLI — Milestone 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single pair:
    python scripts/verify.py --config configs/milestone3.yaml \\
        --image-a face1.jpg --image-b face2.jpg

  Batch from pairs CSV:
    python scripts/verify.py --config configs/milestone3.yaml \\
        --pairs-csv outputs/pairs/val_pairs.csv \\
        --lfw-root data/tfds_cache/downloads/extracted/.../lfw \\
        --max-pairs 20
        """,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image-a", help="Path to first face image (single-pair mode)")
    mode.add_argument("--pairs-csv", help="Path to pairs CSV (batch mode)")

    # Single-pair args
    parser.add_argument("--image-b", help="Path to second face image (single-pair mode)")

    # Batch args
    parser.add_argument("--lfw-root", default=None,
                        help="LFW root directory for resolving relative paths in pairs CSV")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Maximum number of pairs to process in batch mode")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full result for each pair in batch mode")

    # Shared
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold from config")
    parser.add_argument("--output-json", default=None,
                        help="Save result(s) to a JSON file")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.image_a and not args.image_b:
        print("ERROR: --image-b is required when using --image-a")
        sys.exit(1)

    cfg = load_config(args.config)
    m3 = cfg.get("milestone3", {})

    threshold = args.threshold or m3.get("locked_threshold")
    if threshold is None:
        print(
            "ERROR: No threshold set.\n"
            "  Pass --threshold <value> or set milestone3.locked_threshold in config."
        )
        sys.exit(1)

    print(f"\n[verify] Initialising FaceEmbedder (keras-facenet / VGGFace2) ...")
    embedder = FaceEmbedder()
    print(f"[verify] Threshold  : {threshold}")

    if args.image_a:
        run_single(embedder, threshold, args)
    else:
        run_batch(embedder, threshold, args)


if __name__ == "__main__":
    main()
