"""
scripts/load_test.py — Concurrency and load test

Runs face verification inference under multiple concurrent threads and reports throughput and latency distribution (mean, p50, p95, p99).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.utils import find_lfw_root, load_config
from src.embedder import FaceEmbedder


def _worker(task: dict, embedder: FaceEmbedder) -> dict:
    """Process one pair in a thread, sharing the embedder instance across threads."""
    try:
        result = embedder.verify_pair(
            task["path_a"], task["path_b"], task["threshold"]
        )
        result["pair_index"] = task["pair_index"]
        result["status"] = "ok"
        return result
    except Exception as e:
        return {
            "pair_index": task["pair_index"],
            "status": "error",
            "error": str(e),
            "latency_total_ms": 0.0,
        }


def run_load_test(
    pairs: list[dict],
    threshold: float,
    n_workers: int,
    embedder: FaceEmbedder,
) -> dict:
    tasks = [
        {
            "path_a":     p["path_a"],
            "path_b":     p["path_b"],
            "pair_index": p["pair_index"],
            "threshold":  threshold,
        }
        for p in pairs
    ]

    print(f"\n[load_test] Starting {len(tasks)} tasks across {n_workers} thread(s) ...")
    wall_start = time.perf_counter()

    results = []
    errors = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, t, embedder): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            if res["status"] == "error":
                errors += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(f"  Progress: {completed}/{len(tasks)} ({errors} errors)", end="\r")

    wall_elapsed = time.perf_counter() - wall_start
    print()

    ok_results = [r for r in results if r["status"] == "ok"]
    latencies = [r["latency_total_ms"] for r in ok_results]

    if latencies:
        lat_arr = np.array(latencies)
        stats = {
            "mean_ms":   round(float(np.mean(lat_arr)), 2),
            "median_ms": round(float(np.median(lat_arr)), 2),
            "p95_ms":    round(float(np.percentile(lat_arr, 95)), 2),
            "p99_ms":    round(float(np.percentile(lat_arr, 99)), 2),
            "min_ms":    round(float(np.min(lat_arr)), 2),
            "max_ms":    round(float(np.max(lat_arr)), 2),
        }
    else:
        stats = {}

    throughput = round(len(ok_results) / wall_elapsed, 3) if wall_elapsed > 0 else 0.0

    return {
        "n_requested":    len(tasks),
        "n_completed":    len(ok_results),
        "n_errors":       errors,
        "n_workers":      n_workers,
        "wall_time_s":    round(wall_elapsed, 3),
        "throughput_rps": throughput,
        "latency":        stats,
        "results":        results,
    }


def print_summary(report: dict) -> None:
    lat = report["latency"]
    print(f"\n{'='*60}")
    print(f"  Load Test Summary")
    print(f"{'='*60}")
    print(f"  Requests     : {report['n_completed']} ok / "
          f"{report['n_errors']} errors / "
          f"{report['n_requested']} total")
    print(f"  Workers      : {report['n_workers']} threads")
    print(f"  Wall time    : {report['wall_time_s']:.2f} s")
    print(f"  Throughput   : {report['throughput_rps']:.3f} req/s")
    if lat:
        print(f"  Latency (ms) :")
        print(f"    Mean       : {lat['mean_ms']:.1f}")
        print(f"    Median p50 : {lat['median_ms']:.1f}")
        print(f"    p95        : {lat['p95_ms']:.1f}")
        print(f"    p99        : {lat['p99_ms']:.1f}")
        print(f"    Min / Max  : {lat['min_ms']:.1f} / {lat['max_ms']:.1f}")
    print(f"{'='*60}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrency load test for face verification — Milestone 3"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of concurrent worker threads (default: 2)")
    parser.add_argument("--n-pairs", type=int, default=50,
                        help="Number of pairs to process (default: 50)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold from config")
    parser.add_argument("--extracted-root", default=None,
                        help="Override TFDS extracted root path")
    parser.add_argument("--output-json", default=None,
                        help="Save summary to JSON (default: outputs/load_test_results.json)")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="Which pairs split to use (default: val)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    m3 = cfg.get("milestone3", {})

    threshold = args.threshold or m3.get("locked_threshold")
    if threshold is None:
        print(
            "ERROR: No threshold set.\n"
            "  Pass --threshold <value> or set "
            "milestone3.locked_threshold in config."
        )
        sys.exit(1)

    outputs_dir = PROJECT_ROOT / cfg.get("paths", {}).get("outputs", "outputs")
    pairs_csv = outputs_dir / "pairs" / f"{args.split}_pairs.csv"

    if not pairs_csv.exists():
        print(f"ERROR: Pairs CSV not found: {pairs_csv}")
        sys.exit(1)

    if args.extracted_root:
        extracted_root = Path(args.extracted_root)
    else:
        data_cache = cfg.get("paths", {}).get("data_cache", "data/tfds_cache")
        extracted_root = PROJECT_ROOT / data_cache / "downloads" / "extracted"

    lfw_root = find_lfw_root(extracted_root)

    df = pd.read_csv(pairs_csv).head(args.n_pairs)

    def resolve(p: str) -> str:
        path = Path(p)
        parts = list(path.parts)
        if parts and parts[0].lower() == "lfw":
            path = Path(*parts[1:])
        return str(lfw_root / path)

    pairs = [
        {"path_a": resolve(row.left_path), "path_b": resolve(row.right_path), "pair_index": i}
        for i, row in enumerate(df.itertuples(index=False))
    ]

    print(f"\n[load_test] Config    : {args.config}")
    print(f"[load_test] Split     : {args.split}")
    print(f"[load_test] Pairs     : {len(pairs)}")
    print(f"[load_test] Workers   : {args.workers} threads")
    print(f"[load_test] Threshold : {threshold}")
    print("\n[load_test] Initialising FaceEmbedder (shared across threads) ...")
    embedder = FaceEmbedder()

    report = run_load_test(
        pairs=pairs,
        threshold=threshold,
        n_workers=args.workers,
        embedder=embedder,
    )

    print_summary(report)

    out_path = (
        Path(args.output_json)
        if args.output_json
        else outputs_dir / "load_test_results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in report.items() if k != "results"}
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[load_test] Summary saved -> {out_path}")


if __name__ == "__main__":
    main()
