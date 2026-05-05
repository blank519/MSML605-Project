"""
scripts/profile_latency.py — Hardware-aware latency profiling for Milestone 4.

Measures per-stage latency (preprocessing, embedding, scoring) and
batch-size sensitivity for the FaceNet-based face verification pipeline.

All timings use time.perf_counter() for high-resolution wall-clock measurement.
Each configuration is repeated multiple times and statistics are computed
across repetitions to account for warm-up and variance.

Usage:
    python scripts/profile_latency.py --config configs/milestone3.yaml
    python scripts/profile_latency.py --config configs/milestone3.yaml ^
        --n-pairs 20 --n-repeats 5 --output-json outputs/profiling_results.json

Outputs:
    outputs/profiling_results.json  — full results for the report
    Printed summary table           — copy into the profiling report
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.utils import find_lfw_root, load_config
from src.embedder import FaceEmbedder


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------

def get_hardware_info() -> dict:
    """Collect basic hardware and software environment info."""
    import platform
    info = {
        "os":          platform.system() + " " + platform.release(),
        "python":      platform.python_version(),
        "cpu":         platform.processor() or "AMD Ryzen 7 7800X3D 8-Core Processor",
        "cpu_cores":   8,
        "cpu_threads": 16,
    }
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        info["gpu_available"] = len(gpus) > 0
        info["gpu_devices"] = [g.name for g in gpus]
    except Exception:
        info["tensorflow"] = "unknown"
        info["gpu_available"] = False
    return info


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_stage(fn, *args, n_repeats: int = 3) -> dict:
    """Run fn(*args) n_repeats times and return latency statistics (ms)."""
    latencies = []
    result = None
    for i in range(n_repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        t1 = time.perf_counter()
        if i > 0:  # skip first run (cold start / JIT warm-up)
            latencies.append((t1 - t0) * 1000)

    if not latencies:
        latencies = [(t1 - t0) * 1000]  # fallback if n_repeats == 1

    arr = np.array(latencies)
    return {
        "mean_ms":   round(float(np.mean(arr)), 2),
        "std_ms":    round(float(np.std(arr)), 2),
        "min_ms":    round(float(np.min(arr)), 2),
        "max_ms":    round(float(np.max(arr)), 2),
        "n_repeats": len(latencies),
        "_result":   result,
    }


# ---------------------------------------------------------------------------
# Stage-level profiling (single pair)
# ---------------------------------------------------------------------------

def profile_single_pair(
    embedder: FaceEmbedder,
    path_a: str,
    path_b: str,
    threshold: float,
    n_repeats: int = 5,
) -> dict:
    """Profile each stage separately for a single pair."""

    # Stage 1 — Preprocessing
    preprocess_stats = time_stage(
        lambda: (embedder.preprocess(path_a), embedder.preprocess(path_b)),
        n_repeats=n_repeats,
    )
    arr_a, arr_b = preprocess_stats.pop("_result")

    # Stage 2 — Embedding (both images combined)
    embed_stats = time_stage(
        lambda: (embedder.embed(arr_a), embedder.embed(arr_b)),
        n_repeats=n_repeats,
    )
    ea, eb = embed_stats.pop("_result")

    # Stage 3 — Similarity + decision + confidence
    score_stats = time_stage(
        lambda: embedder.score(embedder.similarity(ea, eb)),
        n_repeats=n_repeats,
    )
    score_stats.pop("_result")

    # End-to-end (full verify_pair call)
    e2e_stats = time_stage(
        embedder.verify_pair, path_a, path_b, threshold,
        n_repeats=n_repeats,
    )
    e2e_result = e2e_stats.pop("_result")

    return {
        "preprocess": preprocess_stats,
        "embed":      embed_stats,
        "score":      score_stats,
        "end_to_end": e2e_stats,
        "sample_result": {
            k: v for k, v in e2e_result.items()
            if k not in ("image_a", "image_b")
        },
    }


# ---------------------------------------------------------------------------
# Batch-size sensitivity profiling
# ---------------------------------------------------------------------------

def profile_batch_sizes(
    embedder: FaceEmbedder,
    image_arrays: list,
    batch_sizes: list[int],
    n_repeats: int = 3,
) -> list[dict]:
    """
    Measure embedding throughput and latency for different batch sizes.

    keras-facenet accepts a batch of images as (N, 160, 160, 3).
    We time how long it takes to embed N images at once and compute
    per-image latency and throughput.
    """
    results = []

    for bs in batch_sizes:
        # Use the first bs images (repeat if fewer available)
        batch = np.stack([image_arrays[i % len(image_arrays)] for i in range(bs)])

        def _embed_batch():
            try:
                return embedder._facenet.embeddings(batch, verbose=0)
            except TypeError:
                return embedder._facenet.embeddings(batch)

        stats = time_stage(_embed_batch, n_repeats=n_repeats)
        stats.pop("_result")

        per_image_ms = round(stats["mean_ms"] / bs, 2)
        throughput = round(1000 / stats["mean_ms"] * bs, 2)  # images/sec

        results.append({
            "batch_size":       bs,
            "total_ms_mean":    stats["mean_ms"],
            "total_ms_std":     stats["std_ms"],
            "per_image_ms":     per_image_ms,
            "throughput_img_s": throughput,
            "n_repeats":        stats["n_repeats"],
        })

        print(f"  Batch size {bs:>3}: "
              f"total={stats['mean_ms']:>7.1f}ms  "
              f"per_image={per_image_ms:>7.1f}ms  "
              f"throughput={throughput:>6.1f} img/s")

    return results


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_stage_table(stage_results: dict) -> None:
    print(f"\n{'='*65}")
    print(f"  Per-Stage Latency (single pair, CPU)")
    print(f"{'='*65}")
    print(f"  {'Stage':<20} {'Mean (ms)':>10} {'Std (ms)':>10} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for stage in ("preprocess", "embed", "score", "end_to_end"):
        s = stage_results[stage]
        label = stage.replace("_", " ").title()
        print(f"  {label:<20} {s['mean_ms']:>10.2f} {s['std_ms']:>10.2f} "
              f"{s['min_ms']:>8.2f} {s['max_ms']:>8.2f}")
    print(f"{'='*65}")

    # Show percentage breakdown
    total = stage_results["end_to_end"]["mean_ms"]
    print(f"\n  Stage contribution to end-to-end latency ({total:.1f} ms total):")
    for stage, label in [
        ("preprocess", "Preprocessing"),
        ("embed",      "Embedding"),
        ("score",      "Scoring"),
    ]:
        pct = stage_results[stage]["mean_ms"] / total * 100
        print(f"    {label:<15} : {stage_results[stage]['mean_ms']:>7.2f} ms  ({pct:.1f}%)")


def print_batch_table(batch_results: list[dict]) -> None:
    print(f"\n{'='*65}")
    print(f"  Batch-Size Sensitivity (embedding stage, CPU)")
    print(f"{'='*65}")
    print(f"  {'Batch':>6} {'Total (ms)':>12} {'Per Image (ms)':>15} {'Throughput (img/s)':>19}")
    print(f"  {'-'*6} {'-'*12} {'-'*15} {'-'*19}")
    for r in batch_results:
        print(f"  {r['batch_size']:>6} "
              f"{r['total_ms_mean']:>12.1f} "
              f"{r['per_image_ms']:>15.1f} "
              f"{r['throughput_img_s']:>19.1f}")
    print(f"{'='*65}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latency profiling for face verification pipeline — Milestone 4"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--n-pairs", type=int, default=10,
        help="Number of pairs to profile (default: 10)",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=5,
        help="Timing repetitions per stage (default: 5, first is discarded as warm-up)",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16],
        help="Batch sizes to test (default: 1 2 4 8 16)",
    )
    parser.add_argument(
        "--extracted-root", default=None,
        help="Override TFDS extracted root path",
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Save full results to JSON (default: outputs/profiling_results.json)",
    )
    parser.add_argument(
        "--split", default="val", choices=["val", "test"],
        help="Which pairs split to use (default: val)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    m3 = cfg.get("milestone3", {})

    threshold = m3.get("locked_threshold")
    if threshold is None:
        print("ERROR: milestone3.locked_threshold not set in config.")
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

    # Load pairs
    df = pd.read_csv(pairs_csv).head(args.n_pairs)

    def resolve(p: str) -> str:
        path = Path(p)
        parts = list(path.parts)
        if parts and parts[0].lower() == "lfw":
            path = Path(*parts[1:])
        return str(lfw_root / path)

    pairs = [
        (resolve(row.left_path), resolve(row.right_path))
        for row in df.itertuples(index=False)
    ]

    # Hardware info
    hw = get_hardware_info()
    print(f"\n[profile_latency] Environment")
    print(f"  OS          : {hw['os']}")
    print(f"  CPU         : {hw['cpu']}")
    print(f"  Cores       : {hw['cpu_cores']} physical / {hw['cpu_threads']} logical")
    print(f"  TensorFlow  : {hw['tensorflow']}")
    print(f"  GPU         : {'Available' if hw['gpu_available'] else 'Not used (CPU-only profiling)'}")
    print(f"  Threshold   : {threshold}")
    print(f"  Pairs       : {len(pairs)}")
    print(f"  Repeats     : {args.n_repeats} (first discarded as warm-up)")

    # Initialise embedder
    print("\n[profile_latency] Initialising FaceEmbedder ...")
    embedder = FaceEmbedder()
    print("[profile_latency] Ready.\n")

    # ── Single-pair stage profiling ────────────────────────────────────────
    print("[profile_latency] Profiling per-stage latency across pairs ...")
    all_stage_results = []
    for i, (pa, pb) in enumerate(pairs):
        print(f"  Pair {i+1}/{len(pairs)} ...", end="\r")
        result = profile_single_pair(
            embedder, pa, pb, threshold, n_repeats=args.n_repeats
        )
        all_stage_results.append(result)
    print()

    # Aggregate across pairs
    def agg(stage: str, metric: str) -> float:
        return round(float(np.mean([r[stage][metric] for r in all_stage_results])), 2)

    aggregated_stages = {
        stage: {
            "mean_ms": agg(stage, "mean_ms"),
            "std_ms":  agg(stage, "std_ms"),
            "min_ms":  agg(stage, "min_ms"),
            "max_ms":  agg(stage, "max_ms"),
        }
        for stage in ("preprocess", "embed", "score", "end_to_end")
    }

    print_stage_table(aggregated_stages)

    # ── Batch-size sensitivity ─────────────────────────────────────────────
    print("\n[profile_latency] Profiling batch-size sensitivity ...")
    # Pre-load a set of images for batch testing
    image_arrays = [embedder.preprocess(pa) for pa, _ in pairs[:max(args.batch_sizes)]]
    batch_results = profile_batch_sizes(
        embedder, image_arrays, args.batch_sizes, n_repeats=args.n_repeats
    )
    print_batch_table(batch_results)

    # ── Save results ───────────────────────────────────────────────────────
    out_path = (
        Path(args.output_json)
        if args.output_json
        else outputs_dir / "profiling_results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    full_results = {
        "hardware":        hw,
        "config":          args.config,
        "threshold":       threshold,
        "n_pairs":         len(pairs),
        "n_repeats":       args.n_repeats,
        "batch_sizes":     args.batch_sizes,
        "stage_latency":   aggregated_stages,
        "batch_sensitivity": batch_results,
    }

    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n[profile_latency] Results saved -> {out_path}")
    print("[profile_latency] Done.\n")


if __name__ == "__main__":
    main()
