"""
run_threshold_sweep.py — Milestone 2 Step 5.4 (and Step 5.7).

Reads a scored pairs CSV (produced by scripts/score_pairs.py), runs a
threshold sweep over the validation split, logs the run, and saves:
  - sweep.csv           : per-threshold metrics table
  - selected_threshold.json : the TAR@FAR=0.01 chosen threshold + metrics
  - roc_curve.png       : ROC-style plot (TAR vs FAR)
  - tar_far_vs_threshold.png : TAR and FAR as functions of threshold value

Usage
-----
    # Run 1 – baseline sweep
    python scripts/run_threshold_sweep.py --config configs/milestone2.yaml \
        --note "Run 1 – baseline threshold sweep"

    # Run 4 – after data-centric change (v2 scores)
    python scripts/run_threshold_sweep.py --config configs/milestone2.yaml \
        --data-version v2 \
        --scored-csv outputs/scores/val_scored_v2.csv \
        --note "Run 4 – post data-centric change sweep"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.utils import load_config
from src.evaluate import (
    load_scored_pairs,
    threshold_sweep,
    select_threshold_tar_at_far,
    save_sweep_csv,
    save_metrics_json,
)
from src.tracker import RunTracker

#Plotting helpers

def plot_roc(sweep_df: pd.DataFrame, selected: dict, out_path: Path) -> None:
    """Save a ROC-style curve (TAR vs FAR) with the operating point marked."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        sweep_df["FAR"], sweep_df["TAR"],
        color="#2563eb", linewidth=2, label="Verification ROC",
    )
    ax.scatter(
        [selected["FAR"]], [selected["TAR"]],
        color="#dc2626", zorder=5, s=80,
        label=(
            f"Selected threshold\n"
            f"TAR={selected['TAR']:.3f}, FAR={selected['FAR']:.3f}"
        ),
    )
    ax.axvline(
        x=0.01, color="#6b7280", linestyle="--", linewidth=1,
        label="FAR = 0.01 constraint",
    )
    ax.set_xlabel("False Accept Rate (FAR)", fontsize=12)
    ax.set_ylabel("True Accept Rate (TAR)", fontsize=12)
    ax.set_title("Face Verification — ROC Curve (Threshold Sweep)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC plot saved          -> {out_path}")


def plot_tar_far_vs_threshold(
    sweep_df: pd.DataFrame, selected: dict, out_path: Path
) -> None:
    """Plot TAR and FAR as a function of threshold value."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        sweep_df["threshold"], sweep_df["TAR"],
        label="TAR (True Accept Rate)", color="#2563eb", linewidth=2,
    )
    ax.plot(
        sweep_df["threshold"], sweep_df["FAR"],
        label="FAR (False Accept Rate)", color="#ef4444", linewidth=2,
    )
    ax.axvline(
        x=selected["selected_threshold"],
        color="#059669", linestyle="--", linewidth=1.5,
        label=f"Selected threshold = {selected['selected_threshold']:.4f}",
    )

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title("TAR / FAR vs. Decision Threshold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  TAR/FAR plot saved      -> {out_path}")

#CLI

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold sweep — Milestone 2")
    parser.add_argument("--config",       required=True, help="Path to YAML config")
    parser.add_argument(
        "--scored-csv", default=None,
        help="Path to scored pairs CSV (default: outputs/scores/val_scored.csv)",
    )
    parser.add_argument(
        "--split", default=None,
        help="Split label for tracking (default: val, from config)",
    )
    parser.add_argument(
        "--data-version", default=None,
        help="Data version label for tracking (default: from config)",
    )
    parser.add_argument("--note",    default="", help="Short run note")
    parser.add_argument("--n-steps", type=int, default=200,
                        help="Number of threshold steps (default: 200)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg  = load_config(args.config)
    m2   = cfg.get("milestone2", {})

    split        = args.split        or m2.get("threshold_selection_split", "val")
    data_version = args.data_version or m2.get("data_version", "v1")
    target_far   = float(m2.get("target_far", 0.01))
    config_name  = m2.get("config_name", Path(args.config).stem)

    outputs_dir = PROJECT_ROOT / cfg.get("paths", {}).get("outputs", "outputs")

    # Resolve scored CSV path
    if args.scored_csv is not None:
        scored_csv = Path(args.scored_csv)
    else:
        scored_csv = outputs_dir / "scores" / f"{split}_scored.csv"

    if not scored_csv.exists():
        print(
            f"ERROR: Scored CSV not found: {scored_csv}\n"
            f"       Run score_pairs.py first to generate it."
        )
        sys.exit(1)

    print(f"\n[Threshold Sweep]")
    print(f"  Config       : {args.config}")
    print(f"  Split        : {split}")
    print(f"  Scored CSV   : {scored_csv}")
    print(f"  Data version : {data_version}")
    print(f"  Target FAR   : {target_far}")
    print(f"  Steps        : {args.n_steps}")

    df = load_scored_pairs(scored_csv)
    pos = int(df["label"].sum())
    neg = int((df["label"] == 0).sum())
    print(f"  Pairs loaded : {len(df):,}  (pos={pos:,}, neg={neg:,})")
    print(f"  Score range  : [{df['score'].min():.4f}, {df['score'].max():.4f}]")

    # Start tracking
    tracker = RunTracker(log_path=outputs_dir / "runs" / "run_log.json")
    run_id  = tracker.start_run(
        config_name=config_name,
        data_version=data_version,
        split=split,
        note=args.note or f"Threshold sweep on {split} split",
    )
    print(f"  Run ID       : {run_id}")

    try:
        sweep_df = threshold_sweep(df, n_steps=args.n_steps)
        selected = select_threshold_tar_at_far(sweep_df, target_far=target_far)

        print(f"\n  Selection rule : {selected['selection_rule']}")
        print(f"  Selected threshold = {selected['selected_threshold']:.6f}")
        print(f"  TAR = {selected['TAR']:.4f}  |  FAR = {selected['FAR']:.4f}  "
              f"|  accuracy = {selected.get('accuracy', float('nan')):.4f}  "
              f"|  F1 = {selected.get('F1', float('nan')):.4f}")

        # Save artifacts
        run_dir = outputs_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        sweep_csv_path  = run_dir / "sweep.csv"
        selected_json   = run_dir / "selected_threshold.json"
        roc_plot        = run_dir / "roc_curve.png"
        tar_far_plot    = run_dir / "tar_far_vs_threshold.png"

        save_sweep_csv(sweep_df, sweep_csv_path)
        print(f"  Sweep CSV saved         -> {sweep_csv_path}")
        save_metrics_json(selected, selected_json)
        print(f"  Selected threshold JSON -> {selected_json}")
        plot_roc(sweep_df, selected, roc_plot)
        plot_tar_far_vs_threshold(sweep_df, selected, tar_far_plot)

        tracker.finish_run(
            run_id,
            metrics={
                "TAR":                selected["TAR"],
                "FAR":                selected["FAR"],
                "FRR":                selected.get("FRR"),
                "accuracy":           selected.get("accuracy"),
                "F1":                 selected.get("F1"),
                "selected_threshold": selected["selected_threshold"],
                "selection_rule":     selected["selection_rule"],
                "n_pairs":            len(df),
            },
            threshold=selected["selected_threshold"],
            artifacts=[
                str(sweep_csv_path), str(selected_json),
                str(roc_plot),       str(tar_far_plot),
            ],
        )
        tracker.print_summary()
        print(
            f"\n[Done] Run {run_id} complete.\n"
            f"  -> Set milestone2.locked_threshold = "
            f"{selected['selected_threshold']:.6f} in {args.config}\n"
        )

    except Exception as exc:
        tracker.fail_run(run_id, str(exc))
        raise


if __name__ == "__main__":
    main()
