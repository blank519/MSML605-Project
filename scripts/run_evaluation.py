"""
run_evaluation.py — Evaluate the face verifier at a locked threshold on a specified split.
Produces a confusion matrix plot and a metrics JSON, and logs the run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.utils import load_config
from src.evaluate import (
    load_scored_pairs,
    evaluate_at_threshold,
    validate_threshold,
    save_metrics_json,
)
from src.tracker import RunTracker

# Confusion matrix

def plot_confusion_matrix(
    cm: dict, out_path: Path, title: str = ""
) -> None:
    """Save a 2x2 confusion matrix heatmap."""
    matrix = np.array([
        [cm["TN"], cm["FP"]],
        [cm["FN"], cm["TP"]],
    ])
    cell_labels = [["TN", "FP"], ["FN", "TP"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nDifferent", "Predicted\nSame"], fontsize=11)
    ax.set_yticklabels(["Actually\nDifferent", "Actually\nSame"], fontsize=11)

    for i in range(2):
        for j in range(2):
            val   = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(
                j, i,
                f"{cell_labels[i][j]}\n{val:,}",
                ha="center", va="center",
                color=color, fontsize=13, fontweight="bold",
            )

    ax.set_title(title or "Confusion Matrix", fontsize=13, pad=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved  -> {out_path}")

# CLI

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate face verifier at a locked threshold — Milestone 2"
    )
    parser.add_argument("--config",      required=True, help="Path to YAML config")
    parser.add_argument(
        "--split", default=None,
        choices=["val", "test"],
        help="Split to evaluate (default: reporting_split from config)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold (overrides config milestone2.locked_threshold)",
    )
    parser.add_argument(
        "--scored-csv", default=None,
        help="Path to scored pairs CSV (default: outputs/scores/{split}_scored.csv)",
    )
    parser.add_argument(
        "--data-version", default=None,
        help="Data version label for tracking (default: from config)",
    )
    parser.add_argument("--note", default="", help="Short run note")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg  = load_config(args.config)
    m2   = cfg.get("milestone2", {})

    split        = args.split        or m2.get("reporting_split", "test")
    data_version = args.data_version or m2.get("data_version", "v1")
    config_name  = m2.get("config_name", Path(args.config).stem)

    # Threshold resolution: CLI flag > config > error
    threshold = args.threshold
    if threshold is None:
        threshold = m2.get("locked_threshold")
    if threshold is None:
        print(
            "ERROR: No threshold specified.\n"
            "  Pass --threshold <value>, or set "
            "milestone2.locked_threshold in the config after the sweep."
        )
        sys.exit(1)

    validate_threshold(threshold)

    outputs_dir = PROJECT_ROOT / cfg.get("paths", {}).get("outputs", "outputs")

    # Resolve scored CSV
    if args.scored_csv is not None:
        scored_csv = Path(args.scored_csv)
    else:
        scored_csv = outputs_dir / "scores" / f"{split}_scored.csv"

    if not scored_csv.exists():
        print(
            f"ERROR: Scored CSV not found: {scored_csv}\n"
            f"       Run score_pairs.py --split {split} first."
        )
        sys.exit(1)

    print(f"\n[Evaluation at Locked Threshold]")
    print(f"  Config       : {args.config}")
    print(f"  Split        : {split}")
    print(f"  Threshold    : {threshold}")
    print(f"  Scored CSV   : {scored_csv}")
    print(f"  Data version : {data_version}")

    df = load_scored_pairs(scored_csv)
    print(f"  Pairs loaded : {len(df):,}  "
          f"(pos={int(df['label'].sum()):,}, "
          f"neg={int((df['label']==0).sum()):,})")

    tracker = RunTracker(log_path=outputs_dir / "runs" / "run_log.json")
    run_id  = tracker.start_run(
        config_name=config_name,
        data_version=data_version,
        split=split,
        threshold=threshold,
        note=args.note or f"Eval at threshold={threshold:.4f}, {split} split",
    )
    print(f"  Run ID       : {run_id}")

    try:
        result = evaluate_at_threshold(df, threshold=threshold, split_name=split)

        print(f"\n  Results:")
        print(f"    TP = {result['TP']:>7,}   FP = {result['FP']:>7,}")
        print(f"    FN = {result['FN']:>7,}   TN = {result['TN']:>7,}")
        print(f"    TAR      = {result['TAR']:.4f}")
        print(f"    FAR      = {result['FAR']:.4f}")
        print(f"    FRR      = {result['FRR']:.4f}")
        print(f"    Accuracy = {result['accuracy']:.4f}")
        print(f"    F1       = {result['F1']:.4f}")

        run_dir  = outputs_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics_json = run_dir / "metrics.json"
        cm_plot      = run_dir / "confusion_matrix.png"

        save_metrics_json(result, metrics_json)
        print(f"  Metrics JSON saved      -> {metrics_json}")

        plot_confusion_matrix(
            result, cm_plot,
            title=(
                f"Confusion Matrix — {split} split\n"
                f"threshold = {threshold:.4f}"
            ),
        )

        tracker.finish_run(
            run_id,
            metrics={k: result[k] for k in
                     ["TAR", "FAR", "FRR", "accuracy", "F1",
                      "TP", "FP", "TN", "FN", "n_pairs"]},
            threshold=threshold,
            artifacts=[str(metrics_json), str(cm_plot)],
        )

        tracker.print_summary()
        print(f"\n[Done] Run {run_id} complete. Artifacts in: {run_dir}\n")

    except Exception as exc:
        tracker.fail_run(run_id, str(exc))
        raise


if __name__ == "__main__":
    main()
