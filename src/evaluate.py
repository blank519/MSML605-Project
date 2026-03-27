"""
evaluate.py — Core evaluation logic for Milestone 2.

Integrates with the existing codebase:
  - Pairs CSVs have columns: left_path, right_path, label, split
  - Scores come from SiameseVerifier (sigmoid probabilities, higher = same person)
  - tar_at_far() already exists in scripts/metrics.py; we re-export a compatible
    version here so evaluate.py has no circular dependency on scripts/
  - utils.py provides load_config / find_lfw_root
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Validation helpers

def validate_pairs_df(df: pd.DataFrame, context: str = "") -> None:
    """Raise ValueError if the scored-pairs DataFrame is malformed.

    The DataFrame must have at minimum:
      - 'score'  : float similarity/probability output of the model
      - 'label'  : binary ground-truth (0 = different, 1 = same)
    """
    required = {"score", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{context}] DataFrame is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    unique_labels = set(df["label"].dropna().unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"[{context}] Labels must be binary (0/1). Found: {unique_labels}"
        )
    if df["score"].isnull().any():
        raise ValueError(f"[{context}] NaN values detected in 'score' column.")
    if len(df) == 0:
        raise ValueError(f"[{context}] DataFrame is empty.")


def validate_threshold(threshold: float) -> None:
    """Raise if threshold is not a number or is out of a reasonable range."""
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"Threshold must be numeric, got {type(threshold).__name__}.")
    if not (-1e6 <= float(threshold) <= 1e6):
        raise ValueError(f"Threshold {threshold} is outside the allowed range.")


def validate_score_label_lengths(scores: np.ndarray, labels: np.ndarray) -> None:
    """Raise if score and label arrays have different lengths."""
    if len(scores) != len(labels):
        raise ValueError(
            f"Score count ({len(scores)}) != label count ({len(labels)}). "
            "Possible mismatch between pair CSV and model output."
        )

# Loading scored pairs

def load_scored_pairs(pairs_csv: Path | str) -> pd.DataFrame:
    """Load a pairs CSV that already contains a 'score' column.

    Accepts either:
      - The original pairs format (left_path, right_path, label) + a pre-computed
        'score' column, OR
      - A CSV produced by scripts/score_pairs.py with columns score + label.

    Column aliases handled:
      - 'similarity'    -> renamed to 'score'
      - 'same_identity' -> renamed to 'label'
      - 'same'          -> renamed to 'label'
    """
    df = pd.read_csv(pairs_csv)

    if "score" not in df.columns and "similarity" in df.columns:
        df = df.rename(columns={"similarity": "score"})
    for alias in ("same_identity", "same"):
        if "label" not in df.columns and alias in df.columns:
            df = df.rename(columns={alias: "label"})

    validate_pairs_df(df, context=str(pairs_csv))
    return df

# Function to generate confusion matrix

def compute_confusion(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, int]:
    """Return TP/FP/TN/FN at a threshold.

    Score convention (matches SiameseVerifier sigmoid output):
      scores >= threshold  ->  predict same-person (positive = 1)
    """
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def confusion_to_metrics(cm: dict[str, int]) -> dict[str, float]:
    """Derive TAR, FAR, FRR, precision, accuracy, F1 from a confusion dict."""
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    total = tp + fp + tn + fn

    tar       = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    f1 = (
        2 * precision * tar / (precision + tar)
        if (precision + tar) > 0 else 0.0
    )
    return {
        "TAR":       round(tar, 6),
        "FAR":       round(far, 6),
        "FRR":       round(frr, 6),
        "precision": round(precision, 6),
        "accuracy":  round(accuracy, 6),
        "F1":        round(f1, 6),
    }

# Perform thresold sweep evaluation

def threshold_sweep(
    df: pd.DataFrame,
    thresholds: Iterable[float] | None = None,
    n_steps: int = 200,
) -> pd.DataFrame:
    """Evaluate the verifier across a range of thresholds.

    Holds the pair set and score outputs fixed; only the decision boundary moves.

    Args:
        df: DataFrame with 'score' and 'label' columns (pre-validated).
        thresholds: Explicit threshold values; if None, n_steps evenly-spaced values
                    spanning [score_min, score_max] are used.
        n_steps: Number of sweep points when thresholds is None.

    Returns:
        DataFrame with columns: threshold, TP, FP, TN, FN, TAR, FAR, FRR, precision, accuracy, F1
    """
    validate_pairs_df(df, context="threshold_sweep")

    scores = df["score"].to_numpy(dtype=float)
    labels = df["label"].to_numpy(dtype=int)

    if thresholds is None:
        lo, hi = float(scores.min()), float(scores.max())
        thresholds = np.linspace(lo, hi, n_steps)

    rows = []
    for t in thresholds:
        cm = compute_confusion(labels, scores, float(t))
        metrics = confusion_to_metrics(cm)
        rows.append({"threshold": float(t), **cm, **metrics})

    return pd.DataFrame(rows)


# Select TAR@FAR=0.01

def select_threshold_tar_at_far(
    sweep_df: pd.DataFrame,
    target_far: float = 0.01,
) -> dict:
    """Select the operating threshold using the TAR@FAR rule.

    Rule (stated, consistent with scripts/metrics.py tar_at_far()):
      Among all thresholds where FAR <= target_far on the validation split,
      choose the one that maximises TAR (i.e. the most permissive threshold
      that still keeps the false accept rate within budget).

    Parameters
    ----------
    sweep_df   : Output of threshold_sweep().
    target_far : FAR upper bound (default 0.01 = 1 %).

    Returns
    -------
    dict with keys: selected_threshold, TAR, FAR, FRR, accuracy, F1,
                    TP, FP, TN, FN, selection_rule.
    """
    candidates = sweep_df[sweep_df["FAR"] <= target_far]

    if candidates.empty:
        # Fallback: no threshold achieves the FAR constraint on this split.
        # Pick the threshold whose FAR is nearest to the target instead.
        idx = (sweep_df["FAR"] - target_far).abs().idxmin()
        best = sweep_df.loc[idx]
        fallback = True
    else:
        idx = candidates["TAR"].idxmax()
        best = candidates.loc[idx]
        fallback = False

    result = best.to_dict()
    result["selected_threshold"] = result.pop("threshold")
    result["selection_rule"] = (
        f"TAR@FAR={target_far} (fallback: nearest FAR)" if fallback
        else f"TAR@FAR={target_far}"
    )
    return result

# Single threshold evaluation

def evaluate_at_threshold(
    df: pd.DataFrame,
    threshold: float,
    split_name: str = "unknown",
) -> dict:
    """Evaluate at one locked threshold and return a full metrics dict."""
    validate_pairs_df(df, context=f"evaluate_at_threshold({split_name})")
    validate_threshold(threshold)

    scores = df["score"].to_numpy(dtype=float)
    labels = df["label"].to_numpy(dtype=int)
    validate_score_label_lengths(scores, labels)

    cm = compute_confusion(labels, scores, threshold)
    metrics = confusion_to_metrics(cm)

    return {
        "split":     split_name,
        "threshold": threshold,
        "n_pairs":   len(df),
        **cm,
        **metrics,
    }

# Save dataframes and metrics

def save_sweep_csv(sweep_df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(path, index=False)


def save_metrics_json(metrics: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
