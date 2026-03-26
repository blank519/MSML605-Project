"""
tests/test_evaluate_and_tracker.py — Milestone 2 unit + integration tests.

Unit tests cover:
  - compute_confusion()
  - confusion_to_metrics()
  - threshold_sweep()
  - select_threshold_tar_at_far()
  - validate_pairs_df()
  - validate_threshold()
  - validate_score_label_lengths()
  - RunTracker (start / finish / fail / summary / persistence)

Integration test:
  - Tiny end-to-end cycle: synthetic scored pairs -> sweep -> select ->
    evaluate -> save artifacts -> log run.
  - Does NOT require downloading LFW or loading the TF model.
  - Runs in seconds from a clean clone.

Note on consistency with scripts/metrics.py:
  tar_at_far() in scripts/metrics.py finds the threshold via the
  (1 - FAR) quantile of *negative* scores.  select_threshold_tar_at_far()
  in src/evaluate.py works from the sweep grid instead.  Both honour the
  same TAR@FAR=0.01 rule; minor numeric differences between them on the
  same data are expected due to quantile vs grid resolution.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import (
    compute_confusion,
    confusion_to_metrics,
    threshold_sweep,
    select_threshold_tar_at_far,
    validate_pairs_df,
    validate_threshold,
    validate_score_label_lengths,
    evaluate_at_threshold,
    save_sweep_csv,
    save_metrics_json,
    load_scored_pairs,
)
from src.tracker import RunTracker

# Create fixtures

def _make_scored_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Synthetic scored pairs.

    Score = label + small noise, so the verifier is nearly perfect —
    useful for checking that TAR@FAR=0.01 can be satisfied.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    scores = labels.astype(float) + rng.normal(0, 0.1, size=n)
    return pd.DataFrame({"score": scores, "label": labels})


def _make_pairs_csv(tmp_path: Path, n: int = 60, seed: int = 42) -> Path:
    """Write a minimal scored pairs CSV to a temp directory."""
    df = _make_scored_df(n=n, seed=seed)
    # Also add the original pairs columns so load_scored_pairs() is tested
    # with a realistic full-format CSV.
    df["left_path"]  = [f"lfw/person_{i%10}/img_{i:04d}.jpg" for i in range(n)]
    df["right_path"] = [f"lfw/person_{(i+1)%10}/img_{i:04d}.jpg" for i in range(n)]
    df["split"] = "val"
    p = tmp_path / "val_scored.csv"
    df.to_csv(p, index=False)
    return p

# Unit tests — compute_confusion

class TestComputeConfusion:
    def test_perfect_separation(self):
        labels = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        cm = compute_confusion(labels, scores, threshold=0.5)
        assert cm == {"TP": 2, "FP": 0, "TN": 2, "FN": 0}

    def test_all_predicted_positive(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.9, 0.9, 0.9])
        cm = compute_confusion(labels, scores, threshold=0.5)
        assert cm["FP"] == 2 and cm["TN"] == 0 and cm["TP"] == 2 and cm["FN"] == 0

    def test_all_predicted_negative(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.1, 0.1, 0.1, 0.1])
        cm = compute_confusion(labels, scores, threshold=0.5)
        assert cm["TP"] == 0 and cm["FN"] == 2

    def test_score_at_boundary_is_positive(self):
        """Score exactly equal to threshold must be predicted positive."""
        labels = np.array([1])
        scores = np.array([0.5])
        cm = compute_confusion(labels, scores, threshold=0.5)
        assert cm["TP"] == 1

#Unit tests — confusion_to_metrics

class TestConfusionToMetrics:
    def test_perfect(self):
        cm = {"TP": 50, "FP": 0, "TN": 50, "FN": 0}
        m = confusion_to_metrics(cm)
        assert m["TAR"] == 1.0
        assert m["FAR"] == 0.0
        assert m["accuracy"] == 1.0
        assert m["F1"] == 1.0

    def test_zero_denominators_no_crash(self):
        cm = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        m = confusion_to_metrics(cm)
        assert m["TAR"] == 0.0

    def test_tar_far_values(self):
        # 80 true positives out of 100 positives → TAR = 0.80
        # 10 false positives out of 100 negatives → FAR = 0.10
        cm = {"TP": 80, "FP": 10, "TN": 90, "FN": 20}
        m = confusion_to_metrics(cm)
        assert abs(m["TAR"] - 0.80) < 1e-6
        assert abs(m["FAR"] - 0.10) < 1e-6

# Unit tests — validate_pairs_df

class TestValidatePairsDf:
    def test_valid(self):
        df = _make_scored_df()
        validate_pairs_df(df)   # no exception

    def test_missing_score_column(self):
        df = pd.DataFrame({"label": [0, 1]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_pairs_df(df)

    def test_invalid_labels(self):
        df = pd.DataFrame({"score": [0.5, 0.6], "label": [0, 2]})
        with pytest.raises(ValueError, match="binary"):
            validate_pairs_df(df)

    def test_nan_in_score(self):
        df = pd.DataFrame({"score": [0.5, float("nan")], "label": [0, 1]})
        with pytest.raises(ValueError, match="NaN"):
            validate_pairs_df(df)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"score": pd.Series([], dtype=float),
                           "label": pd.Series([], dtype=int)})
        with pytest.raises(ValueError, match="empty"):
            validate_pairs_df(df)

    def test_similarity_column_alias(self, tmp_path):
        """load_scored_pairs should accept 'similarity' as an alias for 'score'."""
        df = pd.DataFrame({"similarity": [0.7, 0.3], "label": [1, 0]})
        p = tmp_path / "test.csv"
        df.to_csv(p, index=False)
        loaded = load_scored_pairs(p)
        assert "score" in loaded.columns

# Unit tests — validate_threshold

class TestValidateThreshold:
    def test_valid(self):
        validate_threshold(0.5)   # no exception

    def test_non_numeric(self):
        with pytest.raises(TypeError):
            validate_threshold("high")

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            validate_threshold(2e7)

# Unit tests — validate_score_label_lengths

class TestValidateScoreLabelLengths:
    def test_equal_lengths_ok(self):
        validate_score_label_lengths(np.array([0.5, 0.6]), np.array([0, 1]))

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError, match="Score count"):
            validate_score_label_lengths(np.array([0.5, 0.6, 0.7]), np.array([0, 1]))

# Unit tests — threshold_sweep and select_threshold_tar_at_far
class TestThresholdSweep:
    def test_output_columns(self):
        df = _make_scored_df(n=100)
        sweep = threshold_sweep(df, n_steps=50)
        for col in ("threshold", "TP", "FP", "TN", "FN", "TAR", "FAR", "F1"):
            assert col in sweep.columns

    def test_n_steps_respected(self):
        df = _make_scored_df(n=100)
        sweep = threshold_sweep(df, n_steps=50)
        assert len(sweep) == 50

    def test_far_monotone_nonincreasing(self):
        """As threshold increases, FAR must be non-increasing."""
        df = _make_scored_df(n=200)
        sweep = threshold_sweep(df, n_steps=100)
        diffs = sweep["FAR"].diff().dropna()
        assert (diffs <= 1e-9).all()

    def test_custom_thresholds(self):
        df = _make_scored_df()
        sweep = threshold_sweep(df, thresholds=[0.0, 0.5, 1.0])
        assert len(sweep) == 3

# Unit tests — select_threshold_tar_at_far

class TestSelectThresholdTarAtFar:
    def test_far_constraint_honoured(self):
        df = _make_scored_df(n=300, seed=1)
        sweep = threshold_sweep(df, n_steps=200)
        result = select_threshold_tar_at_far(sweep, target_far=0.01)
        assert result["FAR"] <= 0.01 + 1e-6

    def test_fallback_when_no_candidate(self):
        """When no threshold achieves the FAR constraint, fall back gracefully."""
        # Uniform scores → sweep FAR is always the same value
        df = pd.DataFrame({"score": [0.5] * 100, "label": [1] * 50 + [0] * 50})
        sweep = threshold_sweep(df, n_steps=5)
        result = select_threshold_tar_at_far(sweep, target_far=0.0)
        assert "selected_threshold" in result

    def test_selection_rule_label(self):
        df = _make_scored_df(n=200)
        sweep = threshold_sweep(df, n_steps=50)
        result = select_threshold_tar_at_far(sweep, target_far=0.01)
        assert "TAR@FAR=0.01" in result["selection_rule"]

    def test_consistency_with_metrics_tar_at_far(self):
        """Verify rough consistency with scripts/metrics.py tar_at_far().

        Both implement the TAR@FAR=0.01 rule.  The grid-sweep approach may
        differ slightly from the quantile approach — we allow a tolerance of
        5 percentage points to account for sweep resolution.
        """
        from scripts.metrics import tar_at_far as metrics_tar_at_far

        df = _make_scored_df(n=500, seed=99)
        sweep = threshold_sweep(df, n_steps=500)
        sweep_result = select_threshold_tar_at_far(sweep, target_far=0.01)

        tar_ref, _ = metrics_tar_at_far(
            df["label"].to_numpy(), df["score"].to_numpy(), far=0.01
        )

        if not np.isnan(tar_ref):
            assert abs(sweep_result["TAR"] - tar_ref) < 0.05, (
                f"TAR mismatch: sweep={sweep_result['TAR']:.4f}, "
                f"metrics.py={tar_ref:.4f}"
            )

# Unit tests — RunTracker

class TestRunTracker:
    def test_start_finish_roundtrip(self, tmp_path):
        tracker = RunTracker(log_path=tmp_path / "run_log.json")
        run_id  = tracker.start_run("cfg", "v1", "val", note="unit test")
        tracker.finish_run(run_id, metrics={"TAR": 0.9, "FAR": 0.01})

        log = json.loads((tmp_path / "run_log.json").read_text())
        assert len(log) == 1
        assert log[0]["run_id"] == run_id
        assert log[0]["status"] == "done"
        assert log[0]["metrics"]["TAR"] == 0.9

    def test_fail_run(self, tmp_path):
        tracker = RunTracker(log_path=tmp_path / "run_log.json")
        run_id  = tracker.start_run("cfg", "v1", "val")
        tracker.fail_run(run_id, "simulated error")
        log = json.loads((tmp_path / "run_log.json").read_text())
        assert log[0]["status"] == "failed"
        assert "simulated error" in log[0]["error"]

    def test_persistence_across_instances(self, tmp_path):
        log_path = tmp_path / "run_log.json"
        tracker  = RunTracker(log_path=log_path)
        for i in range(3):
            rid = tracker.start_run("cfg", "v1", "val", note=f"run {i}")
            tracker.finish_run(rid, metrics={"TAR": 0.8 + i * 0.01})
        # Reload from disk
        tracker2 = RunTracker(log_path=log_path)
        assert len(tracker2.summary_table()) == 3

    def test_unknown_run_id_raises(self, tmp_path):
        tracker = RunTracker(log_path=tmp_path / "run_log.json")
        with pytest.raises(KeyError):
            tracker.finish_run("nonexistent_id", metrics={})

    def test_threshold_recorded(self, tmp_path):
        tracker = RunTracker(log_path=tmp_path / "run_log.json")
        run_id  = tracker.start_run("cfg", "v1", "val")
        tracker.finish_run(run_id, metrics={}, threshold=0.7231)
        log = json.loads((tmp_path / "run_log.json").read_text())
        assert log[0]["threshold"] == pytest.approx(0.7231)

# Integration test — full end-to-end cycle with synthetic data

class TestIntegrationEndToEnd:
    """
    Full pipeline integration test using only synthetic data.
    No LFW download, no model loading, no external dependencies.
    Should run in well under 10 seconds.
    """

    def test_full_milestone2_pipeline(self, tmp_path):
        # 1. Create a synthetic scored CSV (mimics output of score_pairs.py)
        csv_path = _make_pairs_csv(tmp_path, n=80, seed=7)
        df = load_scored_pairs(csv_path)
        assert len(df) == 80
        assert "score" in df.columns
        assert "label" in df.columns

        # 2. Threshold sweep
        sweep_df = threshold_sweep(df, n_steps=100)
        assert len(sweep_df) == 100
        assert all(c in sweep_df.columns for c in
                   ["threshold", "TAR", "FAR", "FRR", "accuracy", "F1"])

        # 3. Select threshold using TAR@FAR=0.01
        selected = select_threshold_tar_at_far(sweep_df, target_far=0.01)
        assert "selected_threshold" in selected
        assert selected["FAR"] <= 0.01 + 1e-6

        # 4. Save sweep CSV and selected threshold JSON
        sweep_path = tmp_path / "sweep.csv"
        sel_path   = tmp_path / "selected_threshold.json"
        save_sweep_csv(sweep_df, sweep_path)
        save_metrics_json(selected, sel_path)
        assert sweep_path.exists()
        reloaded = pd.read_csv(sweep_path)
        assert len(reloaded) == 100

        sel_loaded = json.loads(sel_path.read_text())
        assert sel_loaded["selected_threshold"] == pytest.approx(
            selected["selected_threshold"]
        )

        # 5. Evaluate at locked threshold
        result = evaluate_at_threshold(
            df, threshold=selected["selected_threshold"], split_name="val"
        )
        assert result["n_pairs"] == 80
        assert 0.0 <= result["TAR"] <= 1.0
        assert 0.0 <= result["FAR"] <= 1.0
        assert result["TP"] + result["FP"] + result["TN"] + result["FN"] == 80

        metrics_path = tmp_path / "metrics.json"
        save_metrics_json(result, metrics_path)
        assert metrics_path.exists()

        # 6. Track run + verify persistence
        tracker = RunTracker(log_path=tmp_path / "run_log.json")
        run_id  = tracker.start_run(
            config_name="integration_test",
            data_version="v1",
            split="val",
            note="integration test run",
        )
        tracker.finish_run(
            run_id,
            metrics=result,
            threshold=selected["selected_threshold"],
            artifacts=[str(sweep_path), str(sel_path), str(metrics_path)],
        )
        log = json.loads((tmp_path / "run_log.json").read_text())
        assert log[0]["status"] == "done"
        assert log[0]["metrics"]["n_pairs"] == 80
        assert log[0]["threshold"] == pytest.approx(selected["selected_threshold"])
