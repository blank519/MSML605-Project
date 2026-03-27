"""
tracker.py — Appends every run as a JSON record to outputs/runs/run_log.json.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DEFAULT_LOG = Path("outputs") / "runs" / "run_log.json"


def _git_commit_hash() -> str:
    """Return the short HEAD commit hash, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class RunTracker:
    """Append-only run logger backed by a single JSON file."""

    def __init__(self, log_path: str | Path = _DEFAULT_LOG) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._runs: list[dict] = self._load()

    def start_run(
        self,
        config_name: str,
        data_version: str,
        split: str,
        note: str = "",
        threshold: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Register the start of a new run and return its run_id."""
        run_id = str(uuid.uuid4())[:8]
        record: dict[str, Any] = {
            "run_id":       run_id,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "commit":       _git_commit_hash(),
            "config_name":  config_name,
            "data_version": data_version,
            "split":        split,
            "threshold":    threshold,
            "note":         note,
            "status":       "started",
            "metrics":      {},
        }
        if extra:
            record.update(extra)
        self._runs.append(record)
        self._save()
        return run_id

    def finish_run(
        self,
        run_id: str,
        metrics: dict[str, Any],
        threshold: float | None = None,
        artifacts: list[str] | None = None,
    ) -> None:
        """Update a run record with final metrics and mark it done."""
        record = self._get(run_id)
        record["status"] = "done"
        record["metrics"] = metrics
        if threshold is not None:
            record["threshold"] = threshold
        if artifacts:
            record["artifacts"] = artifacts
        self._save()

    def fail_run(self, run_id: str, error: str) -> None:
        """Mark a run as failed with an error message."""
        record = self._get(run_id)
        record["status"] = "failed"
        record["error"] = error
        self._save()

    def summary_table(self) -> list[dict]:
        """Return a compact list of run summaries sorted by timestamp."""
        keys = [
            "run_id", "timestamp", "commit", "config_name",
            "data_version", "split", "threshold", "note", "status",
        ]
        rows = []
        for r in self._runs:
            row = {k: r.get(k) for k in keys}
            row.update(r.get("metrics", {}))
            rows.append(row)
        return sorted(rows, key=lambda x: x["timestamp"])

    def print_summary(self) -> None:
        """Print a human-readable run table to stdout."""
        rows = self.summary_table()
        if not rows:
            print("No runs logged yet.")
            return
        header_keys = [
            "run_id", "timestamp", "config_name", "split",
            "threshold", "TAR", "FAR", "accuracy", "note",
        ]
        print("\n" + "=" * 110)
        print("  ".join(f"{k:<18}" for k in header_keys))
        print("=" * 110)
        for r in rows:
            print("  ".join(f"{str(r.get(k, '')):<18}" for k in header_keys))
        print()

    def _load(self) -> list[dict]:
        if self.log_path.exists():
            with open(self.log_path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        return []

    def _save(self) -> None:
        with open(self.log_path, "w") as f:
            json.dump(self._runs, f, indent=2)

    def _get(self, run_id: str) -> dict:
        for r in self._runs:
            if r["run_id"] == run_id:
                return r
        raise KeyError(f"Run '{run_id}' not found in log.")
