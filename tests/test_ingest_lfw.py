import sys
import unittest
from pathlib import Path
import types


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ingest_lfw imports tensorflow_datasets at module import time, but these unit
# tests only exercise pure functions (e.g., split_by_image). To avoid requiring
# TFDS to be installed just to run unit tests, we stub the module.
if "tensorflow_datasets" not in sys.modules:
    sys.modules["tensorflow_datasets"] = types.ModuleType("tensorflow_datasets")

import ingest_lfw  # noqa: E402


class TestIngestLfw(unittest.TestCase):
    def _fake_records(self, n: int):
        recs = []
        for i in range(n):
            ident = f"Person_{i % 3}"  # ensure some repeated identities
            fname = f"{ident}_{i:04d}.jpg"
            recs.append({"identity": ident, "filename": fname, "image_path": f"lfw/{ident}/{fname}"})
        return recs

    def test_split_by_image_deterministic_for_same_seed(self):
        records = self._fake_records(12)
        splits1 = ingest_lfw.split_by_image(records, seed=123, val_frac=0.2, test_frac=0.2)
        splits2 = ingest_lfw.split_by_image(records, seed=123, val_frac=0.2, test_frac=0.2)

        self.assertEqual(splits1, splits2)

    def test_split_by_image_changes_with_different_seed(self):
        records = self._fake_records(12)
        splits1 = ingest_lfw.split_by_image(records, seed=123, val_frac=0.2, test_frac=0.2)
        splits2 = ingest_lfw.split_by_image(records, seed=999, val_frac=0.2, test_frac=0.2)

        # Not a strict guarantee, but extremely likely for shuffled order
        self.assertNotEqual(splits1, splits2)

    def test_split_sizes_sum_to_n(self):
        records = self._fake_records(25)
        splits = ingest_lfw.split_by_image(records, seed=1, val_frac=0.2, test_frac=0.2)

        n_total = sum(len(v) for v in splits.values())
        self.assertEqual(n_total, len(records))
        self.assertGreaterEqual(len(splits["train"]), 1)
        self.assertGreaterEqual(len(splits["val"]), 1)
        self.assertGreaterEqual(len(splits["test"]), 1)


if __name__ == "__main__":
    unittest.main()
