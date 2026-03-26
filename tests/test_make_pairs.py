import csv
import sys
import unittest
from pathlib import Path
import tempfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import make_pairs  # noqa: E402


class TestMakePairs(unittest.TestCase):
    def _write_split_csv(self, path: Path, rows: list[dict[str, str]]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["identity", "filename", "image_path"])
            w.writeheader()
            w.writerows(rows)

    def test_generate_pairs_outputs_consistent_labels(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            extracted_root = td / "extracted"
            lfw_root = extracted_root / "some" / "nested" / "lfw"
            splits_dir = td / "outputs" / "splits"
            out_dir = td / "outputs" / "pairs"

            # Create minimal LFW-like structure with actual files so find_lfw_root works
            for ident in ["Alice", "Bob"]:
                (lfw_root / ident).mkdir(parents=True, exist_ok=True)

            # We'll place 2 images per identity in each split
            def touch(rel: str):
                p = lfw_root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"\x00")

            # train split files
            touch("Alice/Alice_0001.jpg")
            touch("Alice/Alice_0002.jpg")
            touch("Bob/Bob_0001.jpg")
            touch("Bob/Bob_0002.jpg")

            # val split files
            touch("Alice/Alice_0003.jpg")
            touch("Alice/Alice_0004.jpg")
            touch("Bob/Bob_0003.jpg")
            touch("Bob/Bob_0004.jpg")

            # test split files
            touch("Alice/Alice_0005.jpg")
            touch("Alice/Alice_0006.jpg")
            touch("Bob/Bob_0005.jpg")
            touch("Bob/Bob_0006.jpg")

            # Write split CSVs (image_path includes lfw/ prefix like ingest script)
            self._write_split_csv(
                splits_dir / "train.csv",
                [
                    {"identity": "Alice", "filename": "Alice_0001.jpg", "image_path": "lfw/Alice/Alice_0001.jpg"},
                    {"identity": "Alice", "filename": "Alice_0002.jpg", "image_path": "lfw/Alice/Alice_0002.jpg"},
                    {"identity": "Bob", "filename": "Bob_0001.jpg", "image_path": "lfw/Bob/Bob_0001.jpg"},
                    {"identity": "Bob", "filename": "Bob_0002.jpg", "image_path": "lfw/Bob/Bob_0002.jpg"},
                ],
            )
            self._write_split_csv(
                splits_dir / "val.csv",
                [
                    {"identity": "Alice", "filename": "Alice_0003.jpg", "image_path": "lfw/Alice/Alice_0003.jpg"},
                    {"identity": "Alice", "filename": "Alice_0004.jpg", "image_path": "lfw/Alice/Alice_0004.jpg"},
                    {"identity": "Bob", "filename": "Bob_0003.jpg", "image_path": "lfw/Bob/Bob_0003.jpg"},
                    {"identity": "Bob", "filename": "Bob_0004.jpg", "image_path": "lfw/Bob/Bob_0004.jpg"},
                ],
            )
            self._write_split_csv(
                splits_dir / "test.csv",
                [
                    {"identity": "Alice", "filename": "Alice_0005.jpg", "image_path": "lfw/Alice/Alice_0005.jpg"},
                    {"identity": "Alice", "filename": "Alice_0006.jpg", "image_path": "lfw/Alice/Alice_0006.jpg"},
                    {"identity": "Bob", "filename": "Bob_0005.jpg", "image_path": "lfw/Bob/Bob_0005.jpg"},
                    {"identity": "Bob", "filename": "Bob_0006.jpg", "image_path": "lfw/Bob/Bob_0006.jpg"},
                ],
            )

            make_pairs.generate_pairs(
                splits_dir=splits_dir,
                extracted_root=extracted_root,
                out_dir=out_dir,
                seed=42,
                n_pos_train=1,
                n_pos_val=1,
                n_pos_test=1,
                neg_per_pos=1.0,
            )

            for split_name in ["train", "val", "test"]:
                df = pd.read_csv(out_dir / f"{split_name}_pairs.csv")
                self.assertTrue(set(df.columns) >= {"left_path", "right_path", "label", "split"})
                self.assertTrue((df["split"] == split_name).all())
                self.assertTrue(df["label"].isin([0, 1]).all())

                # Check label matches identity equality
                for row in df.itertuples(index=False):
                    left_id = Path(row.left_path).parts[0]
                    right_id = Path(row.right_path).parts[0]
                    expected = 1 if left_id == right_id else 0
                    self.assertEqual(int(row.label), expected)


if __name__ == "__main__":
    unittest.main()
