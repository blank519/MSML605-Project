import csv
import sys
import unittest
from pathlib import Path
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import validate_data  # noqa: E402


class TestValidateData(unittest.TestCase):
    def _touch(self, p: Path):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00")

    def _write_csv(self, path: Path, fieldnames: list[str], rows: list[dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    def test_validate_splits_and_pairs_happy_path(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            extracted_root = td / "data" / "tfds_cache" / "downloads" / "extracted"
            lfw_root = extracted_root / "x" / "lfw"
            splits_dir = td / "outputs" / "splits"
            pairs_dir = td / "outputs" / "pairs"

            # Build tiny on-disk dataset (files must exist)
            self._touch(lfw_root / "Alice" / "Alice_0001.jpg")
            self._touch(lfw_root / "Alice" / "Alice_0002.jpg")
            self._touch(lfw_root / "Bob" / "Bob_0001.jpg")
            self._touch(lfw_root / "Bob" / "Bob_0002.jpg")

            # Splits: 2 images in train, 1 in val, 1 in test (no overlaps)
            self._write_csv(
                splits_dir / "train.csv",
                ["identity", "filename", "image_path"],
                [
                    {"identity": "Alice", "filename": "Alice_0001.jpg", "image_path": "lfw/Alice/Alice_0001.jpg"},
                    {"identity": "Bob", "filename": "Bob_0001.jpg", "image_path": "lfw/Bob/Bob_0001.jpg"},
                ],
            )
            self._write_csv(
                splits_dir / "val.csv",
                ["identity", "filename", "image_path"],
                [{"identity": "Alice", "filename": "Alice_0002.jpg", "image_path": "lfw/Alice/Alice_0002.jpg"}],
            )
            self._write_csv(
                splits_dir / "test.csv",
                ["identity", "filename", "image_path"],
                [{"identity": "Bob", "filename": "Bob_0002.jpg", "image_path": "lfw/Bob/Bob_0002.jpg"}],
            )

            split_images = validate_data.validate_splits(splits_dir=splits_dir, extracted_root=extracted_root)

            # Pairs must reference only images in their split; labels must match identity equality
            self._write_csv(
                pairs_dir / "train_pairs.csv",
                ["left_path", "right_path", "label", "split"],
                [
                    {"left_path": "Alice/Alice_0001.jpg", "right_path": "Alice/Alice_0001.jpg", "label": 1, "split": "train"},
                    {"left_path": "Alice/Alice_0001.jpg", "right_path": "Bob/Bob_0001.jpg", "label": 0, "split": "train"},
                ],
            )
            self._write_csv(
                pairs_dir / "val_pairs.csv",
                ["left_path", "right_path", "label", "split"],
                [{"left_path": "Alice/Alice_0002.jpg", "right_path": "Alice/Alice_0002.jpg", "label": 1, "split": "val"}],
            )
            self._write_csv(
                pairs_dir / "test_pairs.csv",
                ["left_path", "right_path", "label", "split"],
                [{"left_path": "Bob/Bob_0002.jpg", "right_path": "Bob/Bob_0002.jpg", "label": 1, "split": "test"}],
            )

            validate_data.validate_pairs(pairs_dir=pairs_dir, extracted_root=extracted_root, split_images=split_images)


if __name__ == "__main__":
    unittest.main()
