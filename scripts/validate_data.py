import argparse
from pathlib import Path

import pandas as pd

from utils import find_lfw_root, load_config


# CSV loading + schema checks
def _load_split_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"identity", "filename", "image_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Split CSV {path} missing required columns: {sorted(missing)}. Found columns: {list(df.columns)}"
        )

    df = df[["identity", "filename", "image_path"]].copy()
    df["identity"] = df["identity"].astype(str)
    df["filename"] = df["filename"].astype(str)
    df["image_path"] = df["image_path"].astype(str)
    return df


def _load_pairs_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"left_path", "right_path", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Pairs CSV {path} missing required columns: {sorted(missing)}. Found columns: {list(df.columns)}"
        )

    df = df[["left_path", "right_path", "label", "split"]].copy()
    df["left_path"] = df["left_path"].astype(str)
    df["right_path"] = df["right_path"].astype(str)
    df["split"] = df["split"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    return df


# Path parsing / normalization
def _strip_lfw_prefix(p: str) -> str:
    parts = Path(p).parts
    if len(parts) > 0 and parts[0].lower() == "lfw":
        return Path(*parts[1:]).as_posix()
    return Path(p).as_posix()


def _identity_from_relpath(rel: str) -> str:
    parts = Path(rel).parts
    if len(parts) < 2:
        raise ValueError(f"Relative image path must look like '<identity>/<filename>': got {rel}")
    return str(parts[0])


def _filename_from_relpath(rel: str) -> str:
    return Path(rel).name


# Split CSV validations
def validate_splits(splits_dir: Path, extracted_root: Path) -> dict[str, set[str]]:
    lfw_root = find_lfw_root(extracted_root)

    split_images: dict[str, set[str]] = {}

    for split_name in ["train", "val", "test"]:
        csv_path = splits_dir / f"{split_name}.csv"
        df = _load_split_csv(csv_path)

        rel_paths = df["image_path"].map(_strip_lfw_prefix)

        # Validation: image_path has expected '<identity>/<filename>' structure.
        bad_format = df[rel_paths.map(lambda p: len(Path(p).parts) < 2)]
        if len(bad_format) > 0:
            ex = bad_format.iloc[0].to_dict()
            raise ValueError(f"Bad image_path format in {csv_path}: example row: {ex}")

        for row, rel in zip(df.itertuples(index=False), rel_paths):
            exp_identity = _identity_from_relpath(rel)
            exp_filename = _filename_from_relpath(rel)

            if str(row.identity) != exp_identity:
                raise ValueError(
                    f"Identity mismatch in {csv_path}: identity column '{row.identity}' != path identity '{exp_identity}' for image_path='{row.image_path}'"
                )
            if str(row.filename) != exp_filename:
                raise ValueError(
                    f"Filename mismatch in {csv_path}: filename column '{row.filename}' != path filename '{exp_filename}' for image_path='{row.image_path}'"
                )

            # Validation: referenced file exists on disk.
            full_path = lfw_root / rel
            if not full_path.exists():
                raise FileNotFoundError(f"Missing image referenced by {csv_path}: {full_path}")

        split_images[split_name] = set(rel_paths.tolist())

    # Validation: no image overlap between any pair of splits.
    overlap_tv = split_images["train"].intersection(split_images["val"])
    overlap_tt = split_images["train"].intersection(split_images["test"])
    overlap_vt = split_images["val"].intersection(split_images["test"])

    if overlap_tv or overlap_tt or overlap_vt:
        msg = ["Image overlap detected between splits:"]
        if overlap_tv:
            msg.append(f"- train ∩ val: {len(overlap_tv)} (example: {next(iter(overlap_tv))})")
        if overlap_tt:
            msg.append(f"- train ∩ test: {len(overlap_tt)} (example: {next(iter(overlap_tt))})")
        if overlap_vt:
            msg.append(f"- val ∩ test: {len(overlap_vt)} (example: {next(iter(overlap_vt))})")
        raise ValueError("\n".join(msg))

    return split_images


# Pair CSV validations
def validate_pairs(pairs_dir: Path, extracted_root: Path, split_images: dict[str, set[str]]) -> None:
    lfw_root = find_lfw_root(extracted_root)

    for split_name in ["train", "val", "test"]:
        csv_path = pairs_dir / f"{split_name}_pairs.csv"
        df = _load_pairs_csv(csv_path)

        # Validation: pair labels are in {0, 1}.
        bad_labels = df[~df["label"].isin([0, 1])]
        if len(bad_labels) > 0:
            ex = bad_labels.iloc[0].to_dict()
            raise ValueError(f"Invalid pair label in {csv_path}: expected 0/1. example row: {ex}")

        # Validation: split column matches the file being validated.
        bad_split = df[df["split"] != split_name]
        if len(bad_split) > 0:
            ex = bad_split.iloc[0].to_dict()
            raise ValueError(
                f"Split column mismatch in {csv_path}: expected split='{split_name}'. example row: {ex}"
            )

        for row in df.itertuples(index=False):
            left_rel = Path(row.left_path).as_posix()
            right_rel = Path(row.right_path).as_posix()

            # Validation: pairs only reference images in the declared split.
            if left_rel not in split_images[split_name]:
                raise ValueError(
                    f"Pair references image not in {split_name} split: {row.left_path} (from {csv_path})"
                )
            if right_rel not in split_images[split_name]:
                raise ValueError(
                    f"Pair references image not in {split_name} split: {row.right_path} (from {csv_path})"
                )

            # Validation: referenced files exist on disk.
            left_full = lfw_root / left_rel
            right_full = lfw_root / right_rel
            if not left_full.exists():
                raise FileNotFoundError(f"Missing left image referenced by {csv_path}: {left_full}")
            if not right_full.exists():
                raise FileNotFoundError(f"Missing right image referenced by {csv_path}: {right_full}")

            # Validation: pair label matches identity equality.
            left_id = _identity_from_relpath(left_rel)
            right_id = _identity_from_relpath(right_rel)
            expected_label = 1 if left_id == right_id else 0
            if int(row.label) != expected_label:
                raise ValueError(
                    f"Incorrect pair label in {csv_path}: got {row.label} but expected {expected_label} for left='{row.left_path}' right='{row.right_path}'"
                )

def validate_all(project_root: Path, config_path: Path | None = None) -> None:
    cfg = load_config(config_path) if config_path is not None else None

    outputs_dir = project_root / (cfg["paths"]["outputs"] if cfg is not None else "outputs")
    splits_dir = outputs_dir / "splits"
    pairs_dir = outputs_dir / "pairs"

    if cfg is None:
        extracted_root = project_root / "data" / "tfds_cache" / "downloads" / "extracted"
    else:
        extracted_root = project_root / cfg["paths"]["data_cache"] / "downloads" / "extracted"

    split_images = validate_splits(splits_dir=splits_dir, extracted_root=extracted_root)
    validate_pairs(pairs_dir=pairs_dir, extracted_root=extracted_root, split_images=split_images)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate split/pair CSVs against extracted LFW images")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / args.config if args.config is not None else None

    validate_all(project_root=project_root, config_path=config_path)
    print("[validate] OK")


if __name__ == "__main__":
    main()
