import pandas as pd
import random
from pathlib import Path
import yaml


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _relative_path(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _find_lfw_root(extracted_root: Path) -> Path:
    if not extracted_root.exists():
        raise FileNotFoundError(f"Extracted root directory not found: {extracted_root}")

    candidates = sorted(
        [p for p in extracted_root.rglob("lfw") if p.is_dir()],
        key=lambda p: p.as_posix(),
    )
    for c in candidates:
        try:
            next(x for x in c.rglob("*") if x.is_file() and _is_image_file(x))
            return c
        except StopIteration:
            continue

    raise FileNotFoundError(
        f"Could not locate an 'lfw' directory containing images under: {extracted_root}"
    )


def _load_split_csv(split_csv: Path) -> pd.DataFrame:
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    df = pd.read_csv(split_csv)
    required = {"identity", "filename", "image_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Split CSV {split_csv} missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[["identity", "filename", "image_path"]].copy()
    df["identity"] = df["identity"].astype(str)
    df["filename"] = df["filename"].astype(str)
    df["image_path"] = df["image_path"].astype(str)
    df = df.sort_values(by=["identity", "filename", "image_path"], kind="mergesort").reset_index(drop=True)
    return df


def _resolve_image_path(lfw_root: Path, image_path: str) -> Path:
    img = Path(image_path)
    if img.is_absolute():
        return img

    parts = list(img.parts)
    if len(parts) > 0 and parts[0].lower() == "lfw":
        img = Path(*parts[1:])

    return lfw_root / img


def _group_by_identity(split_df: pd.DataFrame, lfw_root: Path) -> dict[str, list[Path]]:
    grouping: dict[str, list[Path]] = {}
    for row in split_df.itertuples(index=False):
        identity = str(row.identity)
        img_path = _resolve_image_path(lfw_root, str(row.image_path))
        grouping.setdefault(identity, []).append(img_path)

    for identity in list(grouping.keys()):
        grouping[identity] = sorted(grouping[identity], key=lambda p: p.as_posix())
    return grouping


def _all_positive_pairs(grouped_imgs: dict[str, list[Path]]) -> list[tuple[Path, Path]]:
    """Generate all possible positive pairs from the same identity for deterministic sampling without replacement"""
    pairs: list[tuple[Path, Path]] = []
    for ident in sorted(grouped_imgs.keys()):
        imgs = grouped_imgs[ident]
        if len(imgs) < 2:
            continue
        for i in range(len(imgs) - 1):
            for j in range(i + 1, len(imgs)):
                pairs.append((imgs[i], imgs[j]))
    return pairs


def _sample_positive_pairs(
    by_id: dict[str, list[Path]],
    n_pos: int,
    rng: random.Random,
) -> list[tuple[Path, Path]]:
    """Samples positive pairs by generating all possible positive pairs and randomly selecting n_pos pairs"""
    all_pos = _all_positive_pairs(by_id)
    rng.shuffle(all_pos)
    if n_pos < 0 or n_pos > len(all_pos):
        n_pos = len(all_pos)
    return all_pos[:n_pos]


def _sample_negative_pairs(
    grouped_imgs: dict[str, list[Path]],
    n_neg: int,
    rng: random.Random,
) -> list[tuple[Path, Path]]:
    """Samples negative pairs by randomly selecting 2 labels, then randomly selecting an image from each label. 
    If a pair has already been selected, as tracked in the "seen" set, a new pair is selected.
    """
    labels = [i for i in sorted(grouped_imgs.keys()) if len(grouped_imgs[i]) > 0]
    if len(labels) < 2:
        raise ValueError("Need at least 2 labels to form negative pairs")

    pairs: list[tuple[Path, Path]] = []
    seen: set[tuple[str, str]] = set()

    max_attempts = max(10_000, n_neg * 50)
    attempts = 0
    while len(pairs) < n_neg and attempts < max_attempts:
        attempts += 1

        a, b = rng.sample(labels, 2)
        left = rng.choice(grouped_imgs[a])
        right = rng.choice(grouped_imgs[b])

        key = (left.as_posix(), right.as_posix())
        if key in seen:
            continue
        seen.add(key)
        pairs.append((left, right))

    if len(pairs) < n_neg:
        raise RuntimeError(
            f"Unable to sample {n_neg} unique negative pairs; got {len(pairs)}. "
            "Try reducing n_neg or ensure more identities/images are available."
        )

    return pairs


def _make_pairs_df(
    pos_pairs: list[tuple[Path, Path]],
    neg_pairs: list[tuple[Path, Path]],
    split_name: str,
    path_base: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for l, r in pos_pairs:
        rows.append(
            {
                "left_path": _relative_path(l, path_base),
                "right_path": _relative_path(r, path_base),
                "label": 1,
                "split": split_name,
            }
        )
    for l, r in neg_pairs:
        rows.append(
            {
                "left_path": _relative_path(l, path_base),
                "right_path": _relative_path(r, path_base),
                "label": 0,
                "split": split_name,
            }
        )

    df = pd.DataFrame(rows, columns=["left_path", "right_path", "label", "split"])

    df = df.sort_values(
        by=["label", "left_path", "right_path"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return df


def generate_pairs(
    splits_dir: Path,
    extracted_root: Path,
    out_dir: Path,
    seed: int = 42,
    n_pos_train: int = 20_000,
    n_pos_val: int = 5_000,
    n_pos_test: int = 5_000,
    neg_per_pos: float = 1.0,
) -> None:
    """Generate LFW verification pairs and write them to disk.

    Pair policy (deterministic given the same inputs + seed):
    - **Split source**: image membership is defined by CSV files in `splits_dir`:
      `train.csv`, `val.csv`, and `test.csv` (from `scripts/ingest_lfw.py`).
    - **Image root**: image files are resolved under the TFDS extracted LFW directory
      discovered under `extracted_root` (typically `data/tfds_cache/downloads/extracted`).
    - **Positive pairs**: sampled without replacement from all possible (image_i,
      image_j) combinations within each identity, then shuffled with the fixed seed.
    - **Negative pairs**: sampled by choosing two different identities uniformly,
      then choosing one random image from each identity; duplicates are avoided.
    - **Counts**: for each split, we create `n_pos_*` positive pairs and
      `round(n_pos_* * neg_per_pos)` negative pairs.
    """

    if neg_per_pos < 0:
        raise ValueError("neg_per_pos must be >= 0")

    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    lfw_root = _find_lfw_root(extracted_root)

    train_df = _load_split_csv(splits_dir / "train.csv")
    val_df = _load_split_csv(splits_dir / "val.csv")
    test_df = _load_split_csv(splits_dir / "test.csv")

    train_by_label = _group_by_identity(train_df, lfw_root)
    val_by_label = _group_by_identity(val_df, lfw_root)
    test_by_label = _group_by_identity(test_df, lfw_root)

    n_neg_train = int(round(n_pos_train * neg_per_pos))
    n_neg_val = int(round(n_pos_val * neg_per_pos))
    n_neg_test = int(round(n_pos_test * neg_per_pos))

    pos_train = _sample_positive_pairs(train_by_label, n_pos_train, rng)
    neg_train = _sample_negative_pairs(train_by_label, n_neg_train, rng)
    df_train = _make_pairs_df(pos_train, neg_train, "train", lfw_root)

    pos_val = _sample_positive_pairs(val_by_label, n_pos_val, rng)
    neg_val = _sample_negative_pairs(val_by_label, n_neg_val, rng)
    df_val = _make_pairs_df(pos_val, neg_val, "val", lfw_root)

    pos_test = _sample_positive_pairs(test_by_label, n_pos_test, rng)
    neg_test = _sample_negative_pairs(test_by_label, n_neg_test, rng)
    df_test = _make_pairs_df(pos_test, neg_test, "test", lfw_root)

    df_train.to_csv(out_dir / "train_pairs.csv", index=False)
    df_val.to_csv(out_dir / "val_pairs.csv", index=False)
    df_test.to_csv(out_dir / "test_pairs.csv", index=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate LFW-style verification pairs")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--extracted-root", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pos-train", type=int, default=None)
    parser.add_argument("--pos-val", type=int, default=None)
    parser.add_argument("--pos-test", type=int, default=None)
    parser.add_argument("--neg-per-pos", type=float, default=1.0)

    args = parser.parse_args()

    project_root = _project_root()

    cfg = None
    if args.config is not None:
        cfg = _load_config(project_root / args.config)

    splits_dir = Path(args.splits_dir) if args.splits_dir is not None else project_root / "outputs/splits"
    out_dir = Path(args.out_dir) if args.out_dir is not None else project_root / "outputs/pairs"

    if args.extracted_root is not None:
        extracted_root = Path(args.extracted_root)
    elif cfg is not None:
        extracted_root = project_root / cfg["paths"]["data_cache"] / "downloads" / "extracted"
    else:
        extracted_root = project_root / "data/tfds_cache/downloads/extracted"

    seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 42) if cfg is not None else 42)

    if args.pos_train is not None:
        n_pos_train = int(args.pos_train)
    elif cfg is not None and "pairs" in cfg and "positive_per_split" in cfg["pairs"]:
        n_pos_train = int(cfg["pairs"]["positive_per_split"])
    else:
        n_pos_train = 20_000

    if args.pos_val is not None:
        n_pos_val = int(args.pos_val)
    elif cfg is not None and "pairs" in cfg and "positive_per_split" in cfg["pairs"]:
        n_pos_val = int(cfg["pairs"]["positive_per_split"])
    else:
        n_pos_val = 5_000

    if args.pos_test is not None:
        n_pos_test = int(args.pos_test)
    elif cfg is not None and "pairs" in cfg and "positive_per_split" in cfg["pairs"]:
        n_pos_test = int(cfg["pairs"]["positive_per_split"])
    else:
        n_pos_test = 5_000

    neg_per_pos = args.neg_per_pos
    if cfg is not None and "pairs" in cfg and "negative_per_split" in cfg["pairs"]:
        denom = float(n_pos_train) if n_pos_train > 0 else 1.0
        neg_per_pos = float(cfg["pairs"]["negative_per_split"]) / denom

    generate_pairs(
        splits_dir=splits_dir,
        extracted_root=extracted_root,
        out_dir=out_dir,
        seed=seed,
        n_pos_train=n_pos_train,
        n_pos_val=n_pos_val,
        n_pos_test=n_pos_test,
        neg_per_pos=neg_per_pos,
    )


if __name__ == "__main__":
    main()