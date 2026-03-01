import pandas as pd
import random
from pathlib import Path


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _relative_path(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def _get_label(img_path: Path, split_root: Path) -> str:
    """Infer identity from an LFW-style path.

    Supports either:
    - Directory layout: <split_root>/<person_name>/<image_file>
    - Flat layout: <split_root>/<person_name>_<####>.<ext>
    """

    # Return "directory layout" image label
    try:
        rel = img_path.resolve().relative_to(split_root.resolve())
    except Exception:
        rel = img_path

    if len(rel.parts) >= 2:
        return rel.parts[0]

    # Return "flat layout" image label
    stem = img_path.stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def _group_by_label(split_dir: Path) -> dict[str, list[Path]]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    imgs = [p for p in split_dir.rglob("*") if p.is_file() and _is_image_file(p)]
    imgs = sorted(imgs, key=lambda p: p.as_posix())

    grouping: dict[str, list[Path]] = {}
    for p in imgs:
        label = _get_label(p, split_dir)
        grouping.setdefault(label, []).append(p)

    for label in list(grouping.keys()):
        grouping[label] = sorted(grouping[label], key=lambda p: p.as_posix())

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
    project_root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for l, r in pos_pairs:
        rows.append(
            {
                "left_path": _relative_path(l, project_root),
                "right_path": _relative_path(r, project_root),
                "label": 1,
                "split": split_name,
            }
        )
    for l, r in neg_pairs:
        rows.append(
            {
                "left_path": _relative_path(l, project_root),
                "right_path": _relative_path(r, project_root),
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
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    out_dir: Path,
    seed: int = 42,
    n_pos_train: int = 20_000,
    n_pos_val: int = 5_000,
    n_pos_test: int = 5_000,
    neg_per_pos: float = 1.0,
) -> None:
    """Generate LFW verification pairs and write them to disk.

    Pair policy (deterministic given the same inputs + seed):
    - **Dedicated splits**: `train`, `val`, and `test` are generated from separate
      directories (`train_dir`, `val_dir`, `test_dir`).
    - **Positive pairs**: sampled without replacement from all possible (image_i,
      image_j) combinations within each identity, then shuffled with the fixed seed.
    - **Negative pairs**: sampled by choosing two different identities uniformly,
      then choosing one random image from each identity; duplicates are avoided.
    - **Counts**: for each split, we create `n_pos_*` positive pairs and
      `round(n_pos_* * neg_per_pos)` negative pairs.
    """

    if neg_per_pos < 0:
        raise ValueError("neg_per_pos must be >= 0")

    project_root = _project_root()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    train_by_label = _group_by_label(train_dir)
    val_by_label = _group_by_label(val_dir)
    test_by_label = _group_by_label(test_dir)

    n_neg_train = int(round(n_pos_train * neg_per_pos))
    n_neg_val = int(round(n_pos_val * neg_per_pos))
    n_neg_test = int(round(n_pos_test * neg_per_pos))

    pos_train = _sample_positive_pairs(train_by_label, n_pos_train, rng)
    neg_train = _sample_negative_pairs(train_by_label, n_neg_train, rng)
    df_train = _make_pairs_df(pos_train, neg_train, "train", project_root)

    pos_val = _sample_positive_pairs(val_by_label, n_pos_val, rng)
    neg_val = _sample_negative_pairs(val_by_label, n_neg_val, rng)
    df_val = _make_pairs_df(pos_val, neg_val, "val", project_root)

    pos_test = _sample_positive_pairs(test_by_label, n_pos_test, rng)
    neg_test = _sample_negative_pairs(test_by_label, n_neg_test, rng)
    df_test = _make_pairs_df(pos_test, neg_test, "test", project_root)

    df_train.to_csv(out_dir / "train_pairs.csv", index=False)
    df_val.to_csv(out_dir / "val_pairs.csv", index=False)
    df_test.to_csv(out_dir / "test_pairs.csv", index=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate LFW-style verification pairs")
    parser.add_argument("--train-dir", type=str, default="data/train")
    parser.add_argument("--val-dir", type=str, default="data/val")
    parser.add_argument("--test-dir", type=str, default="data/test")
    parser.add_argument("--out-dir", type=str, default="outputs/pairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos-train", type=int, default=20_000)
    parser.add_argument("--pos-val", type=int, default=5_000)
    parser.add_argument("--pos-test", type=int, default=5_000)
    parser.add_argument("--neg-per-pos", type=float, default=1.0)

    args = parser.parse_args()

    generate_pairs(
        train_dir=Path(args.train_dir),
        val_dir=Path(args.val_dir),
        test_dir=Path(args.test_dir),
        out_dir=Path(args.out_dir),
        seed=args.seed,
        n_pos_train=args.pos_train,
        n_pos_val=args.pos_val,
        n_pos_test=args.pos_test,
        neg_per_pos=args.neg_per_pos,
    )


if __name__ == "__main__":
    main()