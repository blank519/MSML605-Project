import argparse
from pathlib import Path
import sys

import pandas as pd
import tensorflow as tf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.model import SiameseVerifier
from scripts.metrics import TarAtFarCallback


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png"}


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


def _read_image(path: tf.Tensor, image_size: tuple[int, int]) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


def _make_dataset(
    pairs_csv: Path,
    lfw_root: Path,
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    if not pairs_csv.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {pairs_csv}")

    df = pd.read_csv(pairs_csv)
    required = {"left_path", "right_path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Pairs CSV {pairs_csv} missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    left = df["left_path"].astype(str).apply(lambda p: str(lfw_root / p)).to_numpy()
    right = df["right_path"].astype(str).apply(lambda p: str(lfw_root / p)).to_numpy()
    y = df["label"].astype("float32").to_numpy()

    ds = tf.data.Dataset.from_tensor_slices(((left, right), y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    def _map_fn(paths, label):
        left_p, right_p = paths
        img_a = _read_image(left_p, image_size)
        img_b = _read_image(right_p, image_size)
        return (img_a, img_b), label

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def train(
    project_root: Path,
    config_path: Path | None = None,
    pairs_dir: Path | None = None,
    extracted_root: Path | None = None,
    batch_size: int = 32,
    epochs: int = 5,
    image_size: tuple[int, int] = (160, 160),
    learning_rate: float = 1e-3,
    model_out: Path | None = None,
) -> tuple[SiameseVerifier, tf.keras.callbacks.History]:
    cfg = _load_config(config_path) if config_path is not None else None

    outputs_dir = project_root / (cfg["paths"]["outputs"] if cfg is not None else "outputs")
    pairs_dir = pairs_dir if pairs_dir is not None else outputs_dir / "pairs"

    if extracted_root is None:
        if cfg is None:
            extracted_root = project_root / "data" / "tfds_cache" / "downloads" / "extracted"
        else:
            extracted_root = project_root / cfg["paths"]["data_cache"] / "downloads" / "extracted"

    lfw_root = _find_lfw_root(extracted_root)

    train_ds = _make_dataset(
        pairs_csv=pairs_dir / "train_pairs.csv",
        lfw_root=lfw_root,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = _make_dataset(
        pairs_csv=pairs_dir / "val_pairs.csv",
        lfw_root=lfw_root,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    model = SiameseVerifier(input_shape=(image_size[0], image_size[1], 3))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    callbacks: list[tf.keras.callbacks.Callback] = [
        TarAtFarCallback(dataset=val_ds, far=0.01, name="val_tar@far0.01"),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    if model_out is None:
        model_out = outputs_dir / "models" / "siamese_verifier.keras"
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)

    return model, history


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Siamese face verifier on LFW pairs")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--pairs-dir", type=str, default=None)
    parser.add_argument("--extracted-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--image-size", type=int, nargs=2, default=(160, 160))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]

    config_path = project_root / args.config if args.config is not None else None
    pairs_dir = Path(args.pairs_dir) if args.pairs_dir is not None else None
    extracted_root = Path(args.extracted_root) if args.extracted_root is not None else None
    model_out = Path(args.model_out) if args.model_out is not None else None

    train(
        project_root=project_root,
        config_path=config_path,
        pairs_dir=pairs_dir,
        extracted_root=extracted_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        learning_rate=args.lr,
        model_out=model_out,
    )


if __name__ == "__main__":
    main()