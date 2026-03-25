from pathlib import Path

import yaml


def load_config(config_path: Path | str) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png"}


def find_lfw_root(extracted_root: Path) -> Path:
    if not extracted_root.exists():
        raise FileNotFoundError(f"Extracted root directory not found: {extracted_root}")

    candidates = sorted(
        [p for p in extracted_root.rglob("lfw") if p.is_dir()],
        key=lambda p: p.as_posix(),
    )
    for c in candidates:
        try:
            next(x for x in c.rglob("*") if x.is_file() and is_image_file(x))
            return c
        except StopIteration:
            continue

    raise FileNotFoundError(
        f"Could not locate an 'lfw' directory containing images under: {extracted_root}"
    )
