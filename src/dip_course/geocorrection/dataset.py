from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GeoCorrectionPair:
    key: str
    image_path: Path
    mask_path: Path
    family: str


def classify_family(key: str) -> str:
    if key.isdigit():
        return "numeric_book"
    if key.startswith("dewarp"):
        for distortion in ("curved", "fold", "perspective", "rotate"):
            if f"_{distortion}_" in key:
                return f"dewarp_{distortion}"
        return "dewarp_other"
    if key.startswith("IMG_"):
        return "camera_img"
    if key[:4].isdigit():
        return "camera_timestamp"
    return "other"


def discover_pairs(
    data_root: str | Path,
    *,
    image_subdir: str = "images",
    mask_subdir: str = "mask",
) -> list[GeoCorrectionPair]:
    root = Path(data_root)
    image_dir = root / image_subdir
    mask_dir = root / mask_subdir
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {image_subdir}/ and {mask_subdir}/ directories under {root}"
        )
    images = {path.stem: path for path in image_dir.glob("*") if path.is_file()}
    masks = {path.stem: path for path in mask_dir.glob("*") if path.is_file()}
    if not images:
        raise ValueError(f"No input images found in {image_dir}")
    if set(images) != set(masks):
        image_only = sorted(set(images) - set(masks))
        mask_only = sorted(set(masks) - set(images))
        raise ValueError(
            "Image/mask stems do not match: "
            f"image-only={image_only[:10]}, mask-only={mask_only[:10]}"
        )
    return [
        GeoCorrectionPair(
            key=key,
            image_path=images[key],
            mask_path=masks[key],
            family=classify_family(key),
        )
        for key in sorted(images)
    ]
