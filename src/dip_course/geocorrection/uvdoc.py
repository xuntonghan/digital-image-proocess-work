from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MaskCrop:
    image: Image.Image
    mask: np.ndarray
    box_xyxy: tuple[int, int, int, int]


def crop_to_page_mask(
    image: Image.Image,
    mask: Image.Image | np.ndarray,
    *,
    margin_fraction: float = 0.02,
) -> MaskCrop:
    """Crop an RGB image to a binary page mask with a small contextual margin."""
    rgb = image.convert("RGB")
    binary = np.asarray(mask.convert("L") if isinstance(mask, Image.Image) else mask)
    if binary.ndim == 3:
        binary = binary[..., 0]
    if binary.shape != (rgb.height, rgb.width):
        raise ValueError(
            f"Image/mask dimensions differ: {(rgb.height, rgb.width)} vs {binary.shape}"
        )
    binary = binary > 127 if binary.dtype != np.bool_ else binary.copy()
    y, x = np.nonzero(binary)
    if x.size < 64:
        raise ValueError("Document mask contains too few foreground pixels")
    if margin_fraction < 0:
        raise ValueError("margin_fraction must be non-negative")
    margin = round(max(x.max() - x.min() + 1, y.max() - y.min() + 1) * margin_fraction)
    left = max(0, int(x.min()) - margin)
    top = max(0, int(y.min()) - margin)
    right = min(rgb.width, int(x.max()) + 1 + margin)
    bottom = min(rgb.height, int(y.max()) + 1 + margin)
    return MaskCrop(
        image=rgb.crop((left, top, right, bottom)),
        mask=binary[top:bottom, left:right].copy(),
        box_xyxy=(left, top, right, bottom),
    )


def uvdoc_result_to_rgb(value: object) -> np.ndarray:
    """Convert the Transformers UVDoc post-process result to an RGB uint8 array."""
    if not isinstance(value, dict) or "images" not in value:
        raise ValueError("UVDoc post-process result must contain an 'images' entry")
    image = value["images"]
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 UVDoc output, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    # The processor converts RGB inputs to BGR before inference, then flips the
    # final channel axis during post-processing. PIL inputs therefore emerge RGB.
    return array.copy()


def crop_to_foreground(
    image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove empty UVDoc canvas without adding another geometric transform."""
    rgb = np.asarray(image, dtype=np.uint8)
    binary = np.asarray(mask, dtype=bool)
    if rgb.shape[:2] != binary.shape:
        raise ValueError("Image and mask must be aligned")
    y, x = np.nonzero(binary)
    if x.size < 64:
        raise ValueError("Rectified mask contains too few foreground pixels")
    left, right = int(x.min()), int(x.max()) + 1
    top, bottom = int(y.min()), int(y.max()) + 1
    return rgb[top:bottom, left:right].copy(), binary[top:bottom, left:right].copy()
