from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from dip_course.geocorrection.dataset import classify_family, discover_pairs
from dip_course.geocorrection.metrics import boundary_linearity
from dip_course.geocorrection.rectify import find_page_corners, rectify_document


def _synthetic_document() -> tuple[Image.Image, Image.Image]:
    image = Image.new("RGB", (220, 180), (35, 35, 35))
    mask = Image.new("L", image.size, 0)
    polygon = [(38, 28), (188, 42), (172, 156), (24, 142)]
    ImageDraw.Draw(mask).polygon(polygon, fill=255)
    draw = ImageDraw.Draw(image)
    draw.polygon(polygon, fill=(242, 238, 220))
    for offset in range(0, 76, 12):
        draw.line((54, 58 + offset, 164, 68 + offset), fill=(30, 30, 30), width=2)
    return image, mask


def test_finds_ordered_document_corners() -> None:
    _, mask = _synthetic_document()
    corners = find_page_corners(np.asarray(mask))

    assert corners.shape == (4, 2)
    assert np.allclose(corners[0], (38, 28), atol=2)
    assert np.allclose(corners[2], (172, 156), atol=2)


def test_rectification_improves_mask_boundary_linearity() -> None:
    image, mask = _synthetic_document()
    before = boundary_linearity(np.asarray(mask) > 0)["boundary_rmse_normalized"]
    result = rectify_document(image, mask, max_side=300)
    after = boundary_linearity(result.corrected_mask)["boundary_rmse_normalized"]

    assert result.corrected_image.ndim == 3
    assert result.corrected_image.shape[:2] == result.corrected_mask.shape
    assert after < before
    assert result.corrected_mask.mean() > 0.95
    assert result.orientation_quarter_turns == 0


def test_auto_orients_vertical_text_lines() -> None:
    image, mask = _synthetic_document()
    rotated_image = image.rotate(90, expand=True)
    rotated_mask = mask.rotate(90, expand=True)

    result = rectify_document(rotated_image, rotated_mask, max_side=300)

    assert result.orientation_quarter_turns == 1


def test_rejects_mismatched_image_and_mask() -> None:
    image, _ = _synthetic_document()
    mask = Image.new("L", (100, 100), 255)
    with pytest.raises(ValueError, match="dimensions differ"):
        rectify_document(image, mask)


@pytest.mark.parametrize(
    ("key", "family"),
    [
        ("150", "numeric_book"),
        ("IMG_20260326_105848", "camera_img"),
        ("20260401_103937", "camera_timestamp"),
        ("dewarp20260728_curved_029", "dewarp_curved"),
        ("dewarp20260728_fold_020", "dewarp_fold"),
        ("dewarp20260728_perspective_019", "dewarp_perspective"),
        ("dewarp20260728_rotate_018", "dewarp_rotate"),
    ],
)
def test_classifies_filename_family(key: str, family: str) -> None:
    assert classify_family(key) == family


def test_discovers_strict_pairs(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "mask").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "one.jpg")
    Image.new("L", (8, 8), 255).save(tmp_path / "mask" / "one.png")

    pairs = discover_pairs(tmp_path)

    assert len(pairs) == 1
    assert pairs[0].key == "one"
