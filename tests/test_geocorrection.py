from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from dip_course.cli.evaluate_geocorrection_ocr import (
    _count_characters,
    _ocr_values,
)
from dip_course.cli.run_geocorrection import _portable_source_path
from dip_course.cli.visualize_geocorrection_ml import _fit_on_white
from dip_course.geocorrection.dataset import classify_family, discover_pairs
from dip_course.geocorrection.metrics import boundary_linearity
from dip_course.geocorrection.rectify import find_page_corners, rectify_document
from dip_course.geocorrection.uvdoc import (
    crop_to_foreground,
    crop_to_page_mask,
    uvdoc_result_to_rgb,
)


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


def test_discovers_pairs_from_custom_stage_directories(tmp_path: Path) -> None:
    (tmp_path / "rectified").mkdir()
    (tmp_path / "rectified_masks").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "rectified" / "one.jpg")
    Image.new("L", (8, 8), 255).save(
        tmp_path / "rectified_masks" / "one.png"
    )

    pairs = discover_pairs(
        tmp_path,
        image_subdir="rectified",
        mask_subdir="rectified_masks",
    )

    assert pairs[0].image_path.parent.name == "rectified"
    assert pairs[0].mask_path.parent.name == "rectified_masks"


def test_portable_source_path_accepts_external_predicted_mask(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    external = tmp_path / "predictions" / "mask.png"
    external.parent.mkdir()
    external.touch()

    assert _portable_source_path(external, root) == str(external.resolve())


def test_mask_guided_crop_includes_margin() -> None:
    image = Image.new("RGB", (100, 80), (10, 20, 30))
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((20, 10, 79, 69), fill=255)

    crop = crop_to_page_mask(image, mask, margin_fraction=0.1)

    assert crop.box_xyxy == (14, 4, 86, 76)
    assert crop.image.size == (72, 72)
    assert crop.mask.shape == (72, 72)


def test_uvdoc_postprocess_output_is_preserved_as_rgb() -> None:
    expected = np.array([[[1, 2, 3], [10, 20, 30]]], dtype=np.uint8)

    rgb = uvdoc_result_to_rgb({"images": expected})

    assert np.array_equal(rgb, expected)
    assert not np.shares_memory(rgb, expected)


def test_crops_rectified_canvas_to_foreground() -> None:
    image = np.zeros((12, 14, 3), dtype=np.uint8)
    mask = np.zeros((12, 14), dtype=bool)
    mask[2:11, 3:13] = True

    cropped_image, cropped_mask = crop_to_foreground(image, mask)

    assert cropped_image.shape == (9, 10, 3)
    assert cropped_mask.shape == (9, 10)
    assert cropped_mask.all()


def test_ocr_values_count_non_whitespace_characters() -> None:
    values = _ocr_values(
        {
            "res": {
                "rec_texts": ["几何 纠正", "ABC 123"],
                "rec_scores": [0.95, 0.75],
            }
        },
        threshold=0.8,
    )

    assert _count_characters("几何 \n 纠正") == 4
    assert values["line_count"] == 2
    assert values["character_count"] == 10
    assert values["confident_line_count"] == 1
    assert values["confident_character_count"] == 4
    assert values["confidence_weighted_characters"] == pytest.approx(8.3)


def test_ocr_values_reject_mismatched_predictions() -> None:
    with pytest.raises(ValueError, match="counts differ"):
        _ocr_values(
            {"res": {"rec_texts": ["one"], "rec_scores": []}}, threshold=0.8
        )


def test_comparison_thumbnail_preserves_requested_canvas() -> None:
    source = Image.new("RGB", (200, 100), (1, 2, 3))

    fitted = _fit_on_white(source, (80, 80))

    assert fitted.size == (80, 80)
    assert fitted.getpixel((40, 40)) == (1, 2, 3)
    assert fitted.getpixel((40, 5)) == (255, 255, 255)
