from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from dip_course.geocorrection.segmentation import (
    PageMaskUNet,
    binary_segmentation_metrics,
    letterbox_image,
    restore_mask,
    segmentation_loss,
)


def test_letterbox_mask_restores_original_shape() -> None:
    mask = np.zeros((37, 61), dtype=np.uint8)
    mask[5:32, 8:53] = 255
    prepared, transform = letterbox_image(
        Image.fromarray(mask), size=64, is_mask=True
    )
    restored = restore_mask(
        np.asarray(prepared, dtype=np.float32) / 255.0,
        transform,
    )
    assert restored.shape == mask.shape
    assert binary_segmentation_metrics(restored, mask >= 128)["iou"] > 0.95


def test_page_mask_unet_shape_and_loss() -> None:
    model = PageMaskUNet(base_channels=8)
    inputs = torch.rand(2, 3, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64), dtype=torch.float32)
    logits = model(inputs)
    loss = segmentation_loss(logits, targets)
    assert logits.shape == targets.shape
    assert torch.isfinite(loss)
    loss.backward()


def test_binary_segmentation_metrics() -> None:
    target = np.array([[True, True], [False, False]])
    prediction = np.array([[True, False], [True, False]])
    values = binary_segmentation_metrics(prediction, target)
    assert values["iou"] == pytest.approx(1 / 3)
    assert values["dice"] == pytest.approx(0.5)
    assert values["precision"] == pytest.approx(0.5)
    assert values["recall"] == pytest.approx(0.5)
