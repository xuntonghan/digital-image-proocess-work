from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from dip_course.geocorrection.dataset import GeoCorrectionPair


@dataclass(frozen=True)
class LetterboxTransform:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    offset_x: int
    offset_y: int
    size: int


def letterbox_image(
    image: Image.Image,
    *,
    size: int,
    is_mask: bool = False,
) -> tuple[Image.Image, LetterboxTransform]:
    if size <= 0:
        raise ValueError("size must be positive")
    width, height = image.size
    scale = min(size / width, size / height)
    resized_width = max(1, min(size, round(width * scale)))
    resized_height = max(1, min(size, round(height * scale)))
    resampling = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    resized = image.resize((resized_width, resized_height), resampling)
    offset_x = (size - resized_width) // 2
    offset_y = (size - resized_height) // 2
    mode = "L" if is_mask else "RGB"
    canvas = Image.new(mode, (size, size), color=0)
    canvas.paste(resized, (offset_x, offset_y))
    return canvas, LetterboxTransform(
        original_width=width,
        original_height=height,
        resized_width=resized_width,
        resized_height=resized_height,
        offset_x=offset_x,
        offset_y=offset_y,
        size=size,
    )


def restore_mask(
    mask: np.ndarray,
    transform: LetterboxTransform,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    if mask.shape != (transform.size, transform.size):
        raise ValueError("mask shape does not match the letterbox transform")
    cropped = mask[
        transform.offset_y : transform.offset_y + transform.resized_height,
        transform.offset_x : transform.offset_x + transform.resized_width,
    ]
    image = Image.fromarray(np.clip(cropped * 255.0, 0, 255).astype(np.uint8))
    restored = image.resize(
        (transform.original_width, transform.original_height),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(restored, dtype=np.float32) / 255.0 >= threshold


class _ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(8, output_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class PageMaskUNet(nn.Module):
    """Compact U-Net for foreground page segmentation."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        if base_channels <= 0 or base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of 8")
        channels = [base_channels * (2**index) for index in range(5)]
        self.encoder = nn.ModuleList(
            [
                _ConvBlock(3, channels[0]),
                *[
                    _ConvBlock(channels[index - 1], channels[index])
                    for index in range(1, len(channels))
                ],
            ]
        )
        self.decoder = nn.ModuleList(
            [
                _ConvBlock(channels[index] + channels[index - 1], channels[index - 1])
                for index in range(4, 0, -1)
            ]
        )
        self.output = nn.Conv2d(channels[0], 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features: list[torch.Tensor] = []
        values = inputs
        for index, block in enumerate(self.encoder):
            values = block(values)
            features.append(values)
            if index + 1 < len(self.encoder):
                values = F.max_pool2d(values, 2)
        for block, skip in zip(self.decoder, reversed(features[:-1]), strict=True):
            values = F.interpolate(
                values,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            values = block(torch.cat((values, skip), dim=1))
        return self.output(values)


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dimensions = (1, 2, 3)
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()

    prediction_dx = probabilities[..., :, 1:] - probabilities[..., :, :-1]
    target_dx = targets[..., :, 1:] - targets[..., :, :-1]
    prediction_dy = probabilities[..., 1:, :] - probabilities[..., :-1, :]
    target_dy = targets[..., 1:, :] - targets[..., :-1, :]
    edge_loss = 0.5 * (
        F.l1_loss(prediction_dx, target_dx) + F.l1_loss(prediction_dy, target_dy)
    )
    return 0.35 * bce + 0.50 * dice_loss + 0.15 * edge_loss


def binary_segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    predicted = prediction.astype(bool)
    expected = target.astype(bool)
    if predicted.shape != expected.shape:
        raise ValueError("prediction and target shapes must match")
    true_positive = int(np.count_nonzero(predicted & expected))
    false_positive = int(np.count_nonzero(predicted & ~expected))
    false_negative = int(np.count_nonzero(~predicted & expected))
    union = true_positive + false_positive + false_negative
    predicted_boundary = _binary_boundary(predicted)
    expected_boundary = _binary_boundary(expected)
    predicted_tolerance = _binary_dilate(predicted_boundary, iterations=2)
    expected_tolerance = _binary_dilate(expected_boundary, iterations=2)
    predicted_boundary_count = int(np.count_nonzero(predicted_boundary))
    expected_boundary_count = int(np.count_nonzero(expected_boundary))
    boundary_precision = (
        int(np.count_nonzero(predicted_boundary & expected_tolerance))
        / predicted_boundary_count
        if predicted_boundary_count
        else float(expected_boundary_count == 0)
    )
    boundary_recall = (
        int(np.count_nonzero(expected_boundary & predicted_tolerance))
        / expected_boundary_count
        if expected_boundary_count
        else float(predicted_boundary_count == 0)
    )
    return {
        "iou": true_positive / union if union else 1.0,
        "dice": (2.0 * true_positive) / (2.0 * true_positive + false_positive + false_negative)
        if true_positive + false_positive + false_negative
        else 1.0,
        "precision": true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 1.0,
        "recall": true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 1.0,
        "boundary_f1": (
            2.0 * boundary_precision * boundary_recall
            / (boundary_precision + boundary_recall)
            if boundary_precision + boundary_recall
            else 0.0
        ),
    }


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    erosion = np.ones_like(mask, dtype=bool)
    for offset_y in range(3):
        for offset_x in range(3):
            erosion &= padded[
                offset_y : offset_y + mask.shape[0],
                offset_x : offset_x + mask.shape[1],
            ]
    return mask & ~erosion


def _binary_dilate(mask: np.ndarray, *, iterations: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(result, dtype=bool)
        for offset_y in range(3):
            for offset_x in range(3):
                expanded |= padded[
                    offset_y : offset_y + result.shape[0],
                    offset_x : offset_x + result.shape[1],
                ]
        result = expanded
    return result


class PageMaskDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        pairs: list[GeoCorrectionPair],
        *,
        size: int,
        augment: bool,
        seed: int = 20261018,
    ) -> None:
        self.pairs = pairs
        self.size = size
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.pairs)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        pair = self.pairs[index]
        with Image.open(pair.image_path) as source:
            image = source.convert("RGB")
        with Image.open(pair.mask_path) as source:
            mask = source.convert("L")
        generator = random.Random(self.seed + self.epoch * len(self.pairs) + index)
        if self.augment:
            if generator.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = ImageEnhance.Brightness(image).enhance(generator.uniform(0.75, 1.25))
            image = ImageEnhance.Contrast(image).enhance(generator.uniform(0.75, 1.25))
            image = ImageEnhance.Color(image).enhance(generator.uniform(0.80, 1.20))
        image, _ = letterbox_image(image, size=self.size)
        mask, _ = letterbox_image(mask, size=self.size, is_mask=True)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.float32)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(mask_array[None].copy())
        return image_tensor, mask_tensor, pair.key


def save_segmentation_checkpoint(
    path: Path,
    *,
    model: PageMaskUNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    input_size: int,
    base_channels: int,
    split: dict[str, list[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "dip-course-page-mask-unet-v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "model_config": {
                "input_size": input_size,
                "base_channels": base_channels,
            },
            "split": split,
        },
        temporary,
    )
    temporary.replace(path)


def load_segmentation_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[PageMaskUNet, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format") != "dip-course-page-mask-unet-v1":
        raise ValueError("Unsupported page segmentation checkpoint")
    config = checkpoint["model_config"]
    model = PageMaskUNet(base_channels=int(config["base_channels"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint
