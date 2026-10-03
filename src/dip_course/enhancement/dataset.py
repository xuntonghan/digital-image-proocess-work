from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dip_course.io import RawSpec, pack_rggb, read_pair_manifest


class SIEDPredictionDataset(Dataset):
    """Paired cached SIED predictions and full-scene target thumbnails."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        manifest: str | Path,
        prediction_dir: str | Path,
        augment: bool = True,
    ) -> None:
        self.records = read_pair_manifest(data_root, manifest)
        self.augment = augment
        prediction_root = Path(prediction_dir)
        self.predictions: dict[str, Path] = {}
        for path in sorted(prediction_root.glob("*.png")):
            key = path.stem.split("_", 1)[-1]
            if key in self.predictions:
                raise ValueError(
                    f"Multiple cached SIED predictions for key {key}: "
                    f"{self.predictions[key]} and {path}"
                )
            self.predictions[key] = path
        missing = [
            record.key
            for record in self.records
            if record.key not in self.predictions
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Missing {len(missing)} cached SIED predictions in "
                f"{prediction_root}: {preview}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        with Image.open(self.predictions[record.key]) as image:
            prediction_image = image.convert("RGB")
            size = prediction_image.size
            prediction = np.asarray(prediction_image, dtype=np.float32) / 255.0
        with Image.open(record.gt_rgb) as image:
            target_image = image.convert("RGB")
            if target_image.size != size:
                target_image = target_image.resize(size, Image.Resampling.LANCZOS)
            target = np.asarray(target_image, dtype=np.float32) / 255.0

        prediction = prediction.transpose(2, 0, 1)
        target = target.transpose(2, 0, 1)
        if self.augment:
            if random.random() < 0.5:
                prediction = prediction[:, :, ::-1]
                target = target[:, :, ::-1]
            if random.random() < 0.5:
                prediction = prediction[:, ::-1, :]
                target = target[:, ::-1, :]
        return {
            "prediction": torch.from_numpy(np.ascontiguousarray(prediction)).float(),
            "target": torch.from_numpy(np.ascontiguousarray(target)).float(),
            "key": record.key,
        }


class SIEDPairDataset(Dataset):
    """Paired low-light RAW and full-resolution sRGB patches."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        manifest: str | Path,
        raw_spec: RawSpec,
        patch_size: int = 256,
        augment: bool = True,
        gt_cache_dir: str | Path | None = None,
        color_aware_probability: float = 0.0,
        color_aware_candidates: int = 4,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.records = read_pair_manifest(data_root, manifest)
        self.raw_spec = raw_spec
        self.patch_size = patch_size
        self.augment = augment
        self.gt_cache_dir = Path(gt_cache_dir) if gt_cache_dir is not None else None
        if not 0.0 <= color_aware_probability <= 1.0:
            raise ValueError("color_aware_probability must be in [0, 1]")
        if color_aware_candidates <= 0:
            raise ValueError("color_aware_candidates must be positive")
        if color_aware_probability > 0.0 and self.gt_cache_dir is None:
            raise ValueError("color-aware cropping requires a GT cache")
        self.color_aware_probability = color_aware_probability
        self.color_aware_candidates = color_aware_candidates

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        actual_bytes = record.low_raw.stat().st_size
        if actual_bytes != self.raw_spec.expected_bytes:
            raise ValueError(
                f"Unexpected RAW size for {record.low_raw}: {actual_bytes} bytes; "
                f"expected {self.raw_spec.expected_bytes}"
            )
        packed_height = self.raw_spec.height // 2
        packed_width = self.raw_spec.width // 2
        size = self.patch_size
        if size > packed_height or size > packed_width:
            raise ValueError(
                f"patch_size {size} exceeds packed RAW dimensions {packed_height}x{packed_width}"
            )

        cached: np.ndarray | None = None
        if self.gt_cache_dir is not None:
            cache_path = self.gt_cache_dir / f"{record.key}.npy"
            if not cache_path.is_file():
                raise FileNotFoundError(f"Missing GT cache entry: {cache_path}")
            cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            expected_target = (self.raw_spec.height, self.raw_spec.width, 3)
            if cached.shape != expected_target or cached.dtype != np.uint8:
                raise ValueError(
                    f"Invalid GT cache {cache_path}: shape={cached.shape}, "
                    f"dtype={cached.dtype}; expected {expected_target} uint8"
                )

        if self.augment:
            if (
                cached is not None
                and random.random() < self.color_aware_probability
            ):
                candidates = [
                    (
                        random.randint(0, packed_height - size),
                        random.randint(0, packed_width - size),
                    )
                    for _ in range(self.color_aware_candidates)
                ]
                top, left = max(
                    candidates,
                    key=lambda position: self._color_score(
                        cached, position[0], position[1], size
                    ),
                )
            else:
                top = random.randint(0, packed_height - size)
                left = random.randint(0, packed_width - size)
        else:
            top = (packed_height - size) // 2
            left = (packed_width - size) // 2

        raw_memmap = np.memmap(
            record.low_raw,
            dtype=np.uint16,
            mode="r",
            shape=(self.raw_spec.height, self.raw_spec.width),
        )
        raw_crop = np.asarray(
            raw_memmap[
                top * 2 : (top + size) * 2,
                left * 2 : (left + size) * 2,
            ]
        )
        normalized = (
            raw_crop.astype(np.float32) - float(self.raw_spec.black_level)
        ) / float(self.raw_spec.white_level - self.raw_spec.black_level)
        raw_patch = pack_rggb(np.clip(normalized, 0.0, 1.0))

        box = (
            left * 2,
            top * 2,
            (left + size) * 2,
            (top + size) * 2,
        )
        if self.gt_cache_dir is None:
            with Image.open(record.gt_rgb) as image:
                if image.size != (self.raw_spec.width, self.raw_spec.height):
                    raise ValueError(
                        f"Unexpected GT size for {record.gt_rgb}: {image.size}; "
                        f"expected {(self.raw_spec.width, self.raw_spec.height)}"
                    )
                rgb_patch = (
                    np.asarray(image.crop(box).convert("RGB"), dtype=np.float32) / 255.0
                )
        else:
            assert cached is not None
            rgb_patch = (
                np.asarray(cached[box[1] : box[3], box[0] : box[2], :]).astype(
                    np.float32
                )
                / 255.0
            )
        rgb_patch = rgb_patch.transpose(2, 0, 1)

        if self.augment:
            if random.random() < 0.5:
                raw_patch = raw_patch[:, :, ::-1]
                rgb_patch = rgb_patch[:, :, ::-1]
            if random.random() < 0.5:
                raw_patch = raw_patch[:, ::-1, :]
                rgb_patch = rgb_patch[:, ::-1, :]
            # Do not rotate packed Bayer planes without also transforming the
            # CFA offsets. The official SIED loader uses only horizontal and
            # vertical flips; naive 90-degree rotations teach contradictory
            # sub-pixel geometry and visibly soften color edges.

        return {
            "raw": torch.from_numpy(np.ascontiguousarray(raw_patch)).float(),
            "target": torch.from_numpy(np.ascontiguousarray(rgb_patch)).float(),
            "key": record.key,
        }

    @staticmethod
    def _color_score(
        target: np.ndarray, top: int, left: int, size: int
    ) -> float:
        """Approximate colorful-pixel coverage using a sparse GT crop."""
        stride = max(1, size // 32)
        sample = target[
            top * 2 : (top + size) * 2 : stride,
            left * 2 : (left + size) * 2 : stride,
            :,
        ].astype(np.float32)
        maximum = sample.max(axis=2)
        minimum = sample.min(axis=2)
        saturation = (maximum - minimum) / np.maximum(maximum, 1.0)
        return float(np.mean(saturation * saturation, dtype=np.float64))
