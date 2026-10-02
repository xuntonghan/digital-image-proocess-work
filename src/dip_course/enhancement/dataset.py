from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dip_course.io import RawSpec, pack_rggb, read_pair_manifest


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
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.records = read_pair_manifest(data_root, manifest)
        self.raw_spec = raw_spec
        self.patch_size = patch_size
        self.augment = augment
        self.gt_cache_dir = Path(gt_cache_dir) if gt_cache_dir is not None else None

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

        if self.augment:
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
            rotations = random.randint(0, 3)
            if rotations:
                raw_patch = np.rot90(raw_patch, rotations, axes=(1, 2))
                rgb_patch = np.rot90(rgb_patch, rotations, axes=(1, 2))

        return {
            "raw": torch.from_numpy(np.ascontiguousarray(raw_patch)).float(),
            "target": torch.from_numpy(np.ascontiguousarray(rgb_patch)).float(),
            "key": record.key,
        }
