from __future__ import annotations

import shlex
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RawSpec:
    width: int = 3840
    height: int = 2160
    black_level: int = 512
    white_level: int = 16383
    bayer_pattern: str = "RGGB"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("RAW width and height must be positive")
        if self.width % 2 or self.height % 2:
            raise ValueError("RAW width and height must both be even")
        if not 0 <= self.black_level < self.white_level <= 65535:
            raise ValueError("Expected 0 <= black_level < white_level <= 65535")
        if self.bayer_pattern.upper() != "RGGB":
            raise ValueError("The current implementation supports RGGB only")

    @property
    def sample_count(self) -> int:
        return self.width * self.height

    @property
    def expected_bytes(self) -> int:
        return self.sample_count * np.dtype(np.uint16).itemsize


@dataclass(frozen=True)
class PairRecord:
    low_raw: Path
    gt_rgb: Path
    key: str


def resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"Manifest path escapes data root: {relative}")
    return candidate


def read_pair_manifest(data_root: str | Path, manifest: str | Path) -> list[PairRecord]:
    root = Path(data_root)
    records: list[PairRecord] = []
    with Path(manifest).open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = shlex.split(line, posix=True)
            if len(fields) != 2:
                raise ValueError(
                    f"{manifest}:{line_number}: expected LOW_RAW GT_RGB, got {len(fields)} fields"
                )
            low = resolve_inside(root, fields[0])
            gt = resolve_inside(root, fields[1])
            key = Path(fields[0]).stem
            records.append(PairRecord(low_raw=low, gt_rgb=gt, key=key))
    if not records:
        raise ValueError(f"Manifest contains no image pairs: {manifest}")
    return records


def format_manifest_line(low_relative: str | Path, gt_relative: str | Path) -> str:
    return f"{shlex.quote(str(low_relative))} {shlex.quote(str(gt_relative))}\n"


def read_raw_uint16(path: str | Path, spec: RawSpec) -> np.ndarray:
    raw_path = Path(path)
    actual_bytes = raw_path.stat().st_size
    if actual_bytes != spec.expected_bytes:
        raise ValueError(
            f"Unexpected RAW size for {raw_path}: {actual_bytes} bytes; "
            f"expected {spec.expected_bytes} for {spec.width}x{spec.height} uint16"
        )
    return np.fromfile(raw_path, dtype=np.uint16).reshape(spec.height, spec.width)


def normalize_raw(raw: np.ndarray, spec: RawSpec) -> np.ndarray:
    if raw.shape != (spec.height, spec.width):
        raise ValueError(
            f"Unexpected RAW shape {raw.shape}; expected {(spec.height, spec.width)}"
        )
    normalized = (raw.astype(np.float32) - float(spec.black_level)) / float(
        spec.white_level - spec.black_level
    )
    return np.clip(normalized, 0.0, 1.0)


def pack_rggb(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2-D Bayer array, got shape {raw.shape}")
    height, width = raw.shape
    if height % 2 or width % 2:
        raise ValueError("Bayer height and width must both be even")
    return np.stack(
        (
            raw[0:height:2, 0:width:2],  # R
            raw[0:height:2, 1:width:2],  # G on the R row
            raw[1:height:2, 1:width:2],  # B
            raw[1:height:2, 0:width:2],  # G on the B row
        ),
        axis=0,
    ).astype(np.float32, copy=False)


def load_packed_raw(path: str | Path, spec: RawSpec) -> np.ndarray:
    return pack_rggb(normalize_raw(read_raw_uint16(path, spec), spec))


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}")
    output = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGB").save(output_path)


def iter_existing(records: Iterable[PairRecord]) -> Iterable[PairRecord]:
    for record in records:
        if record.low_raw.is_file() and record.gt_rgb.is_file():
            yield record
