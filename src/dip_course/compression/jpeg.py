from __future__ import annotations

import hashlib
import statistics
import time
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

from dip_course.metrics import psnr_rgb, ssim_rgb

_SUBSAMPLING_CODES = {
    "444": 0,
    "422": 1,
    "420": 2,
}


@dataclass(frozen=True)
class JPEGSettings:
    quality: int
    subsampling: str = "420"
    optimize: bool = False
    progressive: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.quality <= 100:
            raise ValueError("JPEG quality must be in [1, 100]")
        if self.subsampling not in _SUBSAMPLING_CODES:
            raise ValueError("JPEG subsampling must be one of: 444, 422, 420")


@dataclass(frozen=True)
class JPEGResult:
    width: int
    height: int
    source_file_bytes: int
    raw_rgb_bytes: int
    compressed_bytes: int
    file_compression_ratio: float
    raw_compression_ratio: float
    file_space_saving_percent: float
    bits_per_pixel: float
    psnr: float
    ssim: float
    encode_ms: float
    decode_ms: float
    jpeg_sha256: str


def compress_jpeg(image: Image.Image, settings: JPEGSettings) -> bytes:
    """Encode an RGB image to a deterministic in-memory JPEG payload."""
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    stream = BytesIO()
    rgb.save(
        stream,
        format="JPEG",
        quality=settings.quality,
        subsampling=_SUBSAMPLING_CODES[settings.subsampling],
        optimize=settings.optimize,
        progressive=settings.progressive,
    )
    return stream.getvalue()


def decompress_jpeg(payload: bytes) -> Image.Image:
    """Decode a JPEG payload and detach the RGB image from its byte stream."""
    with Image.open(BytesIO(payload)) as image:
        image.load()
        return image.convert("RGB").copy()


def benchmark_jpeg(
    image: Image.Image,
    *,
    source_file_bytes: int,
    settings: JPEGSettings,
    repeats: int = 5,
    warmup: int = 1,
) -> tuple[JPEGResult, bytes, Image.Image]:
    """Benchmark JPEG encoding/decoding without including output-file writes."""
    if source_file_bytes <= 0:
        raise ValueError("source_file_bytes must be positive")
    if repeats <= 0 or warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative")

    rgb = image if image.mode == "RGB" else image.convert("RGB")
    rgb.load()
    for _ in range(warmup):
        warmup_payload = compress_jpeg(rgb, settings)
        decompress_jpeg(warmup_payload)

    encode_times: list[float] = []
    payload = b""
    for _ in range(repeats):
        started = time.perf_counter_ns()
        payload = compress_jpeg(rgb, settings)
        encode_times.append((time.perf_counter_ns() - started) / 1_000_000.0)

    decode_times: list[float] = []
    decoded: Image.Image | None = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        decoded = decompress_jpeg(payload)
        decode_times.append((time.perf_counter_ns() - started) / 1_000_000.0)
    assert decoded is not None
    if decoded.size != rgb.size:
        raise RuntimeError(
            f"JPEG round trip changed dimensions: {rgb.size} -> {decoded.size}"
        )

    target = np.asarray(rgb, dtype=np.float32) / 255.0
    prediction = np.asarray(decoded, dtype=np.float32) / 255.0
    width, height = rgb.size
    compressed_bytes = len(payload)
    raw_rgb_bytes = width * height * 3
    result = JPEGResult(
        width=width,
        height=height,
        source_file_bytes=source_file_bytes,
        raw_rgb_bytes=raw_rgb_bytes,
        compressed_bytes=compressed_bytes,
        file_compression_ratio=source_file_bytes / compressed_bytes,
        raw_compression_ratio=raw_rgb_bytes / compressed_bytes,
        file_space_saving_percent=(1.0 - compressed_bytes / source_file_bytes) * 100.0,
        bits_per_pixel=compressed_bytes * 8.0 / (width * height),
        psnr=psnr_rgb(prediction, target),
        ssim=ssim_rgb(prediction, target),
        encode_ms=statistics.median(encode_times),
        decode_ms=statistics.median(decode_times),
        jpeg_sha256=hashlib.sha256(payload).hexdigest(),
    )
    return result, payload, decoded
