"""Image-compression algorithms and benchmark helpers."""

from dip_course.compression.dct_codec import (
    DCTCodecResult,
    DCTCodecSettings,
    benchmark_dct,
    compress_dct,
    decompress_dct,
)
from dip_course.compression.jpeg import (
    JPEGResult,
    JPEGSettings,
    benchmark_jpeg,
    compress_jpeg,
    decompress_jpeg,
)

__all__ = [
    "DCTCodecResult",
    "DCTCodecSettings",
    "JPEGResult",
    "JPEGSettings",
    "benchmark_dct",
    "benchmark_jpeg",
    "compress_dct",
    "compress_jpeg",
    "decompress_dct",
    "decompress_jpeg",
]
