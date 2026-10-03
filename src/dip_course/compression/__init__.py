"""Image-compression algorithms and benchmark helpers."""

from dip_course.compression.jpeg import (
    JPEGResult,
    JPEGSettings,
    benchmark_jpeg,
    compress_jpeg,
    decompress_jpeg,
)

__all__ = [
    "JPEGResult",
    "JPEGSettings",
    "benchmark_jpeg",
    "compress_jpeg",
    "decompress_jpeg",
]
