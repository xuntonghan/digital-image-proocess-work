from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dip_course.compression.jpeg import (
    JPEGSettings,
    benchmark_jpeg,
    compress_jpeg,
    decompress_jpeg,
)
from dip_course.compression.visualize import (
    build_compression_grid,
    build_rate_distortion_plot,
)


def test_jpeg_round_trip_preserves_dimensions() -> None:
    array = np.zeros((32, 48, 3), dtype=np.uint8)
    array[:, :24] = (240, 40, 20)
    image = Image.fromarray(array)

    payload = compress_jpeg(image, JPEGSettings(quality=75, subsampling="420"))
    decoded = decompress_jpeg(payload)

    assert payload.startswith(b"\xff\xd8")
    assert decoded.mode == "RGB"
    assert decoded.size == image.size


def test_jpeg_benchmark_reports_required_metrics() -> None:
    y, x = np.mgrid[:40, :60]
    array = np.stack((x * 4, y * 6, (x + y) * 2), axis=-1).astype(np.uint8)
    image = Image.fromarray(array)

    result, payload, decoded = benchmark_jpeg(
        image,
        source_file_bytes=2_000,
        settings=JPEGSettings(quality=80, subsampling="444"),
        repeats=2,
        warmup=0,
    )

    assert result.compressed_bytes == len(payload)
    assert result.file_compression_ratio == pytest.approx(2_000 / len(payload))
    assert result.raw_compression_ratio == pytest.approx(40 * 60 * 3 / len(payload))
    assert result.bits_per_pixel == pytest.approx(len(payload) * 8 / (40 * 60))
    assert result.psnr > 20.0
    assert 0.0 < result.ssim <= 1.0
    assert result.encode_ms >= 0.0
    assert result.decode_ms >= 0.0
    assert decoded.size == image.size


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quality": 0}, "quality"),
        ({"quality": 101}, "quality"),
        ({"quality": 75, "subsampling": "411"}, "subsampling"),
    ],
)
def test_jpeg_settings_reject_invalid_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        JPEGSettings(**kwargs)


def test_builds_original_and_quality_comparison_grid(tmp_path: Path) -> None:
    original_path = tmp_path / "01.png"
    decoded_path = tmp_path / "01_q50.png"
    Image.new("RGB", (48, 32), (120, 30, 200)).save(original_path)
    Image.new("RGB", (48, 32), (118, 32, 198)).save(decoded_path)
    output = tmp_path / "comparison.png"

    build_compression_grid(
        originals={"01": original_path},
        decoded_by_quality={50: {"01": decoded_path}},
        keys=["01"],
        output=output,
        panel_size=(100, 80),
    )

    with Image.open(output) as grid:
        assert grid.size == (258, 118)


def test_builds_rate_distortion_plot(tmp_path: Path) -> None:
    points = [
        {
            "quality": 50,
            "bits_per_pixel_mean": 0.8,
            "psnr_mean": 31.0,
            "ssim_mean": 0.90,
        },
        {
            "quality": 80,
            "bits_per_pixel_mean": 1.6,
            "psnr_mean": 36.0,
            "ssim_mean": 0.96,
        },
    ]
    output = tmp_path / "rate_distortion.png"

    build_rate_distortion_plot(
        series={"codec-a": points, "codec-b": points},
        output=output,
        size=(800, 360),
    )

    with Image.open(output) as plot:
        assert plot.size == (800, 360)
