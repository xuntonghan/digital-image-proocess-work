from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from dip_course.compression.dct_codec import (
    DCTCodecSettings,
    benchmark_dct,
    compress_dct,
    decompress_dct,
)


def _fixture_image(width: int = 61, height: int = 43) -> Image.Image:
    y, x = np.mgrid[:height, :width]
    array = np.stack(
        (
            (x * 3 + y * 2) % 256,
            (x * 2 + y * 5) % 256,
            (x * 7 + y) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


@pytest.mark.parametrize("subsampling", ["420", "444"])
def test_dct_codec_round_trip_preserves_odd_dimensions(subsampling: str) -> None:
    image = _fixture_image()
    payload = compress_dct(
        image,
        DCTCodecSettings(quality=70, subsampling=subsampling),
    )
    decoded = decompress_dct(payload)

    assert payload.startswith(b"DCTC\x01")
    assert decoded.mode == "RGB"
    assert decoded.size == image.size


def test_dct_benchmark_reports_required_metrics() -> None:
    image = _fixture_image(64, 48)
    result, payload, decoded = benchmark_dct(
        image,
        source_file_bytes=4_000,
        settings=DCTCodecSettings(quality=80),
        repeats=2,
        warmup=0,
    )

    assert result.compressed_bytes == len(payload)
    assert result.file_compression_ratio == pytest.approx(4_000 / len(payload))
    assert result.raw_compression_ratio == pytest.approx(64 * 48 * 3 / len(payload))
    assert result.bits_per_pixel == pytest.approx(len(payload) * 8 / (64 * 48))
    assert result.psnr > 20.0
    assert 0.0 < result.ssim <= 1.0
    assert result.encode_ms >= 0.0
    assert result.decode_ms >= 0.0
    assert decoded.size == image.size


def test_high_quality_improves_fidelity() -> None:
    image = _fixture_image(96, 64)
    low = decompress_dct(compress_dct(image, DCTCodecSettings(quality=30)))
    high = decompress_dct(compress_dct(image, DCTCodecSettings(quality=90)))
    reference = np.asarray(image, dtype=np.float32)
    low_error = np.mean((np.asarray(low, dtype=np.float32) - reference) ** 2)
    high_error = np.mean((np.asarray(high, dtype=np.float32) - reference) ** 2)

    assert high_error < low_error


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quality": 0}, "quality"),
        ({"quality": 101}, "quality"),
        ({"quality": 70, "subsampling": "411"}, "subsampling"),
        ({"quality": 70, "entropy_level": 10}, "entropy_level"),
        ({"quality": 70, "stream_format": "packed"}, "stream_format"),
    ],
)
def test_dct_settings_reject_invalid_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DCTCodecSettings(**kwargs)


def test_decoder_rejects_foreign_and_truncated_payloads() -> None:
    buffer = BytesIO()
    _fixture_image().save(buffer, format="JPEG")
    with pytest.raises(ValueError, match="Not a DCTC"):
        decompress_dct(buffer.getvalue())

    payload = compress_dct(_fixture_image(), DCTCodecSettings(quality=70))
    with pytest.raises(ValueError):
        decompress_dct(payload[:-20])


@pytest.mark.parametrize("subsampling", ["420", "422", "444"])
@pytest.mark.parametrize("stream_format", ["dense", "trimmed", "sparse"])
def test_v2_round_trip(subsampling: str, stream_format: str) -> None:
    image = _fixture_image()
    payload = compress_dct(
        image,
        DCTCodecSettings(
            quality=70,
            subsampling=subsampling,
            stream_format=stream_format,
        ),
    )
    decoded = decompress_dct(payload)
    assert payload.startswith(b"DCTC\x02")
    assert decoded.size == image.size
