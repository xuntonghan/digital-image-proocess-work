from __future__ import annotations

import hashlib
import json
import statistics
import struct
import time
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image

from dip_course.metrics import psnr_rgb, ssim_rgb

_MAGIC_V1 = b"DCTC\x01"
_MAGIC_V2 = b"DCTC\x02"
_BLOCK_SIZE = 8

_LUMA_QUANTIZATION = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)

_CHROMA_QUANTIZATION = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.float32,
)


def _make_dct_matrix() -> np.ndarray:
    matrix = np.empty((_BLOCK_SIZE, _BLOCK_SIZE), dtype=np.float32)
    for frequency in range(_BLOCK_SIZE):
        scale = np.sqrt(1.0 / _BLOCK_SIZE) if frequency == 0 else np.sqrt(
            2.0 / _BLOCK_SIZE
        )
        for sample in range(_BLOCK_SIZE):
            matrix[frequency, sample] = scale * np.cos(
                np.pi * (2 * sample + 1) * frequency / (2 * _BLOCK_SIZE)
            )
    return matrix


def _make_zigzag_indices() -> np.ndarray:
    indices: list[int] = []
    for diagonal in range(2 * _BLOCK_SIZE - 1):
        cells: list[tuple[int, int]] = []
        for row in range(_BLOCK_SIZE):
            column = diagonal - row
            if 0 <= column < _BLOCK_SIZE:
                cells.append((row, column))
        if diagonal % 2 == 0:
            cells.reverse()
        indices.extend(row * _BLOCK_SIZE + column for row, column in cells)
    return np.asarray(indices, dtype=np.intp)


_DCT_MATRIX = _make_dct_matrix()
_ZIGZAG = _make_zigzag_indices()
_INVERSE_ZIGZAG = np.argsort(_ZIGZAG)


@dataclass(frozen=True)
class DCTCodecSettings:
    quality: int
    subsampling: str = "420"
    entropy_level: int = 9
    stream_format: str = "rle"

    def __post_init__(self) -> None:
        if not 1 <= self.quality <= 100:
            raise ValueError("DCT codec quality must be in [1, 100]")
        if self.subsampling not in {"420", "422", "444"}:
            raise ValueError("DCT codec subsampling must be one of: 420, 422, 444")
        if not 0 <= self.entropy_level <= 9:
            raise ValueError("entropy_level must be in [0, 9]")
        if self.stream_format not in {"rle", "dense", "trimmed", "sparse"}:
            raise ValueError(
                "stream_format must be one of: rle, dense, trimmed, sparse"
            )


@dataclass(frozen=True)
class DCTCodecResult:
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
    payload_sha256: str


def _scaled_quantization_table(base: np.ndarray, quality: int) -> np.ndarray:
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    return np.clip(np.floor((base * scale + 50.0) / 100.0), 1, 255).astype(
        np.float32
    )


def _rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb.astype(np.float32)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = -0.168736 * red - 0.331264 * green + 0.5 * blue + 128.0
    cr = 0.5 * red - 0.418688 * green - 0.081312 * blue + 128.0
    return y, cb, cr


def _ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cb_offset = cb - 128.0
    cr_offset = cr - 128.0
    red = y + 1.402 * cr_offset
    green = y - 0.344136 * cb_offset - 0.714136 * cr_offset
    blue = y + 1.772 * cb_offset
    return np.clip(np.rint(np.stack((red, green, blue), axis=-1)), 0, 255).astype(
        np.uint8
    )


def _downsample_420(channel: np.ndarray) -> np.ndarray:
    pad_bottom = channel.shape[0] % 2
    pad_right = channel.shape[1] % 2
    padded = np.pad(channel, ((0, pad_bottom), (0, pad_right)), mode="edge")
    return padded.reshape(
        padded.shape[0] // 2,
        2,
        padded.shape[1] // 2,
        2,
    ).mean(axis=(1, 3))


def _downsample_422(channel: np.ndarray) -> np.ndarray:
    pad_right = channel.shape[1] % 2
    padded = np.pad(channel, ((0, 0), (0, pad_right)), mode="edge")
    return padded.reshape(padded.shape[0], padded.shape[1] // 2, 2).mean(axis=2)


def _channel_to_blocks(channel: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = channel.shape
    padded_height = ((height + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE
    padded_width = ((width + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE
    padded = np.pad(
        channel,
        ((0, padded_height - height), (0, padded_width - width)),
        mode="edge",
    )
    blocks = (
        padded.reshape(
            padded_height // _BLOCK_SIZE,
            _BLOCK_SIZE,
            padded_width // _BLOCK_SIZE,
            _BLOCK_SIZE,
        )
        .transpose(0, 2, 1, 3)
        .reshape(-1, _BLOCK_SIZE, _BLOCK_SIZE)
    )
    return blocks, (height, width)


def _blocks_to_channel(
    blocks: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    height, width = shape
    padded_height = ((height + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE
    padded_width = ((width + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE
    channel = (
        blocks.reshape(
            padded_height // _BLOCK_SIZE,
            padded_width // _BLOCK_SIZE,
            _BLOCK_SIZE,
            _BLOCK_SIZE,
        )
        .transpose(0, 2, 1, 3)
        .reshape(padded_height, padded_width)
    )
    return channel[:height, :width]


def _write_signed_varint(output: bytearray, value: int) -> None:
    encoded = value * 2 if value >= 0 else -value * 2 - 1
    while encoded >= 0x80:
        output.append((encoded & 0x7F) | 0x80)
        encoded >>= 7
    output.append(encoded)


def _read_signed_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(10):
        if offset >= len(data):
            raise ValueError("Truncated DCT coefficient stream")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            decoded = value // 2 if value % 2 == 0 else -(value // 2) - 1
            return decoded, offset
        shift += 7
    raise ValueError("Invalid DCT coefficient varint")


def _encode_coefficients(coefficients: np.ndarray) -> bytes:
    output = bytearray()
    previous_dc = 0
    for block in coefficients:
        dc = int(block[0])
        _write_signed_varint(output, dc - previous_dc)
        previous_dc = dc

        nonzero = np.flatnonzero(block[1:])
        last = int(nonzero[-1]) + 1 if nonzero.size else 0
        index = 1
        while index <= last:
            run = 0
            while index <= last and block[index] == 0:
                run += 1
                index += 1
            if index > last:
                break
            output.append(run)
            _write_signed_varint(output, int(block[index]))
            index += 1
        output.append(0xFF)
    return bytes(output)


def _decode_coefficients(data: bytes, block_count: int) -> np.ndarray:
    coefficients = np.zeros((block_count, 64), dtype=np.int16)
    offset = 0
    previous_dc = 0
    for block_index in range(block_count):
        dc_delta, offset = _read_signed_varint(data, offset)
        dc = previous_dc + dc_delta
        if not np.iinfo(np.int16).min <= dc <= np.iinfo(np.int16).max:
            raise ValueError("Decoded DC coefficient is outside int16 range")
        coefficients[block_index, 0] = dc
        previous_dc = dc

        coefficient_index = 1
        while True:
            if offset >= len(data):
                raise ValueError("Truncated DCT run-length stream")
            run = data[offset]
            offset += 1
            if run == 0xFF:
                break
            coefficient_index += run
            if coefficient_index >= 64:
                raise ValueError("Invalid zero run in DCT coefficient stream")
            value, offset = _read_signed_varint(data, offset)
            if not np.iinfo(np.int16).min <= value <= np.iinfo(np.int16).max:
                raise ValueError("Decoded AC coefficient is outside int16 range")
            coefficients[block_index, coefficient_index] = value
            coefficient_index += 1
    if offset != len(data):
        raise ValueError("Unexpected bytes after DCT coefficient stream")
    return coefficients


def _encode_dense_coefficients(coefficients: np.ndarray) -> bytes:
    values = coefficients.astype("<i2", copy=True)
    dc = values[:, 0].astype(np.int32)
    delta = np.diff(dc, prepend=0)
    if np.any(delta < np.iinfo(np.int16).min) or np.any(
        delta > np.iinfo(np.int16).max
    ):
        raise ValueError("DC difference is outside int16 range")
    values[:, 0] = delta.astype(np.int16)
    return values.tobytes(order="C")


def _decode_dense_coefficients(data: bytes, block_count: int) -> np.ndarray:
    expected_bytes = block_count * 64 * np.dtype("<i2").itemsize
    if len(data) != expected_bytes:
        raise ValueError("Dense DCT coefficient payload length mismatch")
    coefficients = np.frombuffer(data, dtype="<i2").reshape(block_count, 64).copy()
    dc = np.cumsum(coefficients[:, 0].astype(np.int32))
    if np.any(dc < np.iinfo(np.int16).min) or np.any(dc > np.iinfo(np.int16).max):
        raise ValueError("Decoded DC coefficient is outside int16 range")
    coefficients[:, 0] = dc.astype(np.int16)
    return coefficients


def _encode_trimmed_coefficients(coefficients: np.ndarray) -> bytes:
    values = coefficients.astype("<i2", copy=False)
    dc = values[:, 0].astype(np.int32)
    dc_delta = np.diff(dc, prepend=0)
    if np.any(dc_delta < np.iinfo(np.int16).min) or np.any(
        dc_delta > np.iinfo(np.int16).max
    ):
        raise ValueError("DC difference is outside int16 range")
    positions = np.arange(1, 64, dtype=np.uint8)
    nonzero_positions = np.where(values[:, 1:] != 0, positions, 0)
    last = nonzero_positions.max(axis=1).astype(np.uint8)
    included = positions[None, :] <= last[:, None]
    ac_prefixes = values[:, 1:][included].astype("<i2", copy=False)
    return b"".join(
        (
            last.tobytes(order="C"),
            dc_delta.astype("<i2").tobytes(order="C"),
            ac_prefixes.tobytes(order="C"),
        )
    )


def _decode_trimmed_coefficients(data: bytes, block_count: int) -> np.ndarray:
    prefix_bytes = block_count * 3
    if len(data) < prefix_bytes:
        raise ValueError("Truncated trimmed DCT coefficient payload")
    last = np.frombuffer(data, dtype=np.uint8, count=block_count)
    if np.any(last > 63):
        raise ValueError("Invalid trimmed DCT coefficient length")
    dc_delta = np.frombuffer(
        data,
        dtype="<i2",
        count=block_count,
        offset=block_count,
    )
    ac_count = int(last.astype(np.int64).sum())
    if len(data) != prefix_bytes + ac_count * 2:
        raise ValueError("Trimmed DCT coefficient payload length mismatch")
    coefficients = np.zeros((block_count, 64), dtype=np.int16)
    dc = np.cumsum(dc_delta.astype(np.int32))
    if np.any(dc < np.iinfo(np.int16).min) or np.any(dc > np.iinfo(np.int16).max):
        raise ValueError("Decoded DC coefficient is outside int16 range")
    coefficients[:, 0] = dc.astype(np.int16)
    ac_prefixes = np.frombuffer(data, dtype="<i2", offset=prefix_bytes)
    positions = np.arange(1, 64, dtype=np.uint8)
    coefficients[:, 1:][positions[None, :] <= last[:, None]] = ac_prefixes
    return coefficients


def _encode_sparse_coefficients(coefficients: np.ndarray) -> bytes:
    values = coefficients.astype("<i2", copy=False)
    dc = values[:, 0].astype(np.int32)
    dc_delta = np.diff(dc, prepend=0)
    if np.any(dc_delta < np.iinfo(np.int16).min) or np.any(
        dc_delta > np.iinfo(np.int16).max
    ):
        raise ValueError("DC difference is outside int16 range")
    nonzero = values[:, 1:] != 0
    counts = nonzero.sum(axis=1).astype(np.uint8)
    positions = np.broadcast_to(
        np.arange(1, 64, dtype=np.uint8), nonzero.shape
    )[nonzero]
    ac_values = values[:, 1:][nonzero].astype("<i2", copy=False)
    return b"".join(
        (
            counts.tobytes(order="C"),
            dc_delta.astype("<i2").tobytes(order="C"),
            positions.tobytes(order="C"),
            ac_values.tobytes(order="C"),
        )
    )


def _decode_sparse_coefficients(data: bytes, block_count: int) -> np.ndarray:
    prefix_bytes = block_count * 3
    if len(data) < prefix_bytes:
        raise ValueError("Truncated sparse DCT coefficient payload")
    counts = np.frombuffer(data, dtype=np.uint8, count=block_count)
    if np.any(counts > 63):
        raise ValueError("Invalid sparse DCT coefficient count")
    dc_delta = np.frombuffer(
        data,
        dtype="<i2",
        count=block_count,
        offset=block_count,
    )
    ac_count = int(counts.astype(np.int64).sum())
    if len(data) != prefix_bytes + ac_count * 3:
        raise ValueError("Sparse DCT coefficient payload length mismatch")
    positions = np.frombuffer(
        data,
        dtype=np.uint8,
        count=ac_count,
        offset=prefix_bytes,
    )
    if np.any((positions < 1) | (positions > 63)):
        raise ValueError("Invalid sparse DCT coefficient position")
    ac_values = np.frombuffer(
        data,
        dtype="<i2",
        count=ac_count,
        offset=prefix_bytes + ac_count,
    )
    coefficients = np.zeros((block_count, 64), dtype=np.int16)
    dc = np.cumsum(dc_delta.astype(np.int32))
    if np.any(dc < np.iinfo(np.int16).min) or np.any(dc > np.iinfo(np.int16).max):
        raise ValueError("Decoded DC coefficient is outside int16 range")
    coefficients[:, 0] = dc.astype(np.int16)
    block_indices = np.repeat(np.arange(block_count), counts.astype(np.int64))
    coefficients[block_indices, positions] = ac_values
    return coefficients


def _encode_channel(
    channel: np.ndarray,
    quantization: np.ndarray,
    *,
    stream_format: str,
) -> tuple[bytes, list[int]]:
    blocks, shape = _channel_to_blocks(channel)
    transformed = _DCT_MATRIX @ (blocks - 128.0) @ _DCT_MATRIX.T
    quantized = np.rint(transformed / quantization).astype(np.int16)
    zigzagged = quantized.reshape(-1, 64)[:, _ZIGZAG]
    if stream_format == "dense":
        stream = _encode_dense_coefficients(zigzagged)
    elif stream_format == "trimmed":
        stream = _encode_trimmed_coefficients(zigzagged)
    elif stream_format == "sparse":
        stream = _encode_sparse_coefficients(zigzagged)
    else:
        stream = _encode_coefficients(zigzagged)
    return stream, [shape[0], shape[1]]


def _decode_channel(
    stream: bytes,
    shape: tuple[int, int],
    quantization: np.ndarray,
    *,
    stream_format: str,
) -> np.ndarray:
    padded_height = ((shape[0] + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE
    padded_width = ((shape[1] + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE
    block_count = padded_height * padded_width // (_BLOCK_SIZE * _BLOCK_SIZE)
    if stream_format == "dense":
        zigzagged = _decode_dense_coefficients(stream, block_count)
    elif stream_format == "trimmed":
        zigzagged = _decode_trimmed_coefficients(stream, block_count)
    elif stream_format == "sparse":
        zigzagged = _decode_sparse_coefficients(stream, block_count)
    else:
        zigzagged = _decode_coefficients(stream, block_count)
    quantized = zigzagged[:, _INVERSE_ZIGZAG].reshape(-1, 8, 8)
    transformed = quantized.astype(np.float32) * quantization
    blocks = _DCT_MATRIX.T @ transformed @ _DCT_MATRIX + 128.0
    return _blocks_to_channel(blocks, shape)


def compress_dct(image: Image.Image, settings: DCTCodecSettings) -> bytes:
    """Encode an image into the self-contained DCTC coursework format."""
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    rgb = np.asarray(rgb_image, dtype=np.uint8)
    height, width = rgb.shape[:2]
    y, cb, cr = _rgb_to_ycbcr(rgb)
    if settings.subsampling == "420":
        cb = _downsample_420(cb)
        cr = _downsample_420(cr)
    elif settings.subsampling == "422":
        cb = _downsample_422(cb)
        cr = _downsample_422(cr)

    luma_quantization = _scaled_quantization_table(
        _LUMA_QUANTIZATION, settings.quality
    )
    chroma_quantization = _scaled_quantization_table(
        _CHROMA_QUANTIZATION, settings.quality
    )
    streams: list[bytes] = []
    shapes: list[list[int]] = []
    for channel, quantization in (
        (y, luma_quantization),
        (cb, chroma_quantization),
        (cr, chroma_quantization),
    ):
        stream, shape = _encode_channel(
            channel, quantization, stream_format=settings.stream_format
        )
        streams.append(stream)
        shapes.append(shape)

    coefficient_payload = b"".join(streams)
    metadata = {
        "height": height,
        "quality": settings.quality,
        "shapes": shapes,
        "stream_lengths": [len(stream) for stream in streams],
        "stream_format": settings.stream_format,
        "subsampling": settings.subsampling,
        "width": width,
    }
    header = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    entropy_payload = zlib.compress(coefficient_payload, settings.entropy_level)
    magic = _MAGIC_V2 if settings.stream_format != "rle" else _MAGIC_V1
    return magic + struct.pack(">I", len(header)) + header + entropy_payload


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Invalid {name} in DCTC header")
    return value


def decompress_dct(payload: bytes) -> Image.Image:
    """Decode a DCTC v1 or v2 byte stream to an RGB image."""
    if payload.startswith(_MAGIC_V1):
        magic = _MAGIC_V1
        default_stream_format = "rle"
    elif payload.startswith(_MAGIC_V2):
        magic = _MAGIC_V2
        default_stream_format = "trimmed"
    else:
        raise ValueError("Not a DCTC payload")
    if len(payload) < len(magic) + 4:
        raise ValueError("Truncated DCTC payload")
    header_offset = len(magic)
    header_length = struct.unpack(">I", payload[header_offset : header_offset + 4])[0]
    header_start = header_offset + 4
    header_end = header_start + header_length
    if header_length > 64 * 1024 or header_end >= len(payload):
        raise ValueError("Invalid or truncated DCTC header")
    try:
        metadata = json.loads(payload[header_start:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid DCTC metadata") from exc

    width = _positive_integer(metadata.get("width"), "width")
    height = _positive_integer(metadata.get("height"), "height")
    quality = _positive_integer(metadata.get("quality"), "quality")
    subsampling = metadata.get("subsampling")
    if subsampling not in ("420", "422", "444"):
        raise ValueError("Invalid subsampling in DCTC header")
    stream_format = metadata.get("stream_format", default_stream_format)
    settings = DCTCodecSettings(
        quality=quality,
        subsampling=subsampling,
        stream_format=stream_format,
    )
    if magic == _MAGIC_V1 and stream_format != "rle":
        raise ValueError("DCTC v1 must use the RLE coefficient stream")
    if magic == _MAGIC_V2 and stream_format not in {"dense", "trimmed", "sparse"}:
        raise ValueError("DCTC v2 must use a vectorized coefficient stream")
    shapes = metadata.get("shapes")
    lengths = metadata.get("stream_lengths")
    if not isinstance(shapes, list) or len(shapes) != 3:
        raise ValueError("Invalid component shapes in DCTC header")
    if not isinstance(lengths, list) or len(lengths) != 3:
        raise ValueError("Invalid stream lengths in DCTC header")

    expected_shapes = (
        (height, width),
        (
            ((height + 1) // 2, (width + 1) // 2)
            if settings.subsampling == "420"
            else (height, (width + 1) // 2)
            if settings.subsampling == "422"
            else (height, width)
        ),
        (
            ((height + 1) // 2, (width + 1) // 2)
            if settings.subsampling == "420"
            else (height, (width + 1) // 2)
            if settings.subsampling == "422"
            else (height, width)
        ),
    )
    parsed_shapes: list[tuple[int, int]] = []
    for index, shape in enumerate(shapes):
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError("Invalid component shape in DCTC header")
        parsed = (
            _positive_integer(shape[0], "component height"),
            _positive_integer(shape[1], "component width"),
        )
        if parsed != expected_shapes[index]:
            raise ValueError("Component shape does not match DCTC image dimensions")
        parsed_shapes.append(parsed)
    parsed_lengths = [_positive_integer(value, "stream length") for value in lengths]

    try:
        coefficient_payload = zlib.decompress(payload[header_end:])
    except zlib.error as exc:
        raise ValueError("Invalid DCTC entropy payload") from exc
    if len(coefficient_payload) != sum(parsed_lengths):
        raise ValueError("DCTC coefficient payload length mismatch")
    streams: list[bytes] = []
    offset = 0
    for length in parsed_lengths:
        streams.append(coefficient_payload[offset : offset + length])
        offset += length

    luma_quantization = _scaled_quantization_table(_LUMA_QUANTIZATION, quality)
    chroma_quantization = _scaled_quantization_table(_CHROMA_QUANTIZATION, quality)
    y = _decode_channel(
        streams[0],
        parsed_shapes[0],
        luma_quantization,
        stream_format=settings.stream_format,
    )
    cb = _decode_channel(
        streams[1],
        parsed_shapes[1],
        chroma_quantization,
        stream_format=settings.stream_format,
    )
    cr = _decode_channel(
        streams[2],
        parsed_shapes[2],
        chroma_quantization,
        stream_format=settings.stream_format,
    )
    if settings.subsampling == "420":
        cb = np.repeat(np.repeat(cb, 2, axis=0), 2, axis=1)[:height, :width]
        cr = np.repeat(np.repeat(cr, 2, axis=0), 2, axis=1)[:height, :width]
    elif settings.subsampling == "422":
        cb = np.repeat(cb, 2, axis=1)[:height, :width]
        cr = np.repeat(cr, 2, axis=1)[:height, :width]
    return Image.fromarray(_ycbcr_to_rgb(y, cb, cr), mode="RGB")


def benchmark_dct(
    image: Image.Image,
    *,
    source_file_bytes: int,
    settings: DCTCodecSettings,
    repeats: int = 3,
    warmup: int = 1,
) -> tuple[DCTCodecResult, bytes, Image.Image]:
    """Benchmark DCTC encoding/decoding without metric or file-write time."""
    if source_file_bytes <= 0:
        raise ValueError("source_file_bytes must be positive")
    if repeats <= 0 or warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative")
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    rgb.load()
    for _ in range(warmup):
        warmup_payload = compress_dct(rgb, settings)
        decompress_dct(warmup_payload)

    encode_times: list[float] = []
    payload = b""
    for _ in range(repeats):
        started = time.perf_counter_ns()
        payload = compress_dct(rgb, settings)
        encode_times.append((time.perf_counter_ns() - started) / 1_000_000.0)

    decode_times: list[float] = []
    decoded: Image.Image | None = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        decoded = decompress_dct(payload)
        decode_times.append((time.perf_counter_ns() - started) / 1_000_000.0)
    assert decoded is not None

    target = np.asarray(rgb, dtype=np.float32) / 255.0
    prediction = np.asarray(decoded, dtype=np.float32) / 255.0
    width, height = rgb.size
    compressed_bytes = len(payload)
    raw_rgb_bytes = width * height * 3
    result = DCTCodecResult(
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
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )
    return result, payload, decoded
