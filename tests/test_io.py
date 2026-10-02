from pathlib import Path

import numpy as np
import pytest

from dip_course.io import (
    RawSpec,
    normalize_raw,
    pack_rggb,
    read_pair_manifest,
    read_raw_uint16,
)


def test_normalize_and_pack_rggb() -> None:
    spec = RawSpec(width=4, height=2, black_level=10, white_level=110)
    raw = np.asarray([[10, 20, 30, 40], [50, 60, 70, 110]], dtype=np.uint16)
    normalized = normalize_raw(raw, spec)
    packed = pack_rggb(normalized)
    assert packed.shape == (4, 1, 2)
    np.testing.assert_allclose(packed[0], [[0.0, 0.2]])
    np.testing.assert_allclose(packed[1], [[0.1, 0.3]])
    np.testing.assert_allclose(packed[2], [[0.5, 1.0]])
    np.testing.assert_allclose(packed[3], [[0.4, 0.6]])


def test_raw_reader_rejects_wrong_byte_count(tmp_path: Path) -> None:
    path = tmp_path / "bad.raw"
    path.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="Unexpected RAW size"):
        read_raw_uint16(
            path, RawSpec(width=4, height=2, black_level=0, white_level=1023)
        )


def test_manifest_rejects_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    manifest = tmp_path / "pairs.txt"
    manifest.write_text("../outside.raw gt.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes data root"):
        read_pair_manifest(root, manifest)


def test_manifest_supports_quoted_paths(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    manifest = tmp_path / "pairs.txt"
    manifest.write_text("low.raw 'ground truth/1.jpg'\n", encoding="utf-8")
    records = read_pair_manifest(root, manifest)
    assert records[0].gt_rgb == (root / "ground truth/1.jpg").resolve()
