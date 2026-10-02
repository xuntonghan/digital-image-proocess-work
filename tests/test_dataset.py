from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from dip_course.enhancement.dataset import SIEDPairDataset
from dip_course.io import RawSpec


def test_dataset_returns_aligned_raw_and_rgb_patches(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    raw = np.arange(64, dtype=np.uint16).reshape(8, 8)
    raw.tofile(root / "sample.raw")
    Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(root / "sample.png")
    manifest = tmp_path / "pairs.txt"
    manifest.write_text("sample.raw sample.png\n", encoding="utf-8")

    dataset = SIEDPairDataset(
        data_root=root,
        manifest=manifest,
        raw_spec=RawSpec(width=8, height=8, black_level=0, white_level=65535),
        patch_size=2,
        augment=False,
    )
    sample = dataset[0]
    assert tuple(sample["raw"].shape) == (4, 2, 2)
    assert tuple(sample["target"].shape) == (3, 4, 4)
    assert sample["key"] == "sample"


def test_dataset_reads_uint8_gt_cache(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    raw = np.arange(64, dtype=np.uint16).reshape(8, 8)
    raw.tofile(root / "sample.raw")
    target = np.full((8, 8, 3), 77, dtype=np.uint8)
    Image.fromarray(target).save(root / "sample.png")
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "sample.npy", target, allow_pickle=False)
    manifest = tmp_path / "pairs.txt"
    manifest.write_text("sample.raw sample.png\n", encoding="utf-8")

    dataset = SIEDPairDataset(
        data_root=root,
        manifest=manifest,
        raw_spec=RawSpec(width=8, height=8, black_level=0, white_level=65535),
        patch_size=2,
        augment=False,
        gt_cache_dir=cache,
    )
    sample = dataset[0]

    np.testing.assert_allclose(sample["target"].numpy(), 77 / 255.0)


def test_color_aware_crop_prefers_saturated_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    np.zeros((8, 8), dtype=np.uint16).tofile(root / "sample.raw")
    target = np.full((8, 8, 3), 96, dtype=np.uint8)
    target[4:, 4:] = (255, 0, 0)
    Image.fromarray(target).save(root / "sample.png")
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "sample.npy", target, allow_pickle=False)
    manifest = tmp_path / "pairs.txt"
    manifest.write_text("sample.raw sample.png\n", encoding="utf-8")

    coordinates = iter((0, 0, 2, 2))
    monkeypatch.setattr("random.random", lambda: 0.0)
    monkeypatch.setattr("random.randint", lambda _low, _high: next(coordinates))
    dataset = SIEDPairDataset(
        data_root=root,
        manifest=manifest,
        raw_spec=RawSpec(width=8, height=8, black_level=0, white_level=65535),
        patch_size=2,
        augment=True,
        gt_cache_dir=cache,
        color_aware_probability=1.0,
        color_aware_candidates=2,
    )
    sample = dataset[0]
    assert float(sample["target"][0].mean()) == pytest.approx(1.0)
    assert float(sample["target"][1:].mean()) == pytest.approx(0.0)
