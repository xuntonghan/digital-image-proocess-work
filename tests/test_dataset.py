import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from dip_course.enhancement.dataset import (
    SIEDFullSceneDataset,
    SIEDPairDataset,
    SIEDPredictionDataset,
)
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


def test_prediction_dataset_pairs_cached_thumbnail_with_resized_gt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    np.zeros((8, 8), dtype=np.uint16).tofile(root / "scene_with_name.raw")
    target = np.zeros((8, 12, 3), dtype=np.uint8)
    target[..., 1] = 255
    Image.fromarray(target).save(root / "target.png")
    manifest = tmp_path / "pairs.txt"
    manifest.write_text(
        "scene_with_name.raw target.png\n",
        encoding="utf-8",
    )
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction = np.zeros((4, 6, 3), dtype=np.uint8)
    prediction[..., 0] = 128
    Image.fromarray(prediction).save(prediction_dir / "0007_scene_with_name.png")

    dataset = SIEDPredictionDataset(
        data_root=root,
        manifest=manifest,
        prediction_dir=prediction_dir,
        augment=False,
    )
    sample = dataset[0]

    assert tuple(sample["prediction"].shape) == (3, 4, 6)
    assert tuple(sample["target"].shape) == (3, 4, 6)
    assert sample["key"] == "scene_with_name"
    assert float(sample["prediction"][0].mean()) == pytest.approx(128 / 255.0)
    assert float(sample["target"][1].mean()) == pytest.approx(1.0)


def test_prediction_dataset_can_return_packed_raw_guidance(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    raw = np.arange(64, dtype=np.uint16).reshape(8, 8)
    raw.tofile(root / "sample.raw")
    Image.fromarray(np.full((8, 8, 3), 64, dtype=np.uint8)).save(root / "target.png")
    manifest = tmp_path / "pairs.txt"
    manifest.write_text("sample.raw target.png\n", encoding="utf-8")
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    Image.fromarray(np.full((4, 4, 3), 32, dtype=np.uint8)).save(
        prediction_dir / "0000_sample.png"
    )

    dataset = SIEDPredictionDataset(
        data_root=root,
        manifest=manifest,
        prediction_dir=prediction_dir,
        augment=False,
        include_raw=True,
        raw_spec=RawSpec(width=8, height=8, black_level=0, white_level=65535),
    )

    sample = dataset[0]
    assert tuple(sample["raw"].shape) == (4, 4, 4)
    assert torch.all((sample["raw"] >= 0.0) & (sample["raw"] <= 1.0))


def test_prediction_dataset_returns_aligned_native_crops(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    raw = np.arange(64, dtype=np.uint16).reshape(8, 8)
    raw.tofile(root / "sample.raw")
    target = np.zeros((8, 8, 3), dtype=np.uint8)
    target[..., 1] = 255
    Image.fromarray(target).save(root / "target.png")
    manifest = tmp_path / "pairs.txt"
    manifest.write_text("sample.raw target.png\n", encoding="utf-8")
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    Image.fromarray(np.full((8, 8, 3), 64, dtype=np.uint8)).save(
        prediction_dir / "0000_sample.png"
    )

    dataset = SIEDPredictionDataset(
        data_root=root,
        manifest=manifest,
        prediction_dir=prediction_dir,
        augment=False,
        include_raw=True,
        raw_spec=RawSpec(width=8, height=8, black_level=0, white_level=65535),
        crop_size=4,
    )

    sample = dataset[0]
    assert tuple(sample["prediction"].shape) == (3, 4, 4)
    assert tuple(sample["target"].shape) == (3, 4, 4)
    assert tuple(sample["raw"].shape) == (4, 2, 2)
    expected_raw = torch.tensor(
        [
            [[18, 20], [34, 36]],
            [[19, 21], [35, 37]],
            [[27, 29], [43, 45]],
            [[26, 28], [42, 44]],
        ],
        dtype=torch.float32,
    ) / 65535.0
    torch.testing.assert_close(sample["raw"], expected_raw)


def test_prediction_dataset_returns_cached_full_scene_raw_mean(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    np.zeros((8, 8), dtype=np.uint16).tofile(root / "sample.raw")
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(root / "target.png")
    manifest = tmp_path / "pairs.txt"
    manifest.write_text("sample.raw target.png\n", encoding="utf-8")
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
        prediction_dir / "0000_sample.png"
    )
    raw_mean_cache = tmp_path / "raw_means.json"
    raw_mean_cache.write_text(json.dumps({"sample": 0.0004}), encoding="utf-8")

    dataset = SIEDPredictionDataset(
        data_root=root,
        manifest=manifest,
        prediction_dir=prediction_dir,
        augment=False,
        include_raw=True,
        raw_spec=RawSpec(width=8, height=8, black_level=0, white_level=65535),
        crop_size=4,
        raw_mean_cache=raw_mean_cache,
    )

    sample = dataset[0]
    assert float(sample["raw_mean"]) == pytest.approx(0.0004)


def test_full_scene_dataset_returns_native_pair(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    np.arange(64, dtype=np.uint16).reshape(8, 8).tofile(root / "sample.raw")
    Image.fromarray(np.full((8, 8, 3), 96, dtype=np.uint8)).save(root / "target.png")
    manifest = tmp_path / "pairs.txt"
    manifest.write_text("sample.raw target.png\n", encoding="utf-8")

    dataset = SIEDFullSceneDataset(
        data_root=root,
        manifest=manifest,
        raw_spec=RawSpec(width=8, height=8, black_level=0, white_level=65535),
    )

    sample = dataset[0]
    assert tuple(sample["raw"].shape) == (4, 4, 4)
    assert tuple(sample["target"].shape) == (3, 8, 8)


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
