from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from dip_course.cli.train_sied_restorer_context import (
    CorrectedNativeDataset,
    check_split_disjoint,
    passes_metric_guardrails,
    prepare_corrected_cache,
    restore_corrected,
    restore_corrected_tiled,
)
from dip_course.enhancement.refiner import (
    RawGuidedNAFRefiner,
    apply_raw_guided_refiner_tiled,
)
from dip_course.io import PairRecord, RawSpec


def tiny_model() -> RawGuidedNAFRefiner:
    return RawGuidedNAFRefiner(
        width=4, encoder_blocks=(1,), middle_blocks=1, decoder_blocks=(1,),
        photometric_channels=4, context_side=8,
    ).eval()


def test_structural_forward_never_recomputes_crop_photometry(monkeypatch) -> None:
    model = tiny_model()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Photometric branch must use the full-scene cache")

    monkeypatch.setattr(model.photometric, "forward", forbidden)
    corrected = torch.rand(1, 3, 8, 8)
    raw = torch.rand(1, 4, 4, 4)
    output = restore_corrected(model, corrected, raw)
    torch.testing.assert_close(output, corrected)
    output.mean().backward()
    assert model.ending.weight.grad is not None


def test_cached_native_crop_preserves_context_and_full_scene_raw_gain(tmp_path: Path) -> None:
    corrected = np.arange(8 * 8 * 3).reshape(8, 8, 3).astype(np.float16) / 256
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "scene.npy", corrected, allow_pickle=False)
    raw = np.arange(64, dtype=np.uint16).reshape(8, 8)
    raw.tofile(tmp_path / "scene.raw")
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / "gt.png")
    record = PairRecord(tmp_path / "scene.raw", tmp_path / "gt.png", "scene")
    dataset = CorrectedNativeDataset(
        [record], cache, {"scene": 0.0042},
        RawSpec(width=8, height=8, black_level=0, white_level=65535),
        crop_size=4, augment=False,
    )
    sample = dataset[0]
    np.testing.assert_allclose(sample["corrected"].numpy(), corrected[2:6, 2:6].transpose(2, 0, 1))
    assert float(sample["raw_mean"]) == pytest.approx(0.0042)
    assert float(sample["raw"][0, 0, 0]) == pytest.approx(18 / 65535)


def test_corrected_cache_does_not_read_gt_and_resumes(tmp_path: Path) -> None:
    np.zeros((8, 8), dtype=np.uint16).tofile(tmp_path / "scene.raw")
    Image.fromarray(np.full((8, 8, 3), 64, dtype=np.uint8)).save(tmp_path / "prediction.png")
    records = [PairRecord(tmp_path / "scene.raw", tmp_path / "unavailable_gt.jpg", "scene")]
    options = {
        "manifest_sha256": "test-manifest", "spec": RawSpec(width=8, height=8),
        "device": torch.device("cpu"), "precision": "fp32",
    }
    model = tiny_model()
    cache = tmp_path / "cache"
    means = prepare_corrected_cache(
        model, records, {"scene": tmp_path / "prediction.png"}, cache, **options)
    cached = np.load(cache / "scene.npy", allow_pickle=False)
    assert cached.shape == (8, 8, 3)
    assert cached.dtype == np.float16
    original_mtime = (cache / "scene.npy").stat().st_mtime_ns
    second_means = prepare_corrected_cache(
        model, records, {"scene": tmp_path / "prediction.png"}, cache, **options)
    assert means == second_means
    assert (cache / "scene.npy").stat().st_mtime_ns == original_mtime


def test_corrected_cache_rejects_changed_photometry(tmp_path: Path) -> None:
    np.zeros((8, 8), dtype=np.uint16).tofile(tmp_path / "scene.raw")
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / "prediction.png")
    records = [PairRecord(tmp_path / "scene.raw", tmp_path / "no_gt.jpg", "scene")]
    options = {
        "manifest_sha256": "manifest", "spec": RawSpec(width=8, height=8),
        "device": torch.device("cpu"), "precision": "fp32",
    }
    model = tiny_model()
    cache = tmp_path / "cache"
    prepare_corrected_cache(model, records, {"scene": tmp_path / "prediction.png"}, cache, **options)
    with torch.no_grad():
        model.photometric.global_head[-1].bias[0].add_(0.1)
    with pytest.raises(ValueError, match="provenance mismatch"):
        prepare_corrected_cache(model, records, {"scene": tmp_path / "prediction.png"}, cache, **options)


def test_cached_tiling_matches_live_full_scene_photometric_inference() -> None:
    model = tiny_model()
    with torch.no_grad():
        model.photometric.global_head[-1].bias[0] = 0.2
        model.ending.weight.normal_(std=0.01)
    image, raw = torch.rand(1, 3, 16, 16), torch.rand(1, 4, 8, 8)
    with torch.inference_mode():
        corrected = model.photometric(image)
        cached_path = restore_corrected_tiled(model, corrected, raw, tile_size=8, overlap=2)
        live_path = apply_raw_guided_refiner_tiled(model, image, raw, tile_size=8, overlap=2)
    torch.testing.assert_close(cached_path, live_path)


def test_selection_rejects_psnr_gain_that_harms_color() -> None:
    baseline = {
        "psnr_mean": 26.0, "ssim_mean": 0.85, "delta_e2000_mean": 4.5,
        "delta_e2000_p95_mean": 9.0, "lightness_mae_mean": 4.0,
        "saturation_mae_mean": 0.03, "hue_error_degrees_mean": 10.0,
    }
    improved = dict(baseline, psnr_mean=26.5, delta_e2000_mean=4.4)
    assert passes_metric_guardrails(improved, baseline)
    assert not passes_metric_guardrails(dict(improved, delta_e2000_mean=5.0), baseline)
    assert not passes_metric_guardrails(dict(improved, ssim_mean=0.84), baseline)
    assert not passes_metric_guardrails(dict(improved, psnr_mean=float("nan")), baseline)


def test_split_check_rejects_shared_gt_even_when_raw_paths_differ(tmp_path: Path) -> None:
    train = [PairRecord(tmp_path / "one.raw", tmp_path / "gt.png", "one")]
    val = [PairRecord(tmp_path / "two.raw", tmp_path / "gt.png", "two")]
    with pytest.raises(ValueError, match="gt_rgb overlap"):
        check_split_disjoint(train, val)
