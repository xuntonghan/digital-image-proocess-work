from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dip_course.cli.train_sied_restorer import _loss_components
from dip_course.cli.train_sied_restorer_native import native_crop


def test_native_crop_keeps_raw_and_rgb_geometry_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinates = iter((2, 3))
    monkeypatch.setattr("random.randint", lambda _low, _high: next(coordinates))
    base = torch.arange(3 * 16 * 20).reshape(1, 3, 16, 20).float()
    target = base + 1
    raw = torch.arange(4 * 8 * 10).reshape(1, 4, 8, 10).float()

    base_crop, target_crop, raw_crop = native_crop(base, target, raw, 8)

    assert tuple(base_crop.shape) == (1, 3, 8, 8)
    assert tuple(target_crop.shape) == (1, 3, 8, 8)
    assert tuple(raw_crop.shape) == (1, 4, 4, 4)
    torch.testing.assert_close(target_crop, base_crop + 1)


def test_native_crop_requires_even_patch_size() -> None:
    base = torch.zeros(1, 3, 16, 20)
    target = torch.zeros_like(base)
    raw = torch.zeros(1, 4, 8, 10)

    with pytest.raises(ValueError, match="even"):
        native_crop(base, target, raw, 7)


def test_psnr_loss_profile_keeps_mse_dominant() -> None:
    target = torch.zeros(1, 3, 16, 16)
    prediction = torch.full_like(target, 0.1)
    parameters = {"global_exposure": torch.zeros(1, 1)}

    balanced, _ = _loss_components(
        prediction,
        target,
        prediction,
        parameters,
        profile="balanced",
    )
    psnr, components = _loss_components(
        prediction,
        target,
        prediction,
        parameters,
        profile="psnr",
    )

    assert torch.isfinite(psnr)
    assert float(psnr) < float(balanced)
    assert float(components["mse"]) == pytest.approx(0.01)
