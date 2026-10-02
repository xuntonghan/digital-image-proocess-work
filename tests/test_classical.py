import numpy as np

from dip_course.cli.tune_classical import downsample_mean
from dip_course.enhancement.classical import (
    auto_classical_enhance,
    auto_tone_map_rgb,
    classical_enhance,
    demosaic_rggb,
    estimate_gray_world_white_balance,
    tone_map_rgb,
)


def test_constant_bayer_demosaics_to_constant_rgb() -> None:
    raw = np.full((8, 10), 0.25, dtype=np.float32)
    rgb = demosaic_rggb(raw)
    assert rgb.shape == (8, 10, 3)
    np.testing.assert_allclose(rgb, 0.25, atol=1e-6)


def test_classical_identity_parameters_preserve_constant() -> None:
    raw = np.full((8, 10), 0.25, dtype=np.float32)
    rgb = classical_enhance(raw, exposure=1.0, gamma=1.0, white_balance=(1.0, 1.0, 1.0))
    np.testing.assert_allclose(rgb, 0.25, atol=1e-6)


def test_tone_map_rejects_non_rgb_input() -> None:
    with np.testing.assert_raises(ValueError):
        tone_map_rgb(np.zeros((4, 4), dtype=np.float32))


def test_downsample_mean_preserves_constant_rgb() -> None:
    image = np.full((8, 12, 3), 0.25, dtype=np.float32)
    result = downsample_mean(image, 4)

    assert result.shape == (2, 3, 3)
    np.testing.assert_allclose(result, 0.25)


def test_gray_world_white_balance_equalizes_channel_means() -> None:
    image = np.ones((4, 4, 3), dtype=np.float32)
    image *= np.asarray([2.0, 1.0, 0.5], dtype=np.float32)

    gains = estimate_gray_world_white_balance(image)

    np.testing.assert_allclose(gains, (0.5, 1.0, 2.0))


def test_auto_classical_returns_image_and_parameters() -> None:
    raw = np.full((8, 8), 0.01, dtype=np.float32)
    image, parameters = auto_classical_enhance(raw)

    assert image.shape == (8, 8, 3)
    assert float(parameters["exposure"]) > 0


def test_auto_tone_map_rejects_invalid_percentile() -> None:
    image = np.ones((4, 4, 3), dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        auto_tone_map_rgb(image, percentile=100.0)
