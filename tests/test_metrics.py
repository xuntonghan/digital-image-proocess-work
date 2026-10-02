import math

import numpy as np

from dip_course.metrics import color_fidelity_metrics, psnr_rgb, ssim_rgb


def test_identical_images_have_perfect_metrics() -> None:
    image = np.full((16, 16, 3), 0.4, dtype=np.float32)
    assert math.isinf(psnr_rgb(image, image))
    assert ssim_rgb(image, image) == 1.0


def test_changed_image_has_finite_metrics() -> None:
    target = np.zeros((16, 16, 3), dtype=np.float32)
    prediction = np.full_like(target, 0.1)
    assert 19.9 < psnr_rgb(prediction, target) < 20.1
    assert 0.0 <= ssim_rgb(prediction, target) < 1.0


def test_identical_images_have_perfect_color_metrics() -> None:
    image = np.full((16, 16, 3), 0.4, dtype=np.float32)
    metrics = color_fidelity_metrics(image, image)
    assert metrics["delta_e2000_mean"] == 0.0
    assert metrics["saturation_mae"] == 0.0
    assert metrics["hue_error_degrees"] == 0.0
