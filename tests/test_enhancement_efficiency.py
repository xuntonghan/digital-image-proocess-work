from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from dip_course.enhancement.efficiency import (
    LocalChannelMean,
    enable_local_channel_means,
    mean_image_psnr_loss,
)
from dip_course.enhancement.refiner import RawGuidedNAFRefiner


def test_psnr_loss_is_mean_of_per_image_logs_not_log_of_batch_mse():
    target = torch.zeros(2, 3, 4, 4)
    prediction = torch.stack((torch.full((3, 4, 4), 0.1), torch.full((3, 4, 4), 0.5)))
    expected = 5 * (math.log10(0.01 + 1e-8) + math.log10(0.25 + 1e-8))
    assert float(mean_image_psnr_loss(prediction, target)) == pytest.approx(expected)


def test_psnr_loss_has_finite_zero_error_gradients_in_amp_dtype():
    prediction = torch.zeros(1, 3, 4, 4, dtype=torch.float16, requires_grad=True)
    loss = mean_image_psnr_loss(prediction, torch.zeros_like(prediction))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(prediction.grad).all()


def test_local_channel_mean_agrees_with_direct_sliding_average():
    image = torch.rand(2, 4, 9, 10, requires_grad=True)
    actual = LocalChannelMean(4)(image)
    expected = F.pad(F.avg_pool2d(image, 4, stride=1), (1, 2, 1, 2), mode="replicate")
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    actual.mean().backward()
    assert torch.isfinite(image.grad).all()


def test_local_channel_mean_does_not_overflow_half_prefix_sums():
    features = torch.full((1, 2, 512, 512), 2.0, dtype=torch.float16)
    actual = LocalChannelMean(256)(features)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.full_like(actual, 2))


def test_large_window_is_exact_global_pooling():
    image = torch.rand(1, 4, 8, 12)
    torch.testing.assert_close(LocalChannelMean(16)(image), F.adaptive_avg_pool2d(image, 1))


def test_tlc_never_converts_photometry_or_changes_checkpoint_keys():
    model = RawGuidedNAFRefiner(width=4, encoder_blocks=(1,), middle_blocks=1,
                               decoder_blocks=(1,), photometric_channels=4, context_side=8)
    keys = set(model.state_dict())
    photo_ids = [id(module) for module in model.photometric.modules()]
    assert enable_local_channel_means(model, rgb_window=16) == 3
    assert set(model.state_dict()) == keys
    assert [id(module) for module in model.photometric.modules()] == photo_ids
