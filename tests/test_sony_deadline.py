from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def deadline(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / 'cloud'))
    spec = importlib.util.spec_from_file_location('sony_deadline', root / 'cloud/sony_deadline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gtmean_gaussian_formula_matches_direct_kl_definition(deadline):
    generator = torch.Generator().manual_seed(42)
    prediction = torch.rand(3, 3, 12, 16, generator=generator, requires_grad=True)
    target = torch.rand(3, 3, 12, 16, generator=generator) * .6
    weights = prediction.new_tensor([.2989, .587, .114])[None, :, None, None]
    x = (prediction * weights).sum(1).mean((1, 2))
    y = (target * weights).sum(1).mean((1, 2))
    sx, sy = .1 * x, .1 * y
    center, spread = (x + y) / 2, ((sx.square() + sy.square()) / 2).sqrt()
    def kl(mean, sigma):
        return torch.log(spread / sigma) + (sigma.square() + (mean - center).square()) / (2 * spread.square()) - .5
    mix = ((kl(x, sx) + kl(y, sy)) / 2).clamp(0, 1).detach()
    normalized = (prediction * (y / x)[:, None, None, None]).clamp(0, 1)
    expected = (mix * (prediction - target).abs().mean((1, 2, 3)) +
                (1 - mix) * (normalized - target).abs().mean((1, 2, 3))).mean()
    actual = deadline.brightness_training_loss(prediction, target)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-6)
    actual.backward()
    assert torch.isfinite(prediction.grad).all()


def test_gtmean_black_images_have_finite_loss_and_gradient(deadline):
    prediction = torch.zeros(2, 3, 8, 8, requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = deadline.brightness_training_loss(prediction, target)
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(prediction.grad).all()
