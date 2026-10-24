import pytest

torch = pytest.importorskip("torch")

from dip_course.enhancement.adaptive_lut import (
    AdaptiveColorLUT,
    apply_color_lut_chunked,
    apply_rgb_lut,
)


def test_lut_starts_as_identity_and_preserves_rgb_axis_order():
    model = AdaptiveColorLUT(vertices=5)
    image = torch.rand(2, 3, 16, 24)
    image[:, 0, 0, 0], image[:, 1, 0, 0], image[:, 2, 0, 0] = 0.1, 0.4, 0.8
    torch.testing.assert_close(model(image), image, atol=1e-6, rtol=1e-6)


def test_chunked_lut_is_identical_to_full_image_without_chunkwise_context():
    model = AdaptiveColorLUT(vertices=5).eval()
    with torch.no_grad():
        model.residual.normal_(std=0.2)
    image = torch.rand(1, 3, 35, 24)
    torch.testing.assert_close(apply_color_lut_chunked(model, image, rows=8), model(image))


def test_lut_residual_bound_and_backward():
    model = AdaptiveColorLUT(vertices=5, max_delta=0.05)
    with torch.no_grad():
        model.residual.fill_(100)
    image = torch.rand(1, 3, 16, 24)
    prediction = model(image)
    assert (prediction - image).abs().max() <= 0.050001
    prediction.mean().backward()
    assert model.residual.grad is not None
    assert torch.isfinite(model.residual.grad).all()


def test_identity_lut_has_zero_regularization_and_endpoints_are_preserved():
    model = AdaptiveColorLUT(vertices=5)
    assert float(model.regularization().detach()) == 0
    image = torch.tensor([0.0, 1.0]).view(1, 1, 1, 2).expand(1, 3, 1, 2)
    torch.testing.assert_close(model(image), image)


def test_lut_rejects_wrong_batch_or_channels():
    with pytest.raises(ValueError):
        apply_rgb_lut(torch.rand(2, 3, 4, 4), torch.rand(1, 3, 5, 5, 5))
