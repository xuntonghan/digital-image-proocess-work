import pytest

torch = pytest.importorskip("torch")

from dip_course.enhancement.color import hvi_to_rgb, rgb_to_hvi
from dip_course.enhancement.losses import EnhancementLoss
from dip_course.enhancement.model import AdaptiveColorLUT, RawToRGBUNet
from dip_course.enhancement.refiner import ExposureColorRefiner, linear_to_srgb


def test_model_produces_two_times_larger_rgb() -> None:
    model = RawToRGBUNet(base_channels=8, depth=3)
    inputs = torch.rand(2, 4, 32, 48)
    output = model(inputs)
    assert tuple(output.shape) == (2, 3, 64, 96)
    assert torch.all((0.0 <= output) & (output <= 1.0))


def test_model_supports_batch_normalization() -> None:
    model = RawToRGBUNet(base_channels=8, depth=3, normalization="batch")
    inputs = torch.rand(2, 4, 32, 48)
    output = model(inputs)
    assert tuple(output.shape) == (2, 3, 64, 96)


def test_color_faithful_model_produces_valid_rgb() -> None:
    model = RawToRGBUNet(
        base_channels=8,
        depth=3,
        normalization="none",
        output_space="hvi",
        illumination_conditioning=True,
    )
    output = model(torch.rand(1, 4, 32, 48))
    assert tuple(output.shape) == (1, 3, 64, 96)
    assert torch.all((0.0 <= output) & (output <= 1.0))


def test_model_accepts_full_image_scene_context_for_a_tile() -> None:
    model = RawToRGBUNet(
        base_channels=8,
        depth=3,
        normalization="none",
        output_space="hvi",
        illumination_conditioning=True,
    )
    full_image = torch.rand(1, 4, 64, 96)
    context = model.compute_scene_context(full_image)
    output = model(full_image[:, :, :32, :48], scene_context=context)
    assert tuple(context.shape) == (1, 8)
    assert tuple(output.shape) == (1, 3, 64, 96)


def test_adaptive_color_lut_starts_as_identity() -> None:
    refinement = AdaptiveColorLUT(size=9, basis_count=3)
    image = torch.rand(2, 3, 24, 32)
    context = torch.rand(2, 8)
    output = refinement(image, context)
    torch.testing.assert_close(output, image, atol=2e-6, rtol=2e-6)


def test_exposure_color_refiner_starts_as_identity() -> None:
    refiner = ExposureColorRefiner(base_channels=8, context_side=32)
    image = torch.rand(2, 3, 48, 64)
    output, parameters = refiner(image, return_parameters=True)
    torch.testing.assert_close(output, image, atol=2e-6, rtol=2e-6)
    assert tuple(parameters["local_exposure"].shape) == (2, 1, 48, 64)


def test_exposure_color_refiner_accepts_full_resolution_shape() -> None:
    refiner = ExposureColorRefiner(base_channels=8, context_side=32)
    output = refiner(torch.rand(1, 3, 65, 97))
    assert tuple(output.shape) == (1, 3, 65, 97)
    assert torch.all((0.0 <= output) & (output <= 1.0))


def test_exposure_color_refiner_trains_from_frozen_inference_output() -> None:
    refiner = ExposureColorRefiner(base_channels=8, context_side=32)
    with torch.inference_mode():
        frozen_output = torch.rand(1, 3, 32, 48)
    trainable_input = frozen_output.clone()
    refiner(trainable_input).mean().backward()
    assert any(parameter.grad is not None for parameter in refiner.parameters())


def test_linear_to_srgb_has_finite_gradient_at_black() -> None:
    image = torch.zeros(1, requires_grad=True)

    linear_to_srgb(image).sum().backward()

    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


def test_model_supports_adaptive_color_lut() -> None:
    model = RawToRGBUNet(
        base_channels=8,
        depth=3,
        color_lut_refinement=True,
        color_lut_size=9,
        color_lut_basis=3,
    )
    output = model(torch.rand(1, 4, 32, 48))
    assert tuple(output.shape) == (1, 3, 64, 96)
    assert torch.all((0.0 <= output) & (output <= 1.0))


def test_hvi_round_trip_preserves_rgb() -> None:
    rgb = torch.rand(2, 3, 16, 20)
    reconstructed = hvi_to_rgb(rgb_to_hvi(rgb))
    torch.testing.assert_close(reconstructed, rgb, atol=2e-4, rtol=2e-4)


def test_enhancement_loss_is_finite() -> None:
    criterion = EnhancementLoss(ssim_weight=0.2, chroma_weight=1.0)
    target = torch.rand(1, 3, 32, 32)
    prediction = target.clone()
    loss, components = criterion(prediction, target)
    assert torch.isfinite(loss)
    assert float(loss) < 1e-5
    assert float(components["ssim"]) > 0.999
    assert float(components["chroma"]) < 1e-7


def test_chroma_loss_penalizes_color_mismatch() -> None:
    criterion = EnhancementLoss(ssim_weight=0.0, chroma_weight=1.0)
    target = torch.zeros(1, 3, 16, 16)
    prediction = target.clone()
    prediction[:, 0] = 1.0
    loss, components = criterion(prediction, target)
    assert float(components["chroma"]) > 0.3
    assert float(loss) > float(components["l1"])


def test_color_faithful_losses_penalize_palette_shift() -> None:
    criterion = EnhancementLoss(
        ssim_weight=0.0,
        hvi_weight=0.5,
        histogram_weight=0.05,
        edge_weight=0.05,
        color_moment_weight=0.2,
    )
    target = torch.zeros(1, 3, 32, 32)
    target[:, 1] = 0.8
    prediction = torch.zeros_like(target)
    prediction[:, 0] = 0.8
    loss, components = criterion(prediction, target)
    assert torch.isfinite(loss)
    assert float(components["hvi"]) > 0.1
    assert float(components["histogram"]) >= 0.0
    assert float(components["color_moments"]) > 0.1


def test_circular_hue_loss_cannot_be_avoided_by_desaturation() -> None:
    criterion = EnhancementLoss(
        ssim_weight=0.0,
        hue_weight=1.0,
        saturation_weight=1.0,
    )
    target = torch.zeros(1, 3, 16, 16)
    target[:, 0] = 0.8
    prediction = torch.full_like(target, 0.4)
    loss, components = criterion(prediction, target)
    assert torch.isfinite(loss)
    assert float(components["hue"]) > 0.5
    assert float(components["saturation"]) > 0.5
