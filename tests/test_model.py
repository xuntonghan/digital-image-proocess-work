import pytest

torch = pytest.importorskip("torch")

from dip_course.enhancement.losses import EnhancementLoss
from dip_course.enhancement.model import RawToRGBUNet


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
