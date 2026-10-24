import pytest

torch = pytest.importorskip("torch")

from dip_course.enhancement.adaptive_lut import (
    AdaptiveColorLUT,
    apply_color_lut_chunked,
)
from dip_course.enhancement.factorized_lut import PairwiseLowRankLUT


def test_factorized_lut_starts_at_identity_with_live_factor_gradient():
    model = PairwiseLowRankLUT(vertices=5, rank=2)
    image = torch.rand(2, 3, 12, 18)
    torch.testing.assert_close(model(image), image, atol=1e-6, rtol=1e-6)
    model(image).mean().backward()
    assert model.right.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.regularization().item() == 0


def test_factorized_lut_pairwise_axes_are_rg_rb_gb():
    model = PairwiseLowRankLUT(vertices=3, bases=1, rank=1)
    with torch.no_grad():
        model.left.zero_()
        model.right.zero_()
        # Only the RG table's red output varies along its red (column) axis.
        model.left[0, 0, 0].fill_(1)
        model.right[0, 0, 0, 0] = torch.tensor([-1.0, 0.0, 1.0])
    table = model.residual_tables()
    torch.testing.assert_close(table[0, 0, 0], table[0, 0, 2])
    assert table[0, 0, 0, 0, 0] < 0 < table[0, 0, 0, 0, 2]
    assert table[0, 1:].abs().sum() == 0


def test_factorized_lut_native_chunking_and_bound():
    model = PairwiseLowRankLUT(vertices=5, rank=2).eval()
    with torch.no_grad():
        model.right.normal_(std=20)
    image = torch.rand(1, 3, 31, 43)
    prediction = model(image)
    assert (prediction - image).abs().max() <= model.max_delta + 1e-6
    torch.testing.assert_close(apply_color_lut_chunked(model, image, rows=7), prediction)


def test_factorized_lut_has_fewer_parameters_and_round_trips():
    model = PairwiseLowRankLUT()
    assert sum(p.numel() for p in model.parameters()) < sum(
        p.numel() for p in AdaptiveColorLUT().parameters()) * 0.5
    restored = PairwiseLowRankLUT(**model.config)
    restored.load_state_dict(model.state_dict())
    assert set(restored.state_dict()) == set(model.state_dict())
    with pytest.raises(ValueError):
        PairwiseLowRankLUT(rank=18)
