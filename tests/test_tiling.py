import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

from dip_course.enhancement.tiling import predict_tiled, tile_starts


class ConstantUpsampler(nn.Module):
    input_multiple = 4

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.interpolate(inputs[:, :3], scale_factor=2, mode="nearest")


def test_tile_starts_cover_final_edge() -> None:
    assert tile_starts(20, 8, 2) == [0, 6, 12]
    assert tile_starts(21, 8, 2) == [0, 6, 12, 13]


def test_tiled_prediction_covers_non_multiple_dimensions() -> None:
    inputs = torch.ones(1, 4, 10, 14)
    output = predict_tiled(ConstantUpsampler(), inputs, tile_size=8, overlap=2)
    assert tuple(output.shape) == (1, 3, 20, 28)
    torch.testing.assert_close(output, torch.ones_like(output))
