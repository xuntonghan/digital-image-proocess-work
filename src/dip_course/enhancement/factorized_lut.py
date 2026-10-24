"""Low-rank pairwise LUT ablation inspired by SVDLUT (ICCV 2025).

Independent implementation, NOT a reproduction of SVDLUT's spatial/bilateral
architecture or CUDA kernel. Reconstruct a small RGB table once per scene so
the existing trilinear application and whole-scene context stay comparable.
"""
from __future__ import annotations

import torch
from torch import nn

from dip_course.enhancement.adaptive_lut import AdaptiveColorLUT


class PairwiseLowRankLUT(AdaptiveColorLUT):
    """Bounded RG/RB/GB residuals, each factorized before its tanh bound.

    The controller and native lookup are identical to AdaptiveColorLUT. Only
    the parameterization of its basis tables changes. This is not spatially
    aware: identical input colors still receive identical output colors.
    """

    def __init__(self, vertices: int = 17, bases: int = 3, max_delta: float = 0.12,
                 rank: int = 4):
        super().__init__(vertices=vertices, bases=bases, max_delta=max_delta)
        if not 1 <= rank <= vertices:
            raise ValueError("rank must be between 1 and vertices")
        del self.residual
        self.config["rank"] = rank
        # Pair order: RG, RB, GB. Last two dimensions are (row, column).
        self.left = nn.Parameter(torch.randn(bases, 3, 3, vertices, rank) * 0.02)
        self.right = nn.Parameter(torch.zeros(bases, 3, 3, rank, vertices))

    def residual_tables(self) -> torch.Tensor:
        pairwise = self.max_delta * (self.left @ self.right).tanh()
        rg, rb, gb = pairwise.unbind(dim=1)
        # Reconstruct RGB axes (blue, green, red), as in the existing LUT.
        return (rg.unsqueeze(2) + rb.unsqueeze(3) + gb.unsqueeze(4)) / 3

    def image_lut(self, image: torch.Tensor) -> torch.Tensor:
        from torch.nn import functional as F

        from dip_course.enhancement.adaptive_lut import resize_color_context

        context = resize_color_context(image.float())
        height, width = context.shape[-2:]
        scale = min(1.0, 128 / max(height, width))
        context = F.interpolate(context, size=(max(1, round(height * scale)),
                                               max(1, round(width * scale))), mode="area")
        weights = self.controller(context).softmax(dim=1)
        return self.identity.unsqueeze(0) + torch.einsum(
            "nb,bcxyz->ncxyz", weights, self.residual_tables())

    def regularization(self) -> torch.Tensor:
        # Same functional regularizer on reconstructed 3D tables as the control.
        residuals = self.residual_tables()
        luts = self.identity.unsqueeze(0) + residuals
        smooth, monotonic = [], []
        for axis in (-1, -2, -3):
            smooth.append(torch.diff(residuals, dim=axis).square().mean())
            monotonic.append(torch.relu(-torch.diff(luts, dim=axis)).square().mean())
        return torch.stack(smooth).mean() + 10 * torch.stack(monotonic).mean()
