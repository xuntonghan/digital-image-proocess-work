"""Bounded image-adaptive RGB LUT, inspired by Zeng et al., TPAMI 2022.

Independent torch-only implementation; no old C/CUDA extensions or FiveK weights.
https://github.com/HuiZeng/Image-Adaptive-3DLUT
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def resize_color_context(image: torch.Tensor) -> torch.Tensor:
    """Use the same area-512 context in thumbnail training and native inference."""
    height, width = image.shape[-2:]
    if max(height, width) <= 512:
        return image
    scale = 512 / max(height, width)
    return F.interpolate(image, size=(round(height * scale), round(width * scale)), mode="area")


def apply_rgb_lut(image: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3 or lut.ndim != 5 or lut.shape[:2] != image.shape[:2]:
        raise ValueError("Expected N3HW image and N3DDD LUT")
    # grid_sample addresses W,H,D with x,y,z: these axes encode red,green,blue.
    grid = image.float().clamp(0, 1).permute(0, 2, 3, 1).unsqueeze(1) * 2 - 1
    return F.grid_sample(lut.float(), grid, mode="bilinear", padding_mode="border",
                         align_corners=True).squeeze(2)


class AdaptiveColorLUT(nn.Module):
    def __init__(self, vertices: int = 17, bases: int = 3, max_delta: float = 0.12):
        super().__init__()
        if vertices < 2 or bases < 1 or not 0 < max_delta <= 1:
            raise ValueError("Invalid LUT dimensions/bounds")
        self.config = {"vertices": vertices, "bases": bases, "max_delta": max_delta}
        self.max_delta = max_delta
        axis = torch.linspace(0, 1, vertices)
        blue, green, red = torch.meshgrid(axis, axis, axis, indexing="ij")
        self.register_buffer("identity", torch.stack((red, green, blue)))
        self.residual = nn.Parameter(torch.zeros(bases, 3, vertices, vertices, vertices))
        self.controller = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 48, 3, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, bases))

    def image_lut(self, image: torch.Tensor) -> torch.Tensor:
        context = resize_color_context(image.float())
        height, width = context.shape[-2:]
        scale = min(1.0, 128 / max(height, width))
        context = F.interpolate(context, size=(max(1, round(height * scale)),
                                               max(1, round(width * scale))), mode="area")
        weights = self.controller(context).softmax(dim=1)
        residuals = self.max_delta * self.residual.tanh()
        return self.identity.unsqueeze(0) + torch.einsum("nb,bcxyz->ncxyz", weights, residuals)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return apply_rgb_lut(image, self.image_lut(image)).clamp(0, 1)

    def regularization(self) -> torch.Tensor:
        residuals = self.max_delta * self.residual.tanh()
        luts = self.identity.unsqueeze(0) + residuals
        smooth, monotonic = [], []
        for axis in (-1, -2, -3):
            smooth.append(torch.diff(residuals, dim=axis).square().mean())
            monotonic.append(F.relu(-torch.diff(luts, dim=axis)).square().mean())
        return torch.stack(smooth).mean() + 10 * torch.stack(monotonic).mean()


@torch.inference_mode()
def apply_color_lut_chunked(model: AdaptiveColorLUT, image: torch.Tensor,
                            rows: int = 256) -> torch.Tensor:
    if rows <= 0:
        raise ValueError("rows must be positive")
    lut = model.image_lut(image)
    # Coefficients are computed once from the whole scene, not each chunk.
    return torch.cat([apply_rgb_lut(image[:, :, start:start + rows], lut)
                      for start in range(0, image.shape[-2], rows)], dim=-2).clamp(0, 1)
