from __future__ import annotations

import math

import torch


def rgb_to_hvi(rgb: torch.Tensor, *, density: float = 0.2) -> torch.Tensor:
    """Convert NCHW sRGB to the continuous HVI representation from CVPR 2025.

    Hue is represented on a polar plane, avoiding the red wrap-around in HSV.
    The chroma radius is compressed close to black so that very dark noise has
    less influence than it would in ordinary RGB or HSV coordinates.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("HVI conversion expects NCHW RGB tensors")
    if density <= 0:
        raise ValueError("density must be positive")

    rgb = rgb.clamp(0.0, 1.0)
    red, green, blue = rgb.unbind(dim=1)
    value, maximum_index = rgb.max(dim=1)
    minimum = rgb.min(dim=1).values
    delta = value - minimum
    eps = torch.finfo(rgb.dtype).eps

    hue = torch.zeros_like(value)
    non_gray = delta > eps
    red_mask = non_gray & (maximum_index == 0)
    green_mask = non_gray & (maximum_index == 1)
    blue_mask = non_gray & (maximum_index == 2)
    hue = torch.where(red_mask, ((green - blue) / delta.clamp_min(eps)) % 6.0, hue)
    hue = torch.where(
        green_mask, (blue - red) / delta.clamp_min(eps) + 2.0, hue
    )
    hue = torch.where(blue_mask, (red - green) / delta.clamp_min(eps) + 4.0, hue)
    hue = hue / 6.0

    saturation = torch.where(
        value > eps, delta / value.clamp_min(eps), torch.zeros_like(value)
    )
    sensitivity = torch.sin(value * (0.5 * math.pi)).clamp_min(eps).pow(density)
    angle = hue * (2.0 * math.pi)
    horizontal = sensitivity * saturation * torch.cos(angle)
    vertical = sensitivity * saturation * torch.sin(angle)
    return torch.stack((horizontal, vertical, value), dim=1)


def hvi_to_rgb(hvi: torch.Tensor, *, density: float = 0.2) -> torch.Tensor:
    """Convert an NCHW HVI tensor back to display-referred sRGB."""
    if hvi.ndim != 4 or hvi.shape[1] != 3:
        raise ValueError("inverse HVI conversion expects NCHW HVI tensors")
    if density <= 0:
        raise ValueError("density must be positive")

    horizontal = hvi[:, 0].clamp(-1.0, 1.0)
    vertical = hvi[:, 1].clamp(-1.0, 1.0)
    value = hvi[:, 2].clamp(0.0, 1.0)
    eps = torch.finfo(hvi.dtype).eps
    sensitivity = torch.sin(value * (0.5 * math.pi)).clamp_min(eps).pow(density)

    hue = torch.remainder(
        torch.atan2(vertical + eps, horizontal + eps) / (2.0 * math.pi), 1.0
    )
    saturation = (
        torch.sqrt(horizontal.square() + vertical.square() + eps) / sensitivity
    ).clamp(0.0, 1.0)

    sector = torch.floor(hue * 6.0).to(torch.int64).remainder(6)
    fraction = hue * 6.0 - sector.to(hue.dtype)
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)

    candidates = torch.stack(
        (
            torch.stack((value, t, p), dim=1),
            torch.stack((q, value, p), dim=1),
            torch.stack((p, value, t), dim=1),
            torch.stack((p, q, value), dim=1),
            torch.stack((t, p, value), dim=1),
            torch.stack((value, p, q), dim=1),
        ),
        dim=1,
    )
    gather_index = sector[:, None, None].expand(-1, 1, 3, -1, -1)
    return candidates.gather(1, gather_index).squeeze(1).clamp(0.0, 1.0)
