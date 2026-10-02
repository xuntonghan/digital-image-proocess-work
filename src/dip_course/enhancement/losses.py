from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def ssim_index(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"SSIM tensors must have equal shapes: {prediction.shape} vs {target.shape}"
        )
    if prediction.ndim != 4:
        raise ValueError("SSIM expects NCHW tensors")
    if window_size % 2 == 0 or window_size < 3:
        raise ValueError("window_size must be an odd integer >= 3")

    padding = window_size // 2
    mu_x = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(prediction * prediction, window_size, 1, padding) - mu_x**2
    sigma_y = F.avg_pool2d(target * target, window_size, 1, padding) - mu_y**2
    sigma_xy = F.avg_pool2d(prediction * target, window_size, 1, padding) - mu_x * mu_y

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-12)).mean()


def rgb_chroma(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Chroma conversion expects NCHW RGB tensors")
    red, green, blue = image.unbind(dim=1)
    cb = -0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 0.5 * red - 0.418688 * green - 0.081312 * blue
    return torch.stack((cb, cr), dim=1)


class EnhancementLoss(nn.Module):
    def __init__(self, ssim_weight: float = 0.2, chroma_weight: float = 0.0) -> None:
        super().__init__()
        if ssim_weight < 0 or chroma_weight < 0:
            raise ValueError("loss weights must be non-negative")
        self.ssim_weight = ssim_weight
        self.chroma_weight = chroma_weight

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l1 = F.l1_loss(prediction, target)
        ssim = ssim_index(prediction, target)
        chroma = F.l1_loss(rgb_chroma(prediction), rgb_chroma(target))
        total = l1 + self.ssim_weight * (1.0 - ssim) + self.chroma_weight * chroma
        return total, {
            "l1": l1.detach(),
            "ssim": ssim.detach(),
            "chroma": chroma.detach(),
        }
