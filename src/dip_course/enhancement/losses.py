from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from dip_course.enhancement.color import rgb_to_hvi


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


def edge_map(image: torch.Tensor) -> torch.Tensor:
    """Return a channel-wise Laplacian response for detail supervision."""
    if image.ndim != 4:
        raise ValueError("edge_map expects NCHW tensors")
    channels = image.shape[1]
    kernel = image.new_tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    ).reshape(1, 1, 3, 3)
    return F.conv2d(image, kernel.expand(channels, 1, 3, 3), padding=1, groups=channels)


def soft_histogram(
    image: torch.Tensor,
    *,
    bins: int = 32,
    sigma: float = 0.04,
    max_side: int = 64,
) -> torch.Tensor:
    """Differentiable per-channel histogram with bounded memory use."""
    if image.ndim != 4:
        raise ValueError("soft_histogram expects NCHW tensors")
    if bins < 2 or sigma <= 0 or max_side <= 0:
        raise ValueError("histogram parameters are invalid")
    height, width = image.shape[-2:]
    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        size = (max(1, round(height * scale)), max(1, round(width * scale)))
        image = F.adaptive_avg_pool2d(image, size)
    centers = torch.linspace(0.0, 1.0, bins, device=image.device, dtype=image.dtype)
    values = image.clamp(0.0, 1.0).flatten(2).unsqueeze(-1)
    weights = torch.exp(-0.5 * ((values - centers) / sigma).square())
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    histogram = weights.mean(dim=2)
    return histogram / histogram.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def color_histogram_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between global RGB distributions.

    SIED applies a soft-histogram KL loss in latent color space. Here it is
    evaluated on downsampled RGB so the same idea remains practical for 4K
    course images and cannot dominate local reconstruction.
    """
    prediction_histogram = soft_histogram(prediction).clamp_min(1e-8)
    target_histogram = soft_histogram(target).clamp_min(1e-8)
    midpoint = 0.5 * (prediction_histogram + target_histogram)
    prediction_kl = torch.sum(
        prediction_histogram * (prediction_histogram.log() - midpoint.log()), dim=-1
    )
    target_kl = torch.sum(
        target_histogram * (target_histogram.log() - midpoint.log()), dim=-1
    )
    return 0.5 * (prediction_kl + target_kl).mean()


def color_moment_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match global channel mean and contrast to discourage scene-wide casts."""
    dimensions = (2, 3)
    prediction_mean = prediction.mean(dim=dimensions)
    target_mean = target.mean(dim=dimensions)
    prediction_std = prediction.var(dim=dimensions, unbiased=False).add(1e-6).sqrt()
    target_std = target.var(dim=dimensions, unbiased=False).add(1e-6).sqrt()
    return F.l1_loss(prediction_mean, target_mean) + F.l1_loss(
        prediction_std, target_std
    )


def circular_hue_and_saturation_loss(
    prediction_hvi: torch.Tensor, target_hvi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervise hue direction without allowing desaturation as a shortcut."""
    prediction_chroma = prediction_hvi[:, :2]
    target_chroma = target_hvi[:, :2]
    prediction_radius = torch.linalg.vector_norm(prediction_chroma, dim=1)
    target_radius = torch.linalg.vector_norm(target_chroma, dim=1)
    epsilon = 1e-4
    prediction_direction = prediction_chroma / prediction_radius.clamp_min(
        epsilon
    ).unsqueeze(1)
    target_direction = target_chroma / target_radius.clamp_min(epsilon).unsqueeze(1)
    cosine = (prediction_direction * target_direction).sum(dim=1).clamp(-1.0, 1.0)
    color_weight = target_radius.detach()
    hue = ((1.0 - cosine) * color_weight).sum() / color_weight.sum().clamp_min(
        epsilon
    )
    saturation = F.l1_loss(prediction_radius, target_radius)
    return hue, saturation


class EnhancementLoss(nn.Module):
    def __init__(
        self,
        ssim_weight: float = 0.2,
        chroma_weight: float = 0.0,
        hvi_weight: float = 0.0,
        histogram_weight: float = 0.0,
        edge_weight: float = 0.0,
        color_moment_weight: float = 0.0,
        hue_weight: float = 0.0,
        saturation_weight: float = 0.0,
        hvi_density: float = 0.2,
    ) -> None:
        super().__init__()
        weights = (
            ssim_weight,
            chroma_weight,
            hvi_weight,
            histogram_weight,
            edge_weight,
            color_moment_weight,
            hue_weight,
            saturation_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("loss weights must be non-negative")
        if hvi_density <= 0:
            raise ValueError("hvi_density must be positive")
        self.ssim_weight = ssim_weight
        self.chroma_weight = chroma_weight
        self.hvi_weight = hvi_weight
        self.histogram_weight = histogram_weight
        self.edge_weight = edge_weight
        self.color_moment_weight = color_moment_weight
        self.hue_weight = hue_weight
        self.saturation_weight = saturation_weight
        self.hvi_density = hvi_density

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l1 = F.l1_loss(prediction, target)
        ssim = ssim_index(prediction, target)
        chroma = F.l1_loss(rgb_chroma(prediction), rgb_chroma(target))
        prediction_hvi = rgb_to_hvi(prediction, density=self.hvi_density)
        target_hvi = rgb_to_hvi(target, density=self.hvi_density)
        hvi = F.l1_loss(prediction_hvi, target_hvi)
        hue, saturation = circular_hue_and_saturation_loss(
            prediction_hvi, target_hvi
        )
        histogram = color_histogram_loss(prediction, target)
        edge = F.l1_loss(edge_map(prediction), edge_map(target))
        moments = color_moment_loss(prediction, target)
        total = (
            l1
            + self.ssim_weight * (1.0 - ssim)
            + self.chroma_weight * chroma
            + self.hvi_weight * hvi
            + self.histogram_weight * histogram
            + self.edge_weight * edge
            + self.color_moment_weight * moments
            + self.hue_weight * hue
            + self.saturation_weight * saturation
        )
        return total, {
            "l1": l1.detach(),
            "ssim": ssim.detach(),
            "chroma": chroma.detach(),
            "hvi": hvi.detach(),
            "histogram": histogram.detach(),
            "edge": edge.detach(),
            "color_moments": moments.detach(),
            "hue": hue.detach(),
            "saturation": saturation.detach(),
        }
