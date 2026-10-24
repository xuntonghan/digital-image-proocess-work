"""Opt-in research utilities; the frozen inference path is never modified.

TLC concept: Chu et al., ECCV 2022, https://github.com/megvii-research/TLC.
Log-MSE PSNR objective: Chen et al., ECCV 2022, official NAFNet PSNRLoss.
The summed-area and RGB loss below are independent implementations.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from dip_course.enhancement.refiner import NAFBlock, RawGuidedNAFRefiner


def mean_image_psnr_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative arithmetic mean RGB PSNR with FP32 reductions.

    A patch surrogate, not a claim of optimizing the full-image metric exactly.
    Epsilon prevents singular loss/gradients for identical images.
    """
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("Expected equal NCHW RGB tensors")
    mse = (prediction.float() - target.float()).square().mean(dim=(1, 2, 3))
    return (10 / math.log(10)) * (mse + 1e-8).log().mean()


class LocalChannelMean(nn.Module):
    """TLC-style local average with FP32 prefix sums to avoid AMP overflow."""

    def __init__(self, kernel: int):
        super().__init__()
        if kernel <= 0:
            raise ValueError("kernel must be positive")
        self.kernel = kernel

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        height, width = features.shape[-2:]
        if self.kernel >= max(height, width):
            return F.adaptive_avg_pool2d(features, 1)
        kh, kw = min(self.kernel, height), min(self.kernel, width)
        prefix = F.pad(features.float().cumsum(-2).cumsum(-1), (1, 0, 1, 0))
        local = (prefix[..., kh:, kw:] - prefix[..., :-kh, kw:]
                 - prefix[..., kh:, :-kw] + prefix[..., :-kh, :-kw]) / (kh * kw)
        pad_h, pad_w = height - local.shape[-2], width - local.shape[-1]
        return F.pad(local, (pad_w // 2, pad_w - pad_w // 2,
                             pad_h // 2, pad_h - pad_h // 2), mode="replicate").to(features.dtype)


def enable_local_channel_means(model: RawGuidedNAFRefiner, *, rgb_window: int = 512) -> int:
    """Convert structural attention only, preserving photometry/checkpoint keys."""
    if rgb_window <= 0 or rgb_window % model.padder_size:
        raise ValueError("rgb_window must be positive and divisible by the model padder")
    converted = 0
    groups = [(group, level) for level, group in enumerate(model.encoders)]
    groups.append((model.middle, len(model.encoders)))
    groups.extend((group, level) for level, group in
                  zip(reversed(range(len(model.decoders))), model.decoders, strict=True))
    for group, level in groups:
        for block in group:
            if not isinstance(block, NAFBlock):
                raise TypeError("Unexpected structural block")
            block.channel_attention[0] = LocalChannelMean(rgb_window // (2**level))
            converted += 1
    return converted
