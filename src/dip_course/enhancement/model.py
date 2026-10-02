from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for candidate in (8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1


def _normalization_layer(channels: int, normalization: str) -> nn.Module:
    if normalization == "group":
        return nn.GroupNorm(_group_count(channels), channels)
    if normalization == "batch":
        return nn.BatchNorm2d(channels)
    if normalization == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported normalization: {normalization}")


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, normalization: str = "group"
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            _normalization_layer(out_channels, normalization),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            _normalization_layer(out_channels, normalization),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(inputs) + self.skip(inputs))


class DownBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, normalization: str = "group"
    ) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block = ResidualBlock(out_channels, out_channels, normalization)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(inputs))


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        normalization: str = "group",
    ) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.block = ResidualBlock(
            out_channels + skip_channels, out_channels, normalization
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(
            inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        inputs = self.reduce(inputs)
        return self.block(torch.cat((inputs, skip), dim=1))


class RawToRGBUNet(nn.Module):
    """U-Net that maps packed RGGB RAW to a two-times-larger sRGB image."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 32,
        depth: int = 4,
        normalization: str = "group",
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
        if in_channels <= 0 or out_channels <= 0 or base_channels <= 0:
            raise ValueError("channel counts must be positive")

        channels = [base_channels * (2**index) for index in range(depth)]
        self.input_multiple = 2**depth
        self.stem = ResidualBlock(in_channels, channels[0], normalization)
        self.down_blocks = nn.ModuleList(
            DownBlock(channels[index], channels[index + 1], normalization)
            for index in range(depth - 1)
        )
        bottleneck_channels = channels[-1] * 2
        self.bottleneck = nn.Sequential(
            DownBlock(channels[-1], bottleneck_channels, normalization),
            ResidualBlock(bottleneck_channels, bottleneck_channels, normalization),
        )

        decoder: list[UpBlock] = []
        current = bottleneck_channels
        for skip_channels in reversed(channels):
            decoder.append(
                UpBlock(current, skip_channels, skip_channels, normalization)
            )
            current = skip_channels
        self.up_blocks = nn.ModuleList(decoder)
        self.head = nn.Conv2d(channels[0], out_channels * 4, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, raw_packed: torch.Tensor) -> torch.Tensor:
        if raw_packed.ndim != 4:
            raise ValueError(
                f"Expected NCHW tensor, got shape {tuple(raw_packed.shape)}"
            )
        height, width = raw_packed.shape[-2:]
        if height % self.input_multiple or width % self.input_multiple:
            raise ValueError(
                f"Input dimensions must be divisible by {self.input_multiple}; got {height}x{width}"
            )

        features = [self.stem(raw_packed)]
        for down in self.down_blocks:
            features.append(down(features[-1]))
        output = self.bottleneck(features[-1])
        for up, skip in zip(self.up_blocks, reversed(features), strict=True):
            output = up(output, skip)
        return torch.sigmoid(self.pixel_shuffle(self.head(output)))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
