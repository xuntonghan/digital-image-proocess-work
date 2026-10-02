from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from dip_course.enhancement.color import hvi_to_rgb


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


class AdaptiveIlluminationCorrection(nn.Module):
    """Scene-adaptive feature modulation inspired by SIED's AICM."""

    def __init__(self, channels: int, context_channels: int = 8) -> None:
        super().__init__()
        hidden = max(channels // 4, 8)
        self.modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, 1),
        )
        self.context_modulator = nn.Sequential(
            nn.Linear(context_channels, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels * 2),
        )
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)
        nn.init.zeros_(self.context_modulator[-1].weight)
        nn.init.zeros_(self.context_modulator[-1].bias)

    def forward(
        self, inputs: torch.Tensor, scene_context: torch.Tensor
    ) -> torch.Tensor:
        modulation = self.modulator(inputs)
        context = self.context_modulator(scene_context).unsqueeze(-1).unsqueeze(-1)
        scale, bias = (modulation + context).chunk(2, dim=1)
        return inputs * (1.0 + 0.5 * torch.tanh(scale)) + 0.1 * torch.tanh(bias)


class AdaptiveColorLUT(nn.Module):
    """Scene-adaptive 3D LUT initialized as an exact identity transform.

    A small set of residual LUTs is fused from whole-image RAW statistics.  The
    transform is global for a scene, so tiled inference remains seam-free while
    the nonlinear lattice can correct hue and saturation errors that a 3x3
    color matrix cannot represent.
    """

    def __init__(
        self,
        *,
        size: int = 17,
        basis_count: int = 4,
        context_channels: int = 8,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if size < 2 or basis_count <= 0 or context_channels <= 0:
            raise ValueError("LUT size, basis count, and context channels are invalid")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")

        axis = torch.linspace(0.0, 1.0, size)
        red, green, blue = torch.meshgrid(axis, axis, axis, indexing="ij")
        self.register_buffer(
            "identity_lut", torch.stack((red, green, blue), dim=0).unsqueeze(0)
        )
        self.basis_residuals = nn.Parameter(
            torch.zeros(basis_count, 3, size, size, size)
        )
        hidden = max(context_channels * 2, 16)
        self.weight_predictor = nn.Sequential(
            nn.Linear(context_channels, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, basis_count),
        )
        nn.init.zeros_(self.weight_predictor[-1].weight)
        with torch.no_grad():
            self.weight_predictor[-1].bias.copy_(
                torch.linspace(1.5, -1.5, basis_count)
            )
        self.residual_scale = residual_scale

    def forward(
        self, image: torch.Tensor, scene_context: torch.Tensor
    ) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("AdaptiveColorLUT expects an NCHW RGB tensor")
        if scene_context.ndim != 2 or scene_context.shape[0] != image.shape[0]:
            raise ValueError("scene_context must have one row per image")

        weights = torch.softmax(self.weight_predictor(scene_context), dim=1)
        residuals = torch.tanh(self.basis_residuals)
        fused_residual = torch.einsum("nk,kcxyz->ncxyz", weights, residuals)
        lut = self.identity_lut + self.residual_scale * fused_residual

        # grid_sample's coordinate order is (W, H, D), corresponding here to
        # (blue, green, red) for a LUT stored as (red, green, blue).
        grid = image.permute(0, 2, 3, 1)[..., [2, 1, 0]]
        grid = grid.mul(2.0).sub(1.0).unsqueeze(1)
        refined = F.grid_sample(
            lut,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(2)
        return refined.clamp(0.0, 1.0)


class RawToRGBUNet(nn.Module):
    """U-Net that maps packed RGGB RAW to a two-times-larger sRGB image."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 32,
        depth: int = 4,
        normalization: str = "group",
        output_space: str = "rgb",
        illumination_conditioning: bool = False,
        hvi_density: float = 0.2,
        color_lut_refinement: bool = False,
        color_lut_size: int = 17,
        color_lut_basis: int = 4,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
        if in_channels <= 0 or out_channels <= 0 or base_channels <= 0:
            raise ValueError("channel counts must be positive")
        if output_space not in {"rgb", "hvi"}:
            raise ValueError("output_space must be rgb or hvi")
        if output_space == "hvi" and out_channels != 3:
            raise ValueError("HVI output requires exactly three output channels")
        if hvi_density <= 0:
            raise ValueError("hvi_density must be positive")

        channels = [base_channels * (2**index) for index in range(depth)]
        self.input_multiple = 2**depth
        self.output_space = output_space
        self.hvi_density = hvi_density
        self.illumination_conditioning = illumination_conditioning
        self.color_lut_refinement = color_lut_refinement
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
        self.illumination = (
            AdaptiveIlluminationCorrection(bottleneck_channels)
            if illumination_conditioning
            else nn.Identity()
        )

        decoder: list[UpBlock] = []
        current = bottleneck_channels
        for skip_channels in reversed(channels):
            decoder.append(
                UpBlock(current, skip_channels, skip_channels, normalization)
            )
            current = skip_channels
        self.up_blocks = nn.ModuleList(decoder)
        if output_space == "rgb":
            self.head = nn.Conv2d(channels[0], out_channels * 4, 3, padding=1)
            self.chroma_head = None
            self.intensity_head = None
        else:
            self.head = None
            self.chroma_head = nn.Conv2d(channels[0], 2 * 4, 3, padding=1)
            self.intensity_head = nn.Conv2d(channels[0], 4, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.color_refinement = (
            AdaptiveColorLUT(size=color_lut_size, basis_count=color_lut_basis)
            if color_lut_refinement
            else nn.Identity()
        )

    @staticmethod
    def compute_scene_context(raw_packed: torch.Tensor) -> torch.Tensor:
        """Return channel means/stds shared by every tile from one image."""
        means = raw_packed.mean(dim=(2, 3))
        standard_deviations = raw_packed.var(
            dim=(2, 3), unbiased=False
        ).add(1e-8).sqrt()
        return torch.cat((means, standard_deviations), dim=1)

    def forward(
        self,
        raw_packed: torch.Tensor,
        scene_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if raw_packed.ndim != 4:
            raise ValueError(
                f"Expected NCHW tensor, got shape {tuple(raw_packed.shape)}"
            )
        height, width = raw_packed.shape[-2:]
        if height % self.input_multiple or width % self.input_multiple:
            raise ValueError(
                f"Input dimensions must be divisible by {self.input_multiple}; got {height}x{width}"
            )

        if scene_context is None:
            scene_context = self.compute_scene_context(raw_packed)
        if scene_context.shape != (raw_packed.shape[0], raw_packed.shape[1] * 2):
            raise ValueError(
                "scene_context must contain one mean and standard deviation per RAW channel"
            )

        features = [self.stem(raw_packed)]
        for down in self.down_blocks:
            features.append(down(features[-1]))
        output = self.bottleneck(features[-1])
        if self.illumination_conditioning:
            assert isinstance(self.illumination, AdaptiveIlluminationCorrection)
            output = self.illumination(output, scene_context)
        for up, skip in zip(self.up_blocks, reversed(features), strict=True):
            output = up(output, skip)
        if self.output_space == "rgb":
            assert self.head is not None
            rgb = torch.sigmoid(self.pixel_shuffle(self.head(output)))
            if self.color_lut_refinement:
                assert isinstance(self.color_refinement, AdaptiveColorLUT)
                rgb = self.color_refinement(rgb, scene_context)
            return rgb

        assert self.chroma_head is not None and self.intensity_head is not None
        chroma = torch.tanh(self.pixel_shuffle(self.chroma_head(output)))
        intensity = torch.sigmoid(self.pixel_shuffle(self.intensity_head(output)))
        hvi = torch.cat((chroma, intensity), dim=1)
        rgb = hvi_to_rgb(hvi, density=self.hvi_density)
        if self.color_lut_refinement:
            assert isinstance(self.color_refinement, AdaptiveColorLUT)
            rgb = self.color_refinement(rgb, scene_context)
        return rgb


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
