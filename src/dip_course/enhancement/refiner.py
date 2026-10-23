from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """Per-pixel channel normalization used by NAFNet restoration blocks."""

    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.epsilon = epsilon

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mean = image.mean(dim=1, keepdim=True)
        variance = (image - mean).square().mean(dim=1, keepdim=True)
        normalized = (image - mean) * torch.rsqrt(variance + self.epsilon)
        return normalized * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        first, second = features.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    """Activation-free restoration block from the ECCV 2022 NAFNet design."""

    def __init__(
        self,
        channels: int,
        *,
        depthwise_expansion: int = 2,
        ffn_expansion: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        depthwise_channels = channels * depthwise_expansion
        ffn_channels = channels * ffn_expansion
        if depthwise_channels % 2 or ffn_channels % 2:
            raise ValueError("NAFBlock expanded channel counts must be even")

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, depthwise_channels, 1)
        self.depthwise = nn.Conv2d(
            depthwise_channels,
            depthwise_channels,
            3,
            padding=1,
            groups=depthwise_channels,
        )
        self.gate1 = SimpleGate()
        gated_channels = depthwise_channels // 2
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gated_channels, gated_channels, 1),
        )
        self.conv2 = nn.Conv2d(gated_channels, channels, 1)
        self.dropout1 = nn.Dropout2d(dropout) if dropout else nn.Identity()

        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_channels, 1)
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.dropout2 = nn.Dropout2d(dropout) if dropout else nn.Identity()

        # Zero residual scales make every block an exact identity at startup.
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(self.norm1(features))
        residual = self.depthwise(residual)
        residual = self.gate1(residual)
        residual = residual * self.channel_attention(residual)
        residual = self.dropout1(self.conv2(residual))
        features = features + residual * self.beta

        residual = self.conv3(self.norm2(features))
        residual = self.gate2(residual)
        residual = self.dropout2(self.conv4(residual))
        return features + residual * self.gamma


def srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp(0.0, 1.0)
    return torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055).pow(2.4),
    )


def linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp(0.0, 1.0)
    # torch.where evaluates both branches.  Keeping the fractional-power branch
    # away from exactly zero prevents an infinite derivative (and eventual NaN
    # optimizer state) even for pixels that select the linear branch.
    nonlinear = 1.055 * image.clamp_min(1e-6).pow(1.0 / 2.4) - 0.055
    return torch.where(
        image <= 0.0031308,
        image * 12.92,
        nonlinear,
    )


class ExposureColorRefiner(nn.Module):
    """Identity-initialized exposure and color correction for SIED output.

    The refiner deliberately predicts only a smooth illumination field and
    global photometric parameters. It cannot hallucinate high-frequency detail,
    so the denoising and reconstruction produced by the frozen SIED model stay
    intact while brightness, white balance, and palette can be corrected.
    """

    parameter_count = 14

    def __init__(
        self,
        *,
        base_channels: int = 24,
        context_side: int = 256,
        max_global_exposure: float = 1.0,
        max_local_exposure: float = 0.5,
        max_white_balance: float = 0.25,
        max_color_matrix: float = 0.12,
        max_gamma_log: float = 0.25,
    ) -> None:
        super().__init__()
        if base_channels <= 0 or context_side <= 0:
            raise ValueError("base_channels and context_side must be positive")
        limits = (
            max_global_exposure,
            max_local_exposure,
            max_white_balance,
            max_color_matrix,
            max_gamma_log,
        )
        if any(limit <= 0 for limit in limits):
            raise ValueError("refinement limits must be positive")

        self.context_side = context_side
        self.max_global_exposure = max_global_exposure
        self.max_local_exposure = max_local_exposure
        self.max_white_balance = max_white_balance
        self.max_color_matrix = max_color_matrix
        self.max_gamma_log = max_gamma_log

        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 5, stride=2, padding=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        hidden_channels = base_channels * 2
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, self.parameter_count),
        )
        self.local_head = nn.Sequential(
            nn.Conv2d(hidden_channels, base_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1),
        )
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)
        nn.init.zeros_(self.local_head[-1].weight)
        nn.init.zeros_(self.local_head[-1].bias)

    def _context_image(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        scale = min(1.0, self.context_side / max(height, width))
        if scale == 1.0:
            return image
        size = (max(1, round(height * scale)), max(1, round(width * scale)))
        return F.interpolate(image, size=size, mode="area")

    def forward(
        self, image: torch.Tensor, *, return_parameters: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("ExposureColorRefiner expects NCHW RGB tensors")
        context = self._context_image(image.clamp(0.0, 1.0))
        features = self.encoder(context)
        parameters = self.global_head(features)

        global_exposure = self.max_global_exposure * torch.tanh(parameters[:, 0:1])
        gamma_log = self.max_gamma_log * torch.tanh(parameters[:, 1:2])
        white_balance = self.max_white_balance * torch.tanh(parameters[:, 2:5])
        # Remove the common component so exposure, not white balance, controls
        # overall luminance.
        white_balance = white_balance - white_balance.mean(dim=1, keepdim=True)
        matrix_delta = self.max_color_matrix * torch.tanh(parameters[:, 5:14])
        matrix_delta = matrix_delta.reshape(-1, 3, 3)

        local_exposure = self.max_local_exposure * torch.tanh(self.local_head(features))
        local_exposure = F.interpolate(
            local_exposure,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # The global term is responsible for scene exposure. Centering the local
        # map prevents it from learning a second unconstrained global gain.
        local_exposure = local_exposure - local_exposure.mean(dim=(2, 3), keepdim=True)

        linear = srgb_to_linear(image)
        log_gain = global_exposure[:, :, None, None] + local_exposure
        linear = linear * torch.exp(log_gain)
        linear = linear * torch.exp(white_balance[:, :, None, None])
        identity = torch.eye(3, device=image.device, dtype=image.dtype).unsqueeze(0)
        color_matrix = identity + matrix_delta
        linear = torch.einsum("nij,njhw->nihw", color_matrix, linear)
        refined = linear_to_srgb(linear)
        gamma = torch.exp(gamma_log)[:, :, None, None]
        refined = refined.clamp_min(1e-6).pow(gamma).clamp(0.0, 1.0)

        if not return_parameters:
            return refined
        return refined, {
            "global_exposure": global_exposure,
            "gamma_log": gamma_log,
            "white_balance": white_balance,
            "matrix_delta": matrix_delta,
            "local_exposure": local_exposure,
        }


class RawGuidedNAFRefiner(nn.Module):
    """Restore SIED RGB predictions with Bayer-domain detail guidance.

    The official diffusion output supplies the scene-level reconstruction and
    color prior. A small NAFNet encoder-decoder then predicts a bounded RGB
    residual from that output plus the four packed RGGB planes. The global
    photometric branch is evaluated once per image, while the restoration
    branch can be tiled at inference time without changing exposure or color
    between tiles.
    """

    model_type = "raw_guided_naf"

    def __init__(
        self,
        *,
        width: int = 32,
        encoder_blocks: tuple[int, ...] | list[int] = (2, 2, 4),
        middle_blocks: int = 6,
        decoder_blocks: tuple[int, ...] | list[int] = (2, 2, 2),
        max_residual: float = 0.35,
        photometric_channels: int = 24,
        context_side: int = 256,
        dropout: float = 0.0,
        raw_target_mean: float = 0.1,
        raw_max_gain: float = 4096.0,
    ) -> None:
        super().__init__()
        encoder_blocks = tuple(int(value) for value in encoder_blocks)
        decoder_blocks = tuple(int(value) for value in decoder_blocks)
        if width <= 0 or middle_blocks <= 0:
            raise ValueError("width and middle_blocks must be positive")
        if not encoder_blocks or len(encoder_blocks) != len(decoder_blocks):
            raise ValueError("encoder_blocks and decoder_blocks must have equal depth")
        if any(value <= 0 for value in (*encoder_blocks, *decoder_blocks)):
            raise ValueError("NAFNet block counts must be positive")
        if max_residual <= 0:
            raise ValueError("max_residual must be positive")
        if raw_target_mean <= 0 or raw_max_gain < 1:
            raise ValueError("RAW guidance normalization settings are invalid")

        self.width = width
        self.encoder_block_counts = encoder_blocks
        self.middle_block_count = middle_blocks
        self.decoder_block_counts = decoder_blocks
        self.max_residual = max_residual
        self.raw_target_mean = raw_target_mean
        self.raw_max_gain = raw_max_gain
        self.photometric = ExposureColorRefiner(
            base_channels=photometric_channels,
            context_side=context_side,
        )
        # RGB plus four normalized RGGB planes resized to RGB resolution.
        self.intro = nn.Conv2d(7, width, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = width
        for block_count in encoder_blocks:
            self.encoders.append(
                nn.Sequential(
                    *(NAFBlock(channels, dropout=dropout) for _ in range(block_count))
                )
            )
            self.downsamples.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2

        self.middle = nn.Sequential(
            *(NAFBlock(channels, dropout=dropout) for _ in range(middle_blocks))
        )
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for block_count in decoder_blocks:
            self.upsamples.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(
                nn.Sequential(
                    *(NAFBlock(channels, dropout=dropout) for _ in range(block_count))
                )
            )
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)
        self.padder_size = 2 ** len(encoder_blocks)

    def compute_raw_gain(self, packed_raw: torch.Tensor) -> torch.Tensor:
        if packed_raw.ndim != 4 or packed_raw.shape[1] != 4:
            raise ValueError("RawGuidedNAFRefiner expects packed NCHW RGGB tensors")
        raw_mean = packed_raw.clamp(0.0, 1.0).mean(dim=(1, 2, 3), keepdim=True)
        return (self.raw_target_mean / raw_mean.clamp_min(1e-8)).clamp(
            1.0, self.raw_max_gain
        )

    def raw_guidance(
        self,
        packed_raw: torch.Tensor,
        *,
        size: tuple[int, int],
        raw_gain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if packed_raw.ndim != 4 or packed_raw.shape[1] != 4:
            raise ValueError("RawGuidedNAFRefiner expects packed NCHW RGGB tensors")
        if raw_gain is None:
            raw_gain = self.compute_raw_gain(packed_raw)
        if raw_gain.ndim != 4 or raw_gain.shape != (packed_raw.shape[0], 1, 1, 1):
            raise ValueError("raw_gain must have shape (batch, 1, 1, 1)")
        # SIED values at 0.0001 lux average only about 3.7e-4 after black-level
        # subtraction. Adaptive amplification is therefore essential: without
        # it, RGB features dominate the first convolution and RAW guidance is
        # numerically indistinguishable from zero. This mirrors SIED's AICM,
        # while keeping the transform deterministic and checkpointable.
        normalized_raw = (packed_raw.clamp(0.0, 1.0) * raw_gain).clamp(0.0, 1.0)
        return F.interpolate(
            normalized_raw,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def restoration_residual(
        self, corrected: torch.Tensor, raw_guidance: torch.Tensor
    ) -> torch.Tensor:
        if corrected.ndim != 4 or corrected.shape[1] != 3:
            raise ValueError("corrected image must be an NCHW RGB tensor")
        if raw_guidance.ndim != 4 or raw_guidance.shape[1] != 4:
            raise ValueError("raw_guidance must contain four packed RGGB planes")
        if corrected.shape[0] != raw_guidance.shape[0]:
            raise ValueError("RGB and RAW batch sizes differ")
        if corrected.shape[-2:] != raw_guidance.shape[-2:]:
            raw_guidance = F.interpolate(
                raw_guidance,
                size=corrected.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        original_height, original_width = corrected.shape[-2:]
        pad_height = (-original_height) % self.padder_size
        pad_width = (-original_width) % self.padder_size
        combined = torch.cat((corrected, raw_guidance), dim=1)
        if pad_height or pad_width:
            combined = F.pad(combined, (0, pad_width, 0, pad_height), mode="reflect")
        features = self.intro(combined)
        skips: list[torch.Tensor] = []
        for encoder, downsample in zip(self.encoders, self.downsamples, strict=True):
            features = encoder(features)
            skips.append(features)
            features = downsample(features)
        features = self.middle(features)
        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips), strict=True
        ):
            features = upsample(features) + skip
            features = decoder(features)
        residual = self.max_residual * torch.tanh(self.ending(features))
        return residual[:, :, :original_height, :original_width]

    def forward(
        self,
        image: torch.Tensor,
        packed_raw: torch.Tensor,
        *,
        raw_gain: torch.Tensor | None = None,
        return_parameters: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        corrected, parameters = self.photometric(image, return_parameters=True)
        guidance = self.raw_guidance(
            packed_raw,
            size=corrected.shape[-2:],
            raw_gain=raw_gain,
        )
        residual = self.restoration_residual(corrected, guidance)
        refined = (corrected + residual).clamp(0.0, 1.0)
        if not return_parameters:
            return refined
        parameters = dict(parameters)
        parameters["restoration_residual"] = residual
        return refined, parameters


def build_refiner(
    model_type: str,
    model_config: dict[str, object] | None = None,
) -> nn.Module:
    """Construct a checkpointed refiner while keeping legacy weights loadable."""
    config = dict(model_config or {})
    if model_type in {"exposure_color", "ExposureColorRefiner"}:
        return ExposureColorRefiner(**config)
    if model_type in {"raw_guided_naf", "RawGuidedNAFRefiner"}:
        return RawGuidedNAFRefiner(**config)
    raise ValueError(f"Unknown SIED refiner model_type: {model_type}")


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _blend_window(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # A non-zero floor avoids uncovered edge pixels when a tile touches the
    # image boundary. Overlapping interior regions retain smooth cosine blends.
    vertical = torch.hann_window(
        height, periodic=False, device=device, dtype=dtype
    ).clamp_min(1e-3)
    horizontal = torch.hann_window(
        width, periodic=False, device=device, dtype=dtype
    ).clamp_min(1e-3)
    return (vertical[:, None] * horizontal[None, :])[None, None]


@torch.inference_mode()
def apply_raw_guided_refiner_tiled(
    model: RawGuidedNAFRefiner,
    image: torch.Tensor,
    packed_raw: torch.Tensor,
    *,
    tile_size: int = 768,
    overlap: int = 96,
) -> torch.Tensor:
    """Apply the structural branch in overlapping tiles at native resolution."""
    if image.shape[0] != 1 or packed_raw.shape[0] != 1:
        raise ValueError("Tiled refiner inference currently expects batch size one")
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_size must be positive and 0 <= overlap < tile_size")
    corrected = model.photometric(image)
    guidance = model.raw_guidance(packed_raw, size=corrected.shape[-2:])
    height, width = corrected.shape[-2:]
    if max(height, width) <= tile_size:
        residual = model.restoration_residual(corrected, guidance)
        return (corrected + residual).clamp(0.0, 1.0)

    residual_sum = torch.zeros_like(corrected)
    weight_sum = torch.zeros(
        (1, 1, height, width),
        device=corrected.device,
        dtype=corrected.dtype,
    )
    for top in _tile_starts(height, tile_size, overlap):
        bottom = min(top + tile_size, height)
        for left in _tile_starts(width, tile_size, overlap):
            right = min(left + tile_size, width)
            corrected_tile = corrected[:, :, top:bottom, left:right]
            guidance_tile = guidance[:, :, top:bottom, left:right]
            residual = model.restoration_residual(corrected_tile, guidance_tile)
            weight = _blend_window(
                bottom - top,
                right - left,
                device=corrected.device,
                dtype=corrected.dtype,
            )
            residual_sum[:, :, top:bottom, left:right] += residual * weight
            weight_sum[:, :, top:bottom, left:right] += weight
    residual = residual_sum / weight_sum.clamp_min(1e-6)
    return (corrected + residual).clamp(0.0, 1.0)


def refiner_parameter_penalty(parameters: dict[str, torch.Tensor]) -> torch.Tensor:
    """Keep corrections small unless paired supervision justifies them."""
    terms = [
        value.square().mean()
        for name, value in parameters.items()
        if name != "restoration_residual"
    ]
    return torch.stack(terms).mean()
