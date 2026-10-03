from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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

        local_exposure = self.max_local_exposure * torch.tanh(
            self.local_head(features)
        )
        local_exposure = F.interpolate(
            local_exposure,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # The global term is responsible for scene exposure. Centering the local
        # map prevents it from learning a second unconstrained global gain.
        local_exposure = local_exposure - local_exposure.mean(
            dim=(2, 3), keepdim=True
        )

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


def refiner_parameter_penalty(parameters: dict[str, torch.Tensor]) -> torch.Tensor:
    """Keep corrections small unless paired supervision justifies them."""
    terms = [value.square().mean() for value in parameters.values()]
    return torch.stack(terms).mean()
