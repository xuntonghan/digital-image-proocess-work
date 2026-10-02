from __future__ import annotations

import numpy as np

_INTERPOLATION_KERNEL = np.asarray(
    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float32
)


def _convolve3x3(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    padded = np.pad(image, ((1, 1), (1, 1)), mode="reflect")
    output = np.zeros_like(image, dtype=np.float32)
    for row in range(3):
        for column in range(3):
            output += (
                kernel[row, column]
                * padded[row : row + image.shape[0], column : column + image.shape[1]]
            )
    return output


def demosaic_rggb(raw: np.ndarray) -> np.ndarray:
    """Simple normalized-convolution demosaic used as a transparent baseline."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] % 2 or raw.shape[1] % 2:
        raise ValueError(f"Expected an even HxW Bayer image, got {raw.shape}")

    height, width = raw.shape
    masks = np.zeros((3, height, width), dtype=np.float32)
    masks[0, 0::2, 0::2] = 1.0
    masks[1, 0::2, 1::2] = 1.0
    masks[1, 1::2, 0::2] = 1.0
    masks[2, 1::2, 1::2] = 1.0

    channels: list[np.ndarray] = []
    for mask in masks:
        weighted = _convolve3x3(raw * mask, _INTERPOLATION_KERNEL)
        weights = _convolve3x3(mask, _INTERPOLATION_KERNEL)
        channels.append(weighted / np.maximum(weights, 1e-8))
    return np.stack(channels, axis=-1)


def classical_enhance(
    normalized_raw: np.ndarray,
    *,
    exposure: float = 8.0,
    gamma: float = 2.2,
    white_balance: tuple[float, float, float] = (2.0, 1.0, 1.6),
) -> np.ndarray:
    return tone_map_rgb(
        demosaic_rggb(normalized_raw),
        exposure=exposure,
        gamma=gamma,
        white_balance=white_balance,
    )


def tone_map_rgb(
    linear_rgb: np.ndarray,
    *,
    exposure: float = 8.0,
    gamma: float = 2.2,
    white_balance: tuple[float, float, float] = (2.0, 1.0, 1.6),
) -> np.ndarray:
    if exposure <= 0:
        raise ValueError("exposure must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if len(white_balance) != 3 or any(value <= 0 for value in white_balance):
        raise ValueError("white_balance must contain three positive gains")

    rgb = np.asarray(linear_rgb, dtype=np.float32).copy()
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 linear RGB, got {rgb.shape}")
    rgb *= np.asarray(white_balance, dtype=np.float32).reshape(1, 1, 3)
    rgb = np.clip(rgb * float(exposure), 0.0, 1.0)
    rgb = np.power(rgb, 1.0 / float(gamma)).astype(np.float32, copy=False)
    return np.clip(rgb, 0.0, 1.0)


def estimate_gray_world_white_balance(
    linear_rgb: np.ndarray,
    *,
    power: float = 1.0,
    gain_limits: tuple[float, float] = (0.25, 4.0),
) -> tuple[float, float, float]:
    rgb = np.asarray(linear_rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 linear RGB, got {rgb.shape}")
    if power <= 0:
        raise ValueError("power must be positive")
    lower, upper = gain_limits
    if not 0 < lower <= upper:
        raise ValueError("gain_limits must be positive and ordered")

    powered = np.power(np.clip(rgb, 0.0, None), float(power))
    statistics = np.power(
        np.mean(powered, axis=(0, 1), dtype=np.float64), 1.0 / float(power)
    )
    gains = statistics[1] / np.maximum(statistics, 1e-12)
    gains = np.clip(gains, lower, upper)
    gains[1] = 1.0
    return tuple(float(value) for value in gains)


def auto_classical_enhance(
    normalized_raw: np.ndarray,
    *,
    gamma: float = 2.2,
    percentile: float = 99.0,
    target: float = 0.9,
    white_balance_power: float = 1.0,
) -> tuple[np.ndarray, dict[str, object]]:
    return auto_tone_map_rgb(
        demosaic_rggb(normalized_raw),
        gamma=gamma,
        percentile=percentile,
        target=target,
        white_balance_power=white_balance_power,
    )


def auto_tone_map_rgb(
    linear_rgb: np.ndarray,
    *,
    gamma: float = 2.2,
    percentile: float = 99.0,
    target: float = 0.9,
    white_balance_power: float = 1.0,
) -> tuple[np.ndarray, dict[str, object]]:
    if not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100")
    if not 0 < target <= 1:
        raise ValueError("target must be in (0, 1]")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    white_balance = estimate_gray_world_white_balance(
        linear_rgb, power=white_balance_power
    )
    balanced = linear_rgb * np.asarray(white_balance, dtype=np.float32).reshape(1, 1, 3)
    luminance = (
        0.2126 * balanced[..., 0]
        + 0.7152 * balanced[..., 1]
        + 0.0722 * balanced[..., 2]
    )
    observed = float(np.percentile(luminance, percentile))
    exposure = float(target**gamma / max(observed, 1e-8))
    result = tone_map_rgb(
        linear_rgb,
        exposure=exposure,
        gamma=gamma,
        white_balance=white_balance,
    )
    parameters: dict[str, object] = {
        "exposure": exposure,
        "gamma": gamma,
        "white_balance": list(white_balance),
        "percentile": percentile,
        "target": target,
        "white_balance_power": white_balance_power,
    }
    return result, parameters
