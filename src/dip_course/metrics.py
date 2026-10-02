from __future__ import annotations

import math

import numpy as np


def _validate_pair(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Metric inputs must have equal shapes: {prediction.shape} vs {target.shape}"
        )
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB inputs, got {prediction.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Metric inputs contain NaN or infinity")
    return prediction, target


def psnr_rgb(
    prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0
) -> float:
    prediction, target = _validate_pair(prediction, target)
    mse = float(np.mean((prediction - target) ** 2, dtype=np.float64))
    if mse == 0.0:
        return math.inf
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def ssim_rgb(
    prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0
) -> float:
    prediction, target = _validate_pair(prediction, target)
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:  # pragma: no cover - installation guidance
        raise RuntimeError(
            "SSIM requires scikit-image; install the project dependencies"
        ) from exc
    return float(
        structural_similarity(
            prediction,
            target,
            data_range=data_range,
            channel_axis=-1,
        )
    )


def _resize_for_color_metrics(
    image: np.ndarray, *, max_side: int
) -> np.ndarray:
    if max(image.shape[:2]) <= max_side:
        return image
    from skimage.transform import resize

    scale = max_side / max(image.shape[:2])
    shape = (max(1, round(image.shape[0] * scale)), max(1, round(image.shape[1] * scale)))
    return resize(
        image,
        shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def color_fidelity_metrics(
    prediction: np.ndarray, target: np.ndarray, *, max_side: int = 960
) -> dict[str, float]:
    """Measure perceptual color, saturation, hue, and lightness agreement.

    CIEDE2000 is evaluated on a bounded-resolution copy to make full 4K test
    evaluation practical. Spatial fidelity remains covered by full-resolution
    PSNR and SSIM.
    """
    prediction, target = _validate_pair(prediction, target)
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    prediction = _resize_for_color_metrics(prediction, max_side=max_side)
    target = _resize_for_color_metrics(target, max_side=max_side)

    from skimage.color import deltaE_ciede2000, rgb2hsv, rgb2lab

    prediction_lab = rgb2lab(np.clip(prediction, 0.0, 1.0))
    target_lab = rgb2lab(np.clip(target, 0.0, 1.0))
    delta_e = deltaE_ciede2000(prediction_lab, target_lab)

    prediction_hsv = rgb2hsv(np.clip(prediction, 0.0, 1.0))
    target_hsv = rgb2hsv(np.clip(target, 0.0, 1.0))
    hue_distance = np.abs(prediction_hsv[..., 0] - target_hsv[..., 0])
    hue_distance = np.minimum(hue_distance, 1.0 - hue_distance)
    chromatic_mask = (target_hsv[..., 1] >= 0.1) & (target_hsv[..., 2] >= 0.05)
    hue_error = (
        float(np.mean(hue_distance[chromatic_mask], dtype=np.float64) * 360.0)
        if np.any(chromatic_mask)
        else 0.0
    )
    return {
        "delta_e2000_mean": float(np.mean(delta_e, dtype=np.float64)),
        "delta_e2000_p95": float(np.percentile(delta_e, 95.0)),
        "lightness_mae": float(
            np.mean(np.abs(prediction_lab[..., 0] - target_lab[..., 0]))
        ),
        "saturation_mae": float(
            np.mean(np.abs(prediction_hsv[..., 1] - target_hsv[..., 1]))
        ),
        "hue_error_degrees": hue_error,
    }


class LPIPSMetric:
    """Reusable LPIPS evaluator. Importing this class does not require torch until construction."""

    def __init__(self, device: str = "cuda") -> None:
        try:
            import lpips
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "LPIPS requires the training extras: pip install '.[train]'"
            ) from exc
        self._torch = torch
        self.device = torch.device(device)
        self.model = lpips.LPIPS(net="alex").to(self.device).eval()

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        prediction, target = _validate_pair(prediction, target)
        torch = self._torch
        pred_tensor = (
            torch.from_numpy(prediction.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        )
        target_tensor = (
            torch.from_numpy(target.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        )
        with torch.inference_mode():
            value = self.model(pred_tensor * 2.0 - 1.0, target_tensor * 2.0 - 1.0)
        return float(value.item())
