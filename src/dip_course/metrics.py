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
