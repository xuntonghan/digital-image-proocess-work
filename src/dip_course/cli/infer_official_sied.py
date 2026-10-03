from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import statistics
import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from dip_course.cli._common import choose_torch_device, write_json
from dip_course.enhancement.refiner import ExposureColorRefiner
from dip_course.io import (
    RawSpec,
    load_packed_raw,
    load_rgb,
    read_pair_manifest,
    save_rgb,
)
from dip_course.metrics import LPIPSMetric, color_fidelity_metrics, psnr_rgb, ssim_rgb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the official ICCV 2025 SIED model without GT-conditioned inference"
    )
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--black-level", type=int, default=512)
    parser.add_argument("--white-level", type=int, default=16383)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="Average independent DDIM samples before evaluation",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Cache blind predictions without loading GT or computing metrics",
    )
    parser.add_argument(
        "--save-max-side",
        type=int,
        default=0,
        help="Downsample cached output so its longest side is at most this value",
    )
    parser.add_argument(
        "--refiner-checkpoint",
        help="Optional exposure/color refiner trained on frozen SIED predictions",
    )
    return parser


def finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else None


def _strip_prefix(state: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }


def select_official_state_dict(
    checkpoint: Mapping[str, Any], target_keys: set[str]
) -> tuple[dict[str, Any], str]:
    """Select the EMA model weights from an ema-pytorch checkpoint.

    The official release stores both the online network and its EMA copy.  EMA
    key prefixes changed between ema-pytorch versions, so selection is based on
    exact agreement with the instantiated official model rather than a version
    assumption.
    """
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    ema_state = checkpoint.get("ema")
    if isinstance(ema_state, Mapping):
        for prefix in (
            "ema_model.",
            "module.ema_model.",
            "online_model.",
            "module.online_model.",
        ):
            stripped = _strip_prefix(ema_state, prefix)
            if stripped:
                candidates.append((f"ema:{prefix}", stripped))
        candidates.append(("ema:direct", ema_state))
    model_state = checkpoint.get("model")
    if isinstance(model_state, Mapping):
        candidates.append(("model", model_state))

    diagnostics: list[str] = []
    for name, candidate in candidates:
        candidate_keys = set(candidate)
        if candidate_keys == target_keys:
            return dict(candidate), name
        diagnostics.append(
            f"{name} has {len(candidate_keys)} keys "
            f"({len(target_keys - candidate_keys)} missing, "
            f"{len(candidate_keys - target_keys)} unexpected)"
        )
    detail = "; ".join(diagnostics) or "no model or EMA mapping was found"
    raise RuntimeError(f"Could not match official checkpoint weights: {detail}")


def import_official_model(official_repo: str | Path) -> tuple[type[Any], type[Any]]:
    repo = Path(official_repo).resolve()
    if not (repo / "models" / "__init__.py").is_file():
        raise FileNotFoundError(f"Not an official SIED checkout: {repo}")
    sys.path.insert(0, str(repo))
    try:
        models = importlib.import_module("models")
    except ImportError as exc:
        raise RuntimeError(
            "Official SIED dependencies are missing; install the project 'sied' extra"
        ) from exc
    return models.Unet, models.GaussianDiffusion


def load_official_model(
    official_repo: str | Path,
    checkpoint_path: str | Path,
    *,
    sampling_steps: int,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    if sampling_steps <= 0 or sampling_steps > 1000:
        raise ValueError("sampling_steps must be in [1, 1000]")
    unet_type, diffusion_type = import_official_model(official_repo)
    unet = unet_type(
        channels=3,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False,
    )
    diffusion = diffusion_type(
        unet,
        timesteps=1000,
        sampling_timesteps=sampling_steps,
        training_stage="stage2",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Official checkpoint root must be a mapping")
    state, source = select_official_state_dict(checkpoint, set(diffusion.state_dict()))
    diffusion.load_state_dict(state, strict=True)
    del state, checkpoint
    diffusion.to(device).eval()
    return diffusion, source


def load_refiner(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[ExposureColorRefiner, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Refiner checkpoint root must be a mapping")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("Refiner checkpoint does not contain model weights")
    config = checkpoint.get("model_config", {})
    if not isinstance(config, Mapping):
        raise TypeError("Refiner model_config must be a mapping")
    model = ExposureColorRefiner(**dict(config)).to(device)
    model.load_state_dict(dict(state), strict=True)
    model.eval()
    metadata = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "global_step": checkpoint.get("global_step"),
        "metrics": checkpoint.get("metrics"),
        "model_config": dict(config),
    }
    return model, metadata


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def average_prediction_samples(samples: list[torch.Tensor]) -> torch.Tensor:
    if not samples:
        raise ValueError("At least one prediction sample is required")
    total = samples[0]
    for sample in samples[1:]:
        total = total + sample
    return total / len(samples)


def diffusion_seed(base_seed: int, image_index: int, sample_index: int) -> int:
    """Keep the first ensemble member identical to single-sample evaluation."""
    return base_seed + image_index + sample_index * 1_000_003


@torch.inference_mode()
def predict_official_sied(
    diffusion: torch.nn.Module,
    packed_raw: torch.Tensor,
    *,
    precision: str,
) -> torch.Tensor:
    """Predict RGB from low-light RAW only, avoiding the official GT placeholder.

    This is equivalent to the low-RAW branch of the official stage-2 sample
    method.  The released method also encodes the target RGB even though that
    feature is discarded; omitting it makes the evaluation unambiguously blind.
    """
    if packed_raw.ndim != 4 or packed_raw.shape[1] != 4:
        raise ValueError(f"Expected Bx4xHxW packed RAW, got {tuple(packed_raw.shape)}")
    height, width = packed_raw.shape[-2:]
    padded_height = 256 * math.ceil(height / 256)
    padded_width = 256 * math.ceil(width / 256)
    packed_raw = F.pad(
        packed_raw,
        (0, padded_width - width, 0, padded_height - height),
        mode="reflect",
    )

    recon = diffusion.stage1_Net
    with _autocast_context(packed_raw.device, precision):
        raw_pyramid = recon.RAW_encoder(packed_raw)
        raw_feature_16 = raw_pyramid["down16"]
        raw_condition = recon.RAW_channel_down(raw_feature_16)
        raw_condition = torch.sigmoid(recon.RAW_illu(raw_condition))
        predicted_feature = diffusion.ddim_sample(raw_condition * 2.0 - 1.0)
        predicted_feature = recon.RGB_channel_up(predicted_feature)
        packed_rgb = recon.RGB_decoder(predicted_feature + raw_feature_16, raw_pyramid)
        prediction = F.pixel_shuffle(packed_rgb, 2)
    return prediction[:, :, : height * 2, : width * 2].float().clamp_(0.0, 1.0)


def main() -> None:
    args = build_parser().parse_args()
    device = choose_torch_device(args.device)
    if args.precision != "fp32" and device.type != "cuda":
        raise ValueError("fp16/bf16 inference requires a CUDA device")
    if args.ensemble_size <= 0:
        raise ValueError("ensemble-size must be positive")
    if args.save_max_side < 0:
        raise ValueError("save-max-side must be non-negative")
    if args.lpips and args.skip_metrics:
        raise ValueError("--lpips cannot be combined with --skip-metrics")
    if args.save_max_side and not args.skip_metrics:
        raise ValueError("--save-max-side is only valid with --skip-metrics")

    model, weight_source = load_official_model(
        args.official_repo,
        args.checkpoint,
        sampling_steps=args.sampling_steps,
        device=device,
    )
    refiner: ExposureColorRefiner | None = None
    refiner_metadata: dict[str, Any] | None = None
    if args.refiner_checkpoint:
        refiner, refiner_metadata = load_refiner(args.refiner_checkpoint, device)
    spec = RawSpec(
        width=args.width,
        height=args.height,
        black_level=args.black_level,
        white_level=args.white_level,
    )
    records = read_pair_manifest(args.data_root, args.manifest)
    if args.limit > 0:
        records = records[: args.limit]
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    lpips_metric = LPIPSMetric(str(device)) if args.lpips else None
    rows: list[dict[str, object]] = []

    for index, record in enumerate(records):
        packed = load_packed_raw(record.low_raw, spec)
        tensor = torch.from_numpy(packed).unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        prediction_samples: list[torch.Tensor] = []
        seeds: list[int] = []
        for sample_index in range(args.ensemble_size):
            # Keep sample zero identical to single-sample evaluation.  A large
            # fixed stride makes additional samples independent of manifest size.
            seed = diffusion_seed(args.seed, index, sample_index)
            seeds.append(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            sample = predict_official_sied(model, tensor, precision=args.precision)
            prediction_samples.append(sample)
        prediction_tensor = average_prediction_samples(prediction_samples)
        if refiner is not None:
            with torch.inference_mode():
                prediction_tensor = refiner(prediction_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        saved_tensor = prediction_tensor
        if args.save_max_side:
            predicted_height, predicted_width = prediction_tensor.shape[-2:]
            scale = min(1.0, args.save_max_side / max(predicted_height, predicted_width))
            if scale < 1.0:
                saved_tensor = F.interpolate(
                    prediction_tensor,
                    size=(
                        max(1, round(predicted_height * scale)),
                        max(1, round(predicted_width * scale)),
                    ),
                    mode="area",
                )
        saved_prediction = saved_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output_name = f"{index:04d}_{record.key}.png"
        save_rgb(image_dir / output_name, saved_prediction)
        row: dict[str, object] = {
            "index": index,
            "key": record.key,
            "output": str(Path("images") / output_name),
            "seeds": ";".join(str(seed) for seed in seeds),
            "inference_seconds": elapsed,
        }
        if not args.skip_metrics:
            prediction = prediction_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            # The reference is intentionally loaded only after blind prediction.
            target = load_rgb(record.gt_rgb)
            row.update(
                {
                    "psnr": psnr_rgb(prediction, target),
                    "ssim": ssim_rgb(prediction, target),
                }
            )
            row.update(color_fidelity_metrics(prediction, target))
            if lpips_metric is not None:
                row["lpips"] = lpips_metric(prediction, target)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if not rows:
        raise RuntimeError("No records were evaluated")
    fieldnames = list(rows[0])
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": "official_sied_iccv2025_stage2",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "weight_source": weight_source,
        "blind_inference": True,
        "count": len(rows),
        "device": str(device),
        "precision": args.precision,
        "sampling_steps": args.sampling_steps,
        "seed": args.seed,
        "ensemble_size": args.ensemble_size,
        "refiner": refiner_metadata,
        "metrics_evaluated": not args.skip_metrics,
        "save_max_side": args.save_max_side or None,
        "psnr_mean": (
            finite_mean([float(row["psnr"]) for row in rows])
            if not args.skip_metrics
            else None
        ),
        "ssim_mean": (
            finite_mean([float(row["ssim"]) for row in rows])
            if not args.skip_metrics
            else None
        ),
        "delta_e2000_mean": (
            finite_mean([float(row["delta_e2000_mean"]) for row in rows])
            if not args.skip_metrics
            else None
        ),
        "delta_e2000_p95_mean": (
            finite_mean([float(row["delta_e2000_p95"]) for row in rows])
            if not args.skip_metrics
            else None
        ),
        "lightness_mae_mean": (
            finite_mean([float(row["lightness_mae"]) for row in rows])
            if not args.skip_metrics
            else None
        ),
        "saturation_mae_mean": (
            finite_mean([float(row["saturation_mae"]) for row in rows])
            if not args.skip_metrics
            else None
        ),
        "hue_error_degrees_mean": (
            finite_mean([float(row["hue_error_degrees"]) for row in rows])
            if not args.skip_metrics
            else None
        ),
        "lpips_mean": (
            finite_mean([float(row["lpips"]) for row in rows]) if lpips_metric else None
        ),
        "inference_seconds_mean": finite_mean(
            [float(row["inference_seconds"]) for row in rows]
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
