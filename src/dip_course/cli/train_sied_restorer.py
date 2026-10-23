from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dip_course.cli._common import (
    choose_torch_device,
    file_sha256,
    seed_everything,
    write_json,
)
from dip_course.enhancement.dataset import SIEDPredictionDataset
from dip_course.enhancement.losses import (
    color_histogram_loss,
    color_moment_loss,
    edge_map,
    ssim_index,
)
from dip_course.enhancement.refiner import (
    RawGuidedNAFRefiner,
    refiner_parameter_penalty,
)
from dip_course.io import RawSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a RAW-guided NAFNet residual restorer on cached official SIED output"
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument(
        "--val-prediction-dir",
        help="Optional separate cached predictions for validation",
    )
    parser.add_argument(
        "--raw-mean-cache",
        help="Optional JSON mapping keys to full-scene normalized RAW means",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument(
        "--photometric-checkpoint",
        help="Initialize the global exposure/color branch from a legacy refiner",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--crop-size",
        type=int,
        default=0,
        help="Aligned native RGB crop size; zero trains on complete cached images",
    )
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--encoder-blocks", default="2,2,4")
    parser.add_argument("--middle-blocks", type=int, default=6)
    parser.add_argument("--decoder-blocks", default="2,2,2")
    parser.add_argument("--max-residual", type=float, default=0.35)
    parser.add_argument("--raw-target-mean", type=float, default=0.1)
    parser.add_argument("--raw-max-gain", type=float, default=4096.0)
    parser.add_argument(
        "--loss-profile",
        choices=("balanced", "psnr"),
        default="balanced",
        help="Balanced restoration loss or an MSE-dominant PSNR fine-tuning loss",
    )
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def _block_counts(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or any(count <= 0 for count in result):
        raise ValueError(f"Invalid NAFNet block specification: {value!r}")
    return result


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_initial(
    checkpoint_path: str | Path,
    model: RawGuidedNAFRefiner,
) -> dict[str, object]:
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Restorer checkpoint root must be a dictionary")
    if checkpoint.get("model_type") != RawGuidedNAFRefiner.model_type:
        raise ValueError("Initial checkpoint is not a raw_guided_naf refiner")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Restorer checkpoint does not contain model weights")
    model.load_state_dict(state, strict=True)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "global_step": checkpoint.get("global_step"),
        "metrics": checkpoint.get("metrics"),
    }


def _load_photometric(
    checkpoint_path: str | Path,
    model: RawGuidedNAFRefiner,
) -> dict[str, object]:
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Photometric checkpoint root must be a dictionary")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Photometric checkpoint does not contain model weights")
    model.photometric.load_state_dict(state, strict=True)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "global_step": checkpoint.get("global_step"),
        "metrics": checkpoint.get("metrics"),
    }


@torch.no_grad()
def _update_ema(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    decay: float,
) -> None:
    online_parameters = dict(model.named_parameters())
    for name, parameter in ema_model.named_parameters():
        parameter.lerp_(online_parameters[name].detach(), 1.0 - decay)
    online_buffers = dict(model.named_buffers())
    for name, buffer in ema_model.named_buffers():
        buffer.copy_(online_buffers[name])


def _loss_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    base: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    *,
    profile: str = "balanced",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # PSNR is a monotonic transform of MSE, so MSE remains the primary target.
    mse = F.mse_loss(prediction, target)
    charbonnier = torch.sqrt((prediction - target).square() + 1e-6).mean()
    ssim = ssim_index(prediction, target)
    edge = F.l1_loss(edge_map(prediction), edge_map(target))
    moments = color_moment_loss(prediction, target)
    histogram = color_histogram_loss(prediction, target)
    identity = F.l1_loss(prediction, base)
    photometric_penalty = refiner_parameter_penalty(parameters)
    if profile == "balanced":
        weights = {
            "charbonnier": 0.05,
            "ssim": 0.02,
            "edge": 0.01,
            "moments": 0.02,
            "histogram": 0.02,
            "identity": 0.001,
            "photometric": 0.0005,
        }
    elif profile == "psnr":
        # At high quality, auxiliary losses can otherwise become larger than
        # MSE and cap the very metric being optimized. Retain small structural
        # and color terms while making MSE unambiguously dominant.
        weights = {
            "charbonnier": 0.01,
            "ssim": 0.005,
            "edge": 0.002,
            "moments": 0.005,
            "histogram": 0.005,
            "identity": 0.0005,
            "photometric": 0.0001,
        }
    else:
        raise ValueError(f"Unknown loss profile: {profile}")
    loss = (
        mse
        + weights["charbonnier"] * charbonnier
        + weights["ssim"] * (1.0 - ssim)
        + weights["edge"] * edge
        + weights["moments"] * moments
        + weights["histogram"] * histogram
        + weights["identity"] * identity
        + weights["photometric"] * photometric_penalty
    )
    return loss, {
        "mse": mse.detach(),
        "charbonnier": charbonnier.detach(),
        "ssim": ssim.detach(),
        "edge": edge.detach(),
        "color_moments": moments.detach(),
        "color_histogram": histogram.detach(),
        "identity": identity.detach(),
        "photometric_penalty": photometric_penalty.detach(),
    }


@torch.inference_mode()
def validate(
    model: RawGuidedNAFRefiner,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    base_squared_error = 0.0
    refined_squared_error = 0.0
    pixel_count = 0
    base_psnr_sum = 0.0
    refined_psnr_sum = 0.0
    base_ssim_sum = 0.0
    refined_ssim_sum = 0.0
    count = 0
    for batch in loader:
        base = batch["prediction"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        raw = batch["raw"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            refined = model(base, raw)
        for prediction, is_base in ((base, True), (refined, False)):
            squared = (prediction.float() - target.float()).square()
            per_image_mse = squared.mean(dim=(1, 2, 3)).clamp_min(1e-12)
            psnr = -10.0 * torch.log10(per_image_mse)
            similarity = ssim_index(prediction.float(), target.float())
            if is_base:
                base_squared_error += float(squared.sum().item())
                base_psnr_sum += float(psnr.sum().item())
                base_ssim_sum += float(similarity.item()) * prediction.shape[0]
            else:
                refined_squared_error += float(squared.sum().item())
                refined_psnr_sum += float(psnr.sum().item())
                refined_ssim_sum += float(similarity.item()) * prediction.shape[0]
        pixel_count += target.numel()
        count += target.shape[0]
    if not count:
        raise RuntimeError("Validation loader is empty")
    return {
        "base_psnr_mean": base_psnr_sum / count,
        "refined_psnr_mean": refined_psnr_sum / count,
        "base_global_psnr": -10.0 * math.log10(base_squared_error / pixel_count),
        "refined_global_psnr": -10.0 * math.log10(refined_squared_error / pixel_count),
        "base_ssim": base_ssim_sum / count,
        "refined_ssim": refined_ssim_sum / count,
    }


def main() -> None:
    args = build_parser().parse_args()
    positive = {
        "max_steps": args.max_steps,
        "validation_interval": args.validation_interval,
        "checkpoint_interval": args.checkpoint_interval,
        "gradient_accumulation": args.gradient_accumulation,
        "batch_size": args.batch_size,
        "width": args.width,
        "middle_blocks": args.middle_blocks,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"Training fields must be positive: {invalid}")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Optimizer settings are invalid")
    if args.crop_size < 0 or args.crop_size % 2:
        raise ValueError("crop-size must be zero or a positive even integer")
    if args.raw_target_mean <= 0 or args.raw_max_gain < 1:
        raise ValueError("RAW guidance normalization settings are invalid")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema-decay must be in (0, 1)")

    seed_everything(args.seed)
    device = choose_torch_device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(f"CUDA was required but selected device is {device}")
    if args.precision != "fp32" and device.type != "cuda":
        raise ValueError("fp16/bf16 training requires CUDA")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = RawSpec(width=3840, height=2160, black_level=512, white_level=16383)
    train_dataset = SIEDPredictionDataset(
        data_root=args.data_root,
        manifest=args.train_manifest,
        prediction_dir=args.prediction_dir,
        augment=True,
        include_raw=True,
        raw_spec=spec,
        crop_size=args.crop_size or None,
        raw_mean_cache=args.raw_mean_cache,
    )
    val_dataset = SIEDPredictionDataset(
        data_root=args.data_root,
        manifest=args.val_manifest,
        prediction_dir=args.val_prediction_dir or args.prediction_dir,
        augment=False,
        include_raw=True,
        raw_spec=spec,
    )
    if {record.low_raw for record in train_dataset.records} & {
        record.low_raw for record in val_dataset.records
    }:
        raise ValueError("Training and validation manifests overlap")
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model_config: dict[str, object] = {
        "width": args.width,
        "encoder_blocks": _block_counts(args.encoder_blocks),
        "middle_blocks": args.middle_blocks,
        "decoder_blocks": _block_counts(args.decoder_blocks),
        "max_residual": args.max_residual,
        "raw_target_mean": args.raw_target_mean,
        "raw_max_gain": args.raw_max_gain,
    }
    model = RawGuidedNAFRefiner(**model_config).to(device)
    initial_checkpoint = None
    if args.initial_checkpoint:
        initial_checkpoint = _load_initial(args.initial_checkpoint, model)
    photometric_checkpoint = None
    if args.photometric_checkpoint:
        if args.initial_checkpoint:
            raise ValueError(
                "photometric-checkpoint cannot be combined with initial-checkpoint"
            )
        photometric_checkpoint = _load_photometric(args.photometric_checkpoint, model)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_steps,
        eta_min=args.learning_rate * 0.02,
    )
    use_amp = device.type == "cuda" and args.precision != "fp32"
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.precision == "fp16"
    )

    environment = {
        "method": "official_sied_plus_raw_guided_naf",
        "started_at_unix": time.time(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "prediction_dir": str(Path(args.prediction_dir).resolve()),
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "val_manifest_sha256": file_sha256(args.val_manifest),
        "train_pairs": len(train_dataset),
        "val_pairs": len(val_dataset),
        "model_type": RawGuidedNAFRefiner.model_type,
        "model_config": model_config,
        "initial_checkpoint": initial_checkpoint,
        "photometric_checkpoint": photometric_checkpoint,
        "settings": vars(args),
    }
    write_json(output_dir / "run_environment.json", environment)
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)

    global_step = 0
    micro_step = 0
    best_psnr = -math.inf
    log_path = output_dir / "training_log.jsonl"
    optimizer.zero_grad(set_to_none=True)
    model.train()
    while global_step < args.max_steps:
        for batch in train_loader:
            base = batch["prediction"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            raw = batch["raw"].to(device, non_blocking=True)
            raw_gain = None
            if "raw_mean" in batch:
                raw_mean = batch["raw_mean"].to(device, non_blocking=True)
                raw_gain = (
                    model.raw_target_mean / raw_mean.reshape(-1, 1, 1, 1).clamp_min(1e-8)
                ).clamp(1.0, model.raw_max_gain)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                refined, parameters = model(
                    base,
                    raw,
                    raw_gain=raw_gain,
                    return_parameters=True,
                )
                loss, components = _loss_components(
                    refined,
                    target,
                    base,
                    parameters,
                    profile=args.loss_profile,
                )
                scaled_loss = loss / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite restorer loss at step {global_step + 1}"
                )
            scaler.scale(scaled_loss).backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() < scale_before_step:
                continue
            scheduler.step()
            _update_ema(ema_model, model, args.ema_decay)
            global_step += 1
            row: dict[str, object] = {
                "step": global_step,
                "train_loss": float(loss.item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "time_unix": time.time(),
            }
            row.update(
                {
                    f"train_{name}": float(value.item())
                    for name, value in components.items()
                }
            )

            metrics = None
            if global_step == 1 or global_step % args.validation_interval == 0:
                metrics = validate(
                    ema_model,
                    val_loader,
                    device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                row.update(metrics)
                if metrics["refined_psnr_mean"] > best_psnr:
                    best_psnr = metrics["refined_psnr_mean"]
                    _save_checkpoint(
                        output_dir / "best.pt",
                        {
                            "model": ema_model.state_dict(),
                            "model_type": RawGuidedNAFRefiner.model_type,
                            "model_config": model_config,
                            "global_step": global_step,
                            "metrics": metrics,
                            "torch_version": torch.__version__,
                        },
                    )
                model.train()

            if global_step % args.checkpoint_interval == 0:
                _save_checkpoint(
                    output_dir / f"step_{global_step:07d}.pt",
                    {
                        "model": ema_model.state_dict(),
                        "model_type": RawGuidedNAFRefiner.model_type,
                        "model_config": model_config,
                        "global_step": global_step,
                        "metrics": metrics,
                        "torch_version": torch.__version__,
                    },
                )
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if metrics is not None or global_step % 25 == 0:
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if global_step >= args.max_steps:
                break

    _save_checkpoint(
        output_dir / "last.pt",
        {
            "model": ema_model.state_dict(),
            "model_type": RawGuidedNAFRefiner.model_type,
            "model_config": model_config,
            "global_step": global_step,
            "best_psnr": best_psnr,
            "torch_version": torch.__version__,
        },
    )
    write_json(
        output_dir / "summary.json",
        {
            "global_step": global_step,
            "best_refined_psnr_mean": best_psnr,
            "best_checkpoint": str((output_dir / "best.pt").resolve()),
        },
    )


if __name__ == "__main__":
    main()
