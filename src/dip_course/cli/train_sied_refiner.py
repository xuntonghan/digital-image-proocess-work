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
from dip_course.cli.infer_official_sied import (
    diffusion_seed,
    load_official_model,
    predict_official_sied,
)
from dip_course.enhancement.dataset import SIEDPairDataset, SIEDPredictionDataset
from dip_course.enhancement.losses import EnhancementLoss, edge_map, ssim_index
from dip_course.enhancement.refiner import (
    ExposureColorRefiner,
    refiner_parameter_penalty,
)
from dip_course.io import RawSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a lightweight exposure/color refiner on frozen SIED output"
    )
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--gt-cache-dir")
    parser.add_argument(
        "--prediction-dir",
        help="Train from cached full-scene SIED thumbnails instead of online RAW patches",
    )
    parser.add_argument(
        "--initial-checkpoint",
        help="Initialize the refiner from an earlier best.pt before fine-tuning",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--color-aware-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_initial_refiner(
    checkpoint_path: str | Path | None,
    device: torch.device,
) -> tuple[ExposureColorRefiner, dict[str, object] | None, dict[str, object]]:
    """Build the refiner and optionally initialize it from a trusted checkpoint."""
    default_config: dict[str, object] = {
        "base_channels": 24,
        "context_side": 256,
    }
    if checkpoint_path is None:
        return ExposureColorRefiner(**default_config).to(device), None, default_config

    path = Path(checkpoint_path)
    # This path is only accepted through the explicit --initial-checkpoint
    # option.  Project-generated checkpoints include TorchVersion metadata,
    # which PyTorch 2.6+ intentionally rejects in weights-only mode.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Initial refiner checkpoint root must be a dictionary")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Initial refiner checkpoint does not contain model weights")
    stored_config = checkpoint.get("model_config", default_config)
    if not isinstance(stored_config, dict):
        raise TypeError("Initial refiner model_config must be a dictionary")
    model_config = dict(stored_config)
    model = ExposureColorRefiner(**model_config).to(device)
    model.load_state_dict(state, strict=True)
    metadata: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "source_global_step": checkpoint.get("global_step"),
        "source_metrics": checkpoint.get("metrics"),
    }
    return model, metadata, model_config


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.995
) -> None:
    online = dict(model.named_parameters())
    for name, parameter in ema_model.named_parameters():
        parameter.lerp_(online[name].detach(), 1.0 - decay)
    online_buffers = dict(model.named_buffers())
    for name, buffer in ema_model.named_buffers():
        buffer.copy_(online_buffers[name])


@torch.inference_mode()
def validate(
    diffusion: torch.nn.Module | None,
    refiner: ExposureColorRefiner,
    loader: DataLoader,
    criterion: EnhancementLoss,
    device: torch.device,
    *,
    precision: str,
    seed: int,
    cached_predictions: bool,
) -> dict[str, float]:
    refiner.eval()
    base_psnrs: list[float] = []
    refined_psnrs: list[float] = []
    base_ssims: list[float] = []
    refined_ssims: list[float] = []
    base_losses: list[float] = []
    losses: list[float] = []
    for index, batch in enumerate(loader):
        target = batch["target"].to(device, non_blocking=True)
        if cached_predictions:
            base = batch["prediction"].to(device, non_blocking=True)
        else:
            assert diffusion is not None
            raw = batch["raw"].to(device, non_blocking=True)
            sample_seed = diffusion_seed(seed, index, 0)
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
            base = predict_official_sied(diffusion, raw, precision=precision)
        refined = refiner(base)
        base_loss, _ = criterion(base, target)
        loss, _ = criterion(refined, target)
        base_losses.append(float(base_loss.item()))
        losses.append(float(loss.item()))
        for prediction, psnrs, ssims in (
            (base, base_psnrs, base_ssims),
            (refined, refined_psnrs, refined_ssims),
        ):
            mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
            psnrs.extend(float(value) for value in (-10.0 * torch.log10(mse)).cpu())
            ssims.append(float(ssim_index(prediction, target).item()))
    return {
        "base_val_loss": sum(base_losses) / len(base_losses),
        "val_loss": sum(losses) / len(losses),
        "base_psnr": sum(base_psnrs) / len(base_psnrs),
        "refined_psnr": sum(refined_psnrs) / len(refined_psnrs),
        "base_ssim": sum(base_ssims) / len(base_ssims),
        "refined_ssim": sum(refined_ssims) / len(refined_ssims),
    }


def main() -> None:
    args = build_parser().parse_args()
    positive = {
        "max_steps": args.max_steps,
        "patch_size": args.patch_size,
        "sampling_steps": args.sampling_steps,
        "validation_interval": args.validation_interval,
        "checkpoint_interval": args.checkpoint_interval,
        "gradient_accumulation": args.gradient_accumulation,
        "batch_size": args.batch_size,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"training fields must be positive: {invalid}")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("optimizer settings are invalid")

    seed_everything(args.seed)
    device = choose_torch_device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(f"CUDA was required but selected device is {device}")
    if args.precision != "fp32" and device.type != "cuda":
        raise ValueError("fp16/bf16 SIED inference requires CUDA")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cached_predictions = args.prediction_dir is not None
    diffusion: torch.nn.Module | None = None
    weight_source = "cached_official_sied_predictions"
    if not cached_predictions:
        diffusion, weight_source = load_official_model(
            args.official_repo,
            args.checkpoint,
            sampling_steps=args.sampling_steps,
            device=device,
        )
        for parameter in diffusion.parameters():
            parameter.requires_grad_(False)
        if args.batch_size != 1:
            raise ValueError("Online SIED refinement currently requires --batch-size 1")

    spec = RawSpec(width=3840, height=2160, black_level=512, white_level=16383)
    if cached_predictions:
        train_dataset = SIEDPredictionDataset(
            data_root=args.data_root,
            manifest=args.train_manifest,
            prediction_dir=args.prediction_dir,
            augment=True,
        )
        val_dataset = SIEDPredictionDataset(
            data_root=args.data_root,
            manifest=args.val_manifest,
            prediction_dir=args.prediction_dir,
            augment=False,
        )
    else:
        train_dataset = SIEDPairDataset(
            data_root=args.data_root,
            manifest=args.train_manifest,
            raw_spec=spec,
            patch_size=args.patch_size,
            augment=True,
            gt_cache_dir=args.gt_cache_dir,
            color_aware_probability=args.color_aware_probability,
            color_aware_candidates=6,
        )
        val_dataset = SIEDPairDataset(
            data_root=args.data_root,
            manifest=args.val_manifest,
            raw_spec=spec,
            patch_size=args.patch_size,
            augment=False,
            gt_cache_dir=args.gt_cache_dir,
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

    model, initial_checkpoint, model_config = load_initial_refiner(
        args.initial_checkpoint,
        device,
    )
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    criterion = EnhancementLoss(
        ssim_weight=0.15,
        chroma_weight=0.35,
        hvi_weight=0.15,
        histogram_weight=0.03,
        edge_weight=0.03,
        color_moment_weight=0.15,
        hue_weight=0.03,
        saturation_weight=0.08,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.learning_rate * 0.05
    )
    use_amp = device.type == "cuda" and args.precision != "fp32"
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    use_grad_scaler = device.type == "cuda" and args.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    environment = {
        "method": "frozen_official_sied_with_exposure_color_refiner",
        "started_at_unix": time.time(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "weight_source": weight_source,
        "official_repo": str(Path(args.official_repo).resolve()),
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "val_manifest_sha256": file_sha256(args.val_manifest),
        "train_pairs": len(train_dataset),
        "val_pairs": len(val_dataset),
        "blind_inference": True,
        "settings": vars(args),
        "cached_predictions": cached_predictions,
        "initial_checkpoint": initial_checkpoint,
    }
    write_json(output_dir / "run_environment.json", environment)
    print(json.dumps(environment, ensure_ascii=False, indent=2))

    log_path = output_dir / "training_log.jsonl"
    global_step = 0
    micro_step = 0
    best_val_loss = math.inf
    optimizer.zero_grad(set_to_none=True)
    model.train()
    while global_step < args.max_steps:
        for batch in train_loader:
            target = batch["target"].to(device, non_blocking=True)
            if cached_predictions:
                base = batch["prediction"].to(device, non_blocking=True)
            else:
                assert diffusion is not None
                raw = batch["raw"].to(device, non_blocking=True)
                sample_seed = diffusion_seed(args.seed, micro_step, 0)
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                with torch.inference_mode():
                    base = predict_official_sied(
                        diffusion, raw, precision=args.precision
                    )
                # predict_official_sied intentionally returns an inference tensor.
                # Clone outside inference mode so autograd may save it while
                # computing gradients for the refiner's convolution weights.
                base = base.clone()
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                refined, parameters = model(base, return_parameters=True)
                reconstruction, components = criterion(refined, target)
                identity = F.l1_loss(refined, base)
                detail = F.l1_loss(edge_map(refined), edge_map(base))
                parameter_penalty = refiner_parameter_penalty(parameters)
                loss = (
                    reconstruction
                    + 0.015 * identity
                    + 0.01 * detail
                    + 0.002 * parameter_penalty
                )
                scaled_loss = loss / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite refiner loss at optimizer step {global_step + 1}"
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
                print(
                    json.dumps(
                        {
                            "event": "optimizer_step_skipped",
                            "next_step": global_step + 1,
                            "time_unix": time.time(),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            scheduler.step()
            update_ema(ema_model, model)
            global_step += 1
            row: dict[str, object] = {
                "step": global_step,
                "train_loss": float(loss.item()),
                "train_l1": float(components["l1"].item()),
                "train_ssim": float(components["ssim"].item()),
                "train_chroma": float(components["chroma"].item()),
                "train_hvi": float(components["hvi"].item()),
                "train_hue": float(components["hue"].item()),
                "train_identity": float(identity.item()),
                "train_detail": float(detail.item()),
                "parameter_penalty": float(parameter_penalty.item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "time_unix": time.time(),
            }

            if global_step % args.validation_interval == 0 or global_step == 1:
                metrics = validate(
                    diffusion,
                    ema_model,
                    val_loader,
                    criterion,
                    device,
                    precision=args.precision,
                    seed=args.seed + 10_000_019,
                    cached_predictions=cached_predictions,
                )
                row.update(metrics)
                improved = metrics["val_loss"] < best_val_loss
                if improved:
                    best_val_loss = metrics["val_loss"]
                    save_checkpoint(
                        output_dir / "best.pt",
                        {
                            "model": ema_model.state_dict(),
                            "global_step": global_step,
                            "metrics": metrics,
                            "model_config": model_config,
                            "environment": environment,
                        },
                    )
                model.train()

            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)

            if global_step % args.checkpoint_interval == 0:
                save_checkpoint(
                    output_dir / f"step_{global_step:06d}.pt",
                    {
                        "model": ema_model.state_dict(),
                        "online_model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                        "model_config": model_config,
                        "environment": environment,
                    },
                )
            if global_step >= args.max_steps:
                break

    write_json(
        output_dir / "training_summary.json",
        {
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "best_checkpoint": str((output_dir / "best.pt").resolve()),
        },
    )


if __name__ == "__main__":
    main()
