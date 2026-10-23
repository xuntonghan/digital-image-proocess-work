from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import random
import time
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

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
from dip_course.cli.train_sied_restorer import (
    _loss_components,
    _save_checkpoint,
    _update_ema,
)
from dip_course.enhancement.dataset import SIEDFullSceneDataset
from dip_course.enhancement.refiner import (
    RawGuidedNAFRefiner,
    apply_raw_guided_refiner_tiled,
)
from dip_course.io import RawSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune the RAW-guided SIED restorer on native-resolution crops"
    )
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=2_500)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--val-limit", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--tile-overlap", type=int, default=96)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--freeze-photometric", action="store_true")
    parser.add_argument(
        "--loss-profile",
        choices=("balanced", "psnr"),
        default="balanced",
    )
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def load_restorer_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[RawGuidedNAFRefiner, dict[str, object]]:
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Restorer checkpoint root must be a mapping")
    if checkpoint.get("model_type") != RawGuidedNAFRefiner.model_type:
        raise ValueError("Initial checkpoint is not a raw_guided_naf restorer")
    config = checkpoint.get("model_config")
    state = checkpoint.get("model")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise TypeError("Restorer checkpoint is missing model_config or model")
    model = RawGuidedNAFRefiner(**dict(config)).to(device)
    model.load_state_dict(dict(state), strict=True)
    metadata: dict[str, object] = {
        "path": str(checkpoint_path.resolve()),
        "sha256": file_sha256(checkpoint_path),
        "global_step": checkpoint.get("global_step"),
        "metrics": checkpoint.get("metrics"),
        "model_config": dict(config),
    }
    return model, metadata


def native_crop(
    base: torch.Tensor,
    target: torch.Tensor,
    packed_raw: torch.Tensor,
    crop_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = base.shape[-2:]
    if crop_size > height or crop_size > width or crop_size % 2:
        raise ValueError("patch-size must be even and fit inside the RGB image")
    top = 2 * random.randint(0, (height - crop_size) // 2)
    left = 2 * random.randint(0, (width - crop_size) // 2)
    bottom = top + crop_size
    right = left + crop_size
    return (
        base[:, :, top:bottom, left:right],
        target[:, :, top:bottom, left:right],
        packed_raw[:, :, top // 2 : bottom // 2, left // 2 : right // 2],
    )


@torch.inference_mode()
def validate_native(
    diffusion: torch.nn.Module,
    model: RawGuidedNAFRefiner,
    loader: DataLoader,
    device: torch.device,
    *,
    precision: str,
    seed: int,
    tile_size: int,
    tile_overlap: int,
) -> dict[str, float]:
    model.eval()
    base_psnr = 0.0
    refined_psnr = 0.0
    count = 0
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    use_amp = device.type == "cuda" and precision != "fp32"
    for index, batch in enumerate(loader):
        raw = batch["raw"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        sample_seed = diffusion_seed(seed, index, 0)
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)
        base = predict_official_sied(diffusion, raw, precision=precision)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            refined = apply_raw_guided_refiner_tiled(
                model,
                base,
                raw,
                tile_size=tile_size,
                overlap=tile_overlap,
            )
        for prediction, accumulator in ((base, "base"), (refined, "refined")):
            mse = (prediction.float() - target.float()).square().mean().clamp_min(1e-12)
            psnr = float((-10.0 * torch.log10(mse)).item())
            if accumulator == "base":
                base_psnr += psnr
            else:
                refined_psnr += psnr
        count += 1
    if not count:
        raise RuntimeError("Native validation loader is empty")
    return {
        "base_native_psnr": base_psnr / count,
        "refined_native_psnr": refined_psnr / count,
        "native_validation_count": count,
    }


def main() -> None:
    args = build_parser().parse_args()
    positive = {
        "sampling_steps": args.sampling_steps,
        "max_steps": args.max_steps,
        "patch_size": args.patch_size,
        "validation_interval": args.validation_interval,
        "val_limit": args.val_limit,
        "tile_size": args.tile_size,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"Training fields must be positive: {invalid}")
    if args.tile_overlap < 0 or args.tile_overlap >= args.tile_size:
        raise ValueError("tile-overlap must be in [0, tile-size)")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Optimizer settings are invalid")
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

    diffusion, weight_source = load_official_model(
        args.official_repo,
        args.checkpoint,
        sampling_steps=args.sampling_steps,
        device=device,
    )
    for parameter in diffusion.parameters():
        parameter.requires_grad_(False)
    model, initial_metadata = load_restorer_checkpoint(args.initial_checkpoint, device)
    if args.freeze_photometric:
        for parameter in model.photometric.parameters():
            parameter.requires_grad_(False)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    spec = RawSpec(width=3840, height=2160, black_level=512, white_level=16383)
    train_dataset = SIEDFullSceneDataset(
        data_root=args.data_root,
        manifest=args.train_manifest,
        raw_spec=spec,
    )
    val_dataset = SIEDFullSceneDataset(
        data_root=args.data_root,
        manifest=args.val_manifest,
        raw_spec=spec,
    )
    if {record.low_raw for record in train_dataset.records} & {
        record.low_raw for record in val_dataset.records
    }:
        raise ValueError("Training and validation manifests overlap")
    val_dataset = Subset(val_dataset, range(min(args.val_limit, len(val_dataset))))
    loader_options = {
        "batch_size": 1,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
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
        "method": "official_sied_plus_raw_guided_naf_native_finetune",
        "started_at_unix": time.time(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "official_checkpoint": str(Path(args.checkpoint).resolve()),
        "official_checkpoint_sha256": file_sha256(args.checkpoint),
        "weight_source": weight_source,
        "initial_checkpoint": initial_metadata,
        "train_pairs": len(train_dataset),
        "val_pairs": len(val_dataset),
        "settings": vars(args),
    }
    write_json(output_dir / "run_environment.json", environment)
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)

    global_step = 0
    best_psnr = -math.inf
    log_path = output_dir / "training_log.jsonl"
    model.train()
    while global_step < args.max_steps:
        for batch in train_loader:
            raw_full = batch["raw"].to(device, non_blocking=True)
            target_full = batch["target"]
            raw_gain = model.compute_raw_gain(raw_full)
            sample_seed = diffusion_seed(args.seed, global_step, 0)
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
            with torch.inference_mode():
                base_full = predict_official_sied(
                    diffusion, raw_full, precision=args.precision
                )
            base, target, raw = native_crop(
                base_full,
                target_full,
                raw_full,
                args.patch_size,
            )
            base = base.clone()
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
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
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite native loss at step {global_step + 1}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
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
                metrics = validate_native(
                    diffusion,
                    ema_model,
                    val_loader,
                    device,
                    precision=args.precision,
                    seed=args.seed + 10_000_019,
                    tile_size=args.tile_size,
                    tile_overlap=args.tile_overlap,
                )
                row.update(metrics)
                if metrics["refined_native_psnr"] > best_psnr:
                    best_psnr = metrics["refined_native_psnr"]
                    _save_checkpoint(
                        output_dir / "best.pt",
                        {
                            "model": ema_model.state_dict(),
                            "model_type": RawGuidedNAFRefiner.model_type,
                            "model_config": initial_metadata["model_config"],
                            "global_step": global_step,
                            "metrics": metrics,
                            "torch_version": torch.__version__,
                        },
                    )
                model.train()
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
            "model_config": initial_metadata["model_config"],
            "global_step": global_step,
            "best_psnr": best_psnr,
            "torch_version": torch.__version__,
        },
    )
    write_json(
        output_dir / "summary.json",
        {
            "global_step": global_step,
            "best_refined_native_psnr": best_psnr,
            "best_checkpoint": str((output_dir / "best.pt").resolve()),
        },
    )


if __name__ == "__main__":
    main()
