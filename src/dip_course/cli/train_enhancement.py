from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dip_course.cli._common import (
    choose_torch_device,
    file_sha256,
    raw_spec_from_config,
    seed_everything,
    write_json,
)
from dip_course.config import load_experiment_config
from dip_course.enhancement.dataset import SIEDPairDataset
from dip_course.enhancement.losses import EnhancementLoss, ssim_index
from dip_course.enhancement.model import RawToRGBUNet, count_trainable_parameters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the RAW-to-RGB enhancement model"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-steps", type=int, help="Override config for smoke/pilot runs"
    )
    parser.add_argument("--output-dir", help="Override config output directory")
    parser.add_argument(
        "--gt-cache-dir",
        help="Optional directory made by dip-prepare-gt-cache for faster random crops",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume")
    checkpoint_group.add_argument(
        "--init-checkpoint",
        help=(
            "Warm-start compatible model weights without restoring optimizer, "
            "scheduler, or step state"
        ),
    )
    parser.add_argument(
        "--freeze-batch-norm",
        action="store_true",
        help="Keep pretrained BatchNorm running statistics fixed during fine-tuning",
    )
    tuning_group = parser.add_mutually_exclusive_group()
    tuning_group.add_argument(
        "--color-head-only",
        action="store_true",
        help="Train only the RGB output head and illumination-conditioning module",
    )
    tuning_group.add_argument(
        "--color-lut-only",
        action="store_true",
        help="Train only the scene-adaptive 3D LUT color refinement module",
    )
    tuning_group.add_argument(
        "--decoder-color-only",
        action="store_true",
        help="Train the decoder, RGB head, illumination module, and color LUT",
    )
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, decay: float
) -> None:
    online_parameters = dict(model.named_parameters())
    for name, parameter in ema_model.named_parameters():
        parameter.lerp_(online_parameters[name].detach(), 1.0 - decay)
    online_buffers = dict(model.named_buffers())
    for name, buffer in ema_model.named_buffers():
        buffer.copy_(online_buffers[name])


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: EnhancementLoss,
    device: torch.device,
    *,
    freeze_batch_norm: bool = False,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    for batch in loader:
        raw = batch["raw"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        prediction = model(raw)
        loss, _ = criterion(prediction, target)
        mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
        losses.append(float(loss.item()))
        psnrs.extend(float(value) for value in psnr.cpu())
        ssims.append(float(ssim_index(prediction, target).item()))
    model.train(was_training)
    if was_training and freeze_batch_norm:
        freeze_batch_norm_statistics(model)
    return {
        "val_loss": sum(losses) / len(losses),
        "val_psnr": sum(psnrs) / len(psnrs),
        "val_ssim": sum(ssims) / len(ssims),
    }


def freeze_batch_norm_statistics(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def main() -> None:
    args = build_parser().parse_args()
    config = load_experiment_config(args.config)
    training = config.training
    max_steps = training.max_steps if args.max_steps is None else args.max_steps
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    output_dir = Path(args.output_dir or training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(training.seed)
    device = choose_torch_device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(f"CUDA was required but selected device is {device}")
    use_amp = training.amp and device.type == "cuda"

    raw_spec = raw_spec_from_config(config.dataset)
    train_dataset = SIEDPairDataset(
        data_root=args.data_root,
        manifest=args.train_manifest,
        raw_spec=raw_spec,
        patch_size=config.dataset.patch_size,
        augment=config.dataset.augment,
        gt_cache_dir=args.gt_cache_dir,
        color_aware_probability=config.dataset.color_aware_probability,
        color_aware_candidates=config.dataset.color_aware_candidates,
    )
    val_dataset = SIEDPairDataset(
        data_root=args.data_root,
        manifest=args.val_manifest,
        raw_spec=raw_spec,
        patch_size=config.dataset.patch_size,
        augment=False,
        gt_cache_dir=args.gt_cache_dir,
    )
    train_low = {record.low_raw for record in train_dataset.records}
    val_low = {record.low_raw for record in val_dataset.records}
    train_gt = {record.gt_rgb for record in train_dataset.records}
    val_gt = {record.gt_rgb for record in val_dataset.records}
    if train_low & val_low or train_gt & val_gt:
        raise ValueError("Training and validation manifests overlap")
    loader_options = {
        "batch_size": training.batch_size,
        "num_workers": training.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": training.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=len(train_dataset) >= training.batch_size,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_options
    )

    model_config = asdict(config.model)
    model = RawToRGBUNet(**model_config).to(device)
    if config.dataset.patch_size % model.input_multiple:
        raise ValueError(
            f"patch_size must be divisible by {model.input_multiple} for model depth "
            f"{config.model.depth}"
        )
    initialization: dict[str, object] | None = None
    if args.init_checkpoint:
        initial_checkpoint = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False
        )
        initial_state = initial_checkpoint.get("model", initial_checkpoint)
        if not isinstance(initial_state, dict):
            raise TypeError("Initialization checkpoint does not contain model weights")
        incompatible = model.load_state_dict(initial_state, strict=False)
        initialization = {
            "checkpoint": str(Path(args.init_checkpoint).resolve()),
            "source_global_step": initial_checkpoint.get("global_step"),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
        print(json.dumps({"warm_start": initialization}, ensure_ascii=False, indent=2))
    if args.color_head_only:
        if config.model.output_space != "rgb":
            raise ValueError("--color-head-only requires an RGB-output model")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith(("head.", "illumination."))
            )
    if args.color_lut_only:
        if not config.model.color_lut_refinement:
            raise ValueError("--color-lut-only requires model.color_lut_refinement")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("color_refinement."))
    if args.decoder_color_only:
        if config.model.output_space != "rgb":
            raise ValueError("--decoder-color-only requires an RGB-output model")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith(
                    ("up_blocks.", "head.", "illumination.", "color_refinement.")
                )
            )
    criterion = EnhancementLoss(
        ssim_weight=training.ssim_weight,
        chroma_weight=training.chroma_weight,
        hvi_weight=training.hvi_weight,
        histogram_weight=training.histogram_weight,
        edge_weight=training.edge_weight,
        color_moment_weight=training.color_moment_weight,
        hue_weight=training.hue_weight,
        saturation_weight=training.saturation_weight,
        hvi_density=config.model.hvi_density,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps, eta_min=training.min_learning_rate
        )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    global_step = 0
    best_loss = math.inf

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("online_model", checkpoint["model"]))
        ema_model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = int(checkpoint["global_step"])
        best_loss = float(checkpoint.get("best_loss", math.inf))

    environment = {
        "experiment": config.name,
        "started_at_unix": time.time(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "val_manifest_sha256": file_sha256(args.val_manifest),
        "train_pairs": len(train_dataset),
        "val_pairs": len(val_dataset),
        "model_parameters": count_trainable_parameters(model),
        "config": asdict(config),
        "effective_max_steps": max_steps,
        "amp_enabled": use_amp,
        "gt_cache_dir": (
            str(Path(args.gt_cache_dir).resolve()) if args.gt_cache_dir else None
        ),
        "initialization": initialization,
        "freeze_batch_norm": args.freeze_batch_norm,
        "color_head_only": args.color_head_only,
        "color_lut_only": args.color_lut_only,
        "decoder_color_only": args.decoder_color_only,
    }
    write_json(output_dir / "run_environment.json", environment)
    print(json.dumps(environment, ensure_ascii=False, indent=2))

    log_path = output_dir / "training_log.jsonl"
    model.train()
    if args.freeze_batch_norm:
        freeze_batch_norm_statistics(model)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0

    while global_step < max_steps:
        for batch in train_loader:
            raw = batch["raw"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                prediction = model(raw)
                loss, components = criterion(prediction, target)
                scaled_loss = loss / training.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            micro_step += 1
            if micro_step % training.gradient_accumulation:
                continue

            if training.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, training.max_grad_norm
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if training.ema_decay > 0:
                update_ema(ema_model, model, training.ema_decay)
            else:
                ema_model.load_state_dict(model.state_dict())
            global_step += 1
            row: dict[str, object] = {
                "step": global_step,
                "train_loss": float(loss.item()),
                "train_l1": float(components["l1"].item()),
                "train_ssim": float(components["ssim"].item()),
                "train_chroma": float(components["chroma"].item()),
                "train_hvi": float(components["hvi"].item()),
                "train_histogram": float(components["histogram"].item()),
                "train_edge": float(components["edge"].item()),
                "train_color_moments": float(components["color_moments"].item()),
                "train_hue": float(components["hue"].item()),
                "train_saturation": float(components["saturation"].item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "time_unix": time.time(),
            }

            should_validate = (
                global_step == 1 or global_step % training.validation_interval == 0
            )
            if should_validate:
                validation = validate(
                    ema_model,
                    val_loader,
                    criterion,
                    device,
                    freeze_batch_norm=args.freeze_batch_norm,
                )
                row.update(validation)
                if validation["val_loss"] < best_loss:
                    best_loss = validation["val_loss"]
                    save_checkpoint(
                        output_dir / "best.pt",
                        {
                            "model": ema_model.state_dict(),
                            "online_model": model.state_dict(),
                            "ema_model": ema_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
                            "scheduler": (
                                scheduler.state_dict() if scheduler is not None else None
                            ),
                            "global_step": global_step,
                            "best_loss": best_loss,
                            "model_config": model_config,
                            "experiment_config": asdict(config),
                        },
                    )

            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if global_step == 1 or global_step % 50 == 0 or should_validate:
                print(json.dumps(row, ensure_ascii=False), flush=True)

            if (
                global_step % training.checkpoint_interval == 0
                or global_step == max_steps
            ):
                save_checkpoint(
                    output_dir / f"step_{global_step:07d}.pt",
                    {
                        "model": ema_model.state_dict(),
                        "online_model": model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "scheduler": (
                            scheduler.state_dict() if scheduler is not None else None
                        ),
                        "global_step": global_step,
                        "best_loss": best_loss,
                        "model_config": model_config,
                        "experiment_config": asdict(config),
                    },
                )

            if global_step >= max_steps:
                break

    write_json(
        output_dir / "completed.json",
        {"completed": True, "global_step": global_step, "best_loss": best_loss},
    )


if __name__ == "__main__":
    main()
