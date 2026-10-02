from __future__ import annotations

import argparse
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
    parser.add_argument("--resume")
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: EnhancementLoss,
    device: torch.device,
) -> dict[str, float]:
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
    model.train()
    return {
        "val_loss": sum(losses) / len(losses),
        "val_psnr": sum(psnrs) / len(psnrs),
        "val_ssim": sum(ssims) / len(ssims),
    }


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
    criterion = EnhancementLoss(training.ssim_weight, training.chroma_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    global_step = 0
    best_loss = math.inf

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
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
    }
    write_json(output_dir / "run_environment.json", environment)
    print(json.dumps(environment, ensure_ascii=False, indent=2))

    log_path = output_dir / "training_log.jsonl"
    model.train()
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

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            row: dict[str, object] = {
                "step": global_step,
                "train_loss": float(loss.item()),
                "train_l1": float(components["l1"].item()),
                "train_ssim": float(components["ssim"].item()),
                "train_chroma": float(components["chroma"].item()),
                "time_unix": time.time(),
            }

            should_validate = (
                global_step == 1 or global_step % training.validation_interval == 0
            )
            if should_validate:
                validation = validate(model, val_loader, criterion, device)
                row.update(validation)
                if validation["val_loss"] < best_loss:
                    best_loss = validation["val_loss"]
                    save_checkpoint(
                        output_dir / "best.pt",
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
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
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
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
