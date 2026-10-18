from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dip_course.cli._common import write_json
from dip_course.geocorrection.dataset import GeoCorrectionPair, discover_pairs
from dip_course.geocorrection.segmentation import (
    PageMaskDataset,
    PageMaskUNet,
    binary_segmentation_metrics,
    save_segmentation_checkpoint,
    segmentation_loss,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train page-mask U-Net for GeoCorrection")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def stratified_split(
    pairs: list[GeoCorrectionPair], seed: int
) -> tuple[list[GeoCorrectionPair], list[GeoCorrectionPair], list[GeoCorrectionPair]]:
    grouped: defaultdict[str, list[GeoCorrectionPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.family].append(pair)
    generator = random.Random(seed)
    train: list[GeoCorrectionPair] = []
    validation: list[GeoCorrectionPair] = []
    test: list[GeoCorrectionPair] = []
    for family_pairs in grouped.values():
        shuffled = sorted(family_pairs, key=lambda pair: pair.key)
        generator.shuffle(shuffled)
        count = len(shuffled)
        test_count = max(1, round(count * 0.10)) if count >= 3 else 0
        validation_count = max(1, round(count * 0.10)) if count >= 3 else 0
        if test_count + validation_count >= count:
            validation_count = max(0, count - test_count - 1)
        test.extend(shuffled[:test_count])
        validation.extend(shuffled[test_count : test_count + validation_count])
        train.extend(shuffled[test_count + validation_count :])
    return tuple(sorted(group, key=lambda pair: pair.key) for group in (train, validation, test))  # type: ignore[return-value]


@torch.inference_mode()
def evaluate(
    model: PageMaskUNet,
    loader: DataLoader,
    *,
    device: torch.device,
    autocast_enabled: bool,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    metrics: defaultdict[str, list[float]] = defaultdict(list)
    for images, masks, _keys in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            logits = model(images)
            loss = segmentation_loss(logits, masks)
        losses.append(float(loss))
        predictions = logits.sigmoid().float().cpu().numpy() >= 0.5
        targets = masks.cpu().numpy() >= 0.5
        for prediction, target in zip(predictions, targets, strict=True):
            values = binary_segmentation_metrics(prediction[0], target[0])
            for name, value in values.items():
                metrics[name].append(value)
    return {
        "loss": statistics.fmean(losses),
        **{name: statistics.fmean(values) for name, values in metrics.items()},
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    if args.epochs <= 0 or args.batch_size <= 0 or args.input_size % 16:
        raise ValueError("epochs/batch-size must be positive and input-size divisible by 16")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    pairs = discover_pairs(args.data_root)
    train_pairs, validation_pairs, test_pairs = stratified_split(pairs, args.seed)
    split = {
        "train": [pair.key for pair in train_pairs],
        "validation": [pair.key for pair in validation_pairs],
        "test": [pair.key for pair in test_pairs],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "split.json", split)
    train_dataset = PageMaskDataset(
        train_pairs, size=args.input_size, augment=True, seed=args.seed
    )
    validation_dataset = PageMaskDataset(
        validation_pairs, size=args.input_size, augment=False, seed=args.seed
    )
    test_dataset = PageMaskDataset(
        test_pairs, size=args.input_size, augment=False, seed=args.seed
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model = PageMaskUNet(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.03
    )
    autocast_enabled = args.precision == "bf16" and device.type == "cuda"
    history: list[dict[str, float | int]] = []
    best_iou = -math.inf
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        train_losses: list[float] = []
        for images, masks, _keys in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                loss = segmentation_loss(model(images), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        scheduler.step()
        validation = evaluate(
            model,
            validation_loader,
            device=device,
            autocast_enabled=autocast_enabled,
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": statistics.fmean(train_losses),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["iou"] > best_iou:
            best_iou = validation["iou"]
            best_epoch = epoch
            save_segmentation_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=validation,
                input_size=args.input_size,
                base_channels=args.base_channels,
                split=split,
            )
        if epoch - best_epoch >= args.patience:
            break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    test_metrics = evaluate(
        model, test_loader, device=device, autocast_enabled=autocast_enabled
    )
    summary = {
        "method": "page_mask_unet_v1",
        "data_root": str(Path(args.data_root).resolve()),
        "pair_count": len(pairs),
        "split_counts": {key: len(values) for key, values in split.items()},
        "seed": args.seed,
        "input_size": args.input_size,
        "base_channels": args.base_channels,
        "best_epoch": best_epoch,
        "validation": checkpoint["metrics"],
        "test": test_metrics,
        "epochs_completed": len(history),
        "device": str(device),
        "precision": args.precision,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "checkpoint": "best.pt",
        "split": "split.json",
        "history": history,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
