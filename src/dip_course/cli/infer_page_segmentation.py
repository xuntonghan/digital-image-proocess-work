from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dip_course.cli._common import write_json
from dip_course.geocorrection.dataset import discover_pairs
from dip_course.geocorrection.segmentation import (
    binary_segmentation_metrics,
    letterbox_image,
    load_segmentation_checkpoint,
    restore_mask,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict page masks for GeoCorrection")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    device = torch.device(args.device)
    model, checkpoint = load_segmentation_checkpoint(args.checkpoint, device=device)
    input_size = int(checkpoint["model_config"]["input_size"])
    pairs = discover_pairs(args.data_root)
    if args.split != "all":
        allowed = set(checkpoint["split"][args.split])
        pairs = [pair for pair in pairs if pair.key in allowed]
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    autocast_enabled = args.precision == "bf16" and device.type == "cuda"
    with torch.inference_mode():
        for index, pair in enumerate(pairs, start=1):
            with Image.open(pair.image_path) as source:
                image = source.convert("RGB")
            with Image.open(pair.mask_path) as source:
                target = np.asarray(source.convert("L"), dtype=np.uint8) >= 128
            prepared, transform = letterbox_image(image, size=input_size)
            tensor = torch.from_numpy(
                (np.asarray(prepared, dtype=np.float32) / 255.0).transpose(2, 0, 1).copy()
            )[None].to(device)
            started = time.perf_counter_ns()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                probability = model(tensor).sigmoid()[0, 0].float().cpu().numpy()
            runtime_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            predicted = restore_mask(probability, transform, threshold=args.threshold)
            output_path = mask_dir / f"{pair.key}.png"
            Image.fromarray(predicted.astype(np.uint8) * 255).save(output_path)
            row: dict[str, object] = {
                "key": pair.key,
                "family": pair.family,
                "runtime_ms": runtime_ms,
                **binary_segmentation_metrics(predicted, target),
            }
            rows.append(row)
            print(json.dumps({"progress": f"{index}/{len(pairs)}", **row}), flush=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": "page_mask_unet_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "count": len(rows),
        "threshold": args.threshold,
        **{
            f"{key}_mean": statistics.fmean(float(row[key]) for row in rows)
            for key in (
                "iou",
                "dice",
                "precision",
                "recall",
                "boundary_f1",
                "runtime_ms",
            )
        },
        "mask_subdir": "masks",
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
