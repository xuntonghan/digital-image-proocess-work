from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from dip_course.cli._common import choose_torch_device, write_json
from dip_course.enhancement.model import RawToRGBUNet
from dip_course.enhancement.tiling import predict_tiled
from dip_course.io import (
    RawSpec,
    load_packed_raw,
    load_rgb,
    read_pair_manifest,
    save_rgb,
)
from dip_course.metrics import (
    LPIPSMetric,
    color_fidelity_metrics,
    psnr_rgb,
    ssim_rgb,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tiled RAW-to-RGB model inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--black-level", type=int, default=512)
    parser.add_argument("--white-level", type=int, default=16383)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lpips", action="store_true")
    return parser


def finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else None


def main() -> None:
    args = build_parser().parse_args()
    device = choose_torch_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise TypeError("Checkpoint is missing a dictionary model_config")
    model = RawToRGBUNet(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

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
        prediction_tensor = predict_tiled(
            model, tensor, tile_size=args.tile_size, overlap=args.overlap
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        prediction = (
            prediction_tensor.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
            .astype(np.float32)
        )
        target = load_rgb(record.gt_rgb)
        output_name = f"{index:04d}_{record.key}.png"
        save_rgb(image_dir / output_name, prediction)
        row: dict[str, object] = {
            "index": index,
            "key": record.key,
            "output": str(Path("images") / output_name),
            "psnr": psnr_rgb(prediction, target),
            "ssim": ssim_rgb(prediction, target),
            "inference_seconds": elapsed,
        }
        row.update(color_fidelity_metrics(prediction, target))
        if lpips_metric is not None:
            row["lpips"] = lpips_metric(prediction, target)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    fieldnames = list(rows[0])
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": "raw_to_rgb_unet",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "count": len(rows),
        "device": str(device),
        "psnr_mean": finite_mean([float(row["psnr"]) for row in rows]),
        "ssim_mean": finite_mean([float(row["ssim"]) for row in rows]),
        "delta_e2000_mean": finite_mean(
            [float(row["delta_e2000_mean"]) for row in rows]
        ),
        "delta_e2000_p95_mean": finite_mean(
            [float(row["delta_e2000_p95"]) for row in rows]
        ),
        "lightness_mae_mean": finite_mean(
            [float(row["lightness_mae"]) for row in rows]
        ),
        "saturation_mae_mean": finite_mean(
            [float(row["saturation_mae"]) for row in rows]
        ),
        "hue_error_degrees_mean": finite_mean(
            [float(row["hue_error_degrees"]) for row in rows]
        ),
        "lpips_mean": (
            finite_mean([float(row["lpips"]) for row in rows]) if lpips_metric else None
        ),
        "inference_seconds_mean": finite_mean(
            [float(row["inference_seconds"]) for row in rows]
        ),
        "tile_size": args.tile_size,
        "overlap": args.overlap,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
