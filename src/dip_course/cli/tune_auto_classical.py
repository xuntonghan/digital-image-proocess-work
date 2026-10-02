from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from dip_course.cli._common import write_json
from dip_course.cli.tune_classical import downsample_mean
from dip_course.enhancement.classical import (
    demosaic_rggb,
    estimate_gray_world_white_balance,
    tone_map_rgb,
)
from dip_course.io import (
    RawSpec,
    load_rgb,
    normalize_raw,
    read_pair_manifest,
    read_raw_uint16,
)
from dip_course.metrics import psnr_rgb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune target percentile exposure on a validation manifest"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--black-level", type=int, default=512)
    parser.add_argument("--white-level", type=int, default=16383)
    parser.add_argument("--downsample-factor", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gammas", nargs="+", type=float, default=(1.8, 2.0, 2.2))
    parser.add_argument(
        "--percentiles", nargs="+", type=float, default=(98.0, 99.0, 99.5, 99.9)
    )
    parser.add_argument("--targets", nargs="+", type=float, default=(0.8, 0.9, 0.98))
    parser.add_argument("--white-balance-power", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = RawSpec(
        width=args.width,
        height=args.height,
        black_level=args.black_level,
        white_level=args.white_level,
    )
    records = read_pair_manifest(args.data_root, args.manifest)
    if args.limit > 0:
        records = records[: args.limit]

    linear_images = []
    targets = []
    for index, record in enumerate(records, 1):
        normalized = normalize_raw(read_raw_uint16(record.low_raw, spec), spec)
        linear_images.append(
            downsample_mean(demosaic_rggb(normalized), args.downsample_factor)
        )
        targets.append(downsample_mean(load_rgb(record.gt_rgb), args.downsample_factor))
        print(f"Loaded validation pair {index}/{len(records)}: {record.key}")

    candidates: list[dict[str, object]] = []
    for gamma in args.gammas:
        for percentile in args.percentiles:
            for target_value in args.targets:
                values = []
                exposures = []
                for linear, target in zip(linear_images, targets, strict=True):
                    white_balance = estimate_gray_world_white_balance(
                        linear, power=args.white_balance_power
                    )
                    balanced = linear * white_balance
                    luminance = (
                        0.2126 * balanced[..., 0]
                        + 0.7152 * balanced[..., 1]
                        + 0.0722 * balanced[..., 2]
                    )
                    observed = float(np.percentile(luminance, percentile))
                    exposure = float(target_value**gamma / max(observed, 1e-8))
                    prediction = tone_map_rgb(
                        linear,
                        exposure=exposure,
                        gamma=gamma,
                        white_balance=white_balance,
                    )
                    values.append(psnr_rgb(prediction, target))
                    exposures.append(exposure)
                candidates.append(
                    {
                        "gamma": float(gamma),
                        "percentile": float(percentile),
                        "target": float(target_value),
                        "white_balance_power": args.white_balance_power,
                        "psnr_mean_downsampled": statistics.fmean(values),
                        "psnr_std_downsampled": statistics.pstdev(values),
                        "exposure_mean": statistics.fmean(exposures),
                    }
                )

    candidates.sort(key=lambda item: float(item["psnr_mean_downsampled"]), reverse=True)
    report = {
        "method": "validation_grid_search_auto_gray_world",
        "manifest": str(Path(args.manifest).resolve()),
        "count": len(records),
        "downsample_factor": args.downsample_factor,
        "selection_metric": "mean RGB PSNR on downsampled validation images",
        "best": candidates[0],
        "top_candidates": candidates[:10],
        "search": {
            "gammas": list(args.gammas),
            "percentiles": list(args.percentiles),
            "targets": list(args.targets),
            "white_balance_power": args.white_balance_power,
        },
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
