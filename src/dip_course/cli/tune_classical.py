from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from dip_course.cli._common import write_json
from dip_course.enhancement.classical import demosaic_rggb
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
        description="Tune one global classical RAW baseline on a validation manifest"
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
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=(1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8),
    )
    parser.add_argument(
        "--scale-multipliers",
        nargs="+",
        type=float,
        default=(0.75, 0.875, 1.0, 1.125, 1.25),
    )
    return parser


def downsample_mean(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 0:
        raise ValueError("downsample factor must be positive")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB input, got {image.shape}")
    height, width, channels = image.shape
    if height % factor or width % factor:
        raise ValueError(
            f"Image shape {image.shape} is not divisible by factor {factor}"
        )
    return image.reshape(
        height // factor,
        factor,
        width // factor,
        factor,
        channels,
    ).mean(axis=(1, 3), dtype=np.float32)


def estimate_gamma_coefficients(
    linear_images: list[np.ndarray],
    targets: list[np.ndarray],
    gamma: float,
) -> np.ndarray:
    numerator = np.zeros(3, dtype=np.float64)
    denominator = np.zeros(3, dtype=np.float64)
    inverse_gamma = 1.0 / gamma
    for linear, target in zip(linear_images, targets, strict=True):
        base = np.power(np.clip(linear, 0.0, 1.0), inverse_gamma)
        numerator += np.sum(base * target, axis=(0, 1), dtype=np.float64)
        denominator += np.sum(base * base, axis=(0, 1), dtype=np.float64)
    return numerator / np.maximum(denominator, 1e-12)


def main() -> None:
    args = build_parser().parse_args()
    if any(value <= 0 for value in (*args.gammas, *args.scale_multipliers)):
        raise ValueError("gammas and scale multipliers must all be positive")

    spec = RawSpec(
        width=args.width,
        height=args.height,
        black_level=args.black_level,
        white_level=args.white_level,
    )
    records = read_pair_manifest(args.data_root, args.manifest)
    if args.limit > 0:
        records = records[: args.limit]

    linear_images: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index, record in enumerate(records, 1):
        normalized = normalize_raw(read_raw_uint16(record.low_raw, spec), spec)
        linear = demosaic_rggb(normalized)
        target = load_rgb(record.gt_rgb)
        linear_images.append(downsample_mean(linear, args.downsample_factor))
        targets.append(downsample_mean(target, args.downsample_factor))
        print(f"Loaded validation pair {index}/{len(records)}: {record.key}")

    candidates: list[dict[str, object]] = []
    for gamma in args.gammas:
        coefficients = estimate_gamma_coefficients(linear_images, targets, float(gamma))
        combined_scales = np.power(coefficients, float(gamma))
        for multiplier in args.scale_multipliers:
            exposure = float(combined_scales[1] * multiplier)
            white_balance = combined_scales / combined_scales[1]
            inverse_gamma = 1.0 / float(gamma)
            coefficient_multiplier = float(multiplier) ** inverse_gamma
            values = []
            for linear, target in zip(linear_images, targets, strict=True):
                base = np.power(np.clip(linear, 0.0, 1.0), inverse_gamma)
                prediction = np.clip(
                    base * coefficients.reshape(1, 1, 3) * coefficient_multiplier,
                    0.0,
                    1.0,
                ).astype(np.float32, copy=False)
                values.append(psnr_rgb(prediction, target))
            candidates.append(
                {
                    "gamma": float(gamma),
                    "scale_multiplier": float(multiplier),
                    "exposure": exposure,
                    "white_balance": [float(value) for value in white_balance],
                    "psnr_mean_downsampled": statistics.fmean(values),
                    "psnr_std_downsampled": statistics.pstdev(values),
                }
            )

    candidates.sort(key=lambda item: float(item["psnr_mean_downsampled"]), reverse=True)
    best = candidates[0]
    report = {
        "method": "validation_global_least_squares_then_grid_search",
        "manifest": str(Path(args.manifest).resolve()),
        "count": len(records),
        "downsample_factor": args.downsample_factor,
        "selection_metric": "mean RGB PSNR on downsampled validation images",
        "best": best,
        "top_candidates": candidates[:10],
        "search": {
            "gammas": [float(value) for value in args.gammas],
            "scale_multipliers": [float(value) for value in args.scale_multipliers],
        },
        "finite": all(
            math.isfinite(float(candidate["psnr_mean_downsampled"]))
            for candidate in candidates
        ),
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
