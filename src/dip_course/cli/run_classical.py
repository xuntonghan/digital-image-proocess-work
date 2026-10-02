from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from dip_course.cli._common import write_json
from dip_course.enhancement.classical import auto_classical_enhance, classical_enhance
from dip_course.io import (
    RawSpec,
    load_rgb,
    normalize_raw,
    read_pair_manifest,
    read_raw_uint16,
    save_rgb,
)
from dip_course.metrics import LPIPSMetric, psnr_rgb, ssim_rgb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the transparent classical RAW baseline"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--black-level", type=int, default=512)
    parser.add_argument("--white-level", type=int, default=16383)
    parser.add_argument("--exposure", type=float, default=8.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--white-balance", nargs=3, type=float, default=(2.0, 1.0, 1.6))
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--auto-percentile", type=float, default=99.0)
    parser.add_argument("--auto-target", type=float, default=0.9)
    parser.add_argument("--auto-white-balance-power", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else None


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    spec = RawSpec(
        width=args.width,
        height=args.height,
        black_level=args.black_level,
        white_level=args.white_level,
    )
    records = read_pair_manifest(args.data_root, args.manifest)
    if args.limit > 0:
        records = records[: args.limit]
    lpips_metric = LPIPSMetric(args.device) if args.lpips else None
    rows: list[dict[str, object]] = []

    for index, record in enumerate(records):
        raw = normalize_raw(read_raw_uint16(record.low_raw, spec), spec)
        if args.auto:
            prediction, estimated = auto_classical_enhance(
                raw,
                gamma=args.gamma,
                percentile=args.auto_percentile,
                target=args.auto_target,
                white_balance_power=args.auto_white_balance_power,
            )
        else:
            prediction = classical_enhance(
                raw,
                exposure=args.exposure,
                gamma=args.gamma,
                white_balance=tuple(args.white_balance),
            )
            estimated = {
                "exposure": args.exposure,
                "gamma": args.gamma,
                "white_balance": list(args.white_balance),
            }
        target = load_rgb(record.gt_rgb)
        output_name = f"{index:04d}_{record.key}.png"
        save_rgb(image_dir / output_name, prediction)
        row: dict[str, object] = {
            "index": index,
            "key": record.key,
            "output": str(Path("images") / output_name),
            "psnr": psnr_rgb(prediction, target),
            "ssim": ssim_rgb(prediction, target),
            "exposure": estimated["exposure"],
            "white_balance_r": estimated["white_balance"][0],
            "white_balance_g": estimated["white_balance"][1],
            "white_balance_b": estimated["white_balance"][2],
        }
        if lpips_metric is not None:
            row["lpips"] = lpips_metric(prediction, target)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    fieldnames = list(rows[0])
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "method": (
            "classical_rggb_auto_gray_world" if args.auto else "classical_rggb_fixed"
        ),
        "count": len(rows),
        "psnr_mean": finite_mean([float(row["psnr"]) for row in rows]),
        "ssim_mean": finite_mean([float(row["ssim"]) for row in rows]),
        "lpips_mean": (
            finite_mean([float(row["lpips"]) for row in rows]) if lpips_metric else None
        ),
        "parameters": {
            "gamma": args.gamma,
            "exposure": None if args.auto else args.exposure,
            "white_balance": None if args.auto else list(args.white_balance),
            "auto_percentile": args.auto_percentile if args.auto else None,
            "auto_target": args.auto_target if args.auto else None,
            "auto_white_balance_power": (
                args.auto_white_balance_power if args.auto else None
            ),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
