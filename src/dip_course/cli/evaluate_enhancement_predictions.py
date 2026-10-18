from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from dip_course.cli._common import choose_torch_device, write_json
from dip_course.io import load_rgb, read_pair_manifest
from dip_course.metrics import LPIPSMetric, color_fidelity_metrics, psnr_rgb, ssim_rgb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved RGB enhancement predictions against a manifest"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="saved_enhancement_predictions")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lpips", action="store_true")
    return parser


def finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else None


def find_prediction(prediction_dir: Path, index: int, key: str) -> Path:
    exact = prediction_dir / f"{index:04d}_{key}.png"
    if exact.is_file():
        return exact
    matches = sorted(prediction_dir.glob(f"*_{key}.png"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one prediction for key {key!r}, found {len(matches)} in "
            f"{prediction_dir}"
        )
    return matches[0]


def main() -> None:
    args = build_parser().parse_args()
    prediction_dir = Path(args.prediction_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_pair_manifest(args.data_root, args.manifest)
    device = choose_torch_device(args.device)
    lpips_metric = LPIPSMetric(str(device)) if args.lpips else None
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        prediction_path = find_prediction(prediction_dir, index, record.key)
        prediction = load_rgb(prediction_path)
        target = load_rgb(record.gt_rgb)
        row: dict[str, object] = {
            "index": index,
            "key": record.key,
            "prediction": str(prediction_path.resolve()),
            "psnr": psnr_rgb(prediction, target),
            "ssim": ssim_rgb(prediction, target),
        }
        row.update(color_fidelity_metrics(prediction, target))
        if lpips_metric is not None:
            row["lpips"] = lpips_metric(prediction, target)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if not rows:
        raise RuntimeError("No predictions were evaluated")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": args.method,
        "prediction_dir": str(prediction_dir.resolve()),
        "count": len(rows),
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
            finite_mean([float(row["lpips"]) for row in rows])
            if lpips_metric is not None
            else None
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
