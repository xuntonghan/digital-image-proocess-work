from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import __version__ as pillow_version

from dip_course.cli._common import write_json
from dip_course.compression.dct_codec import DCTCodecSettings, benchmark_dct
from dip_course.compression.visualize import (
    build_compression_grid,
    build_rate_distortion_plot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the self-contained JPEG-like DCTC codec on Kodak"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--qualities",
        type=int,
        nargs="+",
        default=(30, 50, 70, 90),
    )
    parser.add_argument("--subsampling", choices=("444", "422", "420"), default="420")
    parser.add_argument("--entropy-level", type=int, default=9)
    parser.add_argument(
        "--stream-format",
        choices=("rle", "dense", "trimmed", "sparse"),
        default="sparse",
        help="v1 RLE or vectorized v2 coefficient stream",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--expected-count", type=int, default=24)
    parser.add_argument(
        "--visualize-keys",
        default="01,04,13,23,24",
        help="Comma-separated Kodak image stems for the comparison grid",
    )
    parser.add_argument(
        "--baseline-summary",
        help="Optional libjpeg summary.json used to calculate paired deltas",
    )
    return parser


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _input_manifest_sha256(paths: list[Path]) -> str:
    aggregate = hashlib.sha256()
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        aggregate.update(f"{digest}  {path.name}\n".encode())
    return aggregate.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_baseline(path: str | None) -> dict[int, dict[str, object]]:
    if path is None:
        return {}
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(row["quality"]): row for row in document["qualities"]}


def main() -> None:
    args = build_parser().parse_args()
    if args.repeats <= 0 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative")
    if len(set(args.qualities)) != len(args.qualities):
        raise ValueError("qualities must not contain duplicates")
    settings = [
        DCTCodecSettings(
            quality=quality,
            subsampling=args.subsampling,
            entropy_level=args.entropy_level,
            stream_format=args.stream_format,
        )
        for quality in sorted(args.qualities)
    ]
    data_dir = Path(args.data_dir)
    images = sorted(data_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG images found in {data_dir}")
    if args.expected_count > 0 and len(images) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} PNG images, found {len(images)}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    original_paths: dict[str, Path] = {}
    decoded_paths: dict[int, dict[str, Path]] = {
        setting.quality: {} for setting in settings
    }
    for image_path in images:
        read_started = time.perf_counter_ns()
        with Image.open(image_path) as source:
            if source.mode != "RGB":
                raise ValueError(
                    f"Expected true-color RGB input at {image_path}, got {source.mode}"
                )
            image = source.copy()
        read_ms = (time.perf_counter_ns() - read_started) / 1_000_000.0
        key = image_path.stem
        original_paths[key] = image_path

        for setting in settings:
            result, payload, decoded = benchmark_dct(
                image,
                source_file_bytes=image_path.stat().st_size,
                settings=setting,
                repeats=args.repeats,
                warmup=args.warmup,
            )
            encoded_path = (
                output_dir / "encoded" / f"q{setting.quality}" / f"{key}.dctc"
            )
            decoded_path = output_dir / "decoded" / f"q{setting.quality}" / f"{key}.png"
            encoded_path.parent.mkdir(parents=True, exist_ok=True)
            decoded_path.parent.mkdir(parents=True, exist_ok=True)
            encoded_path.write_bytes(payload)
            decoded.save(decoded_path)
            decoded_paths[setting.quality][key] = decoded_path
            row: dict[str, object] = {
                "key": key,
                "quality": setting.quality,
                "subsampling": setting.subsampling,
                "entropy_level": setting.entropy_level,
                "stream_format": setting.stream_format,
                "source": str(image_path.relative_to(data_dir)),
                "encoded": str(encoded_path.relative_to(output_dir)),
                "decoded": str(decoded_path.relative_to(output_dir)),
                "read_ms": read_ms,
                **asdict(result),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    _write_csv(output_dir / "metrics.csv", rows)
    baseline = _load_baseline(args.baseline_summary)
    summaries: list[dict[str, object]] = []
    for setting in settings:
        quality_rows = [row for row in rows if row["quality"] == setting.quality]
        source_total = sum(int(row["source_file_bytes"]) for row in quality_rows)
        raw_total = sum(int(row["raw_rgb_bytes"]) for row in quality_rows)
        compressed_total = sum(int(row["compressed_bytes"]) for row in quality_rows)
        summary: dict[str, object] = {
            "quality": setting.quality,
            "subsampling": setting.subsampling,
            "entropy_level": setting.entropy_level,
            "stream_format": setting.stream_format,
            "count": len(quality_rows),
            "source_file_bytes_total": source_total,
            "raw_rgb_bytes_total": raw_total,
            "compressed_bytes_total": compressed_total,
            "file_compression_ratio": source_total / compressed_total,
            "raw_compression_ratio": raw_total / compressed_total,
            "file_space_saving_percent": (1.0 - compressed_total / source_total)
            * 100.0,
            "bits_per_pixel_mean": _mean(quality_rows, "bits_per_pixel"),
            "psnr_mean": _mean(quality_rows, "psnr"),
            "ssim_mean": _mean(quality_rows, "ssim"),
            "read_ms_mean": _mean(quality_rows, "read_ms"),
            "encode_ms_mean": _mean(quality_rows, "encode_ms"),
            "decode_ms_mean": _mean(quality_rows, "decode_ms"),
        }
        if setting.quality in baseline:
            reference = baseline[setting.quality]
            summary["libjpeg_delta"] = {
                "compressed_bytes": compressed_total
                - int(reference["compressed_bytes_total"]),
                "bits_per_pixel": float(summary["bits_per_pixel_mean"])
                - float(reference["bits_per_pixel_mean"]),
                "psnr_db": float(summary["psnr_mean"])
                - float(reference["psnr_mean"]),
                "ssim": float(summary["ssim_mean"])
                - float(reference["ssim_mean"]),
                "encode_time_ratio": float(summary["encode_ms_mean"])
                / float(reference["encode_ms_mean"]),
                "decode_time_ratio": float(summary["decode_ms_mean"])
                / float(reference["decode_ms_mean"]),
            }
        summaries.append(summary)

    keys = [key.strip() for key in args.visualize_keys.split(",") if key.strip()]
    visualization = output_dir / "comparison.png"
    build_compression_grid(
        originals=original_paths,
        decoded_by_quality=decoded_paths,
        keys=keys,
        output=visualization,
        codec_label="DCTC",
    )
    rate_distortion_plot: str | None = None
    if baseline:
        curve_path = output_dir / "rate_distortion.png"
        build_rate_distortion_plot(
            series={
                "libjpeg": list(baseline.values()),
                f"DCTC {'v1' if args.stream_format == 'rle' else 'v2'}": summaries,
            },
            output=curve_path,
        )
        rate_distortion_plot = str(curve_path.relative_to(output_dir))
    summary = {
        "method": f"coursework_dctc_{'v1_rle' if args.stream_format == 'rle' else 'v2_' + args.stream_format}",
        "algorithm": [
            "RGB to YCbCr",
            f"chroma subsampling {args.subsampling}",
            "8x8 orthonormal DCT",
            "JPEG quality-scaled quantization",
            "zigzag scan",
            (
                "vectorized DC differential and trailing-zero-trimmed int16 packing"
                if args.stream_format == "trimmed"
                else "vectorized sparse AC positions and int16 values"
                if args.stream_format == "sparse"
                else "vectorized DC differential and dense int16 coefficient packing"
                if args.stream_format == "dense"
                else "DC differential and AC zero-run coding"
            ),
            "zlib DEFLATE entropy coding",
        ],
        "data_dir": str(data_dir),
        "input_count": len(images),
        "input_manifest_sha256": _input_manifest_sha256(images),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": pillow_version,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "timing_scope": "codec only; excludes metric computation and output-file writes",
        "compression_ratio_definitions": {
            "file": "sum(original PNG bytes) / sum(compressed DCTC bytes)",
            "raw": "sum(width * height * 3 bytes) / sum(compressed DCTC bytes)",
        },
        "baseline_summary": args.baseline_summary,
        "visualization": str(visualization.relative_to(output_dir)),
        "rate_distortion_plot": rate_distortion_plot,
        "visualization_keys": keys,
        "qualities": summaries,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
