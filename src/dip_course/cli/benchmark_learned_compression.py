from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from dip_course.cli._common import write_json
from dip_course.compression.visualize import (
    build_compression_grid,
    build_rate_distortion_plot,
)
from dip_course.metrics import psnr_rgb, ssim_rgb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark an official pretrained CompressAI model on Kodak"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--architecture",
        choices=(
            "bmshj2018-factorized",
            "bmshj2018-hyperprior",
            "mbt2018-mean",
            "mbt2018",
            "cheng2020-anchor",
            "cheng2020-attn",
        ),
        default="cheng2020-anchor",
    )
    parser.add_argument("--qualities", type=int, nargs="+", default=(1, 2, 3, 4, 5, 6))
    parser.add_argument("--metric", choices=("mse", "ms-ssim"), default="mse")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--expected-count", type=int, default=24)
    parser.add_argument("--visualize-keys", default="01,04,13,23,24")
    parser.add_argument("--baseline-summary")
    return parser


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _payload_bytes(strings: list[list[bytes]]) -> int:
    return sum(len(value) for group in strings for value in group)


def _load_model(architecture: str, quality: int, metric: str) -> tuple[Any, str]:
    try:
        import compressai
        from compressai import zoo
    except ImportError as exc:
        raise RuntimeError(
            "CompressAI is required; install the project compression-ml extra"
        ) from exc
    constructors = {
        "bmshj2018-factorized": zoo.bmshj2018_factorized,
        "bmshj2018-hyperprior": zoo.bmshj2018_hyperprior,
        "mbt2018-mean": zoo.mbt2018_mean,
        "mbt2018": zoo.mbt2018,
        "cheng2020-anchor": zoo.cheng2020_anchor,
        "cheng2020-attn": zoo.cheng2020_attn,
    }
    model = constructors[architecture](quality=quality, metric=metric, pretrained=True)
    return model, compressai.__version__


def _load_baseline(path: str | None) -> list[dict[str, object]]:
    if path is None:
        return []
    return json.loads(Path(path).read_text(encoding="utf-8"))["qualities"]


def main() -> None:
    args = build_parser().parse_args()
    if args.repeats <= 0 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative")
    if len(set(args.qualities)) != len(args.qualities):
        raise ValueError("qualities must not contain duplicates")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    data_dir = Path(args.data_dir)
    image_paths = sorted(data_dir.glob("*.png"))
    if args.expected_count > 0 and len(image_paths) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} images, found {len(image_paths)}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    decoded_by_quality: dict[int, dict[str, Path]] = {}
    original_paths = {path.stem: path for path in image_paths}
    compressai_version = "unknown"

    for quality in sorted(args.qualities):
        model, compressai_version = _load_model(args.architecture, quality, args.metric)
        model = model.to(device).eval()
        decoded_by_quality[quality] = {}
        for image_path in image_paths:
            with Image.open(image_path) as source:
                rgb = source.convert("RGB")
                array = np.asarray(rgb, dtype=np.float32) / 255.0
            height, width = array.shape[:2]
            tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())[None].to(device)
            pad_right = (64 - width % 64) % 64
            pad_bottom = (64 - height % 64) % 64
            padded = F.pad(tensor, (0, pad_right, 0, pad_bottom))
            with torch.inference_mode():
                for _ in range(args.warmup):
                    encoded = model.compress(padded)
                    model.decompress(encoded["strings"], encoded["shape"])
                encode_times: list[float] = []
                for _ in range(args.repeats):
                    _synchronize(device)
                    started = time.perf_counter_ns()
                    encoded = model.compress(padded)
                    _synchronize(device)
                    encode_times.append((time.perf_counter_ns() - started) / 1_000_000.0)
                decode_times: list[float] = []
                reconstruction = None
                for _ in range(args.repeats):
                    _synchronize(device)
                    started = time.perf_counter_ns()
                    decoded = model.decompress(encoded["strings"], encoded["shape"])
                    _synchronize(device)
                    decode_times.append((time.perf_counter_ns() - started) / 1_000_000.0)
                    reconstruction = decoded["x_hat"]
            assert reconstruction is not None
            prediction = (
                reconstruction[0, :, :height, :width]
                .clamp(0.0, 1.0)
                .float()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
            )
            payload_bytes = _payload_bytes(encoded["strings"])
            decoded_path = output_dir / "decoded" / f"q{quality}" / image_path.name
            decoded_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.rint(prediction * 255.0).astype(np.uint8)).save(decoded_path)
            decoded_by_quality[quality][image_path.stem] = decoded_path
            row: dict[str, object] = {
                "key": image_path.stem,
                "quality": quality,
                "architecture": args.architecture,
                "metric": args.metric,
                "width": width,
                "height": height,
                "compressed_bytes": payload_bytes,
                "bits_per_pixel": payload_bytes * 8.0 / (width * height),
                "psnr": psnr_rgb(prediction, array),
                "ssim": ssim_rgb(prediction, array),
                "encode_ms": statistics.median(encode_times),
                "decode_ms": statistics.median(decode_times),
                "decoded": decoded_path.relative_to(output_dir).as_posix(),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries: list[dict[str, object]] = []
    for quality in sorted(args.qualities):
        selected = [row for row in rows if row["quality"] == quality]
        summaries.append(
            {
                "quality": quality,
                "count": len(selected),
                "compressed_bytes_total": sum(int(row["compressed_bytes"]) for row in selected),
                **{
                    f"{name}_mean": statistics.fmean(float(row[name]) for row in selected)
                    for name in ("bits_per_pixel", "psnr", "ssim", "encode_ms", "decode_ms")
                },
            }
        )
    keys = [key.strip() for key in args.visualize_keys.split(",") if key.strip()]
    build_compression_grid(
        originals=original_paths,
        decoded_by_quality=decoded_by_quality,
        keys=keys,
        output=output_dir / "comparison.png",
        codec_label=args.architecture,
    )
    baseline = _load_baseline(args.baseline_summary)
    rate_distortion = None
    if baseline:
        build_rate_distortion_plot(
            series={"libjpeg": baseline, args.architecture: summaries},
            output=output_dir / "rate_distortion.png",
        )
        rate_distortion = "rate_distortion.png"
    manifest = hashlib.sha256()
    for path in image_paths:
        manifest.update(hashlib.sha256(path.read_bytes()).digest())
    summary = {
        "method": "pretrained_learned_image_compression",
        "architecture": args.architecture,
        "optimization_metric": args.metric,
        "pretrained": True,
        "kodak_used_for_training": False,
        "input_count": len(image_paths),
        "input_manifest_sha256": manifest.hexdigest(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "compressai": compressai_version,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "timing_scope": "actual entropy-coded compress/decompress calls",
        "visualization": "comparison.png",
        "rate_distortion_plot": rate_distortion,
        "qualities": summaries,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
