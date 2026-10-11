from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from dip_course.cli._common import file_sha256, write_json
from dip_course.geocorrection.dataset import discover_pairs
from dip_course.geocorrection.metrics import boundary_linearity, text_line_metrics
from dip_course.geocorrection.rectify import needs_quarter_turn
from dip_course.geocorrection.uvdoc import (
    crop_to_foreground,
    crop_to_page_mask,
    uvdoc_result_to_rgb,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pretrained Hugging Face UVDoc on GeoCorrection"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="PaddlePaddle/UVDoc_safetensors")
    parser.add_argument("--model-revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--keys")
    parser.add_argument("--mask-margin", type=float, default=0.02)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--hf-endpoint")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    return parser


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _prefixed_metrics(
    prefix: str, image: np.ndarray, mask: np.ndarray, *, max_side: int = 1200
) -> dict[str, object]:
    if max(image.shape[:2]) > max_side:
        scale = max_side / max(image.shape[:2])
        size = (
            max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[0] * scale)),
        )
        image = np.asarray(
            Image.fromarray(image).resize(size, Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        mask = (
            np.asarray(
                Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).resize(
                    size, Image.Resampling.NEAREST
                )
            )
            > 127
        )
    values: dict[str, object] = {}
    values.update(boundary_linearity(mask))
    values.update(text_line_metrics(image, mask))
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _mean(rows: list[dict[str, object]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and np.isfinite(float(row[key]))
    ]
    return statistics.fmean(values) if values else None


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"count": len(rows)}
    for stage in ("input", "uvdoc"):
        for metric in (
            "boundary_rmse_normalized",
            "boundary_p95_normalized",
            "page_fill_ratio",
            "text_line_skew_degrees",
            "hough_line_strength",
        ):
            result[f"{stage}_{metric}_mean"] = _mean(rows, f"{stage}_{metric}")
    for metric in (
        "preprocess_ms",
        "forward_ms",
        "postprocess_ms",
        "runtime_ms",
    ):
        result[f"{metric}_mean"] = _mean(rows, metric)
    return result


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> None:
    args = build_parser().parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    import torch
    from transformers import AutoImageProcessor, AutoModel
    from transformers.utils.hub import cached_file

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    root = Path(args.data_root)
    pairs = discover_pairs(root)
    if args.expected_count > 0 and len(pairs) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} pairs, found {len(pairs)}")
    if args.keys:
        requested = {key.strip() for key in args.keys.split(",") if key.strip()}
        known = {pair.key for pair in pairs}
        missing = requested - known
        if missing:
            raise ValueError(f"Unknown GeoCorrection keys: {sorted(missing)}")
        pairs = [pair for pair in pairs if pair.key in requested]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        pairs = pairs[: args.limit]

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "rectified"
    mask_dir = output_dir / "rectified_masks"
    grid_dir = output_dir / "grids"
    for directory in (image_dir, mask_dir, grid_dir):
        directory.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    ).to(args.device)
    model.eval()
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.warmup:
        height = int(processor.size["height"])
        width = int(processor.size["width"])
        dummy = torch.zeros((1, 3, height, width), device=args.device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(pixel_values=dummy)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
    weight_path = cached_file(
        args.model_id,
        "model.safetensors",
        revision=args.model_revision,
        local_files_only=True,
    )
    weight_sha256 = file_sha256(weight_path) if weight_path else None
    rows: list[dict[str, object]] = []
    for index, pair in enumerate(pairs, start=1):
        started = time.perf_counter_ns()
        with Image.open(pair.image_path) as source:
            image = source.convert("RGB").copy()
        with Image.open(pair.mask_path) as source:
            mask = source.convert("L").copy()
        crop = crop_to_page_mask(image, mask, margin_fraction=args.mask_margin)
        input_rgb = np.asarray(crop.image, dtype=np.uint8)
        preprocess_started = time.perf_counter_ns()
        inputs = processor(images=crop.image, return_tensors="pt").to(args.device)
        preprocess_ms = (time.perf_counter_ns() - preprocess_started) / 1_000_000.0

        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        forward_started = time.perf_counter_ns()
        with torch.inference_mode():
            outputs = model(**inputs)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        forward_ms = (time.perf_counter_ns() - forward_started) / 1_000_000.0

        postprocess_started = time.perf_counter_ns()
        image_result = processor.post_process_document_rectification(
            outputs.last_hidden_state, inputs["original_images"]
        )[0]
        rectified_rgb = uvdoc_result_to_rgb(image_result)
        mask_tensor = torch.from_numpy(crop.mask.astype(np.float32))[None].repeat(3, 1, 1)
        mask_result = processor.post_process_document_rectification(
            outputs.last_hidden_state, [mask_tensor], scale=255.0
        )[0]
        rectified_mask = uvdoc_result_to_rgb(mask_result)[..., 0] > 127
        rectified_rgb, rectified_mask = crop_to_foreground(
            rectified_rgb, rectified_mask
        )
        rectified_rgb[~rectified_mask] = 255
        orientation_quarter_turns = 0
        if needs_quarter_turn(rectified_rgb, rectified_mask):
            rectified_rgb = np.rot90(rectified_rgb, k=1).copy()
            rectified_mask = np.rot90(rectified_mask, k=1).copy()
            orientation_quarter_turns = 1
        postprocess_ms = (time.perf_counter_ns() - postprocess_started) / 1_000_000.0

        image_path = image_dir / f"{pair.key}.jpg"
        mask_path = mask_dir / f"{pair.key}.png"
        grid_path = grid_dir / f"{pair.key}.npy"
        Image.fromarray(rectified_rgb).save(
            image_path, quality=args.jpeg_quality, subsampling=0
        )
        Image.fromarray(rectified_mask.astype(np.uint8) * 255).save(mask_path)
        np.save(
            grid_path,
            outputs.last_hidden_state[0].detach().cpu().numpy().astype(np.float32),
            allow_pickle=False,
        )
        runtime_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        row: dict[str, object] = {
            "key": pair.key,
            "family": pair.family,
            "source": pair.image_path.relative_to(root).as_posix(),
            "source_mask": pair.mask_path.relative_to(root).as_posix(),
            "rectified": image_path.relative_to(output_dir).as_posix(),
            "rectified_mask": mask_path.relative_to(output_dir).as_posix(),
            "grid": grid_path.relative_to(output_dir).as_posix(),
            "crop_box_xyxy": json.dumps(crop.box_xyxy),
            "input_width": input_rgb.shape[1],
            "input_height": input_rgb.shape[0],
            "output_width": rectified_rgb.shape[1],
            "output_height": rectified_rgb.shape[0],
            "orientation_quarter_turns": orientation_quarter_turns,
            "preprocess_ms": preprocess_ms,
            "forward_ms": forward_ms,
            "postprocess_ms": postprocess_ms,
            "runtime_ms": runtime_ms,
            **_prefixed_metrics("input", input_rgb, crop.mask),
            **_prefixed_metrics("uvdoc", rectified_rgb, rectified_mask),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(pairs)}",
                    "key": pair.key,
                    "forward_ms": forward_ms,
                    "runtime_ms": runtime_ms,
                    "skew_before": row["input_text_line_skew_degrees"],
                    "skew_after": row["uvdoc_text_line_skew_degrees"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    _write_csv(output_dir / "metrics.csv", rows)
    by_family: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(row)
    summary = {
        "method": "pretrained_uvdoc_mask_guided_crop",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_weight_sha256": weight_sha256,
        "transformers_version": _package_version("transformers"),
        "torch_version": _package_version("torch"),
        "device": args.device,
        "warmup": args.warmup,
        "input_count": len(rows),
        "mask_usage": "dataset-provided oracle page mask for crop and output masking",
        "mask_margin": args.mask_margin,
        "flat_ground_truth_available": False,
        "metric_max_side": 1200,
        "metric_note": (
            "No PSNR/SSIM is reported because the dataset has no flat ground truth. "
            "Boundary and Hough text-line metrics are no-reference diagnostics."
        ),
        "overall": _aggregate(rows),
        "by_family": {
            family: _aggregate(family_rows)
            for family, family_rows in sorted(by_family.items())
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
