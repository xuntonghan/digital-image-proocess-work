from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from dip_course.cli._common import write_json
from dip_course.geocorrection.dataset import discover_pairs
from dip_course.geocorrection.metrics import boundary_linearity, text_line_metrics
from dip_course.geocorrection.rectify import rectify_document
from dip_course.geocorrection.visualize import build_geometry_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the mask-guided classical GeoCorrection baseline"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-side", type=int, default=1400)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--keys",
        help="Optional comma-separated image stems for a focused run",
    )
    parser.add_argument(
        "--visualize-keys",
        default=(
            "150,20260401_103937,IMG_20260326_105848,"
            "dewarp20260728_curved_029,dewarp20260728_fold_020,"
            "dewarp20260728_perspective_019,dewarp20260728_rotate_018"
        ),
    )
    return parser


def _prefixed_metrics(
    prefix: str, image: np.ndarray, mask: np.ndarray
) -> dict[str, object]:
    values: dict[str, object] = {}
    values.update(boundary_linearity(mask))
    values.update(text_line_metrics(image, mask))
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _numeric_mean(rows: list[dict[str, object]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and np.isfinite(float(row[key]))
    ]
    return statistics.fmean(values) if values else None


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = (
        "boundary_rmse_normalized",
        "boundary_p95_normalized",
        "page_fill_ratio",
        "text_line_skew_degrees",
        "hough_line_strength",
    )
    summary: dict[str, object] = {"count": len(rows)}
    for stage in ("input", "homography", "corrected"):
        for metric in metrics:
            summary[f"{stage}_{metric}_mean"] = _numeric_mean(
                rows, f"{stage}_{metric}"
            )
    summary["runtime_ms_mean"] = _numeric_mean(rows, "runtime_ms")
    return summary


def main() -> None:
    args = build_parser().parse_args()
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
    corrected_dir = output_dir / "corrected"
    mask_dir = output_dir / "corrected_masks"
    homography_dir = output_dir / "homography_selected"
    corrected_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    homography_dir.mkdir(parents=True, exist_ok=True)
    selected = {
        key.strip() for key in args.visualize_keys.split(",") if key.strip()
    }
    rows: list[dict[str, object]] = []
    originals: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    homographies: dict[str, Path] = {}
    corrected_paths: dict[str, Path] = {}

    for index, pair in enumerate(pairs, start=1):
        with Image.open(pair.image_path) as source_image:
            image = source_image.convert("RGB").copy()
        with Image.open(pair.mask_path) as source_mask:
            mask = source_mask.convert("L").copy()
        started = time.perf_counter_ns()
        result = rectify_document(image, mask, max_side=args.max_side)
        runtime_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        corrected_path = corrected_dir / f"{pair.key}.jpg"
        corrected_mask_path = mask_dir / f"{pair.key}.png"
        Image.fromarray(result.corrected_image).save(
            corrected_path, quality=95, subsampling=0
        )
        Image.fromarray(result.corrected_mask.astype(np.uint8) * 255).save(
            corrected_mask_path
        )
        originals[pair.key] = pair.image_path
        masks[pair.key] = pair.mask_path
        corrected_paths[pair.key] = corrected_path
        if pair.key in selected:
            homography_path = homography_dir / f"{pair.key}.jpg"
            Image.fromarray(result.homography_image).save(
                homography_path, quality=95, subsampling=0
            )
            homographies[pair.key] = homography_path
        row: dict[str, object] = {
            "key": pair.key,
            "family": pair.family,
            "source": pair.image_path.relative_to(root).as_posix(),
            "mask": pair.mask_path.relative_to(root).as_posix(),
            "corrected": corrected_path.relative_to(output_dir).as_posix(),
            "input_width": result.input_image.shape[1],
            "input_height": result.input_image.shape[0],
            "output_width": result.corrected_image.shape[1],
            "output_height": result.corrected_image.shape[0],
            "orientation_quarter_turns": result.orientation_quarter_turns,
            "runtime_ms": runtime_ms,
            **_prefixed_metrics("input", result.input_image, result.input_mask),
            **_prefixed_metrics(
                "homography", result.homography_image, result.homography_mask
            ),
            **_prefixed_metrics(
                "corrected", result.corrected_image, result.corrected_mask
            ),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(pairs)}",
                    "key": pair.key,
                    "runtime_ms": runtime_ms,
                    "boundary_before": row["input_boundary_rmse_normalized"],
                    "boundary_after": row["corrected_boundary_rmse_normalized"],
                    "skew_before": row["input_text_line_skew_degrees"],
                    "skew_after": row["corrected_text_line_skew_degrees"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    _write_csv(output_dir / "metrics.csv", rows)
    by_family: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(row)
    visualization_keys = [key for key in selected if key in corrected_paths]
    visualization_keys.sort()
    if visualization_keys:
        build_geometry_grid(
            originals=originals,
            masks=masks,
            homographies=homographies,
            corrected=corrected_paths,
            keys=visualization_keys,
            output=output_dir / "comparison.png",
        )
    summary = {
        "method": "mask_guided_projective_plus_boundary_strip_dewarp",
        "data_root": str(root),
        "input_count": len(rows),
        "mask_usage": "dataset-provided oracle page-region masks",
        "flat_ground_truth_available": False,
        "ocr_status": "not_run_in_classical_local_baseline",
        "max_side": args.max_side,
        "metric_note": (
            "No PSNR/SSIM is reported because the dataset has no flat ground truth. "
            "Boundary and Hough text-line metrics are no-reference diagnostics."
        ),
        "overall": _aggregate(rows),
        "by_family": {
            family: _aggregate(family_rows)
            for family, family_rows in sorted(by_family.items())
        },
        "visualization": "comparison.png" if visualization_keys else None,
        "visualization_keys": visualization_keys,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
