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

from PIL import Image

from dip_course.cli._common import write_json
from dip_course.geocorrection.dataset import GeoCorrectionPair, discover_pairs
from dip_course.geocorrection.uvdoc import crop_to_page_mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate geometry variants with a fixed local PaddleOCR model"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--keys-manifest", required=True)
    parser.add_argument("--classical-dir", required=True)
    parser.add_argument("--uvdoc-dir", required=True)
    parser.add_argument("--hybrid-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--engine", default="transformers")
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--mask-margin", type=float, default=0.02)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--model-source", default="BOS")
    return parser


def _read_keys(path: Path) -> list[str]:
    keys = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("OCR key manifest must contain unique non-empty keys")
    return keys


def _resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    rgb = image.convert("RGB")
    if max(rgb.size) <= max_side:
        return rgb
    scale = max_side / max(rgb.size)
    size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
    return rgb.resize(size, Image.Resampling.LANCZOS)


def _prepare_input_crop(
    pair: GeoCorrectionPair,
    *,
    output: Path,
    max_side: int,
    mask_margin: float,
) -> Path:
    with Image.open(pair.image_path) as source:
        image = source.convert("RGB").copy()
    with Image.open(pair.mask_path) as source:
        mask = source.convert("L").copy()
    crop = crop_to_page_mask(image, mask, margin_fraction=mask_margin)
    resized = _resize_max_side(crop.image, max_side)
    output.parent.mkdir(parents=True, exist_ok=True)
    resized.save(output, quality=95, subsampling=0)
    return output


def _count_characters(text: str) -> int:
    return sum(not character.isspace() for character in text)


def _ocr_values(result: dict[str, object], threshold: float) -> dict[str, object]:
    payload = result.get("res")
    if not isinstance(payload, dict):
        raise TypeError("Unexpected PaddleOCR result without 'res' mapping")
    texts = [str(value) for value in payload.get("rec_texts", [])]
    scores = [float(value) for value in payload.get("rec_scores", [])]
    if len(texts) != len(scores):
        raise ValueError("PaddleOCR text and score counts differ")
    character_counts = [_count_characters(text) for text in texts]
    confident = [score >= threshold for score in scores]
    return {
        "line_count": len(texts),
        "confident_line_count": sum(confident),
        "character_count": sum(character_counts),
        "confident_character_count": sum(
            count for count, keep in zip(character_counts, confident, strict=True) if keep
        ),
        "mean_confidence": statistics.fmean(scores) if scores else None,
        "confidence_sum": sum(scores),
        "confidence_weighted_characters": sum(
            count * score
            for count, score in zip(character_counts, scores, strict=True)
        ),
        "texts": texts,
        "scores": scores,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    line_count = sum(int(row["line_count"]) for row in rows)
    return {
        "image_count": len(rows),
        "line_count_total": line_count,
        "confident_line_count_total": sum(
            int(row["confident_line_count"]) for row in rows
        ),
        "character_count_total": sum(int(row["character_count"]) for row in rows),
        "confident_character_count_total": sum(
            int(row["confident_character_count"]) for row in rows
        ),
        "mean_line_confidence": (
            sum(float(row["confidence_sum"]) for row in rows) / line_count
            if line_count
            else None
        ),
        "confidence_weighted_characters_total": sum(
            float(row["confidence_weighted_characters"]) for row in rows
        ),
        "runtime_ms_mean": statistics.fmean(float(row["runtime_ms"]) for row in rows),
    }


def _comparison(
    stage_rows: list[dict[str, object]], input_rows: list[dict[str, object]]
) -> dict[str, object]:
    baseline = {str(row["key"]): row for row in input_rows}
    deltas = [
        int(row["confident_character_count"])
        - int(baseline[str(row["key"])]["confident_character_count"])
        for row in stage_rows
    ]
    return {
        "confident_character_delta_total": sum(deltas),
        "confident_character_delta_mean": statistics.fmean(deltas),
        "images_improved": sum(delta > 0 for delta in deltas),
        "images_equal": sum(delta == 0 for delta in deltas),
        "images_worse": sum(delta < 0 for delta in deltas),
    }


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", args.model_source)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    from paddleocr import PaddleOCR

    root = Path(args.data_root)
    pairs = discover_pairs(root)
    pair_by_key = {pair.key: pair for pair in pairs}
    keys = _read_keys(Path(args.keys_manifest))
    missing = set(keys) - set(pair_by_key)
    if missing:
        raise ValueError(f"Unknown OCR evaluation keys: {sorted(missing)}")
    output_dir = Path(args.output_dir)
    input_dir = output_dir / "input_crops"
    stage_dirs = {
        "classical": Path(args.classical_dir),
        "uvdoc": Path(args.uvdoc_dir),
        "hybrid": Path(args.hybrid_dir),
    }
    for stage, directory in stage_dirs.items():
        absent = [key for key in keys if not (directory / f"{key}.jpg").is_file()]
        if absent:
            raise FileNotFoundError(f"Missing {stage} images for keys: {absent}")

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        engine=args.engine,
        device=args.device,
    )
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for index, key in enumerate(keys, start=1):
        pair = pair_by_key[key]
        input_path = _prepare_input_crop(
            pair,
            output=input_dir / f"{key}.jpg",
            max_side=args.max_side,
            mask_margin=args.mask_margin,
        )
        paths = {"input": input_path, **{name: path / f"{key}.jpg" for name, path in stage_dirs.items()}}
        for stage, path in paths.items():
            started = time.perf_counter_ns()
            results = list(ocr.predict(str(path)))
            runtime_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            if len(results) != 1:
                raise ValueError(f"Expected one OCR result for {path}, got {len(results)}")
            values = _ocr_values(results[0].json, args.confidence_threshold)
            predictions.append(
                {
                    "key": key,
                    "family": pair.family,
                    "stage": stage,
                    "texts": values.pop("texts"),
                    "scores": values.pop("scores"),
                }
            )
            rows.append(
                {
                    "key": key,
                    "family": pair.family,
                    "stage": stage,
                    "image": str(path),
                    "runtime_ms": runtime_ms,
                    **values,
                }
            )
        print(json.dumps({"progress": f"{index}/{len(keys)}", "key": key}), flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics.csv", rows)
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    by_stage: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    by_stage_family: defaultdict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        stage, family = str(row["stage"]), str(row["family"])
        by_stage[stage].append(row)
        by_stage_family[(stage, family)].append(row)
    input_rows = by_stage["input"]
    summary = {
        "method": "fixed_subset_ppocrv6_diagnostic",
        "subset_count": len(keys),
        "keys_manifest": str(Path(args.keys_manifest)),
        "stages": ["input", "classical", "uvdoc", "hybrid"],
        "confidence_threshold": args.confidence_threshold,
        "max_side": args.max_side,
        "paddleocr_version": _version("paddleocr"),
        "paddlex_version": _version("paddlex"),
        "transformers_version": _version("transformers"),
        "engine": args.engine,
        "device": args.device,
        "models": {
            "detection": "PP-OCRv6_medium_det_safetensors",
            "recognition": "PP-OCRv6_medium_rec_safetensors",
        },
        "accuracy_note": (
            "No transcripts are available, so these are OCR yield/confidence diagnostics, "
            "not character accuracy or CER."
        ),
        "overall": {stage: _aggregate(stage_rows) for stage, stage_rows in by_stage.items()},
        "versus_input": {
            stage: _comparison(stage_rows, input_rows)
            for stage, stage_rows in by_stage.items()
            if stage != "input"
        },
        "by_family": {
            stage: {
                family: _aggregate(family_rows)
                for (family_stage, family), family_rows in by_stage_family.items()
                if family_stage == stage
            }
            for stage in by_stage
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
