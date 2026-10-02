from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dip_course.cli._common import write_json
from dip_course.enhancement.classical import (
    auto_tone_map_rgb,
    demosaic_rggb,
    tone_map_rgb,
)
from dip_course.io import (
    RawSpec,
    load_rgb,
    normalize_raw,
    read_pair_manifest,
    read_raw_uint16,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build RAW/baseline/model/GT comparison grid"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-dir")
    parser.add_argument(
        "--reference-prediction-dir",
        help="Optional earlier-model predictions for a direct before/after comparison",
    )
    parser.add_argument("--count", type=int, default=4)
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
    parser.add_argument("--panel-width", type=int, default=640)
    return parser


def to_pil(image: np.ndarray, panel_width: int, panel_height: int) -> Image.Image:
    array = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(array).resize(
        (panel_width, panel_height), Image.Resampling.LANCZOS
    )


def find_prediction(directory: Path, key: str) -> Path:
    matches = sorted(directory.glob(f"*_{key}.png"))
    if not matches:
        direct = directory / f"{key}.png"
        if direct.is_file():
            return direct
        raise FileNotFoundError(f"No prediction found for key {key} in {directory}")
    if len(matches) > 1:
        raise ValueError(f"Multiple predictions found for key {key}: {matches}")
    return matches[0]


def main() -> None:
    args = build_parser().parse_args()
    if args.count <= 0 or args.panel_width <= 0:
        raise ValueError("count and panel-width must be positive")
    spec = RawSpec(
        width=args.width,
        height=args.height,
        black_level=args.black_level,
        white_level=args.white_level,
    )
    records = read_pair_manifest(args.data_root, args.manifest)[: args.count]
    panel_height = round(args.panel_width * args.height / args.width)
    headers = ["Input RAW preview", "Classical baseline"]
    if args.reference_prediction_dir:
        headers.append("Previous model")
    if args.prediction_dir:
        headers.append(
            "Improved model" if args.reference_prediction_dir else "ML enhancement"
        )
    headers.append("Ground truth")
    header_height = 38
    row_label_width = 90
    canvas = Image.new(
        "RGB",
        (
            row_label_width + args.panel_width * len(headers),
            header_height + panel_height * len(records),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(headers):
        draw.text(
            (row_label_width + column * args.panel_width + 12, 12),
            title,
            fill="black",
        )

    prediction_dir = Path(args.prediction_dir) if args.prediction_dir else None
    reference_prediction_dir = (
        Path(args.reference_prediction_dir)
        if args.reference_prediction_dir
        else None
    )
    for row, record in enumerate(records):
        normalized = normalize_raw(read_raw_uint16(record.low_raw, spec), spec)
        demosaiced = demosaic_rggb(normalized)
        preview = tone_map_rgb(
            demosaiced,
            exposure=1.0,
            gamma=args.gamma,
            white_balance=tuple(args.white_balance),
        )
        if args.auto:
            baseline, _ = auto_tone_map_rgb(
                demosaiced,
                gamma=args.gamma,
                percentile=args.auto_percentile,
                target=args.auto_target,
                white_balance_power=args.auto_white_balance_power,
            )
        else:
            baseline = tone_map_rgb(
                demosaiced,
                exposure=args.exposure,
                gamma=args.gamma,
                white_balance=tuple(args.white_balance),
            )
        panels = [preview, baseline]
        if reference_prediction_dir:
            with Image.open(
                find_prediction(reference_prediction_dir, record.key)
            ) as prediction_image:
                panels.append(
                    np.asarray(prediction_image.convert("RGB"), dtype=np.float32)
                    / 255.0
                )
        if prediction_dir:
            with Image.open(
                find_prediction(prediction_dir, record.key)
            ) as prediction_image:
                panels.append(
                    np.asarray(prediction_image.convert("RGB"), dtype=np.float32)
                    / 255.0
                )
        panels.append(load_rgb(record.gt_rgb))
        y = header_height + row * panel_height
        draw.text((12, y + 12), f"ID {record.key}", fill="black")
        for column, panel in enumerate(panels):
            canvas.paste(
                to_pil(panel, args.panel_width, panel_height),
                (row_label_width + column * args.panel_width, y),
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    metadata = {
        "output": str(output.resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "keys": [record.key for record in records],
        "columns": headers,
        "raw_preview": {"exposure": 1.0, "gamma": args.gamma},
        "classical": {
            "mode": "auto" if args.auto else "fixed",
            "exposure": None if args.auto else args.exposure,
            "gamma": args.gamma,
            "white_balance": None if args.auto else list(args.white_balance),
            "auto_percentile": args.auto_percentile if args.auto else None,
            "auto_target": args.auto_target if args.auto else None,
            "auto_white_balance_power": (
                args.auto_white_balance_power if args.auto else None
            ),
        },
        "prediction_dir": str(prediction_dir.resolve()) if prediction_dir else None,
        "reference_prediction_dir": (
            str(reference_prediction_dir.resolve())
            if reference_prediction_dir
            else None
        ),
    }
    write_json(output.with_suffix(".json"), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
