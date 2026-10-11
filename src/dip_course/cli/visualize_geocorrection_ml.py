from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from dip_course.geocorrection.dataset import discover_pairs
from dip_course.geocorrection.uvdoc import crop_to_page_mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare original, classical, UVDoc, and hybrid geometry outputs"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--classical-dir", required=True)
    parser.add_argument("--uvdoc-dir", required=True)
    parser.add_argument("--hybrid-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--cell-width", type=int, default=360)
    parser.add_argument("--cell-height", type=int, default=260)
    parser.add_argument("--mask-margin", type=float, default=0.02)
    return parser


def _fit_on_white(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("Comparison cell dimensions must be positive")
    fitted = image.convert("RGB").copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.data_root)
    by_key = {pair.key: pair for pair in discover_pairs(root)}
    missing_keys = set(args.keys) - set(by_key)
    if missing_keys:
        raise ValueError(f"Unknown GeoCorrection keys: {sorted(missing_keys)}")

    stage_dirs = {
        "Classical": Path(args.classical_dir),
        "UVDoc": Path(args.uvdoc_dir),
        "Hybrid": Path(args.hybrid_dir),
    }
    columns = ["Original crop", *stage_dirs]
    cell_size = (args.cell_width, args.cell_height)
    gutter = 12
    margin = 16
    label_width = 190
    header_height = 34
    row_label_height = 24
    row_height = args.cell_height + row_label_height + gutter
    width = label_width + len(columns) * (args.cell_width + gutter) + margin
    height = header_height + len(args.keys) * row_height + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for index, label in enumerate(columns):
        x = label_width + index * (args.cell_width + gutter)
        draw.text((x + 4, 8), label, fill="black")

    for row_index, key in enumerate(args.keys):
        pair = by_key[key]
        y = header_height + row_index * row_height
        draw.text((margin, y + 4), key, fill="black")
        draw.text((margin, y + 20), pair.family, fill=(90, 90, 90))
        with Image.open(pair.image_path) as source:
            original = source.convert("RGB").copy()
        with Image.open(pair.mask_path) as source:
            mask = source.convert("L").copy()
        images = [
            crop_to_page_mask(
                original, mask, margin_fraction=args.mask_margin
            ).image,
        ]
        for stage, directory in stage_dirs.items():
            path = directory / f"{key}.jpg"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {stage} output: {path}")
            with Image.open(path) as source:
                images.append(source.convert("RGB").copy())
        for column_index, image in enumerate(images):
            x = label_width + column_index * (args.cell_width + gutter)
            cell = _fit_on_white(image, cell_size)
            canvas.paste(cell, (x, y + row_label_height))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


if __name__ == "__main__":
    main()
