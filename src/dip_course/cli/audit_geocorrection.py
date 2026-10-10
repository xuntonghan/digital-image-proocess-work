from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image

from dip_course.cli._common import file_sha256, write_json
from dip_course.geocorrection.dataset import discover_pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the GeoCorrection image/mask pairs")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--expected-count", type=int, default=200)
    return parser


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.data_root)
    pairs = discover_pairs(root)
    if args.expected_count > 0 and len(pairs) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} pairs, found {len(pairs)}")

    rows: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    sizes: Counter[str] = Counter()
    families: Counter[str] = Counter()
    for pair in pairs:
        with Image.open(pair.image_path) as image:
            image.load()
            if image.mode != "RGB":
                raise ValueError(f"Expected RGB image at {pair.image_path}, got {image.mode}")
            width, height = image.size
        with Image.open(pair.mask_path) as mask:
            mask.load()
            if mask.mode != "L":
                raise ValueError(f"Expected L mask at {pair.mask_path}, got {mask.mode}")
            if mask.size != (width, height):
                raise ValueError(
                    f"Image/mask size mismatch for {pair.key}: {(width, height)} vs {mask.size}"
                )
            colors = mask.getcolors(maxcolors=3)
            if colors is None or {value for _, value in colors} - {0, 255}:
                raise ValueError(f"Mask is not binary 0/255 at {pair.mask_path}")
            foreground_ratio = sum(
                count for count, value in colors if value == 255
            ) / (width * height)
        image_sha256 = file_sha256(pair.image_path)
        mask_sha256 = file_sha256(pair.mask_path)
        relative_image = pair.image_path.relative_to(root).as_posix()
        relative_mask = pair.mask_path.relative_to(root).as_posix()
        aggregate.update(
            f"{image_sha256}  {relative_image}\n{mask_sha256}  {relative_mask}\n".encode()
        )
        rows.append(
            {
                "key": pair.key,
                "family": pair.family,
                "image": relative_image,
                "mask": relative_mask,
                "width": width,
                "height": height,
                "foreground_ratio": foreground_ratio,
                "image_bytes": pair.image_path.stat().st_size,
                "mask_bytes": pair.mask_path.stat().st_size,
                "image_sha256": image_sha256,
                "mask_sha256": mask_sha256,
            }
        )
        sizes[f"{width}x{height}"] += 1
        families[pair.family] += 1

    _write_csv(Path(args.output_csv), rows)
    summary = {
        "dataset": "GeoCorrection",
        "data_root": str(root),
        "pair_count": len(rows),
        "has_flat_ground_truth": False,
        "mask_semantics": "binary document-page region, aligned to the distorted image",
        "aggregate_sha256": aggregate.hexdigest(),
        "image_mode": "RGB",
        "mask_mode": "L binary {0,255}",
        "sizes": dict(sorted(sizes.items())),
        "families": dict(sorted(families.items())),
        "total_image_bytes": sum(int(row["image_bytes"]) for row in rows),
        "total_mask_bytes": sum(int(row["mask_bytes"]) for row in rows),
        "pairs_csv": str(Path(args.output_csv)),
    }
    write_json(args.output_json, summary)
    print(summary)


if __name__ == "__main__":
    main()
