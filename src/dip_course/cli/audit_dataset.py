from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from dip_course.cli._common import write_json
from dip_course.io import RawSpec, read_pair_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit paired SIED-style RAW/RGB data")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--black-level", type=int, default=512)
    parser.add_argument("--white-level", type=int, default=16383)
    parser.add_argument("--limit", type=int, default=0, help="0 checks every pair")
    parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = RawSpec(
        width=args.width,
        height=args.height,
        black_level=args.black_level,
        white_level=args.white_level,
    )
    records = read_pair_manifest(args.data_root, args.manifest)
    selected = records[: args.limit] if args.limit > 0 else records
    errors: list[dict[str, object]] = []
    seen_pairs: set[tuple[Path, Path]] = set()
    duplicate_count = 0
    total_bytes = 0

    for index, record in enumerate(selected):
        pair = (record.low_raw, record.gt_rgb)
        if pair in seen_pairs:
            duplicate_count += 1
        seen_pairs.add(pair)

        if not record.low_raw.is_file():
            errors.append({"index": index, "key": record.key, "error": "missing_raw"})
        else:
            size = record.low_raw.stat().st_size
            total_bytes += size
            if size != spec.expected_bytes:
                errors.append(
                    {
                        "index": index,
                        "key": record.key,
                        "error": "unexpected_raw_bytes",
                        "actual": size,
                        "expected": spec.expected_bytes,
                    }
                )

        if not record.gt_rgb.is_file():
            errors.append({"index": index, "key": record.key, "error": "missing_gt"})
        else:
            total_bytes += record.gt_rgb.stat().st_size
            try:
                with Image.open(record.gt_rgb) as image:
                    if image.size != (spec.width, spec.height):
                        errors.append(
                            {
                                "index": index,
                                "key": record.key,
                                "error": "unexpected_gt_dimensions",
                                "actual": list(image.size),
                                "expected": [spec.width, spec.height],
                            }
                        )
                    image.verify()
            except (OSError, ValueError) as exc:
                errors.append(
                    {
                        "index": index,
                        "key": record.key,
                        "error": "unreadable_gt",
                        "detail": str(exc),
                    }
                )

    report = {
        "valid": not errors and duplicate_count == 0,
        "manifest": str(Path(args.manifest).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "pairs_declared": len(records),
        "pairs_checked": len(selected),
        "duplicate_pairs": duplicate_count,
        "checked_bytes": total_bytes,
        "raw_spec": {
            "width": spec.width,
            "height": spec.height,
            "black_level": spec.black_level,
            "white_level": spec.white_level,
            "expected_raw_bytes": spec.expected_bytes,
        },
        "error_count": len(errors),
        "errors": errors[:100],
        "errors_truncated": len(errors) > 100,
    }
    if args.output_json:
        write_json(args.output_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
