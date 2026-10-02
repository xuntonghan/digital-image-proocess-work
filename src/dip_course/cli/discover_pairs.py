from __future__ import annotations

import argparse
import json
from pathlib import Path

from dip_course.cli._common import file_sha256, write_json
from dip_course.io import format_manifest_line, resolve_inside


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover numerically named RAW/GT pairs"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-count", type=int)
    return parser


def numeric_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError as exc:
        raise ValueError(f"Expected a numeric filename, got {path.name}") from exc


def discover(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.stem in result:
                raise ValueError(f"Duplicate stem in {directory}: {path.stem}")
            result[path.stem] = path
    return result


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.data_root).resolve()
    raw_dir = resolve_inside(root, args.raw_dir)
    gt_dir = resolve_inside(root, args.gt_dir)
    if not raw_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError("RAW or GT directory does not exist")

    raw_files = discover(raw_dir, {".raw"})
    gt_files = discover(gt_dir, {".jpg", ".jpeg", ".png"})
    missing_gt = sorted(set(raw_files) - set(gt_files), key=lambda value: int(value))
    missing_raw = sorted(set(gt_files) - set(raw_files), key=lambda value: int(value))
    if missing_gt or missing_raw:
        raise ValueError(
            f"Unpaired files: missing_gt={missing_gt[:20]}, missing_raw={missing_raw[:20]}"
        )
    if args.expected_count is not None and len(raw_files) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} pairs, discovered {len(raw_files)}"
        )

    ordered_raw = sorted(raw_files.values(), key=numeric_key)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for raw_path in ordered_raw:
        gt_path = gt_files[raw_path.stem]
        lines.append(
            format_manifest_line(
                raw_path.relative_to(root).as_posix(),
                gt_path.relative_to(root).as_posix(),
            )
        )
    output.write_text("".join(lines), encoding="utf-8")
    report = {
        "valid": True,
        "data_root": str(root),
        "raw_dir": str(raw_dir),
        "gt_dir": str(gt_dir),
        "pair_count": len(lines),
        "manifest": str(output.resolve()),
        "manifest_sha256": file_sha256(output),
        "first_key": ordered_raw[0].stem,
        "last_key": ordered_raw[-1].stem,
    }
    write_json(output.with_suffix(".discovery.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
