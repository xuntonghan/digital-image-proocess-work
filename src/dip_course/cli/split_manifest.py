from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dip_course.cli._common import file_sha256, write_json
from dip_course.io import PairRecord, format_manifest_line, read_pair_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/val/test manifests"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260930)
    return parser


def record_line(record: PairRecord, root: Path) -> str:
    low = record.low_raw.relative_to(root.resolve()).as_posix()
    gt = record.gt_rgb.relative_to(root.resolve()).as_posix()
    return format_manifest_line(low, gt)


def write_manifest(path: Path, records: list[PairRecord], root: Path) -> None:
    path.write_text(
        "".join(record_line(record, root) for record in records), encoding="utf-8"
    )


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("train-ratio must be between 0 and 1")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("val-ratio must be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train-ratio + val-ratio must be less than 1")

    root = Path(args.data_root).resolve()
    records = read_pair_manifest(root, args.manifest)
    pairs = [(record.low_raw, record.gt_rgb) for record in records]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Source manifest contains duplicate pairs")
    low_paths = [record.low_raw for record in records]
    if len(set(low_paths)) != len(low_paths):
        raise ValueError("A low-light RAW path appears more than once")

    records.sort(key=lambda record: (str(record.low_raw), str(record.gt_rgb)))
    random.Random(args.seed).shuffle(records)
    total = len(records)
    train_count = int(total * args.train_ratio)
    val_count = int(total * args.val_ratio)
    test_count = total - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(
            "Dataset is too small for the requested non-empty three-way split"
        )

    splits = {
        "train": records[:train_count],
        "val": records[train_count : train_count + val_count],
        "test": records[train_count + val_count :],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for name, split_records in splits.items():
        path = output_dir / f"{args.prefix}_{name}.txt"
        write_manifest(path, split_records, root)
        output_paths[name] = path

    report = {
        "source_manifest": str(Path(args.manifest).resolve()),
        "source_manifest_sha256": file_sha256(args.manifest),
        "data_root": str(root),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1.0 - args.train_ratio - args.val_ratio,
        },
        "counts": {name: len(split) for name, split in splits.items()},
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for name, path in output_paths.items()
        },
        "disjoint": True,
    }
    write_json(output_dir / f"{args.prefix}_split.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
