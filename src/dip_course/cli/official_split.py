from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

from dip_course.cli._common import file_sha256, write_json
from dip_course.io import PairRecord, format_manifest_line, read_pair_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map a course subset onto the official SIED train/test split"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--official-train", required=True)
    parser.add_argument("--official-test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260930)
    return parser


def git_show(repo: Path, revision_path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"HEAD:{revision_path}"],
        check=True,
        capture_output=True,
    )
    return result.stdout


def git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_official_ids(content: bytes) -> set[str]:
    identifiers: set[str] = set()
    for line_number, raw_line in enumerate(content.decode("utf-8").splitlines(), 1):
        fields = raw_line.split()
        if len(fields) != 2:
            raise ValueError(
                f"Official list line {line_number} does not have two fields"
            )
        identifier = Path(fields[0]).stem
        if identifier in identifiers:
            raise ValueError(f"Duplicate identifier in official list: {identifier}")
        identifiers.add(identifier)
    return identifiers


def write_records(path: Path, records: list[PairRecord], root: Path) -> None:
    lines = [
        format_manifest_line(
            record.low_raw.relative_to(root).as_posix(),
            record.gt_rgb.relative_to(root).as_posix(),
        )
        for record in records
    ]
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.validation_ratio < 0.5:
        raise ValueError("validation-ratio must be between 0 and 0.5")
    root = Path(args.data_root).resolve()
    repo = Path(args.official_repo).resolve()
    records = read_pair_manifest(root, args.manifest)
    by_key = {record.key: record for record in records}
    if len(by_key) != len(records):
        raise ValueError("Course manifest contains duplicate filename stems")

    official_train_content = git_show(repo, args.official_train)
    official_test_content = git_show(repo, args.official_test)
    official_train = parse_official_ids(official_train_content)
    official_test = parse_official_ids(official_test_content)
    if official_train & official_test:
        raise ValueError("Official train and test identifiers overlap")

    course_keys = set(by_key)
    mapped_train = course_keys & official_train
    mapped_test = course_keys & official_test
    missing = course_keys - mapped_train - mapped_test
    if missing:
        raise ValueError(
            f"Course keys missing from official split: {sorted(missing)[:20]}"
        )

    ordered_train = [by_key[key] for key in sorted(mapped_train, key=int)]
    random.Random(args.seed).shuffle(ordered_train)
    validation_count = round(len(ordered_train) * args.validation_ratio)
    if validation_count <= 0 or validation_count >= len(ordered_train):
        raise ValueError(
            "Validation split would be empty or consume all training records"
        )
    validation_records = ordered_train[:validation_count]
    training_records = ordered_train[validation_count:]
    test_records = [by_key[key] for key in sorted(mapped_test, key=int)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": output_dir / f"{args.prefix}_train.txt",
        "val": output_dir / f"{args.prefix}_val.txt",
        "test": output_dir / f"{args.prefix}_test.txt",
    }
    split_records = {
        "train": training_records,
        "val": validation_records,
        "test": test_records,
    }
    for name, path in outputs.items():
        write_records(path, split_records[name], root)

    report = {
        "valid": True,
        "official_repo": str(repo),
        "official_commit": git_head(repo),
        "official_train_path": args.official_train,
        "official_train_sha256": hashlib.sha256(official_train_content).hexdigest(),
        "official_test_path": args.official_test,
        "official_test_sha256": hashlib.sha256(official_test_content).hexdigest(),
        "course_manifest": str(Path(args.manifest).resolve()),
        "course_manifest_sha256": file_sha256(args.manifest),
        "seed": args.seed,
        "validation_ratio_within_official_train": args.validation_ratio,
        "counts": {name: len(values) for name, values in split_records.items()},
        "course_total": len(records),
        "official_train_mapped": len(mapped_train),
        "official_test_mapped": len(mapped_test),
        "disjoint": True,
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for name, path in outputs.items()
        },
    }
    write_json(output_dir / f"{args.prefix}_official_split.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
