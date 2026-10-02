from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dip_course.cli._common import file_sha256, write_json
from dip_course.io import read_pair_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert paired GT JPEG files to memory-mapped uint8 NumPy arrays"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_pair_manifest(args.data_root, args.manifest)
    seen: set[str] = set()
    entries = []
    for index, record in enumerate(records, 1):
        if record.key in seen:
            raise ValueError(f"Duplicate cache key in manifest: {record.key}")
        seen.add(record.key)
        with Image.open(record.gt_rgb) as image:
            if image.size != (args.width, args.height):
                raise ValueError(
                    f"Unexpected GT size for {record.gt_rgb}: {image.size}; "
                    f"expected {(args.width, args.height)}"
                )
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        output = output_dir / f"{record.key}.npy"
        np.save(output, array, allow_pickle=False)
        entries.append(
            {
                "key": record.key,
                "source": str(record.gt_rgb),
                "output": output.name,
                "bytes": output.stat().st_size,
            }
        )
        print(f"Cached GT {index}/{len(records)}: {record.key}", flush=True)

    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "count": len(entries),
        "shape": [args.height, args.width, 3],
        "dtype": "uint8",
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "entries": entries,
    }
    write_json(output_dir / "cache_manifest.json", report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "entries"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
