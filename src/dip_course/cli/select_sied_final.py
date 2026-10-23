"""Prepare/seal SIED acceptance without loading a neural model or test image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dip_course.enhancement.finalization import (
    mark_evaluation_started,
    prepare_finalization,
    read_json,
    seal_finalization,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "parameters", "start", "seal"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        print(prepare_finalization(
            root, root / "project/manifests/enhancement/canon_test.txt",
            root / "checkpoints/official/canon_0.0001.pt"))
    elif args.action == "parameters":
        protocol = read_json(root / "final/selection.json")["frozen_protocol"]
        print(protocol["refiner_tile_size"], protocol["refiner_overlap"], protocol["seed"],
              protocol["sampling_steps"], protocol["ensemble_size"])
    elif args.action == "start":
        mark_evaluation_started(root)
    else:
        print(json.dumps(seal_finalization(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
