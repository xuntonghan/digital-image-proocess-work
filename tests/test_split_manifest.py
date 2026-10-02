from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_split_manifest_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    source = tmp_path / "all.txt"
    source.write_text(
        "".join(f"raw/{index}.raw gt/{index}.png\n" for index in range(20)),
        encoding="utf-8",
    )
    output = tmp_path / "splits"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dip_course.cli.split_manifest",
            "--data-root",
            str(root),
            "--manifest",
            str(source),
            "--output-dir",
            str(output),
            "--prefix",
            "canon",
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((output / "canon_split.json").read_text())
    assert report["counts"] == {"train": 16, "val": 2, "test": 2}
    contents = {
        name: set((output / f"canon_{name}.txt").read_text().splitlines())
        for name in ("train", "val", "test")
    }
    assert not contents["train"] & contents["val"]
    assert not contents["train"] & contents["test"]
    assert not contents["val"] & contents["test"]
