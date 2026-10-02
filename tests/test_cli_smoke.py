from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")


def run_command(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def make_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    lines: list[str] = []
    for index in range(3):
        ramp = np.linspace(0, 1023, 32 * 32, dtype=np.uint16).reshape(32, 32)
        raw = np.roll(ramp, index * 3, axis=1)
        raw_name = f"low_{index}.raw"
        gt_name = f"gt_{index}.png"
        raw.tofile(root / raw_name)
        normalized = raw.astype(np.float32) / 1023.0
        rgb = np.stack((normalized, np.sqrt(normalized), normalized**0.8), axis=-1)
        Image.fromarray(np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)).save(
            root / gt_name
        )
        lines.append(f"{raw_name} {gt_name}\n")

    train_manifest = root / "train.txt"
    val_manifest = root / "val.txt"
    train_manifest.write_text("".join(lines[:2]), encoding="utf-8")
    val_manifest.write_text(lines[2], encoding="utf-8")
    return train_manifest, val_manifest


def test_full_local_pipeline(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    train_manifest, val_manifest = make_fixture(data_root)

    audit_json = tmp_path / "audit.json"
    run_command(
        "-m",
        "dip_course.cli.audit_dataset",
        "--data-root",
        str(data_root),
        "--manifest",
        str(train_manifest),
        "--width",
        "32",
        "--height",
        "32",
        "--black-level",
        "0",
        "--white-level",
        "1023",
        "--output-json",
        str(audit_json),
    )
    assert json.loads(audit_json.read_text())["valid"] is True

    classical_output = tmp_path / "classical"
    run_command(
        "-m",
        "dip_course.cli.run_classical",
        "--data-root",
        str(data_root),
        "--manifest",
        str(val_manifest),
        "--output-dir",
        str(classical_output),
        "--width",
        "32",
        "--height",
        "32",
        "--black-level",
        "0",
        "--white-level",
        "1023",
        "--exposure",
        "1",
        "--gamma",
        "1",
        "--white-balance",
        "1",
        "1",
        "1",
    )
    assert (classical_output / "metrics.csv").is_file()

    checkpoint_dir = tmp_path / "checkpoints"
    run_command(
        "-m",
        "dip_course.cli.train_enhancement",
        "--config",
        "tests/fixtures/tiny.toml",
        "--data-root",
        str(data_root),
        "--train-manifest",
        str(train_manifest),
        "--val-manifest",
        str(val_manifest),
        "--output-dir",
        str(checkpoint_dir),
        "--max-steps",
        "1",
        "--device",
        "cpu",
    )
    checkpoint = checkpoint_dir / "best.pt"
    assert checkpoint.is_file()
    assert (
        json.loads((checkpoint_dir / "completed.json").read_text())["completed"] is True
    )

    inference_output = tmp_path / "inference"
    run_command(
        "-m",
        "dip_course.cli.infer_enhancement",
        "--checkpoint",
        str(checkpoint),
        "--data-root",
        str(data_root),
        "--manifest",
        str(val_manifest),
        "--output-dir",
        str(inference_output),
        "--device",
        "cpu",
        "--width",
        "32",
        "--height",
        "32",
        "--black-level",
        "0",
        "--white-level",
        "1023",
        "--tile-size",
        "16",
        "--overlap",
        "4",
    )
    summary = json.loads((inference_output / "summary.json").read_text())
    assert summary["count"] == 1
    assert (inference_output / "metrics.csv").is_file()
    assert len(list((inference_output / "images").glob("*.png"))) == 1
