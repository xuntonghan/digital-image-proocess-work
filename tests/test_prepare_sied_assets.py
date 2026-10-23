"""Preparation must fail before downloads if the user's data already exist."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("relative", ["data/SIED_dataset/0.0001", "data/SIED_dataset/gt",
                                      "checkpoints/official/canon_0.0001.pt"])
def test_preparation_preserves_existing_data(tmp_path: Path, relative: str) -> None:
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    if target.suffix == ".pt":
        sentinel = target
    else:
        target.mkdir()
        sentinel = target / "unique-original.bin"
    sentinel.write_bytes(b"only-copy-must-survive")
    script = Path(__file__).resolve().parents[1] / "cloud/prepare_sied_canon_assets.sh"
    result = subprocess.run(["bash", str(script), str(tmp_path)], check=False,
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "Refusing to replace existing data/weights" in result.stderr
    assert sentinel.read_bytes() == b"only-copy-must-survive"
    assert not (tmp_path / "assets").exists()


def test_preparation_refuses_broken_symlink(tmp_path: Path) -> None:
    target = tmp_path / "data/SIED_dataset/gt"
    target.parent.mkdir(parents=True)
    target.symlink_to(tmp_path / "unavailable-original")
    script = Path(__file__).resolve().parents[1] / "cloud/prepare_sied_canon_assets.sh"
    result = subprocess.run(["bash", str(script), str(tmp_path)], check=False,
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "Refusing to replace existing data/weights" in result.stderr
    assert target.is_symlink()
    assert not (tmp_path / "assets").exists()
