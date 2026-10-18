from pathlib import Path

import pytest

from dip_course.cli.evaluate_enhancement_predictions import find_prediction


def test_find_prediction_prefers_indexed_name(tmp_path: Path) -> None:
    prediction = tmp_path / "0003_scene.png"
    prediction.touch()
    assert find_prediction(tmp_path, 3, "scene") == prediction


def test_find_prediction_accepts_one_matching_suffix(tmp_path: Path) -> None:
    prediction = tmp_path / "0042_scene.png"
    prediction.touch()
    assert find_prediction(tmp_path, 3, "scene") == prediction


def test_find_prediction_rejects_ambiguous_suffix(tmp_path: Path) -> None:
    (tmp_path / "0001_scene.png").touch()
    (tmp_path / "0002_scene.png").touch()
    with pytest.raises(FileNotFoundError, match="Expected one prediction"):
        find_prediction(tmp_path, 3, "scene")
