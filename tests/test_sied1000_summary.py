from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "sied1000", Path(__file__).parents[1] / "cloud/evaluate_sied1000.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(split, key, value):
    return {"split": split, "key": key, **{metric: value for metric in MODULE.METRICS}}


def test_sied1000_mean_is_per_image_not_mean_of_split_means():
    rows = [row("train", str(i), 30.0) for i in range(832)]
    rows += [row("val", str(i), 26.0) for i in range(92)]
    rows += [row("test", str(i), 23.0) for i in range(76)]
    result = MODULE.summarize(rows)
    assert result["count"] == 1000
    assert result["psnr_mean"] == pytest.approx((832 * 30 + 92 * 26 + 76 * 23) / 1000)


def test_sied1000_never_filters_bad_rows():
    with pytest.raises(ValueError, match="Nonfinite"):
        MODULE.summarize([row("train", "1", float("nan"))])


def test_sied1000_rejects_duplicate_and_empty_rows():
    with pytest.raises(ValueError, match="Duplicate"):
        MODULE.summarize([row("train", "1", 20), row("train", "1", 30)])
    with pytest.raises(ValueError, match="empty"):
        MODULE.summarize([])


def test_sied1000_verifies_real_sealed_test_without_model_inference():
    import json

    project = Path(__file__).parents[1]
    final = project / "reports/assets/enhancement/sprint_20261024/final"
    protocol = json.loads((final / "selection.json").read_text())["frozen_protocol"]
    keys = [Path(line.split()[0]).stem for line in
            (project / "manifests/enhancement/canon_test.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    rows = MODULE.read_sealed_test(final, keys, protocol)
    result = MODULE.summarize(rows)
    assert result["count"] == 76
    assert result["psnr_mean"] == pytest.approx(23.295996430040688)


def test_color_cache_upcasts_amp_output_before_area_resize():
    import numpy as np

    torch = pytest.importorskip("torch")
    prediction = np.random.default_rng(3).random((8, 12, 3)).astype(np.float16)
    target = prediction.astype(np.float32)
    pair = MODULE.make_color_training_pair(prediction, target, size=(2, 3))
    expected = torch.nn.functional.interpolate(
        torch.from_numpy(target.transpose(2, 0, 1).copy()).unsqueeze(0),
        size=(2, 3), mode="area").squeeze(0).permute(1, 2, 0).numpy()
    assert pair.dtype == np.float32
    np.testing.assert_array_equal(pair[0], expected)
    np.testing.assert_array_equal(pair[0], pair[1])
