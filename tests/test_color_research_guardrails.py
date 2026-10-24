import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "cloud"))
try:
    RESEARCH = importlib.import_module("research_sied_color")
finally:
    sys.path.pop(0)


def baseline():
    return {"count": 12, "psnr_mean": 26.0, "ssim_mean": 0.85,
            "delta_e2000_mean": 4.5, "delta_e2000_p95_mean": 8.0,
            "lightness_mae_mean": 4.4, "saturation_mae_mean": 0.04,
            "hue_error_degrees_mean": 12.0, "lpips_mean": 0.3}


def test_research_requires_psnr_gain_and_matching_count():
    reference = baseline()
    assert RESEARCH.safe_gain(dict(reference, psnr_mean=26.1), reference)
    assert not RESEARCH.safe_gain(dict(reference, psnr_mean=26.01), reference)
    assert not RESEARCH.safe_gain(dict(reference, count=11, psnr_mean=27.0), reference)


@pytest.mark.parametrize("name,value", [("lpips_mean", 0.32), ("hue_error_degrees_mean", 13),
                                       ("ssim_mean", 0.847), ("psnr_mean", float("nan"))])
def test_research_never_accepts_psnr_gain_with_guardrail_failure(name, value):
    reference = baseline()
    candidate = dict(reference, psnr_mean=27.0)
    candidate[name] = value
    assert not RESEARCH.safe_gain(candidate, reference)


def test_color_pairs_reject_quantized_or_wrong_resolution_cache(tmp_path):
    path = tmp_path / "pair.npy"
    np.save(path, np.zeros((2, 288, 512, 3), dtype=np.float16), allow_pickle=False)
    with pytest.raises(ValueError, match="shape/dtype"):
        RESEARCH.ColorPairs([path])[0]
    np.save(path, np.zeros((2, 8, 8, 3), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="shape/dtype"):
        RESEARCH.ColorPairs([path])[0]
