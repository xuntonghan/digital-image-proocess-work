import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "cloud"))
try:
    FINALIZE = importlib.import_module("finalize_sied_submission1000")
finally:
    sys.path.pop(0)


def metrics(psnr=26.0, ssim=0.85):
    return {"count": 92, "psnr_mean": psnr, "ssim_mean": ssim,
            **{field: 0.3 for field in FINALIZE.METRICS.values()
               if field not in {"psnr_mean", "ssim_mean"}}}


def research():
    first = {"test_evaluated": False, "lut_full_passed": True,
             "lut_full_val": metrics(26.2), "best_lut_sha256": "first"}
    paired = {"test_evaluated": False, "arms": {
        "global3d": {"full_passed": True, "full_val": metrics(26.1), "checkpoint_sha256": "global"},
        "pairwise_rank4": {"full_passed": True, "full_val": metrics(26.3), "checkpoint_sha256": "rank"}}}
    return first, paired


def test_submission_selects_with_full_validation_never_overall_or_test():
    first, paired = research()
    first["overall"] = metrics(99)
    chosen = FINALIZE.select_candidate(first, paired, metrics())
    assert chosen["selected"]["name"] == "pairwise_rank4"
    assert chosen["test_or_overall_used_for_selection"] is False


def test_submission_keeps_ssim_and_color_guardrails():
    first, paired = research()
    paired["arms"]["pairwise_rank4"]["full_val"]["ssim_mean"] = 0.84
    first["lut_full_val"]["lpips_mean"] = 0.4
    assert FINALIZE.select_candidate(first, paired, metrics())["selected"]["name"] == "global3d"
    assert not FINALIZE.eligible(metrics(float("nan")), metrics())
    assert not FINALIZE.eligible(dict(metrics(28), count=12), metrics())


def test_submission_waits_for_both_arms_and_rejects_test_selection():
    first, paired = research()
    paired["arms"].pop("pairwise_rank4")
    with pytest.raises(ValueError, match="Both"):
        FINALIZE.select_candidate(first, paired, metrics())
    first, paired = research()
    paired["test_evaluated"] = True
    with pytest.raises(ValueError, match="test"):
        FINALIZE.select_candidate(first, paired, metrics())


def test_submission_uses_baseline_when_all_candidates_fail():
    first, paired = research()
    first["lut_full_passed"] = False
    for result in paired["arms"].values():
        result["full_passed"] = False
    assert FINALIZE.select_candidate(first, paired, metrics())["selected"]["name"] == "baseline"


def test_submission_ssim_breaks_psnr_tie():
    first, paired = research()
    paired["arms"]["pairwise_rank4"]["full_val"] = metrics(26.2, 0.851)
    assert FINALIZE.select_candidate(first, paired, metrics())["selected"]["name"] == "pairwise_rank4"


def test_submission_requires_1000_unique_same_pipeline_rows():
    expected = {s: [f"{s}_{i}" for i in range(n)] for s, n in
                (("train", 832), ("val", 92), ("test", 76))}
    rows = [{"split": s, "index": i, "key": key, "binding_sha256": "same",
             **{field: 0.5 for field in FINALIZE.METRICS}} for s, keys in expected.items()
            for i, key in enumerate(keys)]
    FINALIZE.require_complete(rows, expected)
    with pytest.raises(ValueError, match="1000"):
        FINALIZE.require_complete(rows[:-1], expected)
    rows[-1]["binding_sha256"] = "different"
    with pytest.raises(ValueError, match="mix"):
        FINALIZE.require_complete(rows, expected)
    rows[-1]["binding_sha256"] = "same"
    rows[-1]["key"] = rows[0]["key"]
    with pytest.raises(ValueError, match="1000"):
        FINALIZE.require_complete(rows, expected)
