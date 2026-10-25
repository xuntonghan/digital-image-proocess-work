from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("sony_prepare", ROOT / "cloud/prepare_sony1000.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_original_sony_1000_partition_is_disjoint():
    splits = MODULE.validate_sony_manifests(ROOT)
    assert {s: len(rows) for s, rows in splits.items()} == {"train": 822, "val": 91, "test": 87}
    assert len({p for rows in splits.values() for p in rows}) == 1000


def test_official_gt_folder_can_map_to_original_course_folder():
    assert MODULE.archive_members(["gt/1.jpg", "gt/2.jpg"], ["gt 2/1.jpg", "gt 2/2.jpg"]) == {
        "gt 2/1.jpg": "gt/1.jpg", "gt 2/2.jpg": "gt/2.jpg"}


def test_archive_ambiguity_and_unsafe_paths_are_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        MODULE.archive_members(["gt/1.jpg", "other/1.jpg"], ["gt 2/1.jpg"])
    with pytest.raises(ValueError, match="Unsafe"):
        MODULE.archive_members(["../1.jpg"], ["gt 2/1.jpg"])


def test_manifest_path_cannot_escape_data_directory(tmp_path):
    p = tmp_path / "pairs.txt"
    p.write_text("../1.raw 'gt 2/1.jpg'\n")
    with pytest.raises(ValueError, match="Unsafe"):
        MODULE.pairs(p)


PIPELINE_SPEC = importlib.util.spec_from_file_location(
    'sony_pipeline', ROOT / 'cloud/sony1000_pipeline.py')


@pytest.fixture
def pipeline(monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(ROOT / 'cloud'))
    module = importlib.util.module_from_spec(PIPELINE_SPEC)
    sys.modules[PIPELINE_SPEC.name] = module
    PIPELINE_SPEC.loader.exec_module(module)
    return module


def test_sony_candidate_requires_native_psnr_and_color_guardrails(pipeline):
    baseline = {'count': 91, **{f: 1.0 for f in pipeline.METRICS.values()}}
    baseline.update(psnr_mean=25.0, ssim_mean=0.85)
    candidate = {**baseline, 'psnr_mean': 25.03}
    assert pipeline.qualifies(candidate, baseline)
    assert not pipeline.qualifies({**candidate, 'count': 12}, baseline)
    assert not pipeline.qualifies({**candidate, 'ssim_mean': 0.84}, baseline)
    assert not pipeline.qualifies({**candidate, 'delta_e2000_mean': 1.03}, baseline)
    assert not pipeline.qualifies({**candidate, 'lpips_mean': float('nan')}, baseline)


def full_rows(pipeline):
    rows, expected, offset = [], {}, 0
    for split, count in pipeline.COUNTS.items():
        expected[split] = [str(offset + i) for i in range(count)]
        for index, key in enumerate(expected[split]):
            rows.append({'split': split, 'index': index, 'key': key,
                         'binding_sha256': 'one-model',
                         'seeds': ';'.join(str(3407 + index + j * 1000003) for j in range(3)),
                         **{f: 1.0 for f in pipeline.METRICS}})
        offset += count
    return rows, expected


def test_final_sony_requires_all_1000_and_one_fixed_model(pipeline):
    rows, expected = full_rows(pipeline)
    pipeline.require_complete(rows, expected)
    with pytest.raises(ValueError, match='1000'):
        pipeline.require_complete(rows[:-1], expected)
    rows[0]['binding_sha256'] = 'another-model'
    with pytest.raises(ValueError, match='mix models'):
        pipeline.require_complete(rows, expected)


def test_final_sony_rejects_wrong_split_local_seeds(pipeline):
    rows, expected = full_rows(pipeline)
    rows[822]['seeds'] = '4229;1004232;2004235'
    with pytest.raises(ValueError, match='split-local'):
        pipeline.require_complete(rows, expected)


def test_final_sony_arithmetic_average_uses_822_91_87_weights(pipeline):
    rows, _ = full_rows(pipeline)
    for row in rows:
        row['psnr'] = {'train': 30.0, 'val': 25.0, 'test': 20.0}[row['split']]
    assert pipeline.summarize(rows)['psnr_mean'] == pytest.approx(
        (822 * 30 + 91 * 25 + 87 * 20) / 1000)


def test_native_fp32_cache_is_lossless(pipeline, tmp_path):
    import numpy as np
    array = np.random.default_rng(4).random((19, 31, 3), dtype=np.float32)
    path = tmp_path / 'native.npz'
    pipeline.compressed_array_save(path, array)
    with np.load(path, allow_pickle=False) as archive:
        restored = archive['rgb']
    assert restored.dtype == np.float32
    assert pipeline.tensor_digest(restored) == pipeline.tensor_digest(array)
    np.testing.assert_array_equal(restored, array)
