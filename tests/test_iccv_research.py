import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "cloud"))
try:
    RESEARCH = importlib.import_module("research_sied_iccv")
finally:
    sys.path.pop(0)


def test_matched_arms_use_same_private_batch_schedule():
    torch.manual_seed(1)
    first = list(RESEARCH.training_batches(35, steps=8, batch=16))
    torch.rand(300)
    second = list(RESEARCH.training_batches(35, steps=8, batch=16))
    assert first == second
    assert len(first) == 8
    assert sorted(item for batch in first[:3] for item in batch) == list(range(35))
    with pytest.raises(ValueError):
        list(RESEARCH.training_batches(0, steps=8))


def test_resident_pairs_preserve_exact_fp32_values_and_indices(tmp_path):
    paths = []
    pairs = []
    for index in range(3):
        pair = np.random.default_rng(index).random((2, 288, 512, 3), dtype=np.float32)
        path = tmp_path / f"{index}.npy"
        np.save(path, pair, allow_pickle=False)
        paths.append(path)
        pairs.append(pair)
    resident = RESEARCH.ResidentPairs(paths, device="cpu")
    image, target = resident.batch([2, 0])
    expected = torch.from_numpy(np.stack([pairs[2], pairs[0]]).transpose(0, 1, 4, 2, 3).copy())
    torch.testing.assert_close(image, expected[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(target, expected[:, 1], atol=0, rtol=0)
    assert resident.bytes == expected[0].numel() * 3 * 4


def test_resident_pairs_reject_nan_or_empty(tmp_path):
    with pytest.raises(ValueError, match="No training"):
        RESEARCH.ResidentPairs([], device="cpu")
    path = tmp_path / "bad.npy"
    np.save(path, np.full((2, 288, 512, 3), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="Nonfinite"):
        RESEARCH.ResidentPairs([path], device="cpu")
