from __future__ import annotations

import pytest
import torch

from dip_course.cli.infer_official_sied import (
    average_prediction_samples,
    build_parser,
    diffusion_seed,
    load_refiner,
    select_official_state_dict,
)
from dip_course.enhancement.refiner import ExposureColorRefiner


def test_averages_inference_tensors_without_inplace_update() -> None:
    with torch.inference_mode():
        samples = [torch.ones(2), torch.full((2,), 3.0)]

    result = average_prediction_samples(samples)

    assert torch.equal(result, torch.full((2,), 2.0))


def test_ensemble_size_defaults_to_one() -> None:
    args = build_parser().parse_args(
        [
            "--official-repo",
            "repo",
            "--checkpoint",
            "model.pt",
            "--data-root",
            "data",
            "--manifest",
            "pairs.txt",
            "--output-dir",
            "output",
        ]
    )

    assert args.ensemble_size == 1
    assert args.skip_metrics is False
    assert args.save_max_side == 0


def test_first_ensemble_seed_matches_single_sample() -> None:
    assert diffusion_seed(3407, 7, 0) == 3414
    assert diffusion_seed(3407, 7, 1) != 3414


def test_selects_ema_model_weights() -> None:
    checkpoint = {
        "ema": {
            "initted": True,
            "step": 10,
            "online_model.layer": "online",
            "ema_model.layer": "ema",
        },
        "model": {"layer": "raw"},
    }

    state, source = select_official_state_dict(checkpoint, {"layer"})

    assert state == {"layer": "ema"}
    assert source == "ema:ema_model."


def test_falls_back_to_model_weights() -> None:
    state, source = select_official_state_dict({"model": {"a": 1, "b": 2}}, {"a", "b"})

    assert state == {"a": 1, "b": 2}
    assert source == "model"


def test_rejects_incompatible_checkpoint() -> None:
    with pytest.raises(RuntimeError, match="Could not match"):
        select_official_state_dict({"model": {"wrong": 1}}, {"expected"})


def test_loads_exposure_color_refiner_checkpoint(tmp_path) -> None:
    source = ExposureColorRefiner(base_channels=8, context_side=32)
    checkpoint = tmp_path / "refiner.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "model_config": {"base_channels": 8, "context_side": 32},
            "global_step": 12,
        },
        checkpoint,
    )

    loaded, metadata = load_refiner(checkpoint, torch.device("cpu"))

    image = torch.rand(1, 3, 24, 32)
    torch.testing.assert_close(loaded(image), image, atol=2e-6, rtol=2e-6)
    assert metadata["global_step"] == 12
