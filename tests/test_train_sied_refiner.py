from pathlib import Path

import torch

from dip_course.cli.train_sied_refiner import load_initial_refiner
from dip_course.enhancement.refiner import ExposureColorRefiner


def test_load_initial_refiner_round_trip(tmp_path: Path) -> None:
    source = ExposureColorRefiner(base_channels=8, context_side=32)
    with torch.no_grad():
        source.global_head[-1].bias.fill_(0.125)
    checkpoint = tmp_path / "refiner.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "global_step": 1000,
            "model_config": {"base_channels": 8, "context_side": 32},
            "torch_version": torch.__version__,
        },
        checkpoint,
    )

    loaded, metadata, config = load_initial_refiner(checkpoint, torch.device("cpu"))

    assert config == {"base_channels": 8, "context_side": 32}
    assert metadata is not None
    assert metadata["source_global_step"] == 1000
    assert metadata["sha256"]
    for expected, actual in zip(source.parameters(), loaded.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_load_initial_refiner_defaults() -> None:
    model, metadata, config = load_initial_refiner(None, torch.device("cpu"))

    assert metadata is None
    assert config == {"base_channels": 24, "context_side": 256}
    assert isinstance(model, ExposureColorRefiner)
