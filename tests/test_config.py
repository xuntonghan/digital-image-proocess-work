from pathlib import Path

import pytest

from dip_course.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    load_experiment_config,
)


def test_load_canon_config() -> None:
    config = load_experiment_config(Path("configs/enhancement/canon.toml"))
    assert config.name == "sied_canon_00001_raw2rgb"
    assert config.dataset.width == 3840
    assert config.model.in_channels == 4
    assert config.dataset.patch_size == 256
    assert config.model.normalization == "batch"
    assert config.training.batch_size == 4
    assert config.training.chroma_weight == 1.0


def test_invalid_config_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive even"):
        DatasetConfig(width=31)
    with pytest.raises(ValueError, match="must be positive"):
        TrainingConfig(batch_size=0)
    with pytest.raises(ValueError, match="normalization"):
        ModelConfig(normalization="instance")
