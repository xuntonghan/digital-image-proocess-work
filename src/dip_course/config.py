from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


@dataclass(frozen=True)
class DatasetConfig:
    width: int = 3840
    height: int = 2160
    black_level: int = 512
    white_level: int = 16383
    bayer_pattern: str = "RGGB"
    patch_size: int = 256
    augment: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError("dataset width and height must be positive even integers")
        if not 0 <= self.black_level < self.white_level <= 65535:
            raise ValueError("dataset black/white levels are invalid")
        if self.bayer_pattern.upper() != "RGGB":
            raise ValueError("only the RGGB Bayer pattern is currently supported")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 4
    out_channels: int = 3
    base_channels: int = 32
    depth: int = 4
    normalization: str = "group"

    def __post_init__(self) -> None:
        if min(self.in_channels, self.out_channels, self.base_channels) <= 0:
            raise ValueError("model channel counts must be positive")
        if self.depth < 2:
            raise ValueError("model depth must be at least 2")
        if self.normalization not in {"group", "batch", "none"}:
            raise ValueError("model normalization must be group, batch, or none")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 4
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_steps: int = 50_000
    validation_interval: int = 1_000
    checkpoint_interval: int = 2_000
    gradient_accumulation: int = 1
    ssim_weight: float = 0.2
    chroma_weight: float = 0.0
    seed: int = 20260930
    amp: bool = True
    output_dir: str = "checkpoints/enhancement"

    def __post_init__(self) -> None:
        positive_integers = {
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "validation_interval": self.validation_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "gradient_accumulation": self.gradient_accumulation,
        }
        invalid = [name for name, value in positive_integers.items() if value <= 0]
        if invalid:
            raise ValueError(f"training fields must be positive: {invalid}")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
            or self.ssim_weight < 0
            or self.chroma_weight < 0
        ):
            raise ValueError(
                "learning rate or non-negative training weights are invalid"
            )
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig


def _only_known(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    known = set(cls.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return values


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("rb") as stream:
        data = tomllib.load(stream)

    allowed_sections = {"experiment", "dataset", "model", "training"}
    unknown_sections = set(data) - allowed_sections
    if unknown_sections:
        raise ValueError(f"Unknown config sections: {sorted(unknown_sections)}")

    experiment = data.get("experiment", {})
    name = experiment.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("[experiment].name must be a non-empty string")

    return ExperimentConfig(
        name=name,
        dataset=DatasetConfig(**_only_known(DatasetConfig, data.get("dataset", {}))),
        model=ModelConfig(**_only_known(ModelConfig, data.get("model", {}))),
        training=TrainingConfig(
            **_only_known(TrainingConfig, data.get("training", {}))
        ),
    )
