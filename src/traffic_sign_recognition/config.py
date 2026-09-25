from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    seed: int = 42
    data_dir: str = "data"
    output_dir: str = "artifacts"
    image_size: int = 64
    batch_size: int = 64
    num_workers: int = 0
    validation_fraction: float = 0.15
    augmentation: bool = True
    model: str = "baseline_cnn"
    pretrained: bool = False
    freeze_backbone: bool = False
    epochs: int = 15
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 4

    def __post_init__(self) -> None:
        if not 0 < self.validation_fraction < 1:
            raise ValueError(
                "validation_fraction must be between 0 and 1"
            )

        if self.model not in {"baseline_cnn", "resnet18"}:
            raise ValueError(f"Unsupported model: {self.model}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open(encoding="utf-8") as config_file:
            values: dict[str, Any] = yaml.safe_load(config_file)

        return cls(**values)