from pathlib import Path

import pandas as pd
import pytest

from src.traffic_sign_recognition.train import save_history
from src.traffic_sign_recognition.utils import (
    save_json,
    set_seed,
)


def test_save_history_creates_csv(
        tmp_path: Path,
) -> None:
    destination = tmp_path / "history.csv"

    save_history(
        [
            {
                "epoch": 1,
                "training_loss": 1.0,
                "training_accuracy": 0.5,
                "validation_loss": 0.9,
                "validation_accuracy": 0.6,
            }
        ],
        destination,
    )

    saved_history = pd.read_csv(destination)

    assert destination.exists()
    assert len(saved_history) == 1
    assert saved_history.iloc[0]["epoch"] == 1


def test_save_history_rejects_empty_history(
        tmp_path: Path,
) -> None:
    with pytest.raises(
            ValueError,
            match="history cannot be empty",
    ):
        save_history(
            [],
            tmp_path / "history.csv",
            )


def test_save_json_creates_parent_directories(
        tmp_path: Path,
) -> None:
    destination = (
            tmp_path / "nested" / "summary.json"
    )

    save_json(
        {"accuracy": 0.9},
        destination,
    )

    assert destination.exists()


def test_set_seed_reproduces_torch_values() -> None:
    import torch

    set_seed(42)
    first_values = torch.rand(3)

    set_seed(42)
    second_values = torch.rand(3)

    assert torch.equal(
        first_values,
        second_values,
    )