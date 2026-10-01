import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Select the best available training device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def save_json(
        values: dict[str, Any],
        destination: Path,
) -> None:
    """Save a dictionary as formatted JSON."""
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open(
            "w",
            encoding="utf-8",
    ) as output_file:
        json.dump(values, output_file, indent=2)