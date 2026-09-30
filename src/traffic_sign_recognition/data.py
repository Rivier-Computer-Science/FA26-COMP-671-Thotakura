from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB
from typing import Any, Callable


NUM_CLASSES = 43

GTSRB_MEAN = (0.3403, 0.3121, 0.3214)
GTSRB_STD = (0.2724, 0.2608, 0.2669)


@dataclass(frozen=True)
class DataLoaders:
    """Training, validation, and test DataLoaders."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader


def build_transform(
        image_size: int = 64,
        augment: bool = False,
) -> transforms.Compose:
    """Create preprocessing with optional training augmentation."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    operations: list[Callable[[Any], Any]] = [
        transforms.Resize((image_size, image_size)),
    ]

    if augment:
        operations.extend(
            [
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.08, 0.08),
                    scale=(0.9, 1.1),
                ),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                ),
            ]
        )

    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=GTSRB_MEAN,
                std=GTSRB_STD,
            ),
        ]
    )

    return transforms.Compose(operations)


def split_indices(
        dataset_size: int,
        validation_fraction: float,
        seed: int,
) -> tuple[list[int], list[int]]:
    """Create deterministic, non-overlapping train and validation indices."""
    if dataset_size <= 1:
        raise ValueError("dataset_size must be greater than 1")

    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be between 0 and 1"
        )

    generator = torch.Generator().manual_seed(seed)

    shuffled_indices = torch.randperm(
        dataset_size,
        generator=generator,
    ).tolist()

    validation_size = round(
        dataset_size * validation_fraction
    )

    # Ensure both subsets contain at least one example.
    validation_size = max(1, validation_size)
    validation_size = min(
        dataset_size - 1,
        validation_size,
        )

    validation_indices = shuffled_indices[:validation_size]
    training_indices = shuffled_indices[validation_size:]

    return training_indices, validation_indices


def create_dataloaders(
        data_dir: str | Path = "data",
        image_size: int = 64,
        batch_size: int = 64,
        validation_fraction: float = 0.15,
        seed: int = 42,
        num_workers: int = 0,
        augmentation: bool = True,
) -> DataLoaders:
    """Download GTSRB and create reproducible DataLoaders."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    training_transform = build_transform(
        image_size=image_size,
        augment=augmentation,
    )

    evaluation_transform = build_transform(
        image_size=image_size,
        augment=False,
    )

    # Training uses random augmentation.
    training_source = GTSRB(
        root=str(data_dir),
        split="train",
        download=True,
        transform=training_transform,
    )

    # Validation uses deterministic preprocessing.
    validation_source = GTSRB(
        root=str(data_dir),
        split="train",
        download=True,
        transform=evaluation_transform,
    )

    # The official test set also uses deterministic preprocessing.
    test_dataset = GTSRB(
        root=str(data_dir),
        split="test",
        download=True,
        transform=evaluation_transform,
    )

    training_indices, validation_indices = split_indices(
        dataset_size=len(training_source),
        validation_fraction=validation_fraction,
        seed=seed,
    )

    training_dataset = Subset(
        training_source,
        training_indices,
    )

    validation_dataset = Subset(
        validation_source,
        validation_indices,
    )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    shuffle_generator = torch.Generator().manual_seed(seed)

    training_loader = DataLoader(
        training_dataset,
        shuffle=True,
        generator=shuffle_generator,
        **loader_options,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_options,
    )

    return DataLoaders(
        train=training_loader,
        validation=validation_loader,
        test=test_loader,
    )