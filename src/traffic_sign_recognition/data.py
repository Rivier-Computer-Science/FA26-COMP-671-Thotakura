from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB


NUM_CLASSES = 43

GTSRB_MEAN = (0.3403, 0.3121, 0.3214)
GTSRB_STD = (0.2724, 0.2608, 0.2669)


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(GTSRB_MEAN, GTSRB_STD),
        ]
    )


def split_indices(
        dataset_size: int,
        validation_fraction: float,
        seed: int,
) -> tuple[list[int], list[int]]:
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

    validation_size = round(dataset_size * validation_fraction)
    validation_indices = shuffled_indices[:validation_size]
    training_indices = shuffled_indices[validation_size:]

    return training_indices, validation_indices


def create_dataloaders(
        data_dir: str | Path,
        image_size: int,
        batch_size: int,
        validation_fraction: float,
        seed: int,
        num_workers: int = 0,
) -> DataLoaders:
    transform = build_transform(image_size)

    complete_training_dataset = GTSRB(
        root=str(data_dir),
        split="train",
        download=True,
        transform=transform,
    )

    test_dataset = GTSRB(
        root=str(data_dir),
        split="test",
        download=True,
        transform=transform,
    )

    training_indices, validation_indices = split_indices(
        dataset_size=len(complete_training_dataset),
        validation_fraction=validation_fraction,
        seed=seed,
    )

    training_dataset = Subset(
        complete_training_dataset,
        training_indices,
    )
    validation_dataset = Subset(
        complete_training_dataset,
        validation_indices,
    )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    return DataLoaders(
        train=DataLoader(
            training_dataset,
            shuffle=True,
            **loader_options,
        ),
        validation=DataLoader(
            validation_dataset,
            shuffle=False,
            **loader_options,
        ),
        test=DataLoader(
            test_dataset,
            shuffle=False,
            **loader_options,
        ),
    )