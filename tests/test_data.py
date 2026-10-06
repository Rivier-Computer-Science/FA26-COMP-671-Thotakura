import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.traffic_sign_recognition.data import (
    build_transform,
    create_dataloaders,
    split_indices,
)


def test_split_is_deterministic() -> None:
    first_train, first_validation = split_indices(100, 0.15, 42)
    second_train, second_validation = split_indices(100, 0.15, 42)

    assert first_train == second_train
    assert first_validation == second_validation


def test_split_sizes_and_no_overlap() -> None:
    training, validation = split_indices(100, 0.15, 42)

    assert len(training) == 85
    assert len(validation) == 15
    assert set(training).isdisjoint(validation)
    assert len(set(training + validation)) == 100


def test_small_dataset_keeps_both_subsets_nonempty() -> None:
    training, validation = split_indices(2, 0.15, 42)

    assert len(training) == 1
    assert len(validation) == 1


@pytest.mark.parametrize(
    "validation_fraction",
    [0.0, 1.0, -0.1, 1.1],
)
def test_rejects_invalid_validation_fraction(
        validation_fraction: float,
) -> None:
    with pytest.raises(
            ValueError,
            match="validation_fraction",
    ):
        split_indices(100, validation_fraction, 42)


def test_transform_produces_expected_shape() -> None:
    image = Image.new("RGB", (40, 30), color="red")

    transformed_image = build_transform(64)(image)

    assert transformed_image.shape == torch.Size([3, 64, 64])
    assert transformed_image.dtype == torch.float32


def test_rejects_invalid_image_size() -> None:
    with pytest.raises(ValueError, match="image_size"):
        build_transform(0)


def test_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        create_dataloaders(batch_size=0)


def test_rejects_negative_num_workers() -> None:
    with pytest.raises(ValueError, match="num_workers"):
        create_dataloaders(num_workers=-1)

def test_augmented_transform_produces_expected_shape() -> None:
    image = Image.new("RGB", (40, 30), color="red")

    transformed_image = build_transform(
        image_size=64,
        augment=True,
    )(image)

    assert transformed_image.shape == torch.Size([3, 64, 64])
    assert transformed_image.dtype == torch.float32


def test_augmentation_pipeline_contains_random_operations() -> None:
    pipeline = build_transform(
        image_size=64,
        augment=True,
    )

    assert any(
        isinstance(operation, transforms.RandomRotation)
        for operation in pipeline.transforms
    )
    assert any(
        isinstance(operation, transforms.RandomAffine)
        for operation in pipeline.transforms
    )
    assert any(
        isinstance(operation, transforms.ColorJitter)
        for operation in pipeline.transforms
    )


def test_evaluation_pipeline_has_no_random_augmentation() -> None:
    pipeline = build_transform(
        image_size=64,
        augment=False,
    )

    random_types = (
        transforms.RandomRotation,
        transforms.RandomAffine,
        transforms.ColorJitter,
    )

    assert not any(
        isinstance(operation, random_types)
        for operation in pipeline.transforms
    )