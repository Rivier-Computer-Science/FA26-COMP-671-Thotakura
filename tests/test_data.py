import torch
from PIL import Image

from src.traffic_sign_recognition.data import (
    build_transform,
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


def test_transform_produces_expected_shape() -> None:
    image = Image.new("RGB", (40, 30), color="red")

    transformed_image = build_transform(64)(image)

    assert transformed_image.shape == torch.Size([3, 64, 64])
    assert transformed_image.dtype == torch.float32