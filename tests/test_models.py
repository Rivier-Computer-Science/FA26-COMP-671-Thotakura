import pytest
import torch

from src.traffic_sign_recognition.models import (
    BaselineCNN,
    create_model,
)


def test_baseline_cnn_output_shape() -> None:
    model = BaselineCNN(num_classes=43)
    images = torch.randn(2, 3, 64, 64)

    output = model(images)

    assert output.shape == torch.Size([2, 43])


def test_model_factory_creates_baseline() -> None:
    model = create_model(
        model_name="baseline_cnn",
        num_classes=43,
    )

    assert isinstance(model, BaselineCNN)


def test_model_has_trainable_parameters() -> None:
    model = BaselineCNN()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert trainable_parameters
    assert sum(
        parameter.numel()
        for parameter in trainable_parameters
    ) > 0


def test_rejects_invalid_number_of_classes() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        BaselineCNN(num_classes=0)


def test_rejects_unknown_model_name() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        create_model("unknown_model")