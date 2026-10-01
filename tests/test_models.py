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

def test_resnet18_output_shape_without_download() -> None:
    model = create_model(
        model_name="resnet18",
        num_classes=43,
        pretrained=False,
        freeze_backbone=False,
    )

    images = torch.randn(2, 3, 64, 64)
    output = model(images)

    assert output.shape == torch.Size([2, 43])


def test_resnet_fine_tuning_enables_all_parameters() -> None:
    model = create_model(
        model_name="resnet18",
        num_classes=43,
        pretrained=False,
        freeze_backbone=False,
    )

    assert all(
        parameter.requires_grad
        for parameter in model.parameters()
    )


def test_frozen_resnet_only_trains_final_layer() -> None:
    model = create_model(
        model_name="resnet18",
        num_classes=43,
        pretrained=False,
        freeze_backbone=True,
    )

    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("fc.")
    ]

    final_layer_parameters = list(model.fc.parameters())

    assert all(
        not parameter.requires_grad
        for parameter in backbone_parameters
    )

    assert all(
        parameter.requires_grad
        for parameter in final_layer_parameters
    )