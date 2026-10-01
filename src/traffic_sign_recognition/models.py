import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .data import NUM_CLASSES


class BaselineCNN(nn.Module):
    """Small CNN baseline for 64x64 traffic-sign images."""

    def __init__(
            self,
            num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes),
        )

    def forward(
            self,
            images: torch.Tensor,
    ) -> torch.Tensor:
        features = self.features(images)
        logits = self.classifier(features)

        return logits


def create_model(
        model_name: str,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = False,
        freeze_backbone: bool = False,
) -> nn.Module:
    """Create a baseline CNN or ResNet-18 classifier."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)

    if model_name == "resnet18":
        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        model = resnet18(weights=weights)

        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False

        input_features = model.fc.in_features
        model.fc = nn.Linear(input_features, num_classes)

        return model

    raise ValueError(f"Unsupported model: {model_name}")