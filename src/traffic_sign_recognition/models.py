import torch
from torch import nn

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
) -> nn.Module:
    """Create a model using its configuration name."""
    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)

    raise ValueError(f"Unsupported model: {model_name}")