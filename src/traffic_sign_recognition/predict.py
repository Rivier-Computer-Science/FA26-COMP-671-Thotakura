"""Run inference on one traffic-sign image."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from .config import ExperimentConfig
from .data import NUM_CLASSES, build_transform
from .labels import class_name
from .models import create_model
from .utils import select_device


def load_checkpoint_model(
        checkpoint_path: Path,
        device: torch.device,
) -> tuple[nn.Module, ExperimentConfig]:
    """Load a trained model and its configuration."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if (
            "model_state" not in checkpoint
            or "config" not in checkpoint
    ):
        raise ValueError(
            "Checkpoint must contain model_state and config"
        )

    config = ExperimentConfig(
        **checkpoint["config"]
    )

    # Pretrained weights are unnecessary because the
    # trained state is loaded immediately afterward.
    model = create_model(
        model_name=config.model,
        num_classes=NUM_CLASSES,
        pretrained=False,
        freeze_backbone=False,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )
    model.eval()

    return model, config


def predict_image(
        image_path: Path,
        checkpoint_path: Path,
        top_k: int = 3,
        device: torch.device | None = None,
) -> list[dict[str, int | float | str]]:
    """Return the top predictions for one image."""
    if not 1 <= top_k <= NUM_CLASSES:
        raise ValueError(
            f"top_k must be between 1 and {NUM_CLASSES}"
        )

    selected_device = device or select_device()

    model, config = load_checkpoint_model(
        checkpoint_path,
        selected_device,
    )

    transform = build_transform(
        image_size=config.image_size,
        augment=False,
    )

    with Image.open(image_path) as image:
        image_tensor = transform(
            image.convert("RGB")
        ).unsqueeze(0)

    with torch.no_grad():
        logits = model(
            image_tensor.to(selected_device)
        )
        probabilities = logits.softmax(dim=1)[0]

    confidence_values, class_ids = probabilities.topk(
        top_k
    )

    confidences = confidence_values.cpu().tolist()
    predicted_ids = class_ids.cpu().tolist()

    return [
        {
            "class_id": class_id,
            "class_name": class_name(class_id),
            "confidence": confidence,
        }
        for class_id, confidence in zip(
            predicted_ids,
            confidences,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify one traffic-sign image."
        )
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to an input image.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a trained model checkpoint.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of predictions to display.",
    )

    arguments = parser.parse_args()

    predictions = predict_image(
        image_path=arguments.image,
        checkpoint_path=arguments.checkpoint,
        top_k=arguments.top_k,
    )

    for rank, prediction in enumerate(
            predictions,
            start=1,
    ):
        print(
            f"{rank}. {prediction['class_name']} "
            f"(class {prediction['class_id']}): "
            f"{prediction['confidence']:.2%}"
        )


if __name__ == "__main__":
    main()