import json
from pathlib import Path

import matplotlib
from tqdm.auto import tqdm

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader

from .data import NUM_CLASSES


def collect_predictions(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    """Collect labels, predictions, and confidence scores."""
    model.eval()

    targets: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []

    with torch.no_grad():
        for images, labels in tqdm(
                loader,
                desc="Testing",
                unit="batch",
        ):
            images = images.to(device)

            logits = model(images)
            probabilities = logits.softmax(dim=1)

            confidence, prediction = probabilities.max(dim=1)

            targets.extend(labels.tolist())
            predictions.extend(
                prediction.cpu().tolist()
            )
            confidences.extend(
                confidence.cpu().tolist()
            )

    if not targets:
        raise ValueError("DataLoader cannot be empty")

    return targets, predictions, confidences


def evaluate_predictions(
        targets: list[int],
        predictions: list[int],
        confidences: list[float],
        output_directory: Path,
        num_classes: int = NUM_CLASSES,
) -> dict[str, float]:
    """Calculate metrics and save evaluation artifacts."""
    if not targets:
        raise ValueError("targets cannot be empty")

    if not (
            len(targets)
            == len(predictions)
            == len(confidences)
    ):
        raise ValueError(
            "targets, predictions, and confidences "
            "must have equal lengths"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = {
        "accuracy": float(
            accuracy_score(targets, predictions)
        ),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }

    predictions_table = pd.DataFrame(
        {
            "target": targets,
            "prediction": predictions,
            "confidence": confidences,
        }
    )

    predictions_table["correct"] = (
            predictions_table["target"]
            == predictions_table["prediction"]
    )

    predictions_table.to_csv(
        output_directory / "predictions.csv",
        index=False,
        )

    class_ids = list(range(num_classes))

    report = classification_report(
        targets,
        predictions,
        labels=class_ids,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        output_directory / "classification_report.csv"
    )

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=class_ids,
    )

    figure, axis = plt.subplots(
        figsize=(14, 12),
    )

    sns.heatmap(
        matrix,
        cmap="Blues",
        ax=axis,
        cbar=True,
    )

    axis.set_title("GTSRB Confusion Matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")

    figure.tight_layout()
    figure.savefig(
        output_directory / "confusion_matrix.png",
        dpi=180,
        )
    plt.close(figure)

    with (
            output_directory / "summary.json"
    ).open("w", encoding="utf-8") as summary_file:
        json.dump(
            metrics,
            summary_file,
            indent=2,
        )

    return metrics


def evaluate_model(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        output_directory: Path,
        num_classes: int = NUM_CLASSES,
) -> dict[str, float]:
    """Evaluate a model and save all artifacts."""
    targets, predictions, confidences = collect_predictions(
        model=model,
        loader=loader,
        device=device,
    )

    return evaluate_predictions(
        targets=targets,
        predictions=predictions,
        confidences=confidences,
        output_directory=output_directory,
        num_classes=num_classes,
    )