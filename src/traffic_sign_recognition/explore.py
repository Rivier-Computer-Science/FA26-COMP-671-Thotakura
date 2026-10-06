import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision.datasets import GTSRB

from .data import NUM_CLASSES


def build_class_distribution(
        training_labels: Iterable[int],
        test_labels: Iterable[int],
) -> pd.DataFrame:
    """Count examples in every GTSRB class."""
    training_counts = pd.Series(
        list(training_labels),
        dtype="int64",
    ).value_counts()

    test_counts = pd.Series(
        list(test_labels),
        dtype="int64",
    ).value_counts()

    class_ids = range(NUM_CLASSES)

    return pd.DataFrame(
        {
            "class_id": class_ids,
            "training_count": [
                int(training_counts.get(class_id, 0))
                for class_id in class_ids
            ],
            "test_count": [
                int(test_counts.get(class_id, 0))
                for class_id in class_ids
            ],
        }
    )


def extract_labels(dataset: GTSRB) -> list[int]:
    """Read labels from torchvision's GTSRB metadata."""
    samples = dataset._samples

    return [
        int(target)
        for _, target in samples
    ]


def save_distribution_plot(
        distribution: pd.DataFrame,
        output_path: Path,
) -> None:
    """Save a grouped training/test class-distribution chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positions = np.arange(NUM_CLASSES)
    bar_width = 0.4

    figure, axis = plt.subplots(figsize=(16, 6))

    axis.bar(
        positions - bar_width / 2,
        distribution["training_count"],
        width=bar_width,
        label="Training",
        )

    axis.bar(
        positions + bar_width / 2,
        distribution["test_count"],
        width=bar_width,
        label="Test",
        )

    axis.set_title("GTSRB Class Distribution")
    axis.set_xlabel("Traffic-sign class ID")
    axis.set_ylabel("Number of images")
    axis.set_xticks(positions)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_sample_images(
        dataset: GTSRB,
        output_path: Path,
) -> None:
    """Save representative examples from the training dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    indices = np.linspace(
        0,
        len(dataset) - 1,
        num=12,
        dtype=int,
        )

    figure, axes = plt.subplots(3, 4, figsize=(10, 8))

    for axis, index in zip(axes.flat, indices):
        image, label = dataset[int(index)]

        axis.imshow(image)
        axis.set_title(f"Class {label}")
        axis.axis("off")

    figure.suptitle("Representative GTSRB Training Images")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    data_directory = Path("data")
    figure_directory = Path("artifacts/figures")
    metric_directory = Path("artifacts/metrics")

    figure_directory.mkdir(parents=True, exist_ok=True)
    metric_directory.mkdir(parents=True, exist_ok=True)

    training_dataset = GTSRB(
        root=str(data_directory),
        split="train",
        download=True,
    )

    test_dataset = GTSRB(
        root=str(data_directory),
        split="test",
        download=True,
    )

    training_labels = extract_labels(training_dataset)
    test_labels = extract_labels(test_dataset)

    distribution = build_class_distribution(
        training_labels,
        test_labels,
    )

    distribution.to_csv(
        metric_directory / "class_distribution.csv",
        index=False,
        )

    save_distribution_plot(
        distribution,
        figure_directory / "class_distribution.png",
        )

    save_sample_images(
        training_dataset,
        figure_directory / "sample_images.png",
        )

    summary = {
        "training_images": len(training_dataset),
        "test_images": len(test_dataset),
        "number_of_classes": NUM_CLASSES,
        "smallest_training_class": int(
            distribution["training_count"].min()
        ),
        "largest_training_class": int(
            distribution["training_count"].max()
        ),
    }

    with (
            metric_directory / "dataset_summary.json"
    ).open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(json.dumps(summary, indent=2))
    print("Exploration artifacts created successfully.")


if __name__ == "__main__":
    main()