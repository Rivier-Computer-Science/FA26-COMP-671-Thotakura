"""Analyze incorrect predictions and difficult traffic-sign classes."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "target",
    "prediction",
    "confidence",
    "correct",
}


def analyze_failures(
        predictions_path: Path,
        output_directory: Path,
        top_classes: int = 10,
) -> dict[str, int | float]:
    """Create tables and plots describing model errors."""
    if top_classes <= 0:
        raise ValueError("top_classes must be positive")

    predictions = pd.read_csv(predictions_path)

    missing_columns = REQUIRED_COLUMNS.difference(
        predictions.columns
    )

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Missing prediction columns: {missing}"
        )

    if predictions.empty:
        raise ValueError(
            "predictions file cannot be empty"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions["correct"] = (
        predictions["correct"].astype(bool)
    )

    incorrect = predictions.loc[
        ~predictions["correct"]
    ].copy()

    incorrect = incorrect.sort_values(
        "confidence",
        ascending=False,
    )

    incorrect.to_csv(
        output_directory / "incorrect_predictions.csv",
        index=False,
        )

    class_summary = (
        predictions.groupby("target", as_index=False)
        .agg(
            examples=("target", "size"),
            correct_predictions=("correct", "sum"),
            mean_confidence=("confidence", "mean"),
        )
    )

    class_summary["errors"] = (
            class_summary["examples"]
            - class_summary["correct_predictions"]
    )

    class_summary["accuracy"] = (
            class_summary["correct_predictions"]
            / class_summary["examples"]
    )

    class_summary["error_rate"] = (
            class_summary["errors"]
            / class_summary["examples"]
    )

    class_summary = class_summary.sort_values(
        ["error_rate", "errors"],
        ascending=False,
    )

    class_summary.to_csv(
        output_directory / "difficult_classes.csv",
        index=False,
        )

    confused_pairs = (
        incorrect.groupby(
            ["target", "prediction"],
            as_index=False,
        )
        .size()
        .sort_values(
            "size",
            ascending=False,
        )
    )

    confused_pairs.to_csv(
        output_directory / "confused_class_pairs.csv",
        index=False,
        )

    difficult = (
        class_summary.head(top_classes)
        .sort_values("error_rate")
    )

    figure, axis = plt.subplots(
        figsize=(10, 6),
    )

    axis.barh(
        difficult["target"].astype(str),
        difficult["error_rate"],
    )

    axis.set_title(
        "Most Difficult GTSRB Classes"
    )
    axis.set_xlabel("Test error rate")
    axis.set_ylabel("True class ID")
    axis.set_xlim(left=0)

    figure.tight_layout()
    figure.savefig(
        output_directory / "difficult_classes.png",
        dpi=180,
        )
    plt.close(figure)

    summary: dict[str, int | float] = {
        "examples": int(len(predictions)),
        "correct_predictions": int(
            predictions["correct"].sum()
        ),
        "incorrect_predictions": int(
            len(incorrect)
        ),
        "error_rate": float(
            len(incorrect) / len(predictions)
        ),
        "high_confidence_errors": int(
            (incorrect["confidence"] >= 0.90).sum()
        ),
    }

    with (
            output_directory / "failure_summary.json"
    ).open("w", encoding="utf-8") as summary_file:
        json.dump(
            summary,
            summary_file,
            indent=2,
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze saved traffic-sign predictions."
        )
    )

    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--top-classes",
        type=int,
        default=10,
    )

    arguments = parser.parse_args()

    summary = analyze_failures(
        predictions_path=arguments.predictions,
        output_directory=arguments.output_dir,
        top_classes=arguments.top_classes,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()