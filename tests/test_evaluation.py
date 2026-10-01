import json
from pathlib import Path

import pytest

from src.traffic_sign_recognition.evaluation import (
    evaluate_predictions,
)


def test_evaluate_predictions_creates_artifacts(
        tmp_path: Path,
) -> None:
    metrics = evaluate_predictions(
        targets=[0, 0, 1, 1],
        predictions=[0, 1, 1, 1],
        confidences=[0.9, 0.6, 0.8, 0.7],
        output_directory=tmp_path,
        num_classes=2,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(
        0.733333,
        rel=1e-5,
    )

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "predictions.csv").exists()
    assert (
            tmp_path / "classification_report.csv"
    ).exists()
    assert (
            tmp_path / "confusion_matrix.png"
    ).exists()

    with (tmp_path / "summary.json").open(
            encoding="utf-8"
    ) as summary_file:
        saved_metrics = json.load(summary_file)

    assert saved_metrics["accuracy"] == pytest.approx(
        0.75
    )


def test_rejects_mismatched_prediction_lengths(
        tmp_path: Path,
) -> None:
    with pytest.raises(
            ValueError,
            match="equal lengths",
    ):
        evaluate_predictions(
            targets=[0, 1],
            predictions=[0],
            confidences=[0.9, 0.8],
            output_directory=tmp_path,
            num_classes=2,
        )


def test_rejects_empty_targets(
        tmp_path: Path,
) -> None:
    with pytest.raises(
            ValueError,
            match="targets cannot be empty",
    ):
        evaluate_predictions(
            targets=[],
            predictions=[],
            confidences=[],
            output_directory=tmp_path,
            num_classes=2,
        )