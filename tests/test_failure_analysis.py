from pathlib import Path

import pandas as pd
import pytest

from src.traffic_sign_recognition.failure_analysis import (
    analyze_failures,
)


def test_failure_analysis_creates_artifacts(
        tmp_path: Path,
) -> None:
    predictions_path = (
            tmp_path / "predictions.csv"
    )

    pd.DataFrame(
        {
            "target": [0, 0, 1, 1],
            "prediction": [0, 1, 1, 2],
            "confidence": [
                0.90,
                0.95,
                0.85,
                0.60,
            ],
            "correct": [
                True,
                False,
                True,
                False,
            ],
        }
    ).to_csv(
        predictions_path,
        index=False,
    )

    output_directory = tmp_path / "analysis"

    summary = analyze_failures(
        predictions_path,
        output_directory,
        top_classes=2,
    )

    assert summary["examples"] == 4
    assert summary["incorrect_predictions"] == 2
    assert summary["error_rate"] == 0.5
    assert summary["high_confidence_errors"] == 1

    assert (
            output_directory
            / "incorrect_predictions.csv"
    ).exists()

    assert (
            output_directory
            / "difficult_classes.csv"
    ).exists()

    assert (
            output_directory
            / "confused_class_pairs.csv"
    ).exists()

    assert (
            output_directory
            / "difficult_classes.png"
    ).exists()

    assert (
            output_directory
            / "failure_summary.json"
    ).exists()


def test_failure_analysis_rejects_missing_columns(
        tmp_path: Path,
) -> None:
    predictions_path = (
            tmp_path / "predictions.csv"
    )

    pd.DataFrame(
        {"target": [0]}
    ).to_csv(
        predictions_path,
        index=False,
    )

    with pytest.raises(
            ValueError,
            match="Missing prediction columns",
    ):
        analyze_failures(
            predictions_path,
            tmp_path / "analysis",
            )


def test_failure_analysis_rejects_invalid_top_classes(
        tmp_path: Path,
) -> None:
    with pytest.raises(
            ValueError,
            match="top_classes",
    ):
        analyze_failures(
            tmp_path / "missing.csv",
            tmp_path / "analysis",
            top_classes=0,
            )