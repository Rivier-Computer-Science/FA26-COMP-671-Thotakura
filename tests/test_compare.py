import json
from pathlib import Path

from src.traffic_sign_recognition.compare import (
    compare_experiments,
    load_experiment_results,
)


def write_summary(
        directory: Path,
        accuracy: float,
        macro_f1: float,
        best_epoch: int,
) -> None:
    directory.mkdir(parents=True)

    with (directory / "summary.json").open(
            "w",
            encoding="utf-8",
    ) as summary_file:
        json.dump(
            {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "best_epoch": best_epoch,
            },
            summary_file,
        )


def test_loads_and_orders_experiment_results(
        tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics"

    write_summary(
        metrics / "baseline_cnn",
        accuracy=0.85,
        macro_f1=0.80,
        best_epoch=10,
        )
    write_summary(
        metrics / "resnet18_finetuned",
        accuracy=0.94,
        macro_f1=0.92,
        best_epoch=8,
        )

    comparison = load_experiment_results(metrics)

    assert len(comparison) == 2
    assert comparison.iloc[0]["experiment"] == (
        "resnet18_finetuned"
    )
    assert comparison.iloc[0]["accuracy"] == 0.94


def test_creates_comparison_artifacts(
        tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics"
    output = tmp_path / "figures"

    write_summary(
        metrics / "baseline_cnn",
        accuracy=0.85,
        macro_f1=0.80,
        best_epoch=10,
        )
    write_summary(
        metrics / "resnet18_finetuned",
        accuracy=0.94,
        macro_f1=0.92,
        best_epoch=8,
        )

    compare_experiments(metrics, output)

    assert (
            output / "model_comparison.csv"
    ).exists()
    assert (
            output / "model_comparison.png"
    ).exists()