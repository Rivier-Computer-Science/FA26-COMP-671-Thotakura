import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_METRICS = {"accuracy", "macro_f1"}


def load_experiment_results(
        metrics_directory: Path,
) -> pd.DataFrame:
    """Load summary files produced by completed experiments."""
    rows: list[dict[str, object]] = []

    for summary_path in sorted(
            metrics_directory.glob("*/summary.json")
    ):
        with summary_path.open(
                encoding="utf-8"
        ) as summary_file:
            summary = json.load(summary_file)

        missing_metrics = (
                REQUIRED_METRICS - summary.keys()
        )

        if missing_metrics:
            missing = ", ".join(sorted(missing_metrics))
            raise ValueError(
                f"{summary_path} is missing: {missing}"
            )

        rows.append(
            {
                "experiment": summary_path.parent.name,
                "accuracy": float(summary["accuracy"]),
                "macro_f1": float(summary["macro_f1"]),
                "best_epoch": summary.get("best_epoch"),
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No summary.json files found in "
            f"{metrics_directory}"
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by="macro_f1",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def save_comparison(
        comparison: pd.DataFrame,
        output_directory: Path,
) -> None:
    """Save a comparison table and grouped metric chart."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        output_directory / "model_comparison.csv",
        index=False,
        )

    figure, axis = plt.subplots(
        figsize=(10, 5),
    )

    comparison.set_index("experiment")[
        ["accuracy", "macro_f1"]
    ].plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title(
        "GTSRB Model Performance Comparison"
    )
    axis.set_xlabel("Experiment")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.tick_params(
        axis="x",
        rotation=15,
    )
    axis.grid(
        axis="y",
        alpha=0.25,
    )
    axis.legend(
        ["Accuracy", "Macro F1"],
    )

    figure.tight_layout()
    figure.savefig(
        output_directory / "model_comparison.png",
        dpi=180,
        )
    plt.close(figure)


def compare_experiments(
        metrics_directory: Path,
        output_directory: Path,
) -> pd.DataFrame:
    comparison = load_experiment_results(
        metrics_directory
    )
    save_comparison(
        comparison,
        output_directory,
    )

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare completed model experiments."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/metrics"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/figures"),
    )

    arguments = parser.parse_args()

    comparison = compare_experiments(
        arguments.metrics,
        arguments.output,
    )

    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()