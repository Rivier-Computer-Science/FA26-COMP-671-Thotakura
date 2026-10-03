import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .config import ExperimentConfig
from .data import NUM_CLASSES, create_dataloaders
from .engine import run_epoch
from .evaluation import evaluate_model
from .models import create_model
from .utils import save_json, select_device, set_seed


def save_history(
        history: list[dict[str, float | int]],
        destination: Path,
) -> None:
    """Save epoch-level training history."""
    if not history:
        raise ValueError("history cannot be empty")

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open(
            "w",
            newline="",
            encoding="utf-8",
    ) as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)


def train_experiment(
        config: ExperimentConfig,
) -> dict[str, float | int]:
    """Train, validate, checkpoint, and test one experiment."""
    torch.set_num_threads(4)
    set_seed(config.seed)
    device = select_device()

    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {device}")

    artifact_directory = Path(config.output_dir)
    checkpoint_directory = (
            artifact_directory / "checkpoints"
    )
    metric_directory = (
            artifact_directory
            / "metrics"
            / config.experiment_name
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    metric_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        asdict(config),
        metric_directory / "config.json",
        )

    loaders = create_dataloaders(
        data_dir=config.data_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
        num_workers=config.num_workers,
        augmentation=config.augmentation,
    )

    model = create_model(
        model_name=config.model,
        num_classes=NUM_CLASSES,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    checkpoint_path = (
            checkpoint_directory
            / f"{config.experiment_name}.pt"
    )

    for epoch in range(1, config.epochs + 1):
        training_result = run_epoch(
            model=model,
            loader=loaders.train,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        validation_result = run_epoch(
            model=model,
            loader=loaders.validation,
            criterion=criterion,
            device=device,
        )

        epoch_values = {
            "epoch": epoch,
            "training_loss": training_result.loss,
            "training_accuracy": (
                training_result.accuracy
            ),
            "validation_loss": (
                validation_result.loss
            ),
            "validation_accuracy": (
                validation_result.accuracy
            ),
        }

        history.append(epoch_values)

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train loss {training_result.loss:.4f} | "
            f"train accuracy "
            f"{training_result.accuracy:.4f} | "
            f"validation loss "
            f"{validation_result.loss:.4f} | "
            f"validation accuracy "
            f"{validation_result.accuracy:.4f}"
        )

        if validation_result.loss < best_validation_loss:
            best_validation_loss = validation_result.loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "validation_loss": (
                        best_validation_loss
                    ),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

            if (
                    epochs_without_improvement
                    >= config.early_stopping_patience
            ):
                print("Early stopping activated.")
                break

    save_history(
        history,
        metric_directory / "history.csv",
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    metrics = evaluate_model(
        model=model,
        loader=loaders.test,
        device=device,
        output_directory=metric_directory,
        num_classes=NUM_CLASSES,
    )

    results: dict[str, float | int] = {
        **metrics,
        "best_epoch": best_epoch,
        "best_validation_loss": (
            best_validation_loss
        ),
    }

    save_json(
        results,
        metric_directory / "summary.json",
        )

    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['macro_f1']:.4f}")
    print(f"Results saved to {metric_directory}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a GTSRB classification model."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to an experiment YAML file.",
    )

    arguments = parser.parse_args()
    config = ExperimentConfig.from_yaml(
        arguments.config
    )

    train_experiment(config)


if __name__ == "__main__":
    main()