from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochResult:
    """Aggregate metrics from one model epoch."""

    loss: float
    accuracy: float


def run_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
) -> EpochResult:
    """Run one training or evaluation epoch."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
                logits.argmax(dim=1) == labels
        ).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("DataLoader cannot be empty")

    return EpochResult(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
    )