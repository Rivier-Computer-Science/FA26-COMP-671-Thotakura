import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.traffic_sign_recognition.engine import run_epoch


def create_test_loader() -> DataLoader:
    images = torch.randn(8, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    dataset = TensorDataset(images, labels)

    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )


def create_test_model() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 2),
    )


def test_evaluation_epoch_returns_metrics() -> None:
    model = create_test_model()
    loader = create_test_loader()
    criterion = nn.CrossEntropyLoss()

    result = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert result.loss >= 0
    assert 0 <= result.accuracy <= 1


def test_training_epoch_updates_parameters() -> None:
    model = create_test_model()
    loader = create_test_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    result = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
        optimizer=optimizer,
    )

    parameters_after = list(model.parameters())

    assert result.loss >= 0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            parameters_after,
        )
    )