from src.traffic_sign_recognition.data import split_indices


def test_split_is_deterministic() -> None:
    first_train, first_validation = split_indices(100, 0.15, 42)
    second_train, second_validation = split_indices(100, 0.15, 42)

    assert first_train == second_train
    assert first_validation == second_validation


def test_split_sizes_and_no_overlap() -> None:
    training, validation = split_indices(100, 0.15, 42)

    assert len(training) == 85
    assert len(validation) == 15
    assert set(training).isdisjoint(validation)