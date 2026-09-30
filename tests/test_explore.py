from src.traffic_sign_recognition.explore import (
    build_class_distribution,
)


def test_build_class_distribution_counts_every_class() -> None:
    distribution = build_class_distribution(
        training_labels=[0, 0, 1, 2, 2, 2],
        test_labels=[0, 1, 1, 2],
    )

    assert len(distribution) == 43

    assert distribution.loc[0, "training_count"] == 2
    assert distribution.loc[1, "training_count"] == 1
    assert distribution.loc[2, "training_count"] == 3

    assert distribution.loc[0, "test_count"] == 1
    assert distribution.loc[1, "test_count"] == 2
    assert distribution.loc[2, "test_count"] == 1

    assert distribution.loc[42, "training_count"] == 0