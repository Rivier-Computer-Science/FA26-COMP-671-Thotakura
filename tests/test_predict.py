from pathlib import Path

import pytest
import torch
from PIL import Image

from src.traffic_sign_recognition.models import (
    create_model,
)
from src.traffic_sign_recognition.predict import (
    predict_image,
)


def test_predict_image_returns_top_predictions(
        tmp_path: Path,
) -> None:
    model = create_model(
        "baseline_cnn"
    )

    checkpoint_path = tmp_path / "model.pt"

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "experiment_name": "test",
                "model": "baseline_cnn",
                "image_size": 64,
            },
        },
        checkpoint_path,
    )

    image_path = tmp_path / "sign.png"

    Image.new(
        "RGB",
        (40, 30),
        color="red",
    ).save(image_path)

    predictions = predict_image(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        top_k=3,
        device=torch.device("cpu"),
    )

    assert len(predictions) == 3

    assert (
            predictions[0]["confidence"]
            >= predictions[1]["confidence"]
    )

    assert set(predictions[0]) == {
        "class_id",
        "class_name",
        "confidence",
    }


@pytest.mark.parametrize(
    "top_k",
    [0, 44],
)
def test_predict_image_rejects_invalid_top_k(
        tmp_path: Path,
        top_k: int,
) -> None:
    with pytest.raises(
            ValueError,
            match="top_k",
    ):
        predict_image(
            image_path=tmp_path / "image.png",
            checkpoint_path=tmp_path / "model.pt",
            top_k=top_k,
            device=torch.device("cpu"),
        )