from pathlib import Path

import joblib

from src.train import train_model


def test_training_saves_loadable_artifacts(tmp_path):
    model_path = tmp_path / "artifacts" / "model.pkl"
    vectorizer_path = tmp_path / "artifacts" / "vectorizer.pkl"

    train_model(model_path, vectorizer_path)

    assert model_path.is_file()
    assert vectorizer_path.is_file()

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    predictions = model.predict(vectorizer.transform(["I loved the experience"]))
    assert len(predictions) == 1
    assert predictions[0] in (0, 1)


def test_importing_training_module_does_not_create_artifacts(tmp_path, monkeypatch):
    # The training module has no import-time training or filesystem writes.
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"
    assert not model_path.exists()
    assert not vectorizer_path.exists()
