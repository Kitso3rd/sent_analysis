import joblib

from src.train import TRAINING_TEXTS, train_model


def test_training_saves_loadable_artifacts_and_evaluation_metrics(tmp_path):
    model_path = tmp_path / "artifacts" / "model.pkl"
    vectorizer_path = tmp_path / "artifacts" / "vectorizer.pkl"

    model, vectorizer, metrics = train_model(model_path, vectorizer_path)

    assert model_path.is_file()
    assert vectorizer_path.is_file()
    assert len(TRAINING_TEXTS) >= 50
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "test_examples"}
    assert metrics["test_examples"] > 0
    for metric in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= metrics[metric] <= 1.0

    saved_model = joblib.load(model_path)
    saved_vectorizer = joblib.load(vectorizer_path)
    predictions = saved_model.predict(saved_vectorizer.transform(["I loved the experience"]))
    assert len(predictions) == 1
    assert predictions[0] in (0, 1)
    assert set(model.classes_) == {0, 1}
    assert vectorizer.transform(["great quality"]).shape[1] > 0


def test_importing_training_module_does_not_create_artifacts(tmp_path):
    # Importing train_model must not trigger training or write default artifacts.
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"
    assert not model_path.exists()
    assert not vectorizer_path.exists()
