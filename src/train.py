from pathlib import Path

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# Small example dataset. Replace with a larger, representative labeled dataset
# before treating the model as anything more than a demo.
TRAINING_TEXTS = [
    "I love this product!",
    "This is the worst service ever.",
    "Amazing experience.",
    "I will never buy this again.",
    "Highly recommend it!",
    "Terrible quality.",
    "Bad customer support.",
    "Good value for money.",
    "The product broke after one use.",
    "Fast delivery and great packaging.",
    "This is disappointing.",
    "Excellent service.",
    "Awful, just awful.",
    "Very bad.",
    "Not worth the price.",
    "Best purchase ever!",
    "Great performance!",
    "Poorly built and weak.",
    "Extremely satisfied.",
    "It does not work",
    "It is just okay",
    "need to improve.",
]
TRAINING_LABELS = [
    1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0,
]

APP_DIR = Path(__file__).resolve().parents[1] / "app"
DEFAULT_MODEL_PATH = APP_DIR / "model.pkl"
DEFAULT_VECTORIZER_PATH = APP_DIR / "vectorizer.pkl"


def train_model(model_path=DEFAULT_MODEL_PATH, vectorizer_path=DEFAULT_VECTORIZER_PATH):
    """Fit the demo sentiment model and save its model and vectorizer artifacts."""
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(TRAINING_TEXTS)

    model = LogisticRegression(random_state=42)
    model.fit(features, TRAINING_LABELS)

    model_path = Path(model_path)
    vectorizer_path = Path(vectorizer_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    return model, vectorizer


if __name__ == "__main__":
    train_model()
    print(f"Model and vectorizer saved to {APP_DIR}.")
