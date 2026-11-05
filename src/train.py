from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


# Hand-labeled example phrases broaden the demo vocabulary, but remain a small,
# synthetic dataset; evaluation results should not be treated as production
# quality estimates.
TRAINING_DATA = [
    ("I love this product", 1),
    ("This is the worst service ever", 0),
    ("Amazing experience from start to finish", 1),
    ("I will never buy this again", 0),
    ("Highly recommend it to everyone", 1),
    ("Terrible quality and poor support", 0),
    ("Good value for money", 1),
    ("The product broke after one use", 0),
    ("Fast delivery and great packaging", 1),
    ("This is disappointing", 0),
    ("Excellent customer service", 1),
    ("Awful experience, just awful", 0),
    ("Best purchase I have ever made", 1),
    ("Not worth the price", 0),
    ("Great performance and easy to use", 1),
    ("Poorly built and feels very weak", 0),
    ("I am extremely satisfied", 1),
    ("It does not work properly", 0),
    ("The quality exceeded my expectations", 1),
    ("Customer support was rude and unhelpful", 0),
    ("Setup was simple and quick", 1),
    ("The instructions were confusing", 0),
    ("Works perfectly for my needs", 1),
    ("The app crashes constantly", 0),
    ("Comfortable, reliable, and well designed", 1),
    ("It stopped working within a week", 0),
    ("A pleasant surprise and a great deal", 1),
    ("I regret spending money on this", 0),
    ("The team resolved my issue quickly", 1),
    ("My issue was ignored for days", 0),
    ("Clear instructions and helpful features", 1),
    ("The materials feel cheap", 0),
    ("I would happily purchase this again", 1),
    ("I would not recommend this", 0),
    ("The experience was smooth and enjoyable", 1),
    ("Long wait and disappointing results", 0),
    ("Excellent value and dependable quality", 1),
    ("The item arrived damaged", 0),
    ("It made my daily routine much easier", 1),
    ("Very frustrating to use", 0),
    ("Friendly staff and prompt service", 1),
    ("The staff were dismissive", 0),
    ("Everything worked exactly as promised", 1),
    ("It failed to meet basic expectations", 0),
    ("A thoughtful design with useful details", 1),
    ("The battery barely lasts an hour", 0),
    ("Excellent results, I am impressed", 1),
    ("The repair did not solve the problem", 0),
    ("Easy ordering and on-time delivery", 1),
    ("The order arrived late and incomplete", 0),
    ("Very happy with my purchase", 1),
    ("Cheaply made and uncomfortable", 0),
    ("A reliable product that works well", 1),
    ("This was a complete waste of time", 0),
    ("The update made everything faster", 1),
    ("The latest update made it worse", 0),
    ("Helpful, responsive, and professional", 1),
    ("Slow response and no useful answer", 0),
    ("I enjoy using this every day", 1),
    ("I am unhappy with the overall quality", 0),
]

TRAINING_TEXTS = [text for text, _label in TRAINING_DATA]
TRAINING_LABELS = [label for _text, label in TRAINING_DATA]

APP_DIR = Path(__file__).resolve().parents[1] / "app"
DEFAULT_MODEL_PATH = APP_DIR / "model.pkl"
DEFAULT_VECTORIZER_PATH = APP_DIR / "vectorizer.pkl"


def train_model(model_path=DEFAULT_MODEL_PATH, vectorizer_path=DEFAULT_VECTORIZER_PATH):
    """Evaluate a held-out split, then train and save the demo model on all data."""
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        TRAINING_TEXTS,
        TRAINING_LABELS,
        test_size=0.25,
        random_state=42,
        stratify=TRAINING_LABELS,
    )

    evaluation_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    train_features = evaluation_vectorizer.fit_transform(train_texts)
    test_features = evaluation_vectorizer.transform(test_texts)
    evaluation_model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )
    evaluation_model.fit(train_features, train_labels)
    predictions = evaluation_model.predict(test_features)
    metrics = {
        "accuracy": accuracy_score(test_labels, predictions),
        "precision": precision_score(test_labels, predictions, zero_division=0),
        "recall": recall_score(test_labels, predictions, zero_division=0),
        "f1": f1_score(test_labels, predictions, zero_division=0),
        "test_examples": len(test_labels),
    }

    # Refit on all available examples for the artifact used by the demo.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    features = vectorizer.fit_transform(TRAINING_TEXTS)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(features, TRAINING_LABELS)

    model_path = Path(model_path)
    vectorizer_path = Path(vectorizer_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    return model, vectorizer, metrics


if __name__ == "__main__":
    _model, _vectorizer, evaluation = train_model()
    print(f"Model and vectorizer saved to {APP_DIR}.")
    print(
        "Held-out demo metrics "
        f"({evaluation['test_examples']} examples): "
        f"accuracy={evaluation['accuracy']:.2f}, "
        f"precision={evaluation['precision']:.2f}, "
        f"recall={evaluation['recall']:.2f}, f1={evaluation['f1']:.2f}"
    )
    print("Note: metrics use a small hand-labeled demo dataset.")
