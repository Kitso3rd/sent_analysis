from pathlib import Path

import joblib
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.pkl"
VECTORIZER_PATH = APP_DIR / "vectorizer.pkl"


def load_artifacts(model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
    """Load the trained classifier and matching text vectorizer."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_sentiment(text, model, vectorizer):
    """Return the model's binary sentiment label for a single text string."""
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    return "positive" if prediction == 1 else "negative"


def render_app(ui, model, vectorizer):
    """Render the local prediction interface using the supplied Streamlit API."""
    ui.title("Sentiment Analysis")
    text = ui.text_input("Enter your text")

    if ui.button("Predict"):
        if not text.strip():
            ui.warning("Please enter some text to analyze.")
            return

        sentiment = predict_sentiment(text, model, vectorizer)
        ui.success(f"Sentiment: {sentiment}")


def main():
    try:
        model, vectorizer = load_artifacts()
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(
            "The sentiment model is not available. Run `python src/train.py` "
            f"to generate it. Details: {error}"
        )
        return

    render_app(st, model, vectorizer)


if __name__ == "__main__":
    main()
