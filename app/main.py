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
    """Return the predicted sentiment and its model probability score."""
    features = vectorizer.transform([text])
    probabilities = model.predict_proba(features)[0]
    prediction_index = probabilities.argmax()
    prediction = model.classes_[prediction_index]
    sentiment = "positive" if prediction == 1 else "negative"
    confidence = float(probabilities[prediction_index])
    return sentiment, confidence


def render_app(ui, model, vectorizer):
    """Render the local prediction interface using the supplied Streamlit API."""
    ui.title("Sentiment Analysis")
    ui.caption("Label guide: Positive = favorable opinion; Negative = unfavorable opinion.")
    text = ui.text_area("Enter your text", height=150)

    if ui.button("Predict"):
        if not text.strip():
            ui.warning("Please enter some text to analyze.")
            return

        sentiment, confidence = predict_sentiment(text, model, vectorizer)
        message = f"Sentiment: {sentiment.title()}"
        if sentiment == "positive":
            ui.success(message)
        else:
            ui.error(message)
        ui.metric("Model confidence", f"{confidence:.1%}")
        ui.caption("Confidence is the model's probability score, not a guarantee of correctness.")


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
