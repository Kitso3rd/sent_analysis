from app.main import predict_sentiment, render_app


class FakeModel:
    classes_ = [0, 1]

    def __init__(self, prediction, probabilities=None):
        self.prediction = prediction
        self.probabilities = probabilities or ([0.8, 0.2] if prediction == 0 else [0.2, 0.8])
        self.seen_features = None

    def predict_proba(self, features):
        self.seen_features = features
        return [self.probabilities]


class FakeVectorizer:
    def __init__(self):
        self.seen_texts = None

    def transform(self, texts):
        self.seen_texts = texts
        return "features"


class FakeStreamlit:
    def __init__(self, text, should_predict=True):
        self.text = text
        self.should_predict = should_predict
        self.messages = []

    def title(self, message):
        self.messages.append(("title", message))

    def caption(self, message):
        self.messages.append(("caption", message))

    def text_area(self, label, height):
        self.messages.append(("input", label, height))
        return self.text

    def button(self, label):
        self.messages.append(("button", label))
        return self.should_predict

    def success(self, message):
        self.messages.append(("success", message))

    def error(self, message):
        self.messages.append(("error", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def metric(self, label, value):
        self.messages.append(("metric", label, value))


def test_predict_sentiment_returns_label_and_class_probability():
    vectorizer = FakeVectorizer()
    assert predict_sentiment("Great!", FakeModel(1, [0.1, 0.9]), vectorizer) == ("positive", 0.9)
    assert vectorizer.seen_texts == ["Great!"]
    assert predict_sentiment("Awful!", FakeModel(0, [0.85, 0.15]), vectorizer) == ("negative", 0.85)


def test_local_app_shows_prediction_confidence_and_label_legend():
    ui = FakeStreamlit("A great experience")
    model = FakeModel(1, [0.15, 0.85])
    vectorizer = FakeVectorizer()

    render_app(ui, model, vectorizer)

    assert vectorizer.seen_texts == ["A great experience"]
    assert model.seen_features == "features"
    assert ("success", "Sentiment: Positive") in ui.messages
    assert ("metric", "Model confidence", "85.0%") in ui.messages
    assert any("Label guide" in message for kind, message in ui.messages if kind == "caption")


def test_local_app_shows_negative_result_as_error():
    ui = FakeStreamlit("A frustrating experience")
    render_app(ui, FakeModel(0, [0.75, 0.25]), FakeVectorizer())
    assert ("error", "Sentiment: Negative") in ui.messages
    assert ("metric", "Model confidence", "75.0%") in ui.messages


def test_local_app_rejects_empty_input():
    ui = FakeStreamlit("  \n")
    vectorizer = FakeVectorizer()

    render_app(ui, FakeModel(1), vectorizer)

    assert ("warning", "Please enter some text to analyze.") in ui.messages
    assert vectorizer.seen_texts is None


def test_local_app_does_not_predict_without_button_click():
    ui = FakeStreamlit("A great experience", should_predict=False)
    vectorizer = FakeVectorizer()

    render_app(ui, FakeModel(1), vectorizer)

    assert vectorizer.seen_texts is None
    assert not any(kind in ("success", "error") for kind, *_ in ui.messages)
