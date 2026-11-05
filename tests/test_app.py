from app.main import predict_sentiment, render_app


class FakeModel:
    def __init__(self, prediction):
        self.prediction = prediction
        self.seen_features = None

    def predict(self, features):
        self.seen_features = features
        return [self.prediction]


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

    def text_input(self, label):
        self.messages.append(("input", label))
        return self.text

    def button(self, label):
        self.messages.append(("button", label))
        return self.should_predict

    def success(self, message):
        self.messages.append(("success", message))

    def warning(self, message):
        self.messages.append(("warning", message))


def test_predict_sentiment_maps_binary_labels():
    vectorizer = FakeVectorizer()
    assert predict_sentiment("Great!", FakeModel(1), vectorizer) == "positive"
    assert vectorizer.seen_texts == ["Great!"]
    assert predict_sentiment("Awful!", FakeModel(0), vectorizer) == "negative"


def test_local_app_shows_prediction():
    ui = FakeStreamlit("A great experience")
    model = FakeModel(1)
    vectorizer = FakeVectorizer()

    render_app(ui, model, vectorizer)

    assert vectorizer.seen_texts == ["A great experience"]
    assert model.seen_features == "features"
    assert ("success", "Sentiment: positive") in ui.messages


def test_local_app_rejects_empty_input():
    ui = FakeStreamlit("  \n")
    model = FakeModel(1)
    vectorizer = FakeVectorizer()

    render_app(ui, model, vectorizer)

    assert ("warning", "Please enter some text to analyze.") in ui.messages
    assert vectorizer.seen_texts is None


def test_local_app_does_not_predict_without_button_click():
    ui = FakeStreamlit("A great experience", should_predict=False)
    model = FakeModel(1)
    vectorizer = FakeVectorizer()

    render_app(ui, model, vectorizer)

    assert vectorizer.seen_texts is None
    assert not any(kind == "success" for kind, _ in ui.messages)
