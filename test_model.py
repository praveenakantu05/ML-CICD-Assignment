from model import model, X_test, y_test


def test_model_exists():
    assert model is not None


def test_predictions():
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)


def test_model_accuracy():
    accuracy = model.score(X_test, y_test)
    assert accuracy > 0.80
