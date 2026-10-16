"""Loads the trained artifacts once and exposes predict_text()."""
import os

import joblib

from preprocess import clean_data

_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
_vectorizer = joblib.load(os.path.join(_ART, "vectorizer.pkl"))
_model = joblib.load(os.path.join(_ART, "model.pkl"))

CLASSES = [str(c) for c in _model.classes_]


def predict_text(text: str) -> dict:
    """Return {"label", "confidence", "probabilities"} for one piece of text.

    Raises ValueError if nothing usable is left after cleaning
    (e.g. the input was only a URL, emojis or numbers).
    """
    cleaned = clean_data(text)
    if not cleaned.strip():
        raise ValueError("No usable words found in the text. Try a full sentence.")
    probs = _model.predict_proba(_vectorizer.transform([cleaned]))[0]
    probabilities = {c: float(p) for c, p in zip(CLASSES, probs)}
    label = max(probabilities, key=probabilities.get)
    return {
        "label": label,
        "confidence": probabilities[label],
        "probabilities": probabilities,
    }
