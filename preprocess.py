"""Shared text cleaning. Used by BOTH train.py and app.py so training and
serving always clean text the same way."""
import re
import string

import nltk
from nltk.stem import PorterStemmer

try:
    from nltk.corpus import stopwords
    _STOP = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    _STOP = set(stopwords.words("english"))

# Negations carry meaning ("not offensive"), so we keep them.
_STOP -= {"no", "nor", "not"}

_stemmer = PorterStemmer()
_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HTML_ENTITY = re.compile(r"&#?\w+;")
_BRACKETS = re.compile(r"\[.*?\]")
_HTML_TAG = re.compile(r"<.*?>+")
_DIGIT_WORDS = re.compile(r"\w*\d\w*")
_PUNCT = re.compile("[%s]" % re.escape(string.punctuation))
_RT = re.compile(r"^\s*rt\b")


def clean_data(text) -> str:
    if text is None or (isinstance(text, float) and text != text):  # None / NaN
        return ""
    text = str(text).lower()
    text = _RT.sub(" ", text)            # retweet marker
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _HTML_ENTITY.sub(" ", text)   # &amp; &#8220; etc.
    text = _BRACKETS.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _PUNCT.sub(" ", text)         # space, not "", so words don't merge
    text = _DIGIT_WORDS.sub(" ", text)
    tokens = [w for w in text.split() if w not in _STOP]
    return " ".join(_stemmer.stem(w) for w in tokens)
