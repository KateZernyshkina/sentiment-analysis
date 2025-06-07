import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def clean_text(text, stopwords_language="russian"):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    try:
        words = word_tokenize(text)
    except Exception:
        nltk.download("punkt_tab")
        words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words(stopwords_language)]
    return " ".join(words)
