import re
import unicodedata


def preprocess(text: str) -> str:
    """Keep sentiment-bearing words, including negation and hashtag text."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.replace("_", " ")
    return " ".join(text.split())
