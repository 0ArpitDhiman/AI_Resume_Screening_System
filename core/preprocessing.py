"""
preprocessing.py
-----------------
Lightweight NLP preprocessing for resumes and job descriptions.

Deliberately avoids NLTK/spaCy model downloads (nltk.download(), spacy
`en_core_web_sm`, etc.) so the project runs offline / on any machine
without an extra download step -- a common pain point for student
submissions. Uses scikit-learn's built-in English stopword list plus a
small custom set of resume/JD boilerplate words, and a regex-based
tokenizer + simple suffix-stripping stemmer.
"""

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# A few resume/JD-specific filler words worth stripping in addition to the
# standard English stopword list.
EXTRA_STOPWORDS = {
    "resume", "cv", "curriculum", "vitae", "responsibilities", "requirements",
    "experience", "years", "year", "job", "role", "position", "candidate",
    "company", "team", "work", "working", "skills", "skill", "please",
    "email", "phone", "address", "references", "available", "request",
}

STOPWORDS = ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS)

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z+#./-]*")


def clean_text(text: str) -> str:
    """Lowercase, strip emails/URLs/phone numbers/special chars, collapse whitespace."""
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\S+@\S+", " ", text)                     # emails
    text = re.sub(r"http\S+|www\.\S+", " ", text)             # URLs
    text = re.sub(r"\+?\d[\d\s\-().]{7,}\d", " ", text)       # phone numbers
    text = re.sub(r"[^a-z0-9+#./\s-]", " ", text)              # keep tech symbols (c++, c#)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    return TOKEN_PATTERN.findall(text)


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def preprocess(text: str) -> str:
    """
    Full pipeline: clean -> tokenize -> remove stopwords -> rejoin.
    Returns a cleaned string ready for TF-IDF vectorization.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)
