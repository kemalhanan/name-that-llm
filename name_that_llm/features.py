"""Feature extraction for author (LLM) attribution.

The EDA (notebooks/1.0-eda.ipynb) showed that punctuation and quote style separate model
families almost by themselves, and vocabulary separates the models within a family. The
features below therefore keep punctuation and unicode intact.
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

# name -> regex; counted per 100 words
PUNCT_PATTERNS = {
    "em_dash": "—",
    "en_dash": "–",
    "semicolon": ";",
    "colon": ":",
    "comma": ",",
    "question": r"\?",
    "exclaim": "!",
    "parens": r"\(",
    "curly_quote": "[‘’“”]",
    "straight_quote": "['\"]",
    "digits": r"\d",
}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z']+")


def style_features(texts) -> pd.DataFrame:
    """Dense stylometric features: punctuation rates, length and lexical diversity."""
    s = pd.Series(list(texts), dtype=object)
    words = s.str.split().str.len().clip(lower=1)
    out = {name: s.str.count(pat) / words * 100 for name, pat in PUNCT_PATTERNS.items()}
    out["has_em_dash"] = s.str.contains("—").astype(float)
    out["has_curly"] = s.str.contains("[‘’“”]").astype(float)
    out["log_words"] = np.log(words)
    sentences = s.map(lambda t: max(1, len([x for x in _SENTENCE_SPLIT.split(t.strip()) if x])))
    out["words_per_sentence"] = words / sentences
    tokens = s.str.lower().map(_WORD.findall)
    out["avg_word_len"] = tokens.map(lambda w: np.mean([len(x) for x in w]) if w else 0.0)
    out["type_token"] = tokens.map(lambda w: len(set(w)) / max(1, len(w)))
    return pd.DataFrame(out).fillna(0.0)


class StyleFeatures(BaseEstimator, TransformerMixin):
    """sklearn wrapper around :func:`style_features`."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return style_features(X).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(style_features(["a b."]).columns, dtype=object)


def build_features(style_weight: float = 1.0) -> FeatureUnion:
    """Character n-grams + word n-grams + scaled stylometric features."""
    return FeatureUnion(
        [
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(1, 5),
                    lowercase=False,
                    sublinear_tf=True,
                    min_df=2,
                    max_features=300_000,
                    dtype=np.float32,
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    lowercase=True,
                    token_pattern=r"(?u)\b\w[\w']*\b",
                    sublinear_tf=True,
                    min_df=2,
                    dtype=np.float32,
                ),
            ),
            ("style", make_pipeline(StyleFeatures(), StandardScaler())),
        ],
        transformer_weights={"char": 1.0, "word": 1.0, "style": style_weight},
    )
