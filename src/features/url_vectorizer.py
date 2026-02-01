from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class UrlVectorizerBundle:
    vectorizer: TfidfVectorizer
    X: np.ndarray


def fit_url_vectorizer(urls: List[str], max_features: int = 8000, ngram_max: int = 5) -> UrlVectorizerBundle:
    v = TfidfVectorizer(analyzer='char', ngram_range=(3, ngram_max), max_features=max_features, lowercase=True)
    X = v.fit_transform(urls).astype(np.float32).toarray()
    return UrlVectorizerBundle(vectorizer=v, X=X)


def transform_url_vectorizer(v: TfidfVectorizer, urls: List[str]) -> np.ndarray:
    return v.transform(urls).astype(np.float32).toarray()
