"""src/keyword_search.py

Name: Hansen Donkor
Index Number: 10012200059

Keyword retrieval using TF-IDF.

Requirements satisfied:
- Implement TF-IDF keyword search
- Provide top-k retrieval with similarity scores

Design notes:
- TF-IDF helps with exact matches (names, numbers, specific phrases).
- We use L2-normalized TF-IDF vectors so dot product == cosine similarity.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger(__name__)


class KeywordSearch:
    """TF-IDF based keyword search."""

    def __init__(
        self,
        stop_words: str | None = "english",
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
    ) -> None:
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.min_df = min_df

        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def fit(self, texts: List[str]) -> None:
        if texts is None:
            raise ValueError("texts cannot be None")
        if len(texts) == 0:
            raise ValueError("texts cannot be empty")

        vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            norm="l2",
        )

        matrix = vectorizer.fit_transform(texts)

        self._vectorizer = vectorizer
        self._matrix = matrix

        logger.info("KeywordSearch fitted (n_docs=%d, vocab=%d)", len(texts), len(vectorizer.vocabulary_))

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Return (doc_idx, similarity_score) pairs."""

        if top_k <= 0:
            return []
        if self._vectorizer is None or self._matrix is None:
            raise RuntimeError("KeywordSearch not fitted")

        query = (query or "").strip()
        if not query:
            return []

        q = self._vectorizer.transform([query])

        # TF-IDF vectors are L2-normalized => dot equals cosine similarity
        scores = (self._matrix @ q.T).toarray().ravel()
        if scores.size == 0:
            return []

        k = min(int(top_k), int(scores.shape[0]))
        idxs = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idxs]
