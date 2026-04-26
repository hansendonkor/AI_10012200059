"""src/retriever.py

Name: Hansen Donkor
Index Number: 10012200059

Hybrid retrieval (dense + keyword).

Requirements satisfied:
- Top-k vector retrieval (via VectorStore)
- Top-k TF-IDF retrieval (via KeywordSearch)
- Hybrid retrieval by merging candidates

Scoring notes:
- We min-max normalize vector and keyword scores separately to 0..1.
- Reranking happens in reranker.py.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .keyword_search import KeywordSearch
from .utils import RetrievalCandidate, min_max_scale
from .vector_store import VectorStore


def hybrid_retrieve(
    query: str,
    query_embedding: np.ndarray,
    vector_store: VectorStore,
    keyword_store: KeywordSearch,
    top_k: int,
) -> List[RetrievalCandidate]:
    """Run hybrid retrieval and return merged candidates.

    Args:
        query: User query.
        query_embedding: 1D embedding vector for the query (ideally normalized).
        vector_store: Built VectorStore.
        keyword_store: Fitted KeywordSearch.
        top_k: Retrieval cutoff.

    Returns:
        List of RetrievalCandidate (unordered). Use reranker to sort.
    """

    if top_k <= 0:
        return []

    query = (query or "").strip()
    if not query:
        return []

    vec_hits = vector_store.search(query_embedding=query_embedding, top_k=top_k)
    kw_hits = keyword_store.search(query=query, top_k=top_k)

    vec_scaled = min_max_scale(vec_hits)
    kw_scaled = min_max_scale(kw_hits)

    merged: Dict[int, RetrievalCandidate] = {}

    for idx, _raw in vec_hits:
        merged[idx] = RetrievalCandidate(
            chunk_idx=idx,
            vector_score=float(vec_scaled.get(idx, 0.0)),
            keyword_score=0.0,
        )

    for idx, _raw in kw_hits:
        if idx not in merged:
            merged[idx] = RetrievalCandidate(chunk_idx=idx)
        merged[idx].keyword_score = float(kw_scaled.get(idx, 0.0))

    return list(merged.values())
