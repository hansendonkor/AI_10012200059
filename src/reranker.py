"""src/reranker.py

Name: Hansen Donkor
Index Number: 10012200059

Custom reranker (innovation feature).

Required formula:

final_score = 0.60 * vector_score + 0.25 * keyword_score + 0.15 * domain_score

Where domain_score is based on whether the query is about elections or the budget.
"""

from __future__ import annotations

from typing import List, Optional

from .utils import RetrievalCandidate


ELECTION_TERMS = {
    "constituency",
    "votes",
    "vote",
    "party",
    "candidate",
    "region",
    "presidential",
    "parliament",
    "mp",
}

BUDGET_TERMS = {
    "tax",
    "revenue",
    "expenditure",
    "spending",
    "deficit",
    "inflation",
    "ministry",
    "policy",
    "fiscal",
    "gdp",
    "debt",
}


def classify_query_domain(query: str) -> str:
    q = query.lower()
    e = sum(1 for t in ELECTION_TERMS if t in q)
    b = sum(1 for t in BUDGET_TERMS if t in q)
    if e == 0 and b == 0:
        return "unknown"
    return "election" if e >= b else "budget"


def chunk_domain_from_source(source: str) -> str:
    # This project’s two sources are domain-specific.
    if source == "csv":
        return "election"
    if source == "pdf":
        return "budget"
    return "unknown"


def domain_match_score(query_domain: str, chunk_domain: str) -> float:
    """Return a domain score in [0, 1]."""

    if query_domain == "unknown" or chunk_domain == "unknown":
        # neutral when unsure
        return 0.5
    return 1.0 if query_domain == chunk_domain else 0.0


def rerank(
    query: str,
    candidates: List[RetrievalCandidate],
    chunk_sources: Optional[List[str]] = None,
    w_vector: float = 0.60,
    w_keyword: float = 0.25,
    w_domain: float = 0.15,
) -> List[RetrievalCandidate]:
    """Rerank retrieval candidates.

    If chunk_sources is provided, applies the full formula including domain_score.
    If chunk_sources is None, computes only the vector+keyword component (domain_score=0).
    """

    query = (query or "").strip()
    q_domain = classify_query_domain(query)

    if chunk_sources is not None:
        return apply_domain_scores(
            query=query,
            candidates=candidates,
            chunk_sources=chunk_sources,
            w_vector=w_vector,
            w_keyword=w_keyword,
            w_domain=w_domain,
        )

    for c in candidates:
        c.domain_score = 0.0
        c.final_score = (w_vector * c.vector_score) + (w_keyword * c.keyword_score)

    candidates.sort(key=lambda x: x.final_score, reverse=True)
    return candidates


def apply_domain_scores(
    query: str,
    candidates: List[RetrievalCandidate],
    chunk_sources: List[str],
    w_vector: float,
    w_keyword: float,
    w_domain: float,
) -> List[RetrievalCandidate]:
    query_domain = classify_query_domain(query)

    for c in candidates:
        source = chunk_sources[c.chunk_idx]
        chunk_domain = chunk_domain_from_source(source)
        dom = domain_match_score(query_domain=query_domain, chunk_domain=chunk_domain)
        c.domain_score = dom
        c.final_score = (w_vector * c.vector_score) + (w_keyword * c.keyword_score) + (w_domain * dom)

    candidates.sort(key=lambda x: x.final_score, reverse=True)
    return candidates
