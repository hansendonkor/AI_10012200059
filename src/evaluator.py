"""src/evaluator.py

Name: Hansen Donkor
Index Number: 10012200059

Evaluation utilities (manual and automated scaffolding).

Requirements satisfied:
- Adversarial queries (ambiguous + misleading)
- Compare RAG vs pure LLM baseline
- Output structured results for the evaluation report

Note:
- Automated metrics for hallucination/accuracy are imperfect without ground-truth labels.
    This module focuses on producing clean, reproducible evaluation outputs, with optional
    manual scoring fields you can fill in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .embedder import Embedder
from .generator import generate_answer
from .keyword_search import KeywordSearch
from .prompt_builder import build_prompt, select_chunks_to_budget
from .reranker import apply_domain_scores
from .retriever import hybrid_retrieve
from .utils import Chunk
from .vector_store import VectorStore


@dataclass
class EvalCase:
    query: str
    expected_notes: str


@dataclass
class EvalOutput:
    query: str
    rag_answer: str
    pure_llm_answer: str
    selected_chunk_ids: List[str]
    retrieved: List[Dict[str, Any]]
    final_prompt: str
    # Optional manual fields (fill these in your report)
    manual_accuracy_1to5: Optional[int] = None
    manual_hallucination: Optional[str] = None  # none/minor/major
    manual_consistency: Optional[str] = None  # consistent/inconsistent


def default_adversarial_cases() -> List[EvalCase]:
    """Two adversarial cases required by the exam + a couple useful extras."""

    return [
        # Ambiguous: forces careful grounding + scope control
        EvalCase(
            query="Who won the election?",
            expected_notes="Ambiguous scope; should ask to clarify or answer narrowly using evidence.",
        ),
        # Misleading: false premise; should refuse or correct using context
        EvalCase(
            query="The 2025 budget statement says income tax is completely abolished. Confirm and cite the page.",
            expected_notes="Should not accept false premise; must cite evidence or say I don't know.",
        ),
        # Numeric precision stress-test
        EvalCase(
            query="Give the exact total budget deficit figure and the page number.",
            expected_notes="Should cite PDF page; if not found, say I don't know.",
        ),
        # Aggregation stress-test for CSV
        EvalCase(
            query="Which candidate won the election in every region?",
            expected_notes="May require aggregation not present; should be careful and grounded.",
        ),
    ]


def score_notes_template() -> Dict[str, str]:
    return {
        "accuracy": "(manual score 1-5)",
        "hallucination": "(manual: none / minor / major)",
        "consistency": "(manual: consistent / inconsistent across runs)",
    }


def evaluate_case(
    *,
    query: str,
    chunks: List[Chunk],
    embedder: Embedder,
    vector_store: VectorStore,
    keyword_store: KeywordSearch,
    top_k: int,
    max_context_tokens: int,
    w_vector: float = 0.60,
    w_keyword: float = 0.25,
    w_domain: float = 0.15,
    llm_provider: str = "vertex",
    llm_model: str = "gemini-1.5-pro",
) -> EvalOutput:
    """Run a single evaluation query and return structured output."""

    q = (query or "").strip()
    q_emb = embedder.embed_query(q, normalize=True)

    candidates = hybrid_retrieve(
        query=q,
        query_embedding=q_emb,
        vector_store=vector_store,
        keyword_store=keyword_store,
        top_k=top_k,
    )

    chunk_sources = [ch.source for ch in chunks]
    reranked = apply_domain_scores(
        query=q,
        candidates=candidates,
        chunk_sources=chunk_sources,
        w_vector=w_vector,
        w_keyword=w_keyword,
        w_domain=w_domain,
    )

    # Select chunks in reranked order, then apply context budget
    reranked_sorted = sorted(reranked, key=lambda c: c.final_score, reverse=True)
    candidate_chunks = [chunks[c.chunk_idx] for c in reranked_sorted[:top_k]]
    selected_chunks = select_chunks_to_budget(candidate_chunks, max_context_tokens=max_context_tokens)

    prompt = build_prompt(question=q, chunks=selected_chunks, max_context_tokens=max_context_tokens)

    rag_answer = generate_answer(provider=llm_provider, model=llm_model, prompt=prompt)
    pure_prompt = f"Answer the question as best you can.\n\nQuestion: {q}"
    pure_answer = generate_answer(provider=llm_provider, model=llm_model, prompt=pure_prompt)

    retrieved_payload: List[Dict[str, Any]] = []
    for c in reranked_sorted[:top_k]:
        ch = chunks[c.chunk_idx]
        retrieved_payload.append(
            {
                "chunk_id": ch.chunk_id,
                "source": ch.source,
                "metadata": ch.metadata,
                "scores": asdict(c),
            }
        )

    return EvalOutput(
        query=q,
        rag_answer=rag_answer,
        pure_llm_answer=pure_answer,
        selected_chunk_ids=[ch.chunk_id for ch in selected_chunks],
        retrieved=retrieved_payload,
        final_prompt=prompt,
    )


def evaluate_suite(
    *,
    cases: List[EvalCase],
    chunks: List[Chunk],
    embedder: Embedder,
    vector_store: VectorStore,
    keyword_store: KeywordSearch,
    top_k: int,
    max_context_tokens: int,
    w_vector: float = 0.60,
    w_keyword: float = 0.25,
    w_domain: float = 0.15,
    llm_provider: str = "vertex",
    llm_model: str = "gemini-1.5-pro",
) -> List[EvalOutput]:
    """Run multiple evaluation cases."""

    outputs: List[EvalOutput] = []
    for case in cases:
        outputs.append(
            evaluate_case(
                query=case.query,
                chunks=chunks,
                embedder=embedder,
                vector_store=vector_store,
                keyword_store=keyword_store,
                top_k=top_k,
                max_context_tokens=max_context_tokens,
                w_vector=w_vector,
                w_keyword=w_keyword,
                w_domain=w_domain,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
        )
    return outputs
