"""src/utils.py

Name: Hansen Donkor
Index Number: 10012200059

Core dataclasses and shared helper utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Document:
    source: str  # "csv" | "pdf"
    text: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalCandidate:
    chunk_idx: int
    vector_score: float = 0.0
    keyword_score: float = 0.0
    domain_score: float = 0.0
    final_score: float = 0.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def approx_token_count(text: str) -> int:
    # Rough heuristic (works reasonably for English): ~4 chars per token
    return max(1, len(text) // 4)


def min_max_scale(pairs: Iterable[Tuple[int, float]]) -> Dict[int, float]:
    pairs_list = list(pairs)
    if not pairs_list:
        return {}

    values = [v for _, v in pairs_list]
    vmin, vmax = min(values), max(values)
    if abs(vmax - vmin) < 1e-12:
        return {i: 0.0 for i, _ in pairs_list}

    return {i: (v - vmin) / (vmax - vmin) for i, v in pairs_list}


def merge_max(existing: float, new: float) -> float:
    return new if new > existing else existing
