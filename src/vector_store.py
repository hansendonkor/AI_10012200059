"""src/vector_store.py

Name: Hansen Donkor
Index Number: 10012200059

Vector store for dense retrieval.

Requirements satisfied:
- Store embeddings in FAISS
- Implement top-k similarity search

Implementation notes:
- When embeddings are L2-normalized, inner product == cosine similarity.
- Uses FAISS when available; otherwise falls back to NumPy dot-product search.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from .utils import Chunk


logger = logging.getLogger(__name__)


try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False


class VectorStore:
    """In-memory vector store with optional FAISS acceleration."""

    def __init__(self) -> None:
        self._index = None
        self._embeddings: np.ndarray | None = None
        self._chunks: List[Chunk] | None = None
        self._dim: int | None = None

    @property
    def faiss_available(self) -> bool:
        return _FAISS_AVAILABLE

    @property
    def size(self) -> int:
        return 0 if self._chunks is None else len(self._chunks)

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("VectorStore not built")
        return self._dim

    def build(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """Build the index.

        Args:
            embeddings: np.ndarray (n, dim) float32, ideally L2-normalized.
            chunks: list of Chunk objects aligned with embeddings.
        """

        if embeddings is None:
            raise ValueError("embeddings cannot be None")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if len(chunks) != int(embeddings.shape[0]):
            raise ValueError("chunks must align with embeddings rows")

        n, dim = int(embeddings.shape[0]), int(embeddings.shape[1])
        if n == 0:
            raise ValueError("Cannot build VectorStore with 0 embeddings")

        emb = np.asarray(embeddings, dtype=np.float32)

        self._chunks = chunks
        self._dim = dim

        if _FAISS_AVAILABLE:
            try:
                index = faiss.IndexFlatIP(dim)
                index.add(emb)
                self._index = index
                self._embeddings = None
                logger.info("VectorStore built with FAISS (n=%d, dim=%d)", n, dim)
                return
            except Exception as e:
                logger.warning("FAISS build failed, falling back to NumPy: %s", e)

        self._index = None
        self._embeddings = emb
        logger.info("VectorStore built with NumPy fallback (n=%d, dim=%d)", n, dim)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """Return (chunk_idx, similarity_score) pairs."""

        if top_k <= 0:
            return []
        if self._chunks is None or self._dim is None:
            raise RuntimeError("VectorStore not built")

        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim != 1 or int(q.shape[0]) != self._dim:
            raise ValueError(f"query_embedding must be shape ({self._dim},)")

        k = min(int(top_k), len(self._chunks))

        if _FAISS_AVAILABLE and self._index is not None:
            q2 = q.reshape(1, -1)
            scores, idxs = self._index.search(q2, k)
            hits: List[Tuple[int, float]] = []
            for i, s in zip(idxs[0].tolist(), scores[0].tolist()):
                if i == -1:
                    continue
                hits.append((int(i), float(s)))
            return hits

        if self._embeddings is None:
            raise RuntimeError("VectorStore missing embeddings")

        # embeddings are assumed normalized; dot == cosine
        sims = self._embeddings @ q
        idxs = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idxs]
