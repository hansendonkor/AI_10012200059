"""src/embedder.py

Name: Hansen Donkor
Index Number: 10012200059

Embedding pipeline using `sentence-transformers`.

Requirements satisfied:
- Custom embedding pipeline (manual RAG component)
- Produces dense vectors for chunks + query

Notes:
- We normalize embeddings to unit length so inner product == cosine similarity.
- Model loading is isolated here to keep other modules lightweight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray  # shape (n, dim), float32
    dim: int


class Embedder:
    """SentenceTransformer-based embedder with batching and normalization."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        try:
            if device:
                self.model = SentenceTransformer(model_name, device=device)
            else:
                self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model {model_name!r}: {e}") from e

        logger.info("Loaded embedding model: %s", model_name)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return vectors / norms

    def embed_texts_result(self, texts: Sequence[str], normalize: bool = True) -> EmbeddingResult:
        """Embed a list of texts.

        Args:
            texts: List of strings.
            normalize: If True, normalize vectors to unit length.

        Returns:
            EmbeddingResult with float32 vectors.
        """

        if texts is None:
            raise ValueError("texts cannot be None")

        texts_list: List[str] = ["" if t is None else str(t) for t in texts]

        if len(texts_list) == 0:
            return EmbeddingResult(vectors=np.zeros((0, 0), dtype=np.float32), dim=0)

        try:
            vecs = self.model.encode(
                texts_list,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}") from e

        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2:
            raise RuntimeError(f"Unexpected embedding shape: {vecs.shape}")

        if normalize:
            vecs = self._normalize(vecs)

        return EmbeddingResult(vectors=vecs, dim=int(vecs.shape[1]))

    def embed_texts(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        """Embed texts and return the raw vector matrix.

        This is the primary API used by the Streamlit app.
        """

        return self.embed_texts_result(texts=texts, normalize=normalize).vectors

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Embed a single query string and return a 1D vector."""

        vecs = self.embed_texts([query], normalize=normalize)
        if vecs.shape[0] != 1:
            raise RuntimeError("embed_query expected one vector")
        return vecs[0]
