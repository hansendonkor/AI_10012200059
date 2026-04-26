"""src/config.py

Name: Hansen Donkor
Index Number: 10012200059

Central configuration for the manual RAG system.

Design goals:
- Keep configuration explicit and explainable (important for grading).
- Load settings from environment variables (works locally and on Streamlit Cloud).
- Validate and normalize values early.

Constraints:
- No LangChain / LlamaIndex / prebuilt RAG frameworks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ID = "intro-to-ai-494310"
LOCATION = "us-central1"


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None else value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"Invalid int for {name}={raw!r}") from e


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid float for {name}={raw!r}") from e


@dataclass(frozen=True)
class AppConfig:
    """Typed configuration for the full RAG pipeline."""

    # Student identifiers (also used in documentation placeholders)
    student_name: str
    index_number: str

    # Paths
    project_root: Path
    data_dir: Path
    docs_dir: Path
    logs_dir: Path

    csv_path: Path
    pdf_path: Path
    log_path: Path

    # Chunking
    chunk_size_chars: int
    chunk_overlap_chars: int

    # Retrieval
    top_k: int

    # Hybrid reranking weights
    w_vector: float
    w_keyword: float
    w_domain: float

    # Prompt/context
    max_context_tokens: int

    # Embeddings
    embedding_model: str

    # PDF extraction
    pdf_engine: str

    # LLM
    llm_provider: str
    project_id: str
    location: str

    @property
    def gemini_model(self) -> str:
        """Backward-compatible model label used by the UI."""
        return "gemini-1.5-pro"

    @staticmethod
    def from_env() -> "AppConfig":
        """Load config from environment variables.

        Supported env vars (examples):
        - STUDENT_NAME, INDEX_NUMBER
        - CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS
        - TOP_K
        - W_VECTOR, W_KEYWORD, W_DOMAIN
        - MAX_CONTEXT_TOKENS
        - EMBEDDING_MODEL
        - PDF_ENGINE (pymupdf | pdfplumber)
        - LLM_PROVIDER (vertex)
        - PROJECT_ID
        - LOCATION
        """

        project_root = Path(__file__).resolve().parents[1]
        data_dir = project_root / "data"
        docs_dir = project_root / "docs"
        logs_dir = project_root / "logs"

        csv_path = data_dir / "Ghana_Election_Result.csv"
        default_pdf_path = data_dir / "2025_Budget_Statement.pdf"
        if not default_pdf_path.exists():
            # Robust fallback: use a PDF found in data/ (prefer names containing 'budget').
            pdf_candidates = sorted(data_dir.glob("*.pdf"))
            if pdf_candidates:
                budgetish = [p for p in pdf_candidates if "budget" in p.name.lower()]
                default_pdf_path = budgetish[0] if budgetish else pdf_candidates[0]

        pdf_path = default_pdf_path
        log_path = logs_dir / "rag_logs.jsonl"

        cfg = AppConfig(
            student_name=_env_str("STUDENT_NAME", "Hansen Donkor"),
            index_number=_env_str("INDEX_NUMBER", "10012200059"),
            project_root=project_root,
            data_dir=data_dir,
            docs_dir=docs_dir,
            logs_dir=logs_dir,
            csv_path=Path(_env_str("CSV_PATH", str(csv_path))),
            pdf_path=Path(_env_str("PDF_PATH", str(pdf_path))),
            log_path=Path(_env_str("LOG_PATH", str(log_path))),
            chunk_size_chars=_env_int("CHUNK_SIZE_CHARS", 1200),
            chunk_overlap_chars=_env_int("CHUNK_OVERLAP_CHARS", 200),
            top_k=_env_int("TOP_K", 12),
            w_vector=_env_float("W_VECTOR", 0.60),
            w_keyword=_env_float("W_KEYWORD", 0.25),
            w_domain=_env_float("W_DOMAIN", 0.15),
            max_context_tokens=_env_int("MAX_CONTEXT_TOKENS", 1800),
            embedding_model=_env_str(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            pdf_engine=_env_str("PDF_ENGINE", "pymupdf").strip().lower(),
            llm_provider=_env_str("LLM_PROVIDER", "vertex").strip().lower(),
            project_id=_env_str("PROJECT_ID", PROJECT_ID).strip(),
            location=_env_str("LOCATION", LOCATION).strip(),
        )

        cfg.validate()
        cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        return cfg

    def validate(self) -> None:
        """Validate configuration values and raise clear errors."""

        if self.chunk_size_chars <= 0:
            raise ValueError("CHUNK_SIZE_CHARS must be > 0")
        if self.chunk_overlap_chars < 0:
            raise ValueError("CHUNK_OVERLAP_CHARS must be >= 0")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be > 0")
        if self.max_context_tokens <= 0:
            raise ValueError("MAX_CONTEXT_TOKENS must be > 0")

        if self.pdf_engine not in {"pymupdf", "pdfplumber"}:
            raise ValueError("PDF_ENGINE must be 'pymupdf' or 'pdfplumber'")

        w_sum = self.w_vector + self.w_keyword + self.w_domain
        if abs(w_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Rerank weights must sum to 1.0, got {w_sum:.4f} "
                f"(W_VECTOR={self.w_vector}, W_KEYWORD={self.w_keyword}, W_DOMAIN={self.w_domain})"
            )

        if self.llm_provider != "vertex":
            raise ValueError("LLM_PROVIDER must be 'vertex'")

        if not self.project_id:
            raise ValueError("PROJECT_ID must not be empty")

        if not self.location:
            raise ValueError("LOCATION must not be empty")
