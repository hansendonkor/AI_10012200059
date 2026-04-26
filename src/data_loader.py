"""src/data_loader.py

Name: Hansen Donkor
Index Number: 10012200059

Loads the two required sources:
1) Ghana election results CSV (structured)
2) 2025 Budget Statement PDF (unstructured)

Output format:
- A list of `Document` objects where each document contains:
  - `source`: "csv" or "pdf"
  - `text`: cleaned-ish raw text (deep cleaning happens in cleaner.py)
  - `metadata`: identifiers (row_index/page), plus helpful labels

This module is intentionally framework-free (manual RAG requirement).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import AppConfig
from .utils import Document


logger = logging.getLogger(__name__)


_COMMON_COLS = {
    "region": ["region"],
    "constituency": ["constituency"],
    "candidate": ["candidate", "name"],
    "party": ["party"],
}


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        hit = lower_map.get(cand.lower())
        if hit:
            return hit
    return None


def _row_to_text(row: pd.Series, columns: List[str]) -> str:
    parts = []
    for col in columns:
        value = row[col]
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str == "" or value_str.lower() == "nan":
            continue
        parts.append(f"{col}: {value_str}")

    # Readable, stable format for retrieval
    return "\n".join(parts)


def _build_csv_label(row: pd.Series, df: pd.DataFrame) -> Dict[str, str]:
    region_col = _find_col(df, _COMMON_COLS["region"])
    constituency_col = _find_col(df, _COMMON_COLS["constituency"])
    candidate_col = _find_col(df, _COMMON_COLS["candidate"])
    party_col = _find_col(df, _COMMON_COLS["party"])

    meta: Dict[str, str] = {}
    label_parts = []

    def add_if(col: Optional[str], key: str) -> None:
        if col:
            v = str(row[col]).strip()
            if v and v.lower() != "nan":
                meta[key] = v
                label_parts.append(v)

    add_if(region_col, "region")
    add_if(constituency_col, "constituency")
    add_if(party_col, "party")
    add_if(candidate_col, "candidate")

    if label_parts:
        meta["label"] = " | ".join(label_parts[:4])

    return meta


def load_election_csv(csv_path: Path) -> List[Document]:
    """Load the election CSV and convert each row to a Document."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. Place Ghana_Election_Result.csv in data/."
        )

    # Robust encoding fallback for exam datasets
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    df = df.fillna("")
    columns = list(df.columns)

    docs: List[Document] = []
    for row_index, row in df.iterrows():
        text = _row_to_text(row, columns=columns)
        if not text.strip():
            continue

        doc_id = f"csv_row_{int(row_index)}"
        meta: Dict[str, object] = {
            "id": doc_id,
            "row_index": int(row_index),
            "source_file": str(csv_path.name),
        }
        meta.update(_build_csv_label(row, df))

        docs.append(Document(source="csv", text=text, metadata=meta))

    logger.info("Loaded %d CSV documents from %s", len(docs), csv_path)
    return docs


def _load_pdf_pymupdf(pdf_path: Path) -> List[Document]:
    import fitz  # PyMuPDF

    docs: List[Document] = []
    with fitz.open(pdf_path) as pdf:
        for i in range(len(pdf)):
            page_num = i + 1
            page = pdf[i]
            text = (page.get_text("text") or "").strip()
            if not text:
                continue

            doc_id = f"pdf_page_{page_num}"
            meta = {
                "id": doc_id,
                "page": page_num,
                "label": f"page {page_num}",
                "source_file": str(pdf_path.name),
            }
            docs.append(Document(source="pdf", text=text, metadata=meta))

    return docs


def _load_pdf_pdfplumber(pdf_path: Path) -> List[Document]:
    import pdfplumber

    docs: List[Document] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            doc_id = f"pdf_page_{page_num}"
            meta = {
                "id": doc_id,
                "page": page_num,
                "label": f"page {page_num}",
                "source_file": str(pdf_path.name),
            }
            docs.append(Document(source="pdf", text=text, metadata=meta))

    return docs


def load_budget_pdf(pdf_path: Path, engine: str = "pymupdf") -> List[Document]:
    """Load the budget PDF (one Document per page).

    Args:
        pdf_path: Path to the PDF.
        engine: "pymupdf" (fast) or "pdfplumber" (sometimes better text extraction).
    """

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}. Place the budget PDF in data/ or set PDF_PATH."
        )

    engine = (engine or "pymupdf").strip().lower()

    if engine == "pymupdf":
        docs = _load_pdf_pymupdf(pdf_path)
    elif engine == "pdfplumber":
        docs = _load_pdf_pdfplumber(pdf_path)
    else:
        raise ValueError("engine must be 'pymupdf' or 'pdfplumber'")

    logger.info("Loaded %d PDF documents from %s using %s", len(docs), pdf_path, engine)
    return docs


def load_sources(config: AppConfig) -> List[Document]:
    """Load both required sources using the current AppConfig."""

    docs: List[Document] = []
    docs.extend(load_election_csv(config.csv_path))
    docs.extend(load_budget_pdf(config.pdf_path, engine=config.pdf_engine))
    return docs
