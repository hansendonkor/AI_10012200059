"""src/logger.py

Name: Hansen Donkor
Index Number: 10012200059

JSONL logging for the manual RAG pipeline.

Requirements satisfied:
- Save logs to logs/rag_logs.jsonl
- Each entry includes timestamp, query, retrieved chunks, scores, prompt, response

Design:
- JSON Lines (one JSON object per line) is append-only and easy to parse.
- This module is independent of Streamlit (usable in scripts/tests).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .utils import utc_now_iso


def _json_default(obj: Any) -> str:
    # Safe fallback for uncommon objects
    return str(obj)


class JsonlLogger:
    """Append-only JSONL logger."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def log(self, event: Dict[str, Any]) -> None:
        record = {"timestamp": utc_now_iso(), **event}
        line = json.dumps(record, ensure_ascii=False, default=_json_default)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def tail(self, n: int = 5) -> List[Dict[str, Any]]:
        if n <= 0:
            return []
        if not self.path.exists():
            return []

        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []

        out: List[Dict[str, Any]] = []
        for ln in lines[-n:]:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return out
