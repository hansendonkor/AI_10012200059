"""src/generator.py

Name: Hansen Donkor
Index Number: 10012200059

LLM generation step (Vertex AI).

Important (exam constraint):
- Only generation uses an external LLM API.
- Retrieval, chunking, reranking, and prompt building are implemented manually.
"""

from __future__ import annotations

import os
from typing import Optional

import vertexai
from vertexai.generative_models import GenerativeModel

from .config import LOCATION, PROJECT_ID

FALLBACK_MESSAGE = (
    "I could not generate a response at the moment. "
    "Please try again in a few seconds."
)
AUTH_MESSAGE = (
    "Vertex AI authentication is not configured. "
    "Install Google Cloud CLI, run `gcloud auth application-default login`, "
    "and ensure PROJECT_ID/LOCATION are correct."
)


class LLMGenerator:
    """Safe Vertex AI text generator."""

    def __init__(self, project_id: str, location: str = LOCATION, model_name: str = "gemini-1.5-pro"):
        self.project_id = (project_id or "").strip()
        self.location = (location or LOCATION).strip()
        self.model_name = (model_name or "gemini-1.5-pro").strip()
        self._model: Optional[GenerativeModel] = None
        self._init_error: Optional[str] = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            if not self.project_id:
                raise ValueError("PROJECT_ID is empty.")
            vertexai.init(project=self.project_id, location=self.location)
            self._model = GenerativeModel(self.model_name)
        except Exception as exc:
            self._init_error = str(exc)
            self._model = None

    def generate(self, prompt: str) -> str:
        """Generate text safely. Never raises to caller."""
        safe_prompt = (prompt or "").strip()
        if not safe_prompt:
            return "Please provide a non-empty prompt."

        if self._model is None:
            if self._init_error and (
                "default credentials" in self._init_error.lower()
                or "credential" in self._init_error.lower()
                or "auth" in self._init_error.lower()
            ):
                return AUTH_MESSAGE
            return FALLBACK_MESSAGE

        try:
            response = self._model.generate_content(safe_prompt)
            text = (getattr(response, "text", "") or "").strip()
            if text:
                return text
            return FALLBACK_MESSAGE
        except Exception as exc:
            err = str(exc).lower()
            if "default credentials" in err or "credential" in err or "auth" in err:
                return AUTH_MESSAGE
            if "publisher model" in err and ("not found" in err or "does not have access" in err):
                fallback_candidates = ["gemini-2.5-flash", "gemini-2.5-pro"]
                for fallback_model in fallback_candidates:
                    if fallback_model == self.model_name:
                        continue
                    try:
                        self._model = GenerativeModel(fallback_model)
                        self.model_name = fallback_model
                        retry_response = self._model.generate_content(safe_prompt)
                        retry_text = (getattr(retry_response, "text", "") or "").strip()
                        if retry_text:
                            return retry_text
                    except Exception:
                        continue
            return FALLBACK_MESSAGE


def generate_answer(
    provider: str,
    model: str,
    prompt: str,
    temperature: float = 0.2,
    max_output_tokens: Optional[int] = None,
) -> str:
    """Compatibility wrapper for existing call sites.

    `provider`, `model`, `temperature`, and `max_output_tokens` are accepted to avoid
    touching the rest of the app, but Vertex AI is used for generation.
    """
    _ = (provider, model, temperature, max_output_tokens)

    project_id = os.getenv("PROJECT_ID", PROJECT_ID).strip() or PROJECT_ID
    location = os.getenv("LOCATION", LOCATION).strip() or LOCATION
    configured_model = (model or "").strip() or os.getenv("GEMINI_MODEL", "gemini-1.5-pro").strip()
    generator = LLMGenerator(
        project_id=project_id,
        location=location,
        model_name=configured_model or "gemini-1.5-pro",
    )
    return generator.generate(prompt=prompt)
