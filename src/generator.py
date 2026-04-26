"""src/generator.py

Name: Hansen Donkor
Index Number: 10012200059

LLM generation step (Vertex AI).

Important (exam constraint):
- Only generation uses an external LLM API.
- Retrieval, chunking, reranking, and prompt building are implemented manually.

Authentication strategy (deployment-safe):
- On Streamlit Cloud: credentials loaded from st.secrets["gcp_service_account"]
- Locally: uses GOOGLE_APPLICATION_CREDENTIALS env var pointing to service account JSON,
  OR falls back to Application Default Credentials (gcloud auth application-default login).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

import vertexai
from vertexai.generative_models import GenerativeModel

FALLBACK_MESSAGE = (
    "I could not generate a response at the moment. "
    "Please try again in a few seconds."
)
AUTH_MESSAGE = (
    "Vertex AI authentication is not configured. "
    "On Streamlit Cloud: add your service account JSON to Streamlit Secrets under [gcp_service_account]. "
    "Locally: set GOOGLE_APPLICATION_CREDENTIALS to the path of your service account JSON file, "
    "or run `gcloud auth application-default login`."
)

_DEFAULT_MODEL = "gemini-2.5-flash"
_DEFAULT_LOCATION = "us-central1"


def _get_credentials():
    """Load Google credentials in order of preference:
    1. Streamlit secrets (gcp_service_account section) — used on Streamlit Cloud
    2. GOOGLE_APPLICATION_CREDENTIALS env var pointing to a JSON file — used locally
    3. Application Default Credentials (gcloud) — local dev fallback
    Returns (credentials, project_id) or (None, None) to use ADC.
    """
    # ── 1. Streamlit Cloud secrets ──────────────────────────────────────────
    try:
        import streamlit as st  # noqa: PLC0415

        if hasattr(st, "secrets") and "gcp_service_account" in st.secrets:
            from google.oauth2 import service_account  # noqa: PLC0415

            sa_info = dict(st.secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(
                sa_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            project_id = sa_info.get("project_id", "")
            return creds, project_id
    except Exception:
        pass

    # ── 2. GOOGLE_APPLICATION_CREDENTIALS env var ────────────────────────────
    gac = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if gac and os.path.isfile(gac):
        try:
            from google.oauth2 import service_account  # noqa: PLC0415

            creds = service_account.Credentials.from_service_account_file(
                gac,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            with open(gac, encoding="utf-8") as f:
                sa_data = json.load(f)
            project_id = sa_data.get("project_id", "")
            return creds, project_id
        except Exception:
            pass

    # ── 3. Application Default Credentials (gcloud) ──────────────────────────
    return None, None


class LLMGenerator:
    """Safe Vertex AI text generator."""

    def __init__(
        self,
        project_id: str,
        location: str = _DEFAULT_LOCATION,
        model_name: str = _DEFAULT_MODEL,
    ):
        self.project_id = (project_id or "").strip()
        self.location = (location or _DEFAULT_LOCATION).strip()
        self.model_name = (model_name or _DEFAULT_MODEL).strip()
        self._model: Optional[GenerativeModel] = None
        self._init_error: Optional[str] = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            if not self.project_id:
                raise ValueError("PROJECT_ID is empty. Set it in Streamlit Secrets or as an env var.")

            creds, sa_project = _get_credentials()

            # Prefer project from env var; fall back to service account's project
            effective_project = self.project_id or sa_project or ""
            if not effective_project:
                raise ValueError("Could not determine GCP project_id from env or credentials.")

            if creds is not None:
                vertexai.init(
                    project=effective_project,
                    location=self.location,
                    credentials=creds,
                )
            else:
                # Fall back to ADC (works with gcloud auth application-default login locally)
                vertexai.init(project=effective_project, location=self.location)

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
            err = (self._init_error or "").lower()
            if any(kw in err for kw in ("credential", "auth", "default credentials", "permission")):
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
            if any(kw in err for kw in ("credential", "auth", "default credentials", "permission")):
                return AUTH_MESSAGE
            if "publisher model" in err and ("not found" in err or "does not have access" in err):
                fallback_candidates = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-pro"]
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

    `provider`, `temperature`, and `max_output_tokens` are accepted to avoid
    touching the rest of the app, but Vertex AI is always used for generation.
    """
    _ = (provider, temperature, max_output_tokens)

    project_id = os.getenv("PROJECT_ID", "").strip()
    location = os.getenv("LOCATION", _DEFAULT_LOCATION).strip() or _DEFAULT_LOCATION
    configured_model = (model or "").strip() or os.getenv("GEMINI_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

    generator = LLMGenerator(
        project_id=project_id,
        location=location,
        model_name=configured_model,
    )
    return generator.generate(prompt=prompt)
