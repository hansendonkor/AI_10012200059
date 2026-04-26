# Academic City Manual RAG Chatbot (Ghana Elections + 2025 Budget)

**Name:** Hansen Donkor 
**Index Number:** 10012200059

This repository implements a **manual Retrieval-Augmented Generation (RAG)** chatbot using:
- `data/Ghana_Election_Result.csv` (structured)
- `data/2025_Budget_Statement.pdf` (unstructured)

Constraints (per exam): **no LangChain, no LlamaIndex, no pre-built RAG pipeline**.

---

## Quickstart (Local)

1. Create environment and install deps:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Add API key:

```bash
copy .env.example .env
# then edit .env
```

Use Vertex AI:
- `LLM_PROVIDER=vertex`
- `PROJECT_ID=intro-to-ai-494310`
- `LOCATION=us-central1`

3. Place the provided files:
- `data/Ghana_Election_Result.csv`
- `data/2025_Budget_Statement.pdf`

4. Run:

```bash
streamlit run app.py
```

---

## What The App Shows (For Marks)

The UI is designed to visibly demonstrate every stage:
1. **Chunking** (fixed vs section-aware)
2. **Embedding** (SentenceTransformers)
3. **Vector search** (FAISS when available, fallback cosine search)
4. **Keyword search** (TF‑IDF)
5. **Hybrid retrieval + reranking** (custom scoring)
6. **Prompt building** (context injection + hallucination control)
7. **LLM generation** (Vertex AI Gemini)
8. **Logging** (JSONL per query, with chunks, scores, and final prompt)

---

## Rubric → Implementation Map (Mark‑by‑Mark)

Use this as a checklist: each rubric item has **code evidence** and **documentation evidence**.

### Part A — Cleaning + Chunking
- Cleaning implementation: `src/cleaner.py`
- Two chunking strategies: `src/chunker.py`
  - Fixed-size chunks (with overlap)
  - Section-aware chunks (PDF heading heuristics; CSV grouping when possible)
- Write-up + justification: `docs/design_decisions.md`
- Chunking comparison logs: `docs/experiment_logs.md`

### Part B — Embeddings + Vector Store + Retrieval Enhancement
- Embeddings: `src/embedder.py`
- Vector store + similarity scoring: `src/vector_store.py`
- Keyword retrieval: `src/keyword_search.py`
- Hybrid retrieval: `src/retriever.py`
- Retrieval enhancement (“innovation” reranker): `src/reranker.py`
- Failure cases + fixes: `docs/experiment_logs.md`

### Part C — Prompting + Hallucination Control
- Prompt template + context injection: `src/prompt_builder.py`
- Context window management: `src/prompt_builder.py`
- Prompt experiments: `docs/experiment_logs.md`

### Part D — End-to-End Manual RAG Pipeline + Visibility
- Manual pipeline wiring: `app.py` + `src/*`
- Stage-by-stage logs: `src/logger.py` → `logs/rag_logs.jsonl`
- UI visibility: retrieved chunks, scores, selected context, final prompt

### Part E — Evaluation + Adversarial Queries + Baseline Comparison
- Evaluation runner: `src/evaluator.py`
- Report template: `docs/evaluation_report.md`
- Required comparisons:
  - RAG vs “pure LLM” (no retrieval)
  - Two adversarial queries
  - Notes on accuracy / hallucination / consistency

### Part F — Architecture + Justification
- Architecture explanation + design rationale: `docs/design_decisions.md`
- Architecture diagram + explanation: `docs/architecture.md` (Mermaid diagram; optional PNG export for screenshots/video)

### Part G — Innovation
- Domain-aware reranking score:
  - `final_score = 0.60*vector + 0.25*keyword + 0.15*domain_match`
  - Implemented in: `src/reranker.py`
- Explain why it helps: `docs/design_decisions.md`

---

## 3‑Day Execution Order (Outputs You Must Produce)

### Day 1 — Data + Chunking + Indexing
Deliverables:
- Working data loading for CSV+PDF (`src/data_loader.py`)
- Cleaning + two chunkers (`src/cleaner.py`, `src/chunker.py`)
- First experiment logs for chunk strategies (`docs/experiment_logs.md`)

Tasks:
- Load CSV rows into readable text records
- Extract PDF text page-by-page
- Implement fixed chunking + overlap
- Implement section-aware chunking for PDF headings
- Sanity-check chunk sizes + edge cases

### Day 2 — Retrieval + Reranking + Prompting
Deliverables:
- Vector store + keyword search + hybrid retrieval (`src/vector_store.py`, `src/keyword_search.py`, `src/retriever.py`)
- Innovation reranker (`src/reranker.py`)
- Prompt template + context budget (`src/prompt_builder.py`)
- Logged failure cases + fixes in `docs/experiment_logs.md`

Tasks:
- Build embeddings + store
- Implement TF‑IDF search
- Merge & normalize scores, then apply reranker
- Build final prompt with citations + “I don’t know” fallback

### Day 3 — App Polish + Evaluation + Deployment + Video
Deliverables:
- Streamlit app showing every stage (`app.py`)
- JSONL logs per query (`logs/rag_logs.jsonl`)
- Evaluation report (`docs/evaluation_report.md`)
- Video script + record <2 minutes (`docs/video_script.md`)
- Deploy to Streamlit Community Cloud

Tasks:
- Wire UI: query → retrieve → select chunks → prompt → generate
- Add baseline “pure LLM” answer
- Run adversarial queries + record findings
- Review/attach the architecture diagram in `docs/architecture.md` (export PNG if you want to show it in the video)

---

## Deployment (Streamlit Community Cloud)
- Push to GitHub
- Ensure Streamlit Cloud runtime has Google Cloud auth configured for Vertex AI
- Ensure `data/` is present in the repo (or document how it’s provided)

---

## Repo Layout

```
ai_INDEXNUMBER/
  app.py
  requirements.txt
  README.md
  .env.example
  data/
  docs/
  src/
  notebooks/
  logs/
```
