# Design Decisions (Manual RAG)

**Name:** Hansen Donkor  
**Index Number:** 10012200059

## Goal
Build a fully manual RAG chatbot (no LangChain/LlamaIndex) that answers questions grounded in:
- Ghana election results CSV (structured)
- 2025 Budget Statement PDF (unstructured)

## Pipeline (Manual)
1. Load sources (CSV + PDF): `src/data_loader.py`
2. Clean/normalize text: `src/cleaner.py`
3. Chunking (two strategies): `src/chunker.py`
4. Embeddings: `src/embedder.py`
5. Vector store + similarity: `src/vector_store.py`
6. Keyword search (TF‑IDF): `src/keyword_search.py`
7. Hybrid retrieval: `src/retriever.py`
8. Reranking (innovation): `src/reranker.py`
9. Prompt building + context budget: `src/prompt_builder.py`
10. LLM generation: `src/generator.py`
11. Logging: `src/logger.py` → `logs/rag_logs.jsonl`

## Part A — Cleaning + Chunking
### Cleaning
- PDF text often has broken line wraps, hyphenation, repeated whitespace.
- Cleaning keeps newlines (needed for section/heading detection) while normalizing whitespace.

**Why:** Cleaner chunks -> better retrieval and fewer hallucinations.

### Chunking strategies
1) **Fixed-size chunks** (with overlap)
- Simple, consistent chunk sizes.
- Overlap preserves context across boundaries.

2) **Section-aware chunks** (PDF)
- Uses heading heuristics (uppercase/CHAPTER/PART patterns) to group text by section.
- Then applies fixed chunking inside each section.

**Why:** Budget statements are structured by headings; keeping sections improves relevance.

**Comparison evidence:** record retrieval quality and failure cases in `docs/experiment_logs.md`.

## Part B — Embeddings + Retrieval
### Embeddings
- SentenceTransformers (`all-MiniLM-L6-v2`) chosen for speed + strong semantic search.

### Vector store
- Uses FAISS IndexFlatIP when available (cosine similarity via normalized vectors).
- Falls back to NumPy cosine similarity when FAISS is unavailable (e.g., Windows pip).

### Keyword search
- TF‑IDF (bigrams) supports exact term matching, names, and numeric phrases.

### Hybrid retrieval
- Union of vector hits + keyword hits.
- Scores min-max normalized to 0..1 before combining.

## Part C — Prompting + Hallucination Control
### Prompt template
- Explicit instruction: answer only from context; otherwise say “I don’t know…”.
- Inline citations encouraged.

### Context window management
- Context blocks are appended until the context budget is reached.
- Budget uses a simple token approximation (chars/4) to keep implementation manual.

## Part D — Full Visibility + Logging
The Streamlit app shows:
- retrieved chunks + their scores
- which chunks were selected
- the final prompt string
- final RAG answer + optional pure LLM baseline

Logs are written as JSONL for each query.

## Part E — Evaluation Design
Required comparisons:
- RAG vs pure LLM baseline
- Two adversarial queries
- Notes on accuracy / hallucination / consistency

**Method:** manual scoring rubric recorded in `docs/evaluation_report.md`.

## Part F — Architecture
A simple architecture diagram should show:
User → Streamlit UI → Chunker → Embedder → Vector Search + Keyword Search → Reranker → Prompt Builder → LLM → Answer (+ Logger)

Create the diagram in Mermaid (embedded in `docs/architecture.md`) and (optionally) export to a PNG for your report/video.

## Part G — Innovation
**Domain-aware reranking**:

$\text{final} = 0.60\cdot\text{vector} + 0.25\cdot\text{keyword} + 0.15\cdot\text{domainMatch}$

Domain match is inferred from the source:
- CSV chunks → election domain
- PDF chunks → budget domain

This improves relevance for ambiguous queries and makes retrieval behavior easier to explain.
