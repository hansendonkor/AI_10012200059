# 2-Minute Video Script (Walkthrough)

**Name:** Hansen Donkor  
**Index Number:** 10012200059

## Target Length
90–120 seconds.

## Script (Time-coded)

**0:00–0:10 (Intro)**
- "Hi, I'm Hansen Donkor, index 10012200059. This is my manual RAG chatbot for Academic City."
- “It uses two sources: Ghana election CSV and the 2025 Budget Statement PDF.”

**0:10–0:35 (Manual pipeline proof)**
- Show the Streamlit app.
- Mention: “No LangChain, no LlamaIndex — I implemented chunking, retrieval, reranking, prompt building manually.”

**0:35–1:10 (Retrieval visibility)**
- Enter a query.
- Click Retrieve.
- Point out:
  - chunking strategy (fixed vs section-aware)
  - retrieved chunks with similarity scores
  - hybrid retrieval (vector + TF-IDF)
  - domain-aware reranking score

**1:10–1:35 (Prompt + answer + logging)**
- Select chunks.
- Show the final prompt.
- Generate answer.
- Show the JSONL logs (retrieved chunks, scores, final prompt, answer).

**1:35–1:55 (Evaluation + baseline)**
- Show “pure LLM baseline” answer vs RAG answer.
- Mention 2 adversarial queries and evaluation report.

**1:55–2:00 (Close)**
- “Repo link and deployed URL are in the README. Thank you.”
