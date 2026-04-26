# Evaluation Report

**Name:** Hansen Donkor  
**Index Number:** 10012200059

## Evaluation Setup
- Data sources:
  - Ghana Election Result CSV
  - 2025 Budget Statement PDF
- Retrieval modes tested:
  1. Pure LLM (no retrieval)
  2. RAG (hybrid retrieval + reranking)

## Metrics (Manual)
For each query, record:
- **Accuracy (1–5):** correctness vs document evidence
- **Hallucination:** none / minor / major
- **Consistency:** consistent / inconsistent across runs
- **Grounding:** did the answer cite chunks and stay within them?

---

## Test Cases

### Normal Queries
| Query | Pure LLM Result | RAG Result | Notes |
|---|---|---|---|
|  |  |  |  |
|  |  |  |  |

### Adversarial Queries (Required)
Include at least two:

1) **Adversarial Query #1:**
- Query:
- Why adversarial:
- Pure LLM:
- RAG:
- Notes:

2) **Adversarial Query #2:**
- Query:
- Why adversarial:
- Pure LLM:
- RAG:
- Notes:

---

## Summary
- Where RAG improves accuracy:
- Where RAG reduces hallucinations:
- Remaining limitations:
- Next improvement (if time):
