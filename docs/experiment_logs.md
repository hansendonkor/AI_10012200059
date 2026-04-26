# Experiment Logs (Manual)

**Name:** Hansen Donkor  
**Index Number:** 10012200059

Use this file to record **manual experiments** for chunking, retrieval, reranking, and prompting.

---

## How To Log Each Experiment
For each experiment, copy/paste this template:

**Date/Time:** 

**Experiment Name:** 

**Change Made:** 

**Parameters:**
- Chunking: fixed/section
- Chunk size chars:
- Overlap chars:
- Top K:
- Weights: vector / keyword / domain
- Prompt notes:

**Test Queries:**
1. 
2. 
3. 

**Observed Retrieval (Top 5):**
- Chunk IDs:
- Notes on relevance:

**Answer Quality:**
- Accuracy (1–5):
- Hallucination (none/minor/major):
- Consistency (consistent/inconsistent):

**Failure Case(s):**
- What went wrong?

**Fix Applied:**
- What changed to improve it?

---

## Chunking Experiments (Part A)
- Compare fixed vs section-aware chunking for PDF queries.
- Record at least one failure caused by bad chunk boundaries.

## Retrieval Experiments (Part B)
- Compare vector-only vs keyword-only vs hybrid.
- Record a query where keyword helps (names, exact phrases) and where vector helps (paraphrase).

## Prompt Experiments (Part C)
- Compare prompts with/without explicit "I don't know" instruction.
- Compare prompts with/without citations.

## Reranking/Innovation Experiments (Part G)
- Show at least one ambiguous query and demonstrate how domain weighting changes top results.
