# Implementation Plan: RAG with Hallucination Mitigation

## Overview
Implementasi RAG-LLM dengan 3 teknik mitigasi halusinasi untuk thesis:
1. Query Rewriting
2. Context Reranking
3. Active Hallucination Detection

Evaluasi menggunakan RAGAS + Active Detection pada dataset PubMedQA.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│ Query Rewriting │ ← Ollama (llama3.2)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Search   │ ← FAISS + Ollama embeddings (nomic-embed-text)
│ (Full RAG)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Context Rerank  │ ← CrossEncoder (sentence-transformers)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Answer Generate │ ← Ollama (llama3.2)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Hallucination   │ ← RAGAS + Active Detection (llama3.2)
│ Detection       │
└─────────────────┘
```

---

## Implementation Steps

### Step 1: Setup Full RAG with PubMedQA
**File:** `RAG-QueryRewriting-Ollama.ipynb` (update existing)

- Load PubMedQA dataset (pqa_labeled: 1000 samples)
- Extract ALL contexts from all samples into a single corpus
- Build FAISS vector index from corpus using Ollama embeddings
- Implement retrieval function that searches across ALL documents

```python
# Corpus structure:
corpus = [
    {"text": "context text...", "pubid": "123", "label": "METHODS", ...},
    {"text": "context text...", "pubid": "123", "label": "RESULTS", ...},
    {"text": "context text...", "pubid": "456", "label": "BACKGROUND", ...},
    ...
]
```

### Step 2: Implement 5 RAG Configurations (sesuai proposal)
| Config | Query Rewriting | Context Reranking | Active Detection |
|--------|-----------------|-------------------|------------------|
| 1. Baseline | ❌ | ❌ | ❌ |
| 2. +QR | ✅ | ❌ | ❌ |
| 3. +CR | ❌ | ✅ | ❌ |
| 4. +AD | ❌ | ❌ | ✅ |
| 5. Combined | ✅ | ✅ | ✅ |

### Step 3: Implement RAGAS Evaluation
**Dependencies:** `ragas`, `langchain`

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # Apakah jawaban sesuai context?
    answer_relevancy,    # Apakah jawaban relevan dengan pertanyaan?
    context_precision,   # Apakah context yang diambil relevan?
    context_recall       # Apakah semua info yang diperlukan ter-retrieve?
)
```

**Metrics untuk thesis:**
- **Faithfulness** = 1 - Hallucination Rate
- **Answer Relevancy** = Seberapa relevan jawaban
- **Context Precision** = Kualitas retrieval

### Step 4: Implement Active Hallucination Detection
```python
VERIFICATION_PROMPT = """
You are a fact-checker. Given the context and answer, determine if each claim
in the answer is supported by the context.

Context: {context}
Answer: {answer}

For each claim in the answer, state:
- SUPPORTED: claim is directly supported by context
- NOT_SUPPORTED: claim contradicts or is not in context
- PARTIALLY_SUPPORTED: claim is partially supported

Final verdict: [SUPPORTED / NOT_SUPPORTED / PARTIALLY_SUPPORTED]
Confidence: [HIGH / MEDIUM / LOW]
"""
```

### Step 5: Run Evaluation Pipeline
**Sample size:** 1000 samples (full PubMedQA labeled dataset)

```python
def evaluate_config(config_name, use_qr, use_cr, use_ad):
    results = []
    for sample in pubmedqa_data:  # All 1000 samples
        # 1. Get RAG response with specified config
        response = rag_pipeline(
            query=sample['question'],
            use_query_rewriting=use_qr,
            use_context_reranking=use_cr
        )

        # 2. RAGAS evaluation
        ragas_scores = evaluate_ragas(
            question=sample['question'],
            answer=response.answer,
            contexts=[c.text for c in response.contexts],
            ground_truth=sample['long_answer']
        )

        # 3. Active Detection (if enabled)
        if use_ad:
            detection = active_hallucination_check(
                answer=response.answer,
                context=response.contexts
            )

        results.append({...})

    return aggregate_results(results)
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `RAG-FullPipeline.ipynb` | CREATE | Main notebook dengan semua teknik |
| `evaluation.py` | CREATE | RAGAS + Active Detection functions |
| `config.py` | CREATE | Configuration constants |

---

## Dependencies to Install
```bash
pip install ragas langchain langchain-community sentence-transformers
pip install datasets faiss-cpu ollama numpy pandas
```

---

## Verification / Testing

1. **Test retrieval quality:**
   - Retrieve top-5 documents for sample questions
   - Check if relevant contexts are retrieved

2. **Test each mitigation technique:**
   - Run baseline vs with-technique
   - Compare faithfulness scores

3. **Full evaluation:**
   - Run all 5 configurations on 1000 samples
   - Generate comparison table for thesis

4. **Expected output:**
   ```
   | Configuration | Faithfulness | Hallucination Rate | Answer Relevancy |
   |---------------|--------------|--------------------|--------------------|
   | Baseline      | 0.72         | 28%                | 0.68               |
   | +QR           | 0.78         | 22%                | 0.74               |
   | +CR           | 0.80         | 20%                | 0.71               |
   | +AD           | 0.85         | 15%                | 0.70               |
   | Combined      | 0.89         | 11%                | 0.76               |
   ```

---

## Questions Resolved
- RAG documents: From ALL PubMedQA contexts (Full RAG approach)
- Ollama role: LLM for generation/rewriting + embeddings
- Hallucination detection: RAGAS faithfulness + Active Detection
