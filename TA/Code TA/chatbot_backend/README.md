# BioRAG Chatbot Backend

FastAPI server that runs the **thesis Kel.3 best config** (Hybrid + Query
Rewriting + Context Reranking with ekspansi akronim) and serves it to the
chatbot UI at `ui/chatbot/`.

## Pipeline

```
question
  → Query rewriting (GPT-4.1-mini, T=0.3)
  → BM25 top-50 + Dense top-50 (Chroma, text-embedding-3-small)
  → RRF (k=60) → top-10
  → Cross-encoder rerank (ms-marco-MiniLM-L-6-v2) → top-5
  → GPT-4.1-mini answer (T=0.2)  ← NATURAL prompt, no yes/no/maybe
```

Retrieval is byte-identical to `notebooks_v2/30_hybrid_dengan_ekspansi/gpt4mini/qr_cr.ipynb`.
Generation uses a conversational prompt (see `prompts.py`) instead of the
classification prompt used in the thesis evaluation.

## Setup (Windows)

1. **Copy `.env.example` to `.env`** and fill in your `OPENAI_API_KEY`.

2. **Create venv + install deps:**
   ```cmd
   cd chatbot_backend
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run:**
   ```cmd
   python main.py
   ```
   Or simply double-click `run.bat`.

Server boots at `http://127.0.0.1:8000`.

The first request triggers index loading (~10-30s) — startup hook warms it
up so users don't pay that latency. Subsequent requests are ~2-4s end-to-end
(QR + retrieval + rerank + generation).

## Endpoints

### `GET /api/health`
Liveness probe. Returns whether each subsystem loaded:
```json
{
  "status": "ok",
  "pipeline_ready": true,
  "indexes_loaded": { "bm25": true, "chroma": true, "reranker": true, "openai": true }
}
```

### `POST /api/chat`
**Request:**
```json
{ "question": "Does vitamin D supplementation boost immunity?" }
```

**Response:**
```json
{
  "answer": "**Yes, there is evidence to support this.**\n\nResearchers found...",
  "pipeline": {
    "original_query": "Does vitamin D...",
    "rewritten_query": "Effect of vitamin D supplementation on immune function in healthy adults",
    "retrieved_docs": [
      {
        "rank": 1,
        "pubid": "20684175",
        "section_label": "RESULTS",
        "text": "Vitamin D supplementation...",
        "bm25_score": 12.34,
        "dense_score": 0.78,
        "rrf_score": 0.032,
        "rerank_score": 8.42
      }
    ],
    "latency_ms": {
      "query_rewrite_ms": 412.3,
      "retrieval_ms": 234.1,
      "fusion_ms": 1.2,
      "rerank_ms": 89.5,
      "generation_ms": 1893.7
    }
  },
  "model": "gpt-4.1-mini"
}
```

## Dependencies on Repo Layout

The backend reads from `../notebooks_v2/indexes/`:
- `pubmedqa_bm25_ekspansi.pkl` — BM25 index + Document list (1.7K chunks)
- `pubmedqa_chroma_ekspansi/` — Chroma persistent DB (collection: `pubmedqa_docs_sh`)

If those files move, update the `INDEXES_DIR` path in `pipeline.py`.

## Frontend Wiring

The Vite dev server at `http://localhost:5174` (ui/chatbot) calls
`http://127.0.0.1:8000/api/chat`. CORS is wide-open in dev — restrict
before deploying.

`ui/chatbot/src/services/chatService.js` falls back to mock answers when the
backend isn't reachable, so the demo still works offline.
