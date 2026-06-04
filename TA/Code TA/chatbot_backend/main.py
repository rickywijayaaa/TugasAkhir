"""
FastAPI app for the BioRAG chatbot backend.

Endpoints:
  GET  /api/health  — quick liveness probe
  POST /api/chat    — run the RAG pipeline on a question

CORS is wide-open by default so the Vite dev server (localhost:5174) can call
this directly. Lock it down before exposing publicly.
"""

from __future__ import annotations

import logging
import os
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import ChatRequest, ChatResponse, HealthResponse
from pipeline import ChatbotPipeline, get_pipeline


# Load .env before anything else reads the environment.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("biorag")


app = FastAPI(
    title="BioRAG Chatbot Backend",
    description="RAG pipeline (Hybrid + QR + CR) over PubMedQA — thesis Kel.3 config.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only — restrict before deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm_pipeline() -> None:
    """Load indexes + models eagerly so the first request isn't slow."""
    try:
        get_pipeline()
        log.info("Pipeline loaded and ready.")
    except Exception as e:  # noqa: BLE001 — surface any startup failure
        log.error("Pipeline failed to load: %s", e)
        log.error(traceback.format_exc())
        # Don't crash — the /api/health endpoint will report not-ready,
        # and /api/chat will return a 503 until things are fixed.


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        pipe = get_pipeline()
        return HealthResponse(
            status="ok",
            pipeline_ready=all(pipe.indexes_loaded.values()),
            indexes_loaded=pipe.indexes_loaded,
        )
    except Exception as e:  # noqa: BLE001
        return HealthResponse(
            status=f"error: {e}",
            pipeline_ready=False,
            indexes_loaded={},
        )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        pipe = get_pipeline()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Pipeline unavailable: {e}")

    try:
        result = pipe.run(question)
    except Exception as e:  # noqa: BLE001
        log.error("chat() failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    log.info(
        "chat ok | q=%r | rewritten=%r | docs=%d | total_ms=%.0f",
        question[:80],
        result["pipeline"]["rewritten_query"][:80],
        len(result["pipeline"]["retrieved_docs"]),
        sum(result["pipeline"]["latency_ms"].values()),
    )
    return ChatResponse(**result)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)
