"""Pydantic schemas for request/response."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class RetrievedDoc(BaseModel):
    rank: int
    pubid: str
    section_label: Optional[str] = None
    text: str
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class ConfidenceBreakdown(BaseModel):
    """Per-signal scores that feed the composite confidence."""

    rerank_signal: float  # 0..1, sigmoid of top-1 rerank score
    dense_signal: float  # 0..1, top-1 dense similarity scaled to [0,1]
    consensus_signal: float  # 0..1, agreement across top-3 rerank scores


class Confidence(BaseModel):
    """Retrieval confidence — the guard-rail signal for hallucination mitigation."""

    score: float  # 0..1 composite
    level: str  # "high" | "moderate" | "low"
    refused: bool  # True if the LLM call was skipped due to low confidence
    breakdown: ConfidenceBreakdown
    thresholds: dict[str, float]  # echoes the high/moderate cutoffs


class PipelineDetails(BaseModel):
    original_query: str
    rewritten_query: str
    retrieved_docs: list[RetrievedDoc]
    latency_ms: dict[str, float] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    pipeline: PipelineDetails
    confidence: Confidence
    model: str = "gpt-4.1-mini"


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    indexes_loaded: dict[str, bool]
