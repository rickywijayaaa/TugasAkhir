"""
RAG pipeline — Hybrid (BM25 + Dense) + Query Rewriting + Context Reranking.

Configuration matches notebooks/30_hybrid_dengan_ekspansi/gpt4mini/qr_cr.ipynb
(thesis Kel.3 "best config"), with one intentional difference: the final answer
prompt is the NATURAL chatbot prompt (no forced yes/no/maybe) — see prompts.py.

Pipeline:
  query
    → QueryRewriter(GPT-4.1-mini, temperature=0.3)
    → BM25 top-50 + Dense top-50 (Chroma, text-embedding-3-small)
    → Reciprocal Rank Fusion (k=60) → top-10
    → CrossEncoder rerank (ms-marco-MiniLM-L-6-v2) → top-5
    → GPT-4.1-mini answer generation (temperature=0.2)
"""

from __future__ import annotations

import os
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import chromadb
from openai import OpenAI
from sentence_transformers import CrossEncoder

from prompts import (
    ANSWER_PROMPT,
    HEDGE_INSTRUCTION,
    LOW_CONFIDENCE_REFUSAL,
    QUERY_REWRITE_PROMPT,
)
import math


# ---------- Config (env-overridable; defaults match thesis Kel.3) -------------

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "pubmedqa_docs_sh")

TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "50"))
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "50"))
TOP_K_AFTER_RRF = int(os.getenv("TOP_K_AFTER_RRF", "10"))
TOP_K_AFTER_RERANK = int(os.getenv("TOP_K_AFTER_RERANK", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
QR_TEMPERATURE = float(os.getenv("QR_TEMPERATURE", "0.3"))
ANSWER_TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0.2"))
SEED = int(os.getenv("SEED", "42"))

# Guard-rail thresholds for hallucination mitigation.
# Composite score is in [0, 1]; bands chosen so a clearly out-of-scope query
# (negative top-1 rerank, low dense sim) lands below 0.4 while an in-scope
# query (top-1 rerank near 0 and dense > 0.5) lands above 0.7.
CONFIDENCE_HIGH_THRESHOLD = float(os.getenv("CONFIDENCE_HIGH_THRESHOLD", "0.7"))
CONFIDENCE_LOW_THRESHOLD = float(os.getenv("CONFIDENCE_LOW_THRESHOLD", "0.4"))


# ---------- Paths --------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
CODE_TA = _HERE.parent  # chatbot_backend/ sibling of notebooks/
INDEXES_DIR = CODE_TA / "notebooks" / "indexes"
BM25_INDEX_PATH = INDEXES_DIR / "pubmedqa_bm25_ekspansi.pkl"
CHROMA_DB_PATH = INDEXES_DIR / "pubmedqa_chroma_ekspansi"


# ---------- Document schema (matches pickle structure) -------------------------


@dataclass
class Document:
    """Document chunk as stored in the BM25 pickle."""

    text: str
    pubid: str = ""
    question: str = ""
    section_label: str = ""
    answer: str = ""
    decision: str = ""

    @classmethod
    def from_any(cls, obj: Any) -> "Document":
        """Coerce whatever the pickle gave us into a Document."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls(
                text=obj.get("text", ""),
                pubid=str(obj.get("pubid", "")),
                question=obj.get("question", ""),
                section_label=obj.get("section_label", ""),
                answer=obj.get("answer", ""),
                decision=obj.get("decision", ""),
            )
        # Last resort — duck-type
        return cls(
            text=getattr(obj, "text", str(obj)),
            pubid=str(getattr(obj, "pubid", "")),
            question=getattr(obj, "question", ""),
            section_label=getattr(obj, "section_label", ""),
            answer=getattr(obj, "answer", ""),
            decision=getattr(obj, "decision", ""),
        )


@dataclass
class Candidate:
    """A retrieved document with all per-stage scores attached."""

    document: Document
    doc_id: int  # index into the corpus
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class _CompatUnpickler(pickle.Unpickler):
    """
    The BM25 pickle was saved from a notebook where `Document` lived in
    `__main__`. When unpickling here, Python looks for `__main__.Document`
    and can't find it. We redirect those lookups to our Document class.
    """

    def find_class(self, module, name):
        if name == "Document" and module in ("__main__", "builtins"):
            return Document
        return super().find_class(module, name)


# ---------- Utilities ---------------------------------------------------------


def tokenize_bm25(text: str) -> list[str]:
    """Same tokenizer used to build the BM25 index."""
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower()).split()


def _sigmoid(x: float) -> float:
    """Stable sigmoid that won't overflow on extreme inputs."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def compute_confidence(
    reranked: list["Candidate"],
) -> tuple[float, str, dict[str, float]]:
    """
    Composite retrieval confidence in [0, 1].

    Three signals are blended:

    1. **rerank_signal** — cross-encoder score on the top-1 doc. The reranker
       is the strongest discriminator empirically: in-scope queries land near
       0 while clearly out-of-scope ones go below -8. Sigmoid centred at -3.
    2. **dense_signal** — top-1 dense cosine similarity, scaled so 0.7 ≈ 1.0.
    3. **consensus_signal** — mean rerank score across the top-3 reranked
       docs. Penalises cases where only doc-1 is decent but the rest are
       garbage (a sign of luck rather than real relevance).

    Returns (composite_score, level_label, per_signal_breakdown).
    """
    if not reranked:
        return 0.0, "low", {"rerank_signal": 0.0, "dense_signal": 0.0, "consensus_signal": 0.0}

    top1 = reranked[0]
    top1_rerank = top1.rerank_score if top1.rerank_score is not None else -10.0
    top1_dense = top1.dense_score if top1.dense_score is not None else 0.0

    # Top-3 average rerank (consensus). Falls back gracefully if fewer than 3.
    top_n = reranked[: min(3, len(reranked))]
    mean_rerank = sum(
        (c.rerank_score if c.rerank_score is not None else -10.0) for c in top_n
    ) / len(top_n)

    rerank_signal = _sigmoid(0.5 * (top1_rerank + 3.0))
    dense_signal = max(0.0, min(1.0, top1_dense / 0.7))
    consensus_signal = _sigmoid(0.4 * (mean_rerank + 4.0))

    composite = (
        0.5 * rerank_signal + 0.3 * dense_signal + 0.2 * consensus_signal
    )

    if composite >= CONFIDENCE_HIGH_THRESHOLD:
        level = "high"
    elif composite >= CONFIDENCE_LOW_THRESHOLD:
        level = "moderate"
    else:
        level = "low"

    return composite, level, {
        "rerank_signal": rerank_signal,
        "dense_signal": dense_signal,
        "consensus_signal": consensus_signal,
    }


def reciprocal_rank_fusion(
    rank_lists: list[list[int]], k: int = RRF_K
) -> list[tuple[int, float]]:
    """Combine multiple ranked lists with RRF. Returns [(doc_id, score), ...]."""
    scores: dict[int, float] = {}
    for rl in rank_lists:
        for rank, doc_id in enumerate(rl):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------- The pipeline -------------------------------------------------------


class ChatbotPipeline:
    """
    Stateful RAG pipeline. Heavy artifacts (BM25 index, Chroma client,
    cross-encoder, OpenAI client) are loaded once at startup and reused per
    request.
    """

    def __init__(self) -> None:
        self.indexes_loaded: dict[str, bool] = {
            "bm25": False,
            "chroma": False,
            "reranker": False,
            "openai": False,
        }
        self.bm25_index = None
        self.documents: list[Document] = []
        self.chroma_collection = None
        self.cross_encoder: Optional[CrossEncoder] = None
        self.openai_client: Optional[OpenAI] = None

    # --- Load ----------------------------------------------------------------

    def load_all(self) -> None:
        self._load_bm25()
        self._load_chroma()
        self._load_reranker()
        self._init_openai()

    def _load_bm25(self) -> None:
        if not BM25_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {BM25_INDEX_PATH}\n"
                "Make sure notebooks/indexes/pubmedqa_bm25_ekspansi.pkl exists."
            )
        with open(BM25_INDEX_PATH, "rb") as f:
            saved = _CompatUnpickler(f).load()
        self.bm25_index = saved["bm25"]
        self.documents = [Document.from_any(d) for d in saved["documents"]]
        self.indexes_loaded["bm25"] = True
        print(f"[pipeline] BM25 loaded: {len(self.documents)} documents")

    def _load_chroma(self) -> None:
        if not CHROMA_DB_PATH.exists():
            raise FileNotFoundError(
                f"Chroma DB not found: {CHROMA_DB_PATH}\n"
                "Make sure notebooks/indexes/pubmedqa_chroma_ekspansi/ exists."
            )
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        try:
            self.chroma_collection = client.get_collection(name=CHROMA_COLLECTION)
        except Exception:
            # Fall back to the first available collection if the expected
            # name isn't present (defensive — name has shifted historically).
            collections = client.list_collections()
            if not collections:
                raise
            self.chroma_collection = collections[0]
            print(
                f"[pipeline] WARN: collection '{CHROMA_COLLECTION}' missing, "
                f"using '{self.chroma_collection.name}' instead"
            )
        self.indexes_loaded["chroma"] = True
        count = self.chroma_collection.count()
        print(f"[pipeline] Chroma loaded: {count} embeddings")

    def _load_reranker(self) -> None:
        self.cross_encoder = CrossEncoder(RERANKER_MODEL)
        self.indexes_loaded["reranker"] = True
        print(f"[pipeline] CrossEncoder loaded: {RERANKER_MODEL}")

    def _init_openai(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY missing. Copy .env.example to .env and fill it in."
            )
        self.openai_client = OpenAI(api_key=api_key)
        self.indexes_loaded["openai"] = True
        print("[pipeline] OpenAI client initialised")

    # --- Stages --------------------------------------------------------------

    def rewrite_query(self, query: str) -> str:
        prompt = QUERY_REWRITE_PROMPT.format(query=query)
        resp = self.openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=QR_TEMPERATURE,
            max_tokens=150,
            seed=SEED,
        )
        return resp.choices[0].message.content.strip().replace("\n", " ")

    def embed_query(self, query: str) -> list[float]:
        resp = self.openai_client.embeddings.create(
            model=EMBED_MODEL, input=[query]
        )
        return resp.data[0].embedding

    def search_bm25(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokens = tokenize_bm25(query)
        scores = self.bm25_index.get_scores(tokens)
        # rank descending
        idx_sorted = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [(i, float(scores[i])) for i in idx_sorted]

    def search_dense(
        self, query: str, top_k: int
    ) -> list[tuple[int, float]]:
        query_embedding = self.embed_query(query)
        result = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances", "metadatas"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0] or [{}] * len(ids)
        out: list[tuple[int, float]] = []
        for raw_id, dist, meta in zip(ids, distances, metadatas):
            # Try metadata first (most reliable), then parse from id.
            doc_id = self._resolve_doc_id(raw_id, meta)
            if doc_id is None:
                continue
            similarity = 1.0 - float(dist)
            out.append((doc_id, similarity))
        return out

    def _resolve_doc_id(
        self, raw_id: str, metadata: dict | None
    ) -> Optional[int]:
        """Map a Chroma id/metadata back to our corpus index."""
        if metadata:
            for key in ("doc_id", "idx", "index"):
                v = metadata.get(key)
                if v is not None:
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        pass
        # Common pattern: id is the integer index as a string, e.g. "42" or "doc_42".
        m = re.search(r"(\d+)", str(raw_id))
        if m:
            return int(m.group(1))
        return None

    def rerank(
        self, query: str, candidates: list[Candidate], top_k: int
    ) -> list[Candidate]:
        if not candidates:
            return []
        pairs = [(query, c.document.text) for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        for c, s in zip(candidates, scores):
            c.rerank_score = float(s)
        return sorted(candidates, key=lambda c: -(c.rerank_score or 0.0))[:top_k]

    def generate_answer(
        self, question: str, contexts: list[str], hedge: bool = False
    ) -> str:
        context_block = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{c}" for i, c in enumerate(contexts)
        )
        prompt = ANSWER_PROMPT.format(context=context_block, question=question)
        if hedge:
            # Insert the hedge instruction just before "Answer (...)" closing line.
            prompt = prompt + "\n" + HEDGE_INSTRUCTION
        resp = self.openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=ANSWER_TEMPERATURE,
            max_tokens=600,
            seed=SEED,
        )
        return resp.choices[0].message.content.strip()

    # --- Orchestration -------------------------------------------------------

    def run(self, question: str) -> dict:
        """Full pipeline. Returns a dict ready to be wrapped in ChatResponse."""
        timings: dict[str, float] = {}

        # 1. Query rewriting (QR)
        t0 = time.perf_counter()
        rewritten = self.rewrite_query(question)
        timings["query_rewrite_ms"] = (time.perf_counter() - t0) * 1000

        # 2. Retrieval (BM25 + Dense)
        t0 = time.perf_counter()
        bm25_hits = self.search_bm25(rewritten, TOP_K_BM25)
        dense_hits = self.search_dense(rewritten, TOP_K_DENSE)
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        # 3. RRF fusion
        t0 = time.perf_counter()
        bm25_ranked_ids = [doc_id for doc_id, _ in bm25_hits]
        dense_ranked_ids = [doc_id for doc_id, _ in dense_hits]
        fused = reciprocal_rank_fusion(
            [bm25_ranked_ids, dense_ranked_ids], k=RRF_K
        )[:TOP_K_AFTER_RRF]
        bm25_score_map = dict(bm25_hits)
        dense_score_map = dict(dense_hits)
        candidates: list[Candidate] = []
        for doc_id, rrf_score in fused:
            if doc_id < 0 or doc_id >= len(self.documents):
                continue
            candidates.append(
                Candidate(
                    document=self.documents[doc_id],
                    doc_id=doc_id,
                    bm25_score=bm25_score_map.get(doc_id),
                    dense_score=dense_score_map.get(doc_id),
                    rrf_score=rrf_score,
                )
            )
        timings["fusion_ms"] = (time.perf_counter() - t0) * 1000

        # 4. Cross-encoder rerank (CR)
        t0 = time.perf_counter()
        reranked = self.rerank(rewritten, candidates, TOP_K_AFTER_RERANK)
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

        # 4b. Confidence — guard rail for hallucination mitigation.
        # Computed AFTER rerank so we use the cross-encoder's final ordering.
        confidence_score, confidence_level, conf_breakdown = compute_confidence(reranked)

        # 5. Answer generation — branches on confidence level.
        t0 = time.perf_counter()
        contexts = [c.document.text for c in reranked]
        refused = False
        if not contexts:
            answer = (
                "I couldn't find relevant biomedical research for your question "
                "in the available corpus. Please try rephrasing or asking about "
                "a different health topic."
            )
            refused = True
        elif confidence_level == "low":
            # Hard guard: skip the LLM entirely on clearly off-topic queries.
            # This is the strongest hallucination-mitigation lever — no model
            # call means no chance for the model to fabricate.
            answer = LOW_CONFIDENCE_REFUSAL
            refused = True
        else:
            # Moderate confidence triggers the hedging instruction. High goes
            # through cleanly.
            answer = self.generate_answer(
                question,
                contexts,
                hedge=(confidence_level == "moderate"),
            )
        timings["generation_ms"] = (time.perf_counter() - t0) * 1000

        retrieved_docs = [
            {
                "rank": i + 1,
                "pubid": c.document.pubid,
                "section_label": c.document.section_label or None,
                "text": c.document.text,
                "bm25_score": c.bm25_score,
                "dense_score": c.dense_score,
                "rrf_score": c.rrf_score,
                "rerank_score": c.rerank_score,
            }
            for i, c in enumerate(reranked)
        ]

        return {
            "answer": answer,
            "pipeline": {
                "original_query": question,
                "rewritten_query": rewritten,
                "retrieved_docs": retrieved_docs,
                "latency_ms": timings,
            },
            "confidence": {
                "score": confidence_score,
                "level": confidence_level,
                "refused": refused,
                "breakdown": conf_breakdown,
                "thresholds": {
                    "high": CONFIDENCE_HIGH_THRESHOLD,
                    "low": CONFIDENCE_LOW_THRESHOLD,
                },
            },
            "model": LLM_MODEL,
        }


# ---------- Module-level singleton --------------------------------------------

_pipeline_singleton: Optional[ChatbotPipeline] = None


def get_pipeline() -> ChatbotPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = ChatbotPipeline()
        _pipeline_singleton.load_all()
    return _pipeline_singleton
