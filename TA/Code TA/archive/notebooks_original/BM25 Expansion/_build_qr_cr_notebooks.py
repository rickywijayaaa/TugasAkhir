"""Generate 3 notebooks for SH index + QR/CR/QR+CR variants:
  05_qr_openai_sh.ipynb       - Multi-query QR (Ma et al 2023 style) + SH index
  06_cr_openai_sh.ipynb       - CrossEncoder reranking + SH index
  07_qr_cr_openai_sh.ipynb    - Multi-query QR + CrossEncoder rerank + SH index

QR follows paper: Ma, X. et al. (2023). "Query Rewriting for Retrieval-Augmented
Large Language Models" arXiv:2305.14283
- Few-shot prompt with 3 demos
- "Think step by step" CoT
- Multi-query output split by ';' ended with '**'
- All rank lists fused via RRF
"""
import json, uuid
from pathlib import Path

HERE = Path(__file__).parent


def md(src):
    return {'cell_type':'markdown','id':str(uuid.uuid4())[:8],
            'metadata':{},'source':src.splitlines(keepends=True)}


def code(src):
    return {'cell_type':'code','id':str(uuid.uuid4())[:8],
            'metadata':{},'execution_count':None,'outputs':[],
            'source':src.splitlines(keepends=True)}


def nb(cells):
    return {
        'cells': cells,
        'metadata': {
            'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python','version':'3.11'},
        },
        'nbformat': 4, 'nbformat_minor': 5,
    }


# ============================================================
# COMMON CELLS (shared across 3 notebooks)
# ============================================================
COMMON_IMPORTS = '''import os, sys, json, pickle, time, re, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime

from openai import OpenAI
from rank_bm25 import BM25Okapi
from datasets import load_dataset
import chromadb

warnings.filterwarnings("ignore")


@dataclass
class Document:
    text: str; pubid: str; question: str
    section_label: str; answer: str; decision: str

@dataclass
class RetrievalResult:
    document: Document; score: float; doc_id: int = -1
    bm25_score: float = 0.0; dense_score: float = 0.0
    rrf_score: float = 0.0; reranker_score: float = 0.0

import __main__
__main__.Document = Document

# # os.environ["OPENAI_API_KEY"] = "<REDACTED — set via shell env or .env file>"
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY belum di-set.")

print(f"Python: {sys.version.split()[0]} | NumPy: {np.__version__} | chromadb: {chromadb.__version__}")
'''


def config_cell(config_name, has_qr=False, has_cr=False):
    label_parts = []
    if has_qr: label_parts.append("Multi-Query QR")
    if has_cr: label_parts.append("CrossEncoder Rerank")
    method_label = " + ".join(label_parts) if label_parts else "Baseline"

    return f'''LLM_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"
TOP_K_BM25 = 50; TOP_K_DENSE = 50; TOP_K_RETRIEVAL = 5
TOP_K_RERANKER = 20  # candidates before CrossEncoder rerank (only used if CR=True)

DATASET_NAME = "qiaojin/PubMedQA"; DATASET_SUBSET = "pqa_labeled"
MAX_SAMPLES = 500
TEMPERATURE = 0.0; SEED = 42
MAX_REWRITE_QUERIES = 3  # max queries from rewriter

HERE = Path(".").resolve()
NOTEBOOKS_DIR = HERE.parent if HERE.name == "BM25 Expansion" else HERE
PROJECT_ROOT = NOTEBOOKS_DIR.parent
BM25_INDEX_PATH = NOTEBOOKS_DIR / "pubmedqa_bm25_sh.pkl"
CHROMA_DB_PATH = NOTEBOOKS_DIR / "pubmedqa_chroma_sh"
RESULTS_DIR = PROJECT_ROOT / "results" / "BM25_Expansion"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_NAME = "{config_name}"
PHASE1_PATH = RESULTS_DIR / f"{{CONFIG_NAME}}_phase1_answers.json"
PHASE2_PATH = RESULTS_DIR / f"{{CONFIG_NAME}}_phase2_custom.json"

print("Konfigurasi:")
print(f"  LLM         : {{LLM_MODEL}}")
print(f"  Method      : {method_label}")
print(f"  BM25 index  : {{BM25_INDEX_PATH.name}} (Schwartz-Hearst)")
print(f"  Chroma path : {{CHROMA_DB_PATH.name}} (Schwartz-Hearst)")
print(f"  Sampel      : {{MAX_SAMPLES}}")
print(f"  Output      : {{PHASE1_PATH}}")

assert BM25_INDEX_PATH.exists(), f"BM25 SH tidak ditemukan. Run build_sh_index.py."
assert CHROMA_DB_PATH.exists(), f"Chroma SH tidak ditemukan. Run notebook 03 dulu."
'''


LOAD_DATA_CELL = '''def tokenize_bm25(text):
    return re.sub(r"[^a-zA-Z0-9\\s]", " ", text.lower()).split()

with open(BM25_INDEX_PATH, "rb") as f:
    saved = pickle.load(f)
bm25_index = saved["bm25"]; documents = saved["documents"]
print(f"Loaded BM25 SH: {len(documents)} chunks")

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
chroma_collection = chroma_client.get_collection(name="pubmedqa_docs_sh")
print(f"Loaded Chroma SH: {chroma_collection.count()} vectors")

ds = load_dataset(DATASET_NAME, DATASET_SUBSET)["train"]
pubmedqa_data = ds.select(range(MAX_SAMPLES))
print(f"Loaded {len(pubmedqa_data)} samples")
'''


OPENAI_CLIENT_CELL = '''client = OpenAI(api_key=api_key)


def openai_generate(prompt, max_tokens=300, temperature=TEMPERATURE):
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens, seed=SEED,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                time.sleep((attempt + 1) * 10)
            elif "500" in err or "502" in err or "503" in err:
                time.sleep((attempt + 1) * 5)
            else:
                raise
    raise RuntimeError("OpenAI gagal")


def openai_embed(texts):
    for attempt in range(5):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                time.sleep((attempt + 1) * 10)
            elif "500" in err or "502" in err or "503" in err:
                time.sleep((attempt + 1) * 5)
            else:
                raise
    raise RuntimeError("OpenAI embed gagal")


_test = openai_generate("Reply OK", max_tokens=5)
print(f"OpenAI ready: {_test!r}")
'''


# QR Prompt (Ma et al 2023 style) — used in QR and QR+CR notebooks
QR_PROMPT_CELL = '''# ============================================================
# Query Rewriting - Few-Shot ala Paper Ma dkk. (2023)
# ============================================================
# Format prompt mengikuti Tabel 1 paper:
#   - Instruksi "Think step by step ..."
#   - Output: queries dipisah ";" diakhiri "**"
#   - 1..n rewritten queries (multi-query untuk dekomposisi)
#   - 3 demonstration examples khusus domain biomedis (PubMedQA)

QUERY_REWRITE_PROMPT = (
    "Think step by step to answer this question, and provide search engine "
    "queries for knowledge that you need. Split the queries with ';' and "
    "end the queries with '**'.\\n\\n"
    "Question: Does aspirin reduce the risk of MI in patients with diabetes?\\n"
    "Answer: aspirin myocardial infarction prevention diabetes; "
    "low-dose aspirin cardiovascular risk type 2 diabetes patients **\\n\\n"
    "Question: Is metformin effective for T2DM in elderly patients with renal impairment?\\n"
    "Answer: metformin type 2 diabetes mellitus efficacy elderly; "
    "metformin renal impairment safety; "
    "metformin contraindications kidney function older adults **\\n\\n"
    "Question: Can regular exercise lower blood pressure in older adults?\\n"
    "Answer: physical exercise hypertension blood pressure reduction elderly; "
    "aerobic exercise antihypertensive effect older adults **\\n\\n"
    "Question: {query}\\n"
    "Answer:"
)


def parse_rewriter_output(text, max_queries=MAX_REWRITE_QUERIES):
    """Parse output rewriter ala paper: '<q1>; <q2>; ... **'."""
    if not text:
        return []
    if "**" in text:
        text = text.split("**", 1)[0]
    text = re.sub(r"^\\s*Answer\\s*:\\s*", "", text, flags=re.IGNORECASE)
    queries = [q.strip() for q in text.split(";")]
    queries = [q for q in queries if len(q) >= 3]
    return queries[:max_queries]


def rewrite_query(query):
    """Multi-query reformulation. Returns list of queries."""
    prompt = QUERY_REWRITE_PROMPT.format(query=query)
    try:
        raw = openai_generate(prompt, max_tokens=200, temperature=0.3)
        queries = parse_rewriter_output(raw)
        return queries if queries else [query]
    except Exception as e:
        print(f"  [QR Error] {e} -- pakai query asli")
        return [query]


# Test QR
print("Contoh Query Rewriting (few-shot, multi-query):")
for q in ["Does aspirin reduce the risk of MI?",
          "Can exercise prevent T2DM?",
          "Is statin therapy beneficial for stroke prevention in patients over 75?"]:
    rws = rewrite_query(q)
    print(f"\\nAsli      : {q}")
    for i, rw in enumerate(rws, 1):
        print(f"Rewrite {i} : {rw}")
'''


# CrossEncoder cell (used in CR and QR+CR notebooks)
CROSSENCODER_CELL = '''# ============================================================
# Context Reranking dengan CrossEncoder
# ============================================================
from sentence_transformers import CrossEncoder

print("Loading CrossEncoder ms-marco-MiniLM-L-6-v2...")
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder = CrossEncoder(RERANKER_MODEL)
print("CrossEncoder ready")


def rerank_with_cross_encoder(query, candidates, k_final):
    """Rerank list of RetrievalResult with CrossEncoder."""
    if not candidates:
        return []
    pairs = [(query, r.document.text) for r in candidates]
    scores = cross_encoder.predict(pairs)
    for r, s in zip(candidates, scores):
        r.reranker_score = float(s)
    return sorted(candidates, key=lambda r: -r.reranker_score)[:k_final]
'''


# Retrieve baseline (no QR, no CR) - hybrid only
RETRIEVE_BASELINE = '''def retrieve_dense(query, k=TOP_K_DENSE):
    qvec = openai_embed([query])[0]
    res = chroma_collection.query(query_embeddings=[qvec], n_results=k, include=["distances"])
    doc_ids = [int(i) for i in res["ids"][0]]
    distances = res["distances"][0]
    scores = [1.0 - d for d in distances]
    return list(zip(doc_ids, scores))


def retrieve_bm25_raw(query, k=TOP_K_BM25):
    tokens = tokenize_bm25(query)
    scores = bm25_index.get_scores(tokens)
    top_k = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_k]


def reciprocal_rank_fusion(rank_lists, k=60):
    scores = {}
    for rl in rank_lists:
        for rank, doc_id in enumerate(rl):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
'''


# QR retrieve (multi-query RRF) - used in QR notebook
RETRIEVE_QR = RETRIEVE_BASELINE + '''

def retrieve_hybrid(query, k_final=TOP_K_RETRIEVAL):
    """Multi-query QR + Hybrid retrieval. Each query produces BM25+Dense rank list,
    all 2*N rank lists fused via single RRF."""
    rewritten_queries = rewrite_query(query)

    all_rank_lists = []
    bm25_score_lookup = {}
    dense_score_lookup = {}

    for rq in rewritten_queries:
        bm25_results = retrieve_bm25_raw(rq, k=TOP_K_BM25)
        dense_results = retrieve_dense(rq, k=TOP_K_DENSE)
        all_rank_lists.append([d for d, _ in bm25_results])
        all_rank_lists.append([d for d, _ in dense_results])
        for doc_id, score in bm25_results:
            bm25_score_lookup[doc_id] = max(bm25_score_lookup.get(doc_id, 0.0), score)
        for doc_id, score in dense_results:
            dense_score_lookup[doc_id] = max(dense_score_lookup.get(doc_id, 0.0), score)

    fused = reciprocal_rank_fusion(all_rank_lists)
    out = []
    for doc_id, rrf in fused[:k_final]:
        out.append(RetrievalResult(
            document=documents[doc_id], score=rrf, doc_id=doc_id,
            bm25_score=bm25_score_lookup.get(doc_id, 0.0),
            dense_score=dense_score_lookup.get(doc_id, 0.0), rrf_score=rrf,
        ))
    return out, rewritten_queries


# Smoke test
test_q = pubmedqa_data[0]["question"]
test_r, test_rws = retrieve_hybrid(test_q)
print(f"Query: {test_q}")
print(f"Rewrite ({len(test_rws)} queries):")
for i, rw in enumerate(test_rws, 1):
    print(f"  [{i}] {rw}")
print(f"\\nTop-{TOP_K_RETRIEVAL} chunks:")
for i, r in enumerate(test_r, 1):
    print(f"  [{i}] RRF={r.rrf_score:.4f} | BM25={r.bm25_score:6.2f} | Dense={r.dense_score:.3f} | "
          f"({r.document.section_label[:25]}) pubid={r.document.pubid}")
'''


# CR retrieve (single query but rerank top-20 to top-5)
RETRIEVE_CR = RETRIEVE_BASELINE + '''

def retrieve_hybrid(query, k_final=TOP_K_RETRIEVAL):
    """Single query Hybrid + CrossEncoder rerank. Get top-20 hybrid, rerank to top-5."""
    bm25_results = retrieve_bm25_raw(query, k=TOP_K_BM25)
    dense_results = retrieve_dense(query, k=TOP_K_DENSE)
    bm25_scores = dict(bm25_results); dense_scores = dict(dense_results)
    fused = reciprocal_rank_fusion([
        [d for d, _ in bm25_results], [d for d, _ in dense_results]
    ])
    # Get top-K_RERANKER candidates
    candidates = []
    for doc_id, rrf in fused[:TOP_K_RERANKER]:
        candidates.append(RetrievalResult(
            document=documents[doc_id], score=rrf, doc_id=doc_id,
            bm25_score=bm25_scores.get(doc_id, 0.0),
            dense_score=dense_scores.get(doc_id, 0.0), rrf_score=rrf,
        ))
    # Rerank with CrossEncoder
    reranked = rerank_with_cross_encoder(query, candidates, k_final=k_final)
    return reranked, query


# Smoke test
test_q = pubmedqa_data[0]["question"]
test_r, _ = retrieve_hybrid(test_q)
print(f"Query: {test_q}")
print(f"\\nTop-{TOP_K_RETRIEVAL} chunks (after CrossEncoder rerank from top-{TOP_K_RERANKER}):")
for i, r in enumerate(test_r, 1):
    print(f"  [{i}] CE={r.reranker_score:6.3f} | BM25={r.bm25_score:6.2f} | Dense={r.dense_score:.3f} | "
          f"({r.document.section_label[:25]}) pubid={r.document.pubid}")
'''


# QR + CR retrieve (multi-query QR + rerank)
RETRIEVE_QRCR = RETRIEVE_BASELINE + '''

def retrieve_hybrid(query, k_final=TOP_K_RETRIEVAL):
    """Multi-query QR + Hybrid + CrossEncoder rerank."""
    rewritten_queries = rewrite_query(query)

    all_rank_lists = []
    bm25_score_lookup = {}
    dense_score_lookup = {}

    for rq in rewritten_queries:
        bm25_results = retrieve_bm25_raw(rq, k=TOP_K_BM25)
        dense_results = retrieve_dense(rq, k=TOP_K_DENSE)
        all_rank_lists.append([d for d, _ in bm25_results])
        all_rank_lists.append([d for d, _ in dense_results])
        for doc_id, score in bm25_results:
            bm25_score_lookup[doc_id] = max(bm25_score_lookup.get(doc_id, 0.0), score)
        for doc_id, score in dense_results:
            dense_score_lookup[doc_id] = max(dense_score_lookup.get(doc_id, 0.0), score)

    fused = reciprocal_rank_fusion(all_rank_lists)
    # Get top-K_RERANKER for reranker
    candidates = []
    for doc_id, rrf in fused[:TOP_K_RERANKER]:
        candidates.append(RetrievalResult(
            document=documents[doc_id], score=rrf, doc_id=doc_id,
            bm25_score=bm25_score_lookup.get(doc_id, 0.0),
            dense_score=dense_score_lookup.get(doc_id, 0.0), rrf_score=rrf,
        ))

    # Rerank with original query (more focused)
    reranked = rerank_with_cross_encoder(query, candidates, k_final=k_final)
    return reranked, rewritten_queries


# Smoke test
test_q = pubmedqa_data[0]["question"]
test_r, test_rws = retrieve_hybrid(test_q)
print(f"Query: {test_q}")
print(f"Rewrite ({len(test_rws)} queries):")
for i, rw in enumerate(test_rws, 1):
    print(f"  [{i}] {rw}")
print(f"\\nTop-{TOP_K_RETRIEVAL} chunks (multi-query QR + CrossEncoder rerank):")
for i, r in enumerate(test_r, 1):
    print(f"  [{i}] CE={r.reranker_score:6.3f} | BM25={r.bm25_score:6.2f} | Dense={r.dense_score:.3f} | "
          f"({r.document.section_label[:25]}) pubid={r.document.pubid}")
'''


GENERATION_CELL = '''GENERATION_PROMPT = (
    "You are a medical research assistant. "
    "Answer a biomedical yes/no/maybe question based solely on the provided scientific abstracts.\\n\\n"
    "Context from medical literature:\\n{context}\\n\\n"
    "Question: {question}\\n\\n"
    "Instructions:\\n"
    "- Carefully read the context and assess whether it supports or refutes the question.\\n"
    "- Provide a brief explanation (2-3 sentences) using ONLY the information above.\\n"
    "- End your response with EXACTLY ONE of these words on its own line: yes, no, or maybe.\\n"
    "  - yes   : the evidence supports the hypothesis, even if not perfectly conclusive\\n"
    "  - no    : the evidence refutes or does not support the hypothesis\\n"
    "  - maybe : ONLY if the evidence is directly contradictory (some findings say yes,\\n"
    "            others say no), or if the context contains no relevant information at all\\n"
    "- IMPORTANT: If the evidence leans in one direction, even partially, choose yes or no.\\n"
    "  Do NOT use maybe simply because the evidence is limited or not 100%% certain.\\n\\n"
    "Answer:"
)


def generate_answer(query, retrieved):
    context = "\\n\\n".join(
        f"[{i}] ({r.document.section_label}): {r.document.text}"
        for i, r in enumerate(retrieved, 1)
    )
    return openai_generate(
        GENERATION_PROMPT.format(context=context, question=query),
        max_tokens=300, temperature=TEMPERATURE
    )


def extract_label(answer):
    lines = [l.strip().lower() for l in answer.split("\\n") if l.strip()]
    for line in reversed(lines[-3:]):
        word = re.sub(r"[^a-z]", "", line)
        if word in ("yes", "no", "maybe"):
            return word
    for label in ("yes", "no", "maybe"):
        if re.search(r"\\b" + label + r"\\b", answer.lower()):
            return label
    return "maybe"
'''


PHASE1_CELL_TEMPLATE = '''if PHASE1_PATH.exists():
    with open(PHASE1_PATH, "r", encoding="utf-8") as f:
        results_p1 = json.load(f)["results"]
    start_from = len(results_p1)
    print(f"Resume Fase 1: {{start_from}}/{{MAX_SAMPLES}} sudah selesai.")
else:
    results_p1, start_from = [], 0
    print(f"Memulai Fase 1: {{MAX_SAMPLES}} sampel.")

if start_from < MAX_SAMPLES:
    t0 = time.time()
    for i in range(start_from, MAX_SAMPLES):
        s = pubmedqa_data[i]
        q, gt, ref = s["question"], s["final_decision"], s["long_answer"]

        retrieved, used_q = retrieve_hybrid(q)
        answer = generate_answer(q, retrieved)
        predicted = extract_label(answer)

        results_p1.append({{
            "idx": i, "pubid": str(s["pubid"]), "question": q,
            {used_q_field}
            "ground_truth": gt, "predicted_label": predicted,
            "is_correct": predicted == gt, "answer": answer,
            "contexts": [r.document.text for r in retrieved],
            "context_pubids": [r.document.pubid for r in retrieved],
            "context_sections": [r.document.section_label for r in retrieved],
            "reference": ref,
            "retrieval_scores": [r.bm25_score for r in retrieved],
            "dense_scores": [r.dense_score for r in retrieved],
            "rrf_scores": [r.rrf_score for r in retrieved],
            "reranker_scores": [r.reranker_score for r in retrieved],
        }})

        if (i + 1) % 10 == 0 or i == MAX_SAMPLES - 1:
            with open(PHASE1_PATH, "w", encoding="utf-8") as f:
                json.dump({{"config": CONFIG_NAME, "llm_model": LLM_MODEL,
                           "embed_model": EMBED_MODEL,
                           "timestamp": datetime.now().isoformat(),
                           "max_samples": MAX_SAMPLES, "completed": i+1,
                           "results": results_p1}}, f, indent=2, ensure_ascii=False)
            done = i + 1
            acc = sum(r["is_correct"] for r in results_p1) / done
            eta = (time.time() - t0) / (i + 1 - start_from) * (MAX_SAMPLES - i - 1) / 60
            print(f"  [{{done:3d}}/{{MAX_SAMPLES}}] acc={{acc:.1%}} | ETA {{eta:.1f}} mnt")

print(f"\\nFase 1 selesai -> {{PHASE1_PATH}}")
'''


PHASE1_ANALYSIS_CELL = '''with open(PHASE1_PATH, "r", encoding="utf-8") as f:
    results_p1 = json.load(f)["results"]

n = len(results_p1)
n_correct = sum(r["is_correct"] for r in results_p1)
print(f"Schwartz-Hearst + {METHOD_LABEL} ({n} sampel)")
print("=" * 60)
print(f"Label Accuracy : {n_correct}/{n} = {n_correct/n:.1%}")
print()
print("Per-label accuracy:")
for lbl in ["yes", "no", "maybe"]:
    sub = [r for r in results_p1 if r["ground_truth"] == lbl]
    if sub:
        c = sum(r["is_correct"] for r in sub)
        print(f"  {lbl:>5}: {c}/{len(sub)} = {c/len(sub):.1%}")

# Comparison with other SH variants
print(f"\\n=== COMPARISON dalam SH family ===")
configs = {
    "SH baseline       ": RESULTS_DIR / "sh_baseline_openai_phase1_answers.json",
    "SH + QR           ": RESULTS_DIR / "sh_qr_openai_phase1_answers.json",
    "SH + CR           ": RESULTS_DIR / "sh_cr_openai_phase1_answers.json",
    "SH + QR + CR      ": RESULTS_DIR / "sh_qr_cr_openai_phase1_answers.json",
}
for name, path in configs.items():
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)["results"]
        a = sum(r["is_correct"] for r in d) / len(d)
        marker = "  <-- THIS" if path == PHASE1_PATH else ""
        print(f"  {name}: {a:.1%}{marker}")
'''


PHASE2_EVAL_CELL = '''def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\\s+", text.strip())
    return [s.strip() for s in parts if len(s.strip()) >= 15]


def _llm_yes_no(prompt):
    try:
        resp = openai_generate(prompt, max_tokens=10, temperature=0.0)
        return "yes" in resp.lower()[:15]
    except Exception:
        return False


def compute_faithfulness(answer, contexts):
    sents = _split_sentences(answer)
    if not sents: return 0.0
    ctx_text = "\\n".join(f"[{i+1}] {c[:1500]}" for i, c in enumerate(contexts))
    tmpl = ("Context:\\n{ctx}\\n\\nStatement: {sent}\\n\\n"
            "Is this statement directly supported by the context above? "
            "Answer with only \\"yes\\" or \\"no\\".")
    return sum(_llm_yes_no(tmpl.format(ctx=ctx_text, sent=s)) for s in sents) / len(sents)


def compute_context_recall(reference, contexts):
    sents = _split_sentences(reference)
    if not sents: return 0.0
    ctx_text = "\\n".join(f"[{i+1}] {c[:1500]}" for i, c in enumerate(contexts))
    tmpl = ("Context:\\n{ctx}\\n\\nStatement: {sent}\\n\\n"
            "Is this statement supported by the context above? "
            "Answer with only \\"yes\\" or \\"no\\".")
    return sum(_llm_yes_no(tmpl.format(ctx=ctx_text, sent=s)) for s in sents) / len(sents)


def compute_answer_relevancy(question, answer):
    sents = _split_sentences(answer)
    if not sents: return 0.0
    tmpl = ("Question: {q}\\n\\nStatement: {sent}\\n\\n"
            "Is this statement relevant to answering the question above? "
            "Answer with only \\"yes\\" or \\"no\\".")
    return sum(_llm_yes_no(tmpl.format(q=question, sent=s)) for s in sents) / len(sents)


def compute_context_precision(question, contexts, reference):
    if not contexts: return 0.0
    tmpl = ("Question: {q}\\n\\nGround truth answer: {ref}\\n\\nRetrieved context: {ctx}\\n\\n"
            "Does this context contain information useful for correctly answering "
            "the question based on the ground truth? Answer with only \\"yes\\" or \\"no\\".")
    relevance = [1 if _llm_yes_no(tmpl.format(q=question, ref=reference[:800], ctx=c[:1500])) else 0
                 for c in contexts]
    total = sum(relevance)
    if total == 0: return 0.0
    prec_sum = 0.0; rel_count = 0
    for k, rel in enumerate(relevance):
        if rel:
            rel_count += 1; prec_sum += rel_count / (k + 1)
    return prec_sum / total


def evaluate_custom(q, ans, ctx, ref):
    return {
        "faithfulness": compute_faithfulness(ans, ctx),
        "context_recall": compute_context_recall(ref, ctx),
        "answer_relevancy": compute_answer_relevancy(q, ans),
        "context_precision": compute_context_precision(q, ctx, ref),
    }


print("Evaluator ready.")
'''


PHASE2_RUN_CELL = '''REQ = ["faithfulness", "context_recall", "answer_relevancy", "context_precision"]

with open(PHASE1_PATH, "r", encoding="utf-8") as f:
    p1_results = json.load(f)["results"][:MAX_SAMPLES]

if PHASE2_PATH.exists():
    with open(PHASE2_PATH, "r", encoding="utf-8") as f:
        p2_results = json.load(f)["results"]
    done = {r["idx"] for r in p2_results if all(m in r for m in REQ)}
else:
    p2_results, done = [], set()

remaining = [r for r in p1_results if r["idx"] not in done]
print(f"Phase 2: {len(done)}/{MAX_SAMPLES} done, {len(remaining)} remaining.")

if remaining:
    t0 = time.time()
    for i, r in enumerate(remaining):
        scores = evaluate_custom(r["question"], r["answer"], r["contexts"], r["reference"])
        p2_results.append({
            "idx": r["idx"], "ground_truth": r["ground_truth"],
            "predicted_label": r["predicted_label"], "is_correct": r["is_correct"],
            **scores,
        })
        if (i + 1) % 10 == 0 or i == len(remaining) - 1:
            with open(PHASE2_PATH, "w", encoding="utf-8") as f:
                json.dump({"config": CONFIG_NAME,
                           "timestamp": datetime.now().isoformat(),
                           "max_samples": MAX_SAMPLES,
                           "metrics": REQ, "evaluator": "custom_zero_nan_4metrics",
                           "results": p2_results}, f, indent=2, ensure_ascii=False)
            done_n = i + 1
            avg = {m: sum(x[m] for x in p2_results) / len(p2_results) for m in REQ}
            eta = (time.time() - t0) / (i + 1) * (len(remaining) - i - 1) / 60
            print(f"  [{done_n:3d}/{len(remaining)}] "
                  f"f={avg['faithfulness']:.3f} cr={avg['context_recall']:.3f} "
                  f"ar={avg['answer_relevancy']:.3f} cp={avg['context_precision']:.3f} | ETA {eta:.1f}m")

print(f"\\nFase 2 selesai -> {PHASE2_PATH}")
'''


# ============================================================
# Build helpers
# ============================================================
def build_notebook(filename, title_md, config_name, has_qr, has_cr, retrieve_cell):
    method_label = []
    if has_qr: method_label.append("QR")
    if has_cr: method_label.append("CR")
    method_str = "+".join(method_label) if method_label else "Baseline"

    used_q_field = '"rewritten_queries": used_q if isinstance(used_q, list) else [used_q],' if has_qr else ''

    cells = [
        md(title_md),
        code(COMMON_IMPORTS),
        code(config_cell(config_name, has_qr, has_cr)),
        code(LOAD_DATA_CELL),
        code(OPENAI_CLIENT_CELL),
    ]
    if has_qr:
        cells.append(code(QR_PROMPT_CELL))
    if has_cr:
        cells.append(code(CROSSENCODER_CELL))
    cells.extend([
        code(retrieve_cell),
        code(GENERATION_CELL),
        md(f'## Phase 1 — Generate jawaban 500 sampel (resumable, {15 if not has_qr else 25} menit)'),
        code(PHASE1_CELL_TEMPLATE.format(used_q_field=used_q_field)),
        md('## Phase 1 — Quick analysis'),
        code(PHASE1_ANALYSIS_CELL.replace('{METHOD_LABEL}', method_str)),
        md('## Phase 2 — Custom evaluator 4 metrik (resumable, 25 menit)'),
        code(PHASE2_EVAL_CELL),
        code(PHASE2_RUN_CELL),
    ])

    out = HERE / filename
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(nb(cells), f, indent=1, ensure_ascii=False)
    print(f'Wrote: {filename}  ({len(cells)} cells)')


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    # Notebook 05: SH + QR (Multi-query Ma et al)
    build_notebook(
        '05_qr_openai_sh.ipynb',
        '''# 05 - SH + Query Rewriting (Ma et al 2023 Multi-Query)

**Setup**: Schwartz-Hearst index + Multi-Query QR (paper Ma dkk. 2023).

QR mengikuti paper:
- "Think step by step" CoT prompting
- 3 demonstrasi few-shot biomedis
- Output: 1-3 queries split by ";", end with "**"
- Setiap query menjalankan BM25+Dense, semua rank list digabung via RRF

**Output**: `results/BM25_Expansion/sh_qr_openai_phase{1,2}.json`

**Estimasi**:
- Phase 1: ~25 menit (extra LLM call per sample untuk rewrite)
- Phase 2: ~25 menit
- Total: ~50 menit, ~$15
''',
        'sh_qr_openai',
        has_qr=True, has_cr=False,
        retrieve_cell=RETRIEVE_QR,
    )

    # Notebook 06: SH + CR (CrossEncoder)
    build_notebook(
        '06_cr_openai_sh.ipynb',
        '''# 06 - SH + Context Reranking (CrossEncoder)

**Setup**: Schwartz-Hearst index + CrossEncoder reranking.

CR mengambil top-20 dari hybrid retrieval, lalu rerank menggunakan
`ms-marco-MiniLM-L-6-v2` untuk ambil top-5 final.

**Output**: `results/BM25_Expansion/sh_cr_openai_phase{1,2}.json`

**Estimasi**:
- Phase 1: ~17 menit (CrossEncoder ringan, CPU OK)
- Phase 2: ~25 menit
- Total: ~42 menit, ~$11
''',
        'sh_cr_openai',
        has_qr=False, has_cr=True,
        retrieve_cell=RETRIEVE_CR,
    )

    # Notebook 07: SH + QR + CR
    build_notebook(
        '07_qr_cr_openai_sh.ipynb',
        '''# 07 - SH + QR + CR (Multi-Query QR + CrossEncoder Rerank)

**Setup**: Schwartz-Hearst index + Multi-Query QR + CrossEncoder rerank.

Pipeline:
1. Multi-query QR (Ma et al)
2. Setiap query menjalankan BM25+Dense, fused via RRF -> top-20 candidates
3. CrossEncoder rerank (pakai original query) -> top-5 final

**Output**: `results/BM25_Expansion/sh_qr_cr_openai_phase{1,2}.json`

**Estimasi**:
- Phase 1: ~30 menit (QR + CR overhead)
- Phase 2: ~25 menit
- Total: ~55 menit, ~$15
''',
        'sh_qr_cr_openai',
        has_qr=True, has_cr=True,
        retrieve_cell=RETRIEVE_QRCR,
    )

    print('\nAll 3 notebooks generated.')
