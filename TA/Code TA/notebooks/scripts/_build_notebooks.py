"""Generate 2 notebooks for BM25 Expansion full experiment:
  01_rebuild_chroma_expanded.ipynb - re-embed all chunks with expanded text
  02_baseline_openai_expanded.ipynb - run full baseline OpenAI 500 samples + comparison

Usage: python _build_notebooks.py
"""
import json, uuid
from pathlib import Path

HERE = Path(__file__).parent


def md(src):
    return {'cell_type':'markdown', 'id':str(uuid.uuid4())[:8],
            'metadata':{}, 'source': src.splitlines(keepends=True)}


def code(src):
    return {'cell_type':'code', 'id':str(uuid.uuid4())[:8],
            'metadata':{}, 'execution_count':None, 'outputs':[],
            'source': src.splitlines(keepends=True)}


def nb(cells, title=''):
    return {
        'cells': cells,
        'metadata': {
            'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python','version':'3.11'},
        },
        'nbformat': 4, 'nbformat_minor': 5,
    }


# ============================================================
# NOTEBOOK 1: REBUILD CHROMA EXPANDED
# ============================================================
nb1_cells = [
    md('''# 01 - Rebuild Chroma DB dengan Acronym-Expanded Text

**Tujuan**: re-embed semua 1.706 chunks dengan text yang sudah di-expand acronym-nya,
lalu simpan ke ChromaDB baru di `notebooks/pubmedqa_chroma_expanded`.

**Estimasi**:
- Cost API: ~$0.005 (1706 chunks x 150 tokens avg x $0.02/1M)
- Waktu: ~5 menit (batch 100)

**Input**: `notebooks/pubmedqa_bm25_expanded.pkl` (sudah dibuat oleh build_expanded_index.py)

**Output**: `notebooks/pubmedqa_chroma_expanded/` (folder ChromaDB baru)
'''),

    code('''import os, sys, time, pickle
from pathlib import Path
from dataclasses import dataclass

# Set OpenAI API key (Anda bisa juga set via env var)
# # os.environ["OPENAI_API_KEY"] = "<REDACTED — set via shell env or .env file>"

from openai import OpenAI
import chromadb

# ============================================================
# Document class (compat dengan pickle dari main notebook)
# ============================================================
@dataclass
class Document:
    text: str; pubid: str; question: str
    section_label: str; answer: str; decision: str

import __main__
__main__.Document = Document

# ============================================================
# Paths
# ============================================================
HERE = Path(".").resolve()
NOTEBOOKS_DIR = HERE.parent if HERE.name == "BM25 Expansion" else HERE
BM25_EXPANDED_PATH = NOTEBOOKS_DIR / "pubmedqa_bm25_expanded.pkl"
CHROMA_EXPANDED_PATH = NOTEBOOKS_DIR / "pubmedqa_chroma_expanded"

print(f"BM25 expanded path : {BM25_EXPANDED_PATH} (exists: {BM25_EXPANDED_PATH.exists()})")
print(f"Chroma target path : {CHROMA_EXPANDED_PATH}")
print(f"Will create new collection \\"pubmedqa_docs_expanded\\"")
'''),

    code('''# Load expanded BM25 to get all 1706 expanded chunks
with open(BM25_EXPANDED_PATH, "rb") as f:
    saved = pickle.load(f)
documents = saved["documents"]
print(f"Loaded {len(documents)} expanded chunks")
print()
print("Sample expanded chunk (paper 11729377, METHODS section):")
for d in documents:
    if d.pubid == "11729377" and d.section_label == "METHODS":
        print(d.text[:300])
        break
'''),

    code('''# Setup OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY belum di-set. Set via env var atau os.environ.")

client = OpenAI(api_key=api_key)
EMBED_MODEL = "text-embedding-3-small"  # 1536 dim

# Smoke test
_emb = client.embeddings.create(model=EMBED_MODEL, input=["aspirin reduces heart attack risk"])
print(f"Embedding dim: {len(_emb.data[0].embedding)}  (expected 1536)")
'''),

    code('''# ============================================================
# Build Chroma collection dengan embedding expanded chunks
# ============================================================
def openai_embed(texts, model=EMBED_MODEL, max_retries=5):
    """Wrapper dengan retry."""
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = (attempt + 1) * 10
                print(f"  [Rate limit] Tunggu {wait}s...")
                time.sleep(wait)
            elif "500" in err or "502" in err or "503" in err:
                time.sleep((attempt + 1) * 5)
            else:
                raise
    raise RuntimeError("OpenAI embeddings gagal setelah 5 retry.")


# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path=str(CHROMA_EXPANDED_PATH))
COLL_NAME = "pubmedqa_docs_expanded"

# Hapus kalau sudah ada (rebuild bersih)
try:
    chroma_client.delete_collection(name=COLL_NAME)
    print(f"Deleted existing collection \\"{COLL_NAME}\\"")
except Exception:
    pass

collection = chroma_client.create_collection(
    name=COLL_NAME,
    metadata={"hnsw:space": "cosine"}
)
print(f"Created new collection \\"{COLL_NAME}\\"")

# Batch embed
BATCH = 100
t0 = time.time()
total = len(documents)

for start in range(0, total, BATCH):
    batch_docs = documents[start:start + BATCH]
    batch_texts = [d.text[:8000] for d in batch_docs]  # truncate untuk safety
    batch_ids = [str(start + i) for i in range(len(batch_docs))]
    batch_meta = [{"pubid": d.pubid, "section": d.section_label} for d in batch_docs]

    embeddings = openai_embed(batch_texts)
    collection.add(
        ids=batch_ids,
        documents=batch_texts,
        metadatas=batch_meta,
        embeddings=embeddings,
    )
    done = start + len(batch_docs)
    elapsed = time.time() - t0
    eta = elapsed / done * (total - done) / 60 if done < total else 0
    print(f"  [{done}/{total}] embedded | {elapsed:.0f}s elapsed | ETA {eta:.1f} mnt")

print(f"\\nSelesai dalam {(time.time()-t0)/60:.1f} menit")
print(f"Total chunks di Chroma: {collection.count()}")
'''),

    code('''# ============================================================
# Verifikasi: query test untuk paper LRT/SLT (idx 16)
# ============================================================
test_query = "Is there still a need for living-related liver transplantation in children?"

qvec = openai_embed([test_query])[0]
results = collection.query(
    query_embeddings=[qvec],
    n_results=10,
    include=["distances", "metadatas"]
)

print(f"Query: {test_query}")
print(f"\\nDense top-10 (Chroma EXPANDED):")
print(f"{'rank':>4} {'doc_id':>7} {'pubid':>10} {'section':<28} {'sim':>6}")
for rank, (did, dist, meta) in enumerate(zip(results["ids"][0], results["distances"][0], results["metadatas"][0]), 1):
    sim = 1 - dist
    src = "*SOURCE*" if meta["pubid"] == "11729377" else ""
    print(f"{rank:>4} {did:>7} {meta['pubid']:>10} {meta['section']:<28} {sim:.3f} {src}")
'''),

    md('''## Verifikasi sukses

Kalau output di atas menunjukkan **METHODS dan RESULTS dari paper sumber (11729377)**
masuk top-10, artinya re-embed berhasil dan dense retrieval sekarang juga terbantu
oleh acronym expansion.

**Lanjut ke notebook 02** untuk run full baseline OpenAI dengan kedua index expanded.
'''),
]


# ============================================================
# NOTEBOOK 2: BASELINE OPENAI EXPANDED
# ============================================================
nb2_cells = [
    md('''# 02 - Baseline OpenAI dengan BM25 + Chroma Expanded

**Tujuan**: re-run baseline OpenAI 500 sampel dengan retriever yang menggunakan
BM25 expanded + Chroma expanded. Bandingkan dengan baseline original.

**Setup identik dengan main notebook 02.1 Baseline - OpenAI.ipynb** kecuali:
- BM25 index: `pubmedqa_bm25_expanded.pkl` (acronym expanded)
- Chroma DB: `pubmedqa_chroma_expanded` (re-embedded dengan expanded text)
- Output: `results/bm25expanded_baseline_openai_phase{1,2}.json`

**Estimasi**:
- Phase 1 (500 calls): ~$0.50, ~15 menit
- Phase 2 (500 sampel x ~14 evaluator calls): ~$10, ~25 menit
- Total: ~$11, ~40 menit
'''),

    code('''import os, sys, json, pickle, time, re, warnings
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

# Document class for pickle compat
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

# Set OpenAI API key
# # os.environ["OPENAI_API_KEY"] = "<REDACTED — set via shell env or .env file>"
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY belum di-set.")

print(f"Python: {sys.version.split()[0]} | NumPy: {np.__version__} | chromadb: {chromadb.__version__}")
'''),

    code('''# ============================================================
# Konfigurasi
# ============================================================
LLM_MODEL   = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"

TOP_K_BM25 = 50; TOP_K_DENSE = 50; TOP_K_RETRIEVAL = 5

DATASET_NAME = "qiaojin/PubMedQA"; DATASET_SUBSET = "pqa_labeled"
MAX_SAMPLES = 500
TEMPERATURE = 0.0; SEED = 42

# Paths - point to EXPANDED indexes
HERE = Path(".").resolve()
NOTEBOOKS_DIR = HERE.parent if HERE.name == "BM25 Expansion" else HERE
PROJECT_ROOT = NOTEBOOKS_DIR.parent
BM25_INDEX_PATH = NOTEBOOKS_DIR / "pubmedqa_bm25_expanded.pkl"
CHROMA_DB_PATH = NOTEBOOKS_DIR / "pubmedqa_chroma_expanded"
RESULTS_DIR = PROJECT_ROOT / "results" / "BM25_Expansion"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_NAME = "bm25expanded_baseline_openai"
PHASE1_PATH = RESULTS_DIR / f"{CONFIG_NAME}_phase1_answers.json"
PHASE2_PATH = RESULTS_DIR / f"{CONFIG_NAME}_phase2_custom.json"

print("Konfigurasi:")
print(f"  LLM            : {LLM_MODEL}")
print(f"  Embedder       : {EMBED_MODEL}")
print(f"  BM25 index     : {BM25_INDEX_PATH.name} (EXPANDED)")
print(f"  Chroma path    : {CHROMA_DB_PATH.name} (EXPANDED)")
print(f"  Sampel         : {MAX_SAMPLES}")
print(f"  Output         : {PHASE1_PATH}")
print(f"  Resume support : YES (jika file output sudah ada)")

assert BM25_INDEX_PATH.exists(), f"BM25 index tidak ditemukan. Run build_expanded_index.py dulu."
assert CHROMA_DB_PATH.exists(), f"Chroma expanded tidak ditemukan. Run notebook 01 dulu."
'''),

    code('''# ============================================================
# Load BM25 expanded + Chroma expanded
# ============================================================
def tokenize_bm25(text):
    return re.sub(r"[^a-zA-Z0-9\\s]", " ", text.lower()).split()

with open(BM25_INDEX_PATH, "rb") as f:
    saved = pickle.load(f)
bm25_index = saved["bm25"]
documents = saved["documents"]
print(f"Loaded BM25 expanded: {len(documents)} chunks")

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
chroma_collection = chroma_client.get_collection(name="pubmedqa_docs_expanded")
print(f"Loaded Chroma expanded: {chroma_collection.count()} vectors")

# Load PubMedQA samples
ds = load_dataset(DATASET_NAME, DATASET_SUBSET)["train"]
pubmedqa_data = ds.select(range(MAX_SAMPLES))
print(f"Loaded {len(pubmedqa_data)} samples (first {MAX_SAMPLES})")
'''),

    code('''# ============================================================
# OpenAI client + helper functions
# ============================================================
client = OpenAI(api_key=api_key)

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
    raise RuntimeError("OpenAI gagal setelah 5 retry")


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


# Smoke test
_test = openai_generate("Reply with exactly: OK", max_tokens=5)
print(f"OpenAI test: {_test!r}")
'''),

    code('''# ============================================================
# Hybrid retrieval (BM25 expanded + Dense expanded + RRF)
# ============================================================
def retrieve_dense(query, k=TOP_K_DENSE):
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


def retrieve_hybrid(query, k_final=TOP_K_RETRIEVAL):
    bm25_results = retrieve_bm25_raw(query, k=TOP_K_BM25)
    dense_results = retrieve_dense(query, k=TOP_K_DENSE)
    bm25_scores = dict(bm25_results); dense_scores = dict(dense_results)
    fused = reciprocal_rank_fusion([
        [d for d, _ in bm25_results], [d for d, _ in dense_results]
    ])
    out = []
    for doc_id, rrf in fused[:k_final]:
        out.append(RetrievalResult(
            document=documents[doc_id], score=rrf, doc_id=doc_id,
            bm25_score=bm25_scores.get(doc_id, 0.0),
            dense_score=dense_scores.get(doc_id, 0.0), rrf_score=rrf,
        ))
    return out


# Smoke test on idx 16
test_q = pubmedqa_data[16]["question"]
test_r = retrieve_hybrid(test_q)
print(f"Query: {test_q}")
print(f"\\nTop-{TOP_K_RETRIEVAL} chunks (BM25 expanded + Chroma expanded):")
for i, r in enumerate(test_r, 1):
    src = " *SOURCE*" if r.document.pubid == "11729377" else ""
    print(f"  [{i}] RRF={r.rrf_score:.4f} | BM25={r.bm25_score:6.2f} | Dense={r.dense_score:.3f} | "
          f"({r.document.section_label[:25]}) pubid={r.document.pubid}{src}")
'''),

    code('''# ============================================================
# Generation prompt + answer (SAMA persis dengan baseline original)
# ============================================================
GENERATION_PROMPT = (
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


# Test on idx 16 (case study)
test_ans = generate_answer(test_q, test_r)
print("Generated answer for idx 16 (LRT/SLT case):")
print(test_ans)
print(f"\\nPredicted: {extract_label(test_ans)} (GT: {pubmedqa_data[16]['final_decision']})")
'''),

    md('''## Phase 1 — Generate jawaban 500 sampel (resumable)

Estimasi 15 menit.
'''),

    code('''# Resume support
if PHASE1_PATH.exists():
    with open(PHASE1_PATH, "r", encoding="utf-8") as f:
        results_p1 = json.load(f)["results"]
    start_from = len(results_p1)
    print(f"Resume Fase 1: {start_from}/{MAX_SAMPLES} sudah selesai.")
else:
    results_p1, start_from = [], 0
    print(f"Memulai Fase 1: {MAX_SAMPLES} sampel.")

if start_from < MAX_SAMPLES:
    t0 = time.time()
    for i in range(start_from, MAX_SAMPLES):
        s = pubmedqa_data[i]
        q, gt, ref = s["question"], s["final_decision"], s["long_answer"]

        retrieved = retrieve_hybrid(q)
        answer = generate_answer(q, retrieved)
        predicted = extract_label(answer)

        results_p1.append({
            "idx": i, "pubid": str(s["pubid"]), "question": q,
            "ground_truth": gt, "predicted_label": predicted,
            "is_correct": predicted == gt, "answer": answer,
            "contexts": [r.document.text for r in retrieved],
            "context_pubids": [r.document.pubid for r in retrieved],
            "context_sections": [r.document.section_label for r in retrieved],
            "reference": ref,
            "retrieval_scores": [r.bm25_score for r in retrieved],
            "dense_scores": [r.dense_score for r in retrieved],
            "rrf_scores": [r.rrf_score for r in retrieved],
        })

        if (i + 1) % 10 == 0 or i == MAX_SAMPLES - 1:
            with open(PHASE1_PATH, "w", encoding="utf-8") as f:
                json.dump({"config": CONFIG_NAME, "llm_model": LLM_MODEL,
                           "embed_model": EMBED_MODEL,
                           "timestamp": datetime.now().isoformat(),
                           "max_samples": MAX_SAMPLES, "completed": i+1,
                           "results": results_p1}, f, indent=2, ensure_ascii=False)
            done = i + 1
            acc = sum(r["is_correct"] for r in results_p1) / done
            eta = (time.time() - t0) / (i + 1 - start_from) * (MAX_SAMPLES - i - 1) / 60
            print(f"  [{done:3d}/{MAX_SAMPLES}] acc={acc:.1%} | ETA {eta:.1f} mnt")

print(f"\\nFase 1 selesai -> {PHASE1_PATH}")
'''),

    md('''## Phase 1 — Analisis cepat
'''),

    code('''with open(PHASE1_PATH, "r", encoding="utf-8") as f:
    results_p1 = json.load(f)["results"]

n = len(results_p1)
n_correct = sum(r["is_correct"] for r in results_p1)
print(f"BM25 Expanded — Baseline OpenAI ({n} sampel)")
print("=" * 60)
print(f"Label Accuracy : {n_correct}/{n} = {n_correct/n:.1%}")
print()
print("Per-label accuracy:")
for lbl in ["yes", "no", "maybe"]:
    sub = [r for r in results_p1 if r["ground_truth"] == lbl]
    if sub:
        c = sum(r["is_correct"] for r in sub)
        print(f"  {lbl:>5}: {c}/{len(sub)} = {c/len(sub):.1%}")

# Quick comparison with original baseline
orig_path = PROJECT_ROOT / "results" / "baseline_openai_phase1_answers.json"
if orig_path.exists():
    with open(orig_path, "r", encoding="utf-8") as f:
        orig = json.load(f)["results"]
    orig_acc = sum(r["is_correct"] for r in orig) / len(orig)
    delta = (n_correct/n - orig_acc) * 100
    print(f"\\n--- COMPARISON vs baseline original ---")
    print(f"  Original BM25 baseline : {orig_acc:.1%}")
    print(f"  Expanded BM25 baseline : {n_correct/n:.1%}")
    print(f"  Δ (delta)              : {delta:+.1f} pp")

# Check idx 16 specifically
idx16 = next((r for r in results_p1 if r["idx"] == 16), None)
if idx16:
    print(f"\\n--- IDX 16 (LRT/SLT case study) ---")
    print(f"  GT: {idx16['ground_truth']} | Pred: {idx16['predicted_label']}")
    print(f"  Source paper sections retrieved:")
    for i, (pubid, section) in enumerate(zip(idx16["context_pubids"], idx16["context_sections"]), 1):
        marker = " <-- SOURCE" if pubid == "11729377" else ""
        print(f"    [{i}] pubid={pubid} ({section}){marker}")
'''),

    md('''## Phase 2 — Custom evaluator 4 metrik (resumable)

Estimasi 25 menit.
'''),

    code('''# Custom 4-metric evaluator (sama persis dengan main notebook)
def _split_sentences(text):
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
    total_rel = sum(relevance)
    if total_rel == 0: return 0.0
    prec_sum = 0.0; rel_count = 0
    for k, rel in enumerate(relevance):
        if rel:
            rel_count += 1; prec_sum += rel_count / (k + 1)
    return prec_sum / total_rel


def evaluate_custom(q, ans, ctx, ref):
    return {
        "faithfulness": compute_faithfulness(ans, ctx),
        "context_recall": compute_context_recall(ref, ctx),
        "answer_relevancy": compute_answer_relevancy(q, ans),
        "context_precision": compute_context_precision(q, ctx, ref),
    }


# Smoke test
_r = evaluate_custom(
    "Does aspirin prevent heart attacks?",
    "Aspirin helps prevent heart attacks. It reduces clotting.",
    ["Aspirin reduces blood clotting and is used for heart attack prevention."],
    "Aspirin is used for heart attack prevention by reducing blood clotting.",
)
print("Smoke test 4 metrics:", {k: f"{v:.3f}" for k, v in _r.items()})
'''),

    code('''# ============================================================
# Phase 2 evaluator loop (resumable)
# ============================================================
REQ = ["faithfulness", "context_recall", "answer_relevancy", "context_precision"]

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
'''),

    md('''## Comparison Final: BM25 Expanded vs Original
'''),

    code('''# ============================================================
# Comparison final
# ============================================================
def summarize(p1_path, p2_path, label):
    with open(p1_path, "r", encoding="utf-8") as f:
        p1 = json.load(f)["results"]
    with open(p2_path, "r", encoding="utf-8") as f:
        p2 = json.load(f)["results"]
    n = len(p1)
    acc = sum(r["is_correct"] for r in p1) / n
    accs_per_label = {}
    for lbl in ["yes", "no", "maybe"]:
        sub = [r for r in p1 if r["ground_truth"] == lbl]
        accs_per_label[lbl] = sum(r["is_correct"] for r in sub) / len(sub) if sub else 0
    bal_acc = sum(accs_per_label.values()) / 3
    metrics = {m: sum(r[m] for r in p2) / len(p2) for m in
               ["faithfulness", "context_recall", "answer_relevancy", "context_precision"]}
    return {"label": label, "n": n, "acc": acc, "bal_acc": bal_acc,
            "yes_acc": accs_per_label["yes"], "no_acc": accs_per_label["no"],
            "maybe_acc": accs_per_label["maybe"], **metrics}


orig = summarize(
    PROJECT_ROOT / "results" / "baseline_openai_phase1_answers.json",
    PROJECT_ROOT / "results" / "baseline_openai_phase2_custom.json",
    "Original BM25"
)
expanded = summarize(PHASE1_PATH, PHASE2_PATH, "Expanded BM25 + Chroma")

print("=" * 70)
print("COMPARISON: BM25 Original vs BM25+Chroma Expanded")
print("=" * 70)
print(f"{'Metric':<22} {'Original':>12} {'Expanded':>12} {'Δ (pp)':>10}")
print("-" * 70)

for key, name in [
    ("acc", "Overall accuracy"),
    ("bal_acc", "Balanced accuracy"),
    ("yes_acc", "Acc yes class"),
    ("no_acc", "Acc no class"),
    ("maybe_acc", "Acc maybe class"),
    ("faithfulness", "Faithfulness"),
    ("context_recall", "Context recall"),
    ("answer_relevancy", "Answer relevancy"),
    ("context_precision", "Context precision"),
]:
    delta = (expanded[key] - orig[key]) * 100
    arrow = " ↑" if delta > 0.5 else (" ↓" if delta < -0.5 else " ·")
    print(f"  {name:<20} {orig[key]:>11.3f}  {expanded[key]:>11.3f}  {delta:+8.2f}{arrow}")

# Check idx 16 specifically
idx16_orig = next((r for r in json.load(open(PROJECT_ROOT/"results"/"baseline_openai_phase1_answers.json"))["results"] if r["idx"] == 16), None)
idx16_exp = next((r for r in json.load(open(PHASE1_PATH))["results"] if r["idx"] == 16), None)
if idx16_orig and idx16_exp:
    print()
    print("=" * 70)
    print("IDX 16 (LRT/SLT case study) BEFORE vs AFTER expansion")
    print("=" * 70)
    print(f"  GT: {idx16_orig['ground_truth']}")
    print(f"  Original   pred: {idx16_orig['predicted_label']} (correct: {idx16_orig['is_correct']})")
    print(f"  Expanded   pred: {idx16_exp['predicted_label']} (correct: {idx16_exp['is_correct']})")
    print(f"\\n  Source paper sections retrieved (Original):")
    if "context_pubids" in idx16_orig:
        for i, (pid, sec) in enumerate(zip(idx16_orig["context_pubids"], idx16_orig.get("context_sections", [])), 1):
            mark = " <-- SOURCE" if pid == "11729377" else ""
            print(f"    [{i}] pubid={pid}{mark}")
    print(f"  Source paper sections retrieved (Expanded):")
    for i, (pid, sec) in enumerate(zip(idx16_exp["context_pubids"], idx16_exp["context_sections"]), 1):
        mark = " <-- SOURCE" if pid == "11729377" else ""
        print(f"    [{i}] pubid={pid} ({sec}){mark}")
'''),

    md('''## Hasil interpretasi

Lihat tabel comparison di atas. Patokan untuk klaim laporan TA:

- **Δ overall accuracy** > +2pp = signifikan (worth claiming)
- **Δ overall accuracy** -1 sampai +1 pp = neutral (acronym tidak jadi bottleneck utama)
- **Δ overall accuracy** < -1 pp = noise penalty melebihi gain (perlu investigasi false positive)

**Per-class analysis penting**: kalau gain overall kecil tapi maybe accuracy naik banyak,
itu juga klaim valid karena maybe paling underrepresented di prediksi.

**Sample idx 16**: kalau pred berubah dari `maybe` ke `yes` (correct), itu validasi
bahwa acronym expansion betul-betul mempengaruhi inference LLM untuk kasus konkrit
yang kita drill-down.
'''),
]


# ============================================================
# WRITE NOTEBOOKS
# ============================================================
nb1 = nb(nb1_cells)
nb2 = nb(nb2_cells)

p1 = HERE / '01_rebuild_chroma_expanded.ipynb'
p2 = HERE / '02_baseline_openai_expanded.ipynb'

with open(p1, 'w', encoding='utf-8') as f:
    json.dump(nb1, f, indent=1, ensure_ascii=False)
with open(p2, 'w', encoding='utf-8') as f:
    json.dump(nb2, f, indent=1, ensure_ascii=False)

print(f'Wrote: {p1.name}  ({len(nb1_cells)} cells)')
print(f'Wrote: {p2.name}  ({len(nb2_cells)} cells)')
