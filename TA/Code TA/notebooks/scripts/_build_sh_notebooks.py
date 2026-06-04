"""Generate 2 notebooks for Schwartz-Hearst experiment.

Output:
  03_rebuild_chroma_sh.ipynb       - re-embed all chunks with SH-expanded text
  04_baseline_openai_sh.ipynb      - run full baseline OpenAI 500 samples + comparison
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
# NOTEBOOK 03 — REBUILD CHROMA SH
# ============================================================
nb3_cells = [
    md('''# 03 - Rebuild Chroma DB dengan Schwartz-Hearst Expansion

**Tujuan**: re-embed semua 1.706 chunks dengan text yang sudah di-expand pakai
algoritma Schwartz-Hearst (handle akronim dengan internal letter seperti HBO,
VEGF, EGFR, dll), lalu simpan ke ChromaDB baru.

**Improvement over naive expansion**:
- Detect 673 acronyms (vs naive 425, +58%)
- Handle "internal letter" pattern: HBO -> Hyperbaric oxygenation
- Verified fix: PMID 7482275 (HBO case) RESULTS rank 371 -> 3

**Estimasi**:
- Cost API: ~$0.005 (1706 chunks x 150 tokens avg)
- Waktu: ~5 menit (batch 100)

**Input**: `notebooks/pubmedqa_bm25_sh.pkl` (sudah dibuat oleh build_sh_index.py)
**Output**: `notebooks/pubmedqa_chroma_sh/` (folder ChromaDB baru)
'''),

    code('''import os, sys, time, pickle
from pathlib import Path
from dataclasses import dataclass

# # os.environ["OPENAI_API_KEY"] = "<REDACTED — set via shell env or .env file>"

from openai import OpenAI
import chromadb


@dataclass
class Document:
    text: str; pubid: str; question: str
    section_label: str; answer: str; decision: str

import __main__
__main__.Document = Document


HERE = Path(".").resolve()
NOTEBOOKS_DIR = HERE.parent if HERE.name == "BM25 Expansion" else HERE
BM25_SH_PATH = NOTEBOOKS_DIR / "pubmedqa_bm25_sh.pkl"
CHROMA_SH_PATH = NOTEBOOKS_DIR / "pubmedqa_chroma_sh"

print(f"BM25 SH path     : {BM25_SH_PATH} (exists: {BM25_SH_PATH.exists()})")
print(f"Chroma target    : {CHROMA_SH_PATH}")
print(f"Collection name  : pubmedqa_docs_sh")

assert BM25_SH_PATH.exists(), "BM25 SH index belum dibuat. Run: python build_sh_index.py"
'''),

    code('''# Load expanded BM25 chunks
with open(BM25_SH_PATH, "rb") as f:
    saved = pickle.load(f)
documents = saved["documents"]
print(f"Loaded {len(documents)} SH-expanded chunks")
print()
print("Sample chunk PMID 7482275 RESULTS (HBO test case):")
for d in documents:
    if d.pubid == "7482275" and d.section_label == "RESULTS":
        print(d.text[:400])
        break
'''),

    code('''api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY belum di-set.")

client = OpenAI(api_key=api_key)
EMBED_MODEL = "text-embedding-3-small"

_emb = client.embeddings.create(model=EMBED_MODEL, input=["test"])
print(f"Embedding dim: {len(_emb.data[0].embedding)} (expected 1536)")
'''),

    code('''def openai_embed(texts, model=EMBED_MODEL, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                time.sleep((attempt + 1) * 10)
            elif "500" in err or "502" in err or "503" in err:
                time.sleep((attempt + 1) * 5)
            else:
                raise
    raise RuntimeError("OpenAI embeddings gagal")


chroma_client = chromadb.PersistentClient(path=str(CHROMA_SH_PATH))
COLL_NAME = "pubmedqa_docs_sh"

try:
    chroma_client.delete_collection(name=COLL_NAME)
    print(f"Deleted existing collection")
except Exception:
    pass

collection = chroma_client.create_collection(
    name=COLL_NAME, metadata={"hnsw:space": "cosine"}
)
print(f"Created collection \\"{COLL_NAME}\\"")

BATCH = 100
t0 = time.time()
total = len(documents)

for start in range(0, total, BATCH):
    batch = documents[start:start + BATCH]
    batch_texts = [d.text[:8000] for d in batch]
    batch_ids = [str(start + i) for i in range(len(batch))]
    batch_meta = [{"pubid": d.pubid, "section": d.section_label} for d in batch]

    embeddings = openai_embed(batch_texts)
    collection.add(
        ids=batch_ids, documents=batch_texts,
        metadatas=batch_meta, embeddings=embeddings,
    )
    done = start + len(batch)
    elapsed = time.time() - t0
    eta = elapsed / done * (total - done) / 60 if done < total else 0
    print(f"  [{done}/{total}] embedded | {elapsed:.0f}s elapsed | ETA {eta:.1f} mnt")

print(f"\\nSelesai dalam {(time.time()-t0)/60:.1f} menit")
print(f"Total chunks di Chroma: {collection.count()}")
'''),

    code('''# Verifikasi: query test untuk HBO case (idx 30)
test_query = "Necrotizing fasciitis: an indication for hyperbaric oxygenation therapy?"

qvec = openai_embed([test_query])[0]
results = collection.query(
    query_embeddings=[qvec], n_results=10,
    include=["distances", "metadatas"]
)

print(f"Query: {test_query}")
print(f"\\nDense top-10 (Chroma SH):")
print(f"{'rank':>4} {'doc_id':>7} {'pubid':>10} {'section':<28} {'sim':>6}")
for rank, (did, dist, meta) in enumerate(zip(
    results["ids"][0], results["distances"][0], results["metadatas"][0]), 1):
    sim = 1 - dist
    src = " *SOURCE*" if meta["pubid"] == "7482275" else ""
    print(f"{rank:>4} {did:>7} {meta['pubid']:>10} {meta['section']:<28} {sim:.3f}{src}")
'''),

    md('''## Verifikasi sukses

Kalau **METHODS dan RESULTS dari PMID 7482275** (HBO case) muncul di top-10 Dense,
artinya re-embed berhasil dan dense retrieval sekarang juga terbantu oleh
Schwartz-Hearst expansion.

Lanjut ke notebook `04_baseline_openai_sh.ipynb` untuk full eksperimen.
'''),
]


# ============================================================
# NOTEBOOK 04 — BASELINE OPENAI SH
# ============================================================
nb4_cells = [
    md('''# 04 - Baseline OpenAI dengan Schwartz-Hearst Expansion

**Tujuan**: re-run baseline OpenAI 500 sampel dengan retriever yang menggunakan
BM25 SH + Chroma SH. Bandingkan dengan baseline naive expansion (notebook 02)
dan baseline original.

**3-way comparison**:
- Baseline original (notebook 02.1 main): 69.2%
- Naive expansion (notebook 02 BM25Exp): 72.8% (+3.6pp)
- Schwartz-Hearst expansion (this notebook): ? (target: 73-75%)

**Estimasi**:
- Phase 1 (500 calls): ~$0.50, 15 menit
- Phase 2 (500 sampel x ~14 evaluator calls): ~$10, 25 menit
- Total: ~$11, 40 menit
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
'''),

    code('''LLM_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"
TOP_K_BM25 = 50; TOP_K_DENSE = 50; TOP_K_RETRIEVAL = 5

DATASET_NAME = "qiaojin/PubMedQA"; DATASET_SUBSET = "pqa_labeled"
MAX_SAMPLES = 500
TEMPERATURE = 0.0; SEED = 42

HERE = Path(".").resolve()
NOTEBOOKS_DIR = HERE.parent if HERE.name == "BM25 Expansion" else HERE
PROJECT_ROOT = NOTEBOOKS_DIR.parent
BM25_INDEX_PATH = NOTEBOOKS_DIR / "pubmedqa_bm25_sh.pkl"
CHROMA_DB_PATH = NOTEBOOKS_DIR / "pubmedqa_chroma_sh"
RESULTS_DIR = PROJECT_ROOT / "results" / "BM25_Expansion"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_NAME = "sh_baseline_openai"
PHASE1_PATH = RESULTS_DIR / f"{CONFIG_NAME}_phase1_answers.json"
PHASE2_PATH = RESULTS_DIR / f"{CONFIG_NAME}_phase2_custom.json"

print("Konfigurasi:")
print(f"  LLM         : {LLM_MODEL}")
print(f"  BM25 index  : {BM25_INDEX_PATH.name} (Schwartz-Hearst)")
print(f"  Chroma path : {CHROMA_DB_PATH.name} (Schwartz-Hearst)")
print(f"  Sampel      : {MAX_SAMPLES}")
print(f"  Output      : {PHASE1_PATH}")

assert BM25_INDEX_PATH.exists(), f"BM25 SH tidak ditemukan. Run build_sh_index.py."
assert CHROMA_DB_PATH.exists(), f"Chroma SH tidak ditemukan. Run notebook 03."
'''),

    code('''def tokenize_bm25(text):
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
'''),

    code('''client = OpenAI(api_key=api_key)


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
'''),

    code('''def retrieve_dense(query, k=TOP_K_DENSE):
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


# Smoke test on idx 30 (HBO case)
test_q = pubmedqa_data[30]["question"]
test_r = retrieve_hybrid(test_q)
print(f"Query: {test_q}")
print(f"\\nTop-{TOP_K_RETRIEVAL} chunks (BM25 SH + Chroma SH):")
for i, r in enumerate(test_r, 1):
    src = " *SOURCE*" if r.document.pubid == "7482275" else ""
    print(f"  [{i}] RRF={r.rrf_score:.4f} | BM25={r.bm25_score:6.2f} | Dense={r.dense_score:.3f} | "
          f"({r.document.section_label[:25]}) pubid={r.document.pubid}{src}")
'''),

    code('''GENERATION_PROMPT = (
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


# Smoke test on idx 30
test_ans = generate_answer(test_q, test_r)
print(f"Generated answer for idx 30 (HBO/NF case, GT={pubmedqa_data[30]['final_decision']}):")
print(test_ans)
print(f"\\nPredicted: {extract_label(test_ans)}")
'''),

    md('''## Phase 1 — Generate jawaban 500 sampel (resumable, 15 menit)
'''),

    code('''if PHASE1_PATH.exists():
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

    md('''## Phase 1 — Quick analysis & comparison
'''),

    code('''with open(PHASE1_PATH, "r", encoding="utf-8") as f:
    results_p1 = json.load(f)["results"]

n = len(results_p1)
n_correct = sum(r["is_correct"] for r in results_p1)
print(f"Schwartz-Hearst — Baseline OpenAI ({n} sampel)")
print("=" * 60)
print(f"Label Accuracy : {n_correct}/{n} = {n_correct/n:.1%}")
print()
print("Per-label accuracy:")
for lbl in ["yes", "no", "maybe"]:
    sub = [r for r in results_p1 if r["ground_truth"] == lbl]
    if sub:
        c = sum(r["is_correct"] for r in sub)
        print(f"  {lbl:>5}: {c}/{len(sub)} = {c/len(sub):.1%}")

# 3-way comparison
print(f"\\n=== 3-WAY COMPARISON ===")
configs = {
    "Original baseline       ": PROJECT_ROOT / "results" / "baseline_openai_phase1_answers.json",
    "Naive Expansion (V1)    ": RESULTS_DIR / "bm25expanded_baseline_openai_phase1_answers.json",
    "Schwartz-Hearst (V2)    ": PHASE1_PATH,
}
for name, path in configs.items():
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)["results"]
        a = sum(r["is_correct"] for r in d) / len(d)
        print(f"  {name}: {a:.1%}")

# Check idx 30 (HBO case)
idx30 = next((r for r in results_p1 if r["idx"] == 30), None)
if idx30:
    print(f"\\n=== IDX 30 (HBO/NF case study) ===")
    print(f"  GT: {idx30['ground_truth']} | Pred: {idx30['predicted_label']} ({'CORRECT' if idx30['is_correct'] else 'WRONG'})")
    print(f"  Source paper sections retrieved:")
    for i, (pid, sec) in enumerate(zip(idx30["context_pubids"], idx30["context_sections"]), 1):
        mark = " <-- SOURCE" if pid == "7482275" else ""
        print(f"    [{i}] pubid={pid} ({sec}){mark}")
'''),

    md('''## Phase 2 — Custom evaluator 4 metrik (resumable, 25 menit)
'''),

    code('''def _split_sentences(text):
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
'''),

    code('''REQ = ["faithfulness", "context_recall", "answer_relevancy", "context_precision"]

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

    md('''## 3-Way Comparison Final
'''),

    code('''def summarize(p1_path, p2_path, label):
    if not p1_path.exists() or not p2_path.exists():
        return None
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
    metrics = {m: sum(r[m] for r in p2) / len(p2) for m in REQ}
    return {"label": label, "n": n, "acc": acc, "bal_acc": bal_acc,
            "yes_acc": accs_per_label["yes"], "no_acc": accs_per_label["no"],
            "maybe_acc": accs_per_label["maybe"], **metrics}


orig = summarize(
    PROJECT_ROOT / "results" / "baseline_openai_phase1_answers.json",
    PROJECT_ROOT / "results" / "baseline_openai_phase2_custom.json",
    "Original"
)
naive = summarize(
    RESULTS_DIR / "bm25expanded_baseline_openai_phase1_answers.json",
    RESULTS_DIR / "bm25expanded_baseline_openai_phase2_custom.json",
    "Naive Exp (V1)"
)
sh = summarize(PHASE1_PATH, PHASE2_PATH, "Schwartz-Hearst (V2)")

print("=" * 90)
print("3-WAY COMPARISON: Original vs Naive Expansion vs Schwartz-Hearst")
print("=" * 90)
print(f"{'Metric':<22} {'Original':>12} {'Naive V1':>12} {'SH V2':>12} "
      f"{'Δ V2-orig':>12} {'Δ V2-V1':>11}")
print("-" * 90)

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
    if not (orig and naive and sh):
        continue
    d_orig = (sh[key] - orig[key]) * 100
    d_v1 = (sh[key] - naive[key]) * 100
    print(f"  {name:<20} {orig[key]:>11.3f}  {naive[key]:>11.3f}  {sh[key]:>11.3f}  "
          f"{d_orig:+10.2f}  {d_v1:+10.2f}")

# Check idx 30 specifically
print(f"\\n{'=' * 90}")
print("IDX 30 (HBO/NF case - was wrong in Naive Expansion)")
print('=' * 90)
def get_idx30(p):
    if not p.exists(): return None
    with open(p, "r", encoding="utf-8") as f:
        return next((r for r in json.load(f)["results"] if r["idx"] == 30), None)

orig_30 = get_idx30(PROJECT_ROOT / "results" / "baseline_openai_phase1_answers.json")
naive_30 = get_idx30(RESULTS_DIR / "bm25expanded_baseline_openai_phase1_answers.json")
sh_30 = get_idx30(PHASE1_PATH)

if orig_30 and naive_30 and sh_30:
    print(f"  GT: {orig_30['ground_truth']}")
    print(f"  Original    pred: {orig_30['predicted_label']:5s} ({'OK' if orig_30['is_correct'] else 'WRONG'})")
    print(f"  Naive V1    pred: {naive_30['predicted_label']:5s} ({'OK' if naive_30['is_correct'] else 'WRONG'})")
    print(f"  SH V2       pred: {sh_30['predicted_label']:5s} ({'OK' if sh_30['is_correct'] else 'WRONG'})")
    if "context_pubids" in sh_30:
        print(f"\\n  SH V2 source paper sections retrieved:")
        for i, (pid, sec) in enumerate(zip(sh_30["context_pubids"], sh_30["context_sections"]), 1):
            mark = " <-- SOURCE" if pid == "7482275" else ""
            print(f"    [{i}] pubid={pid} ({sec}){mark}")
'''),

    md('''## Interpretasi hasil

**Klaim defensible untuk laporan TA**:

1. **Schwartz-Hearst > Naive Expansion** kalau Δ V2-V1 > +0.5 pp untuk overall accuracy.
   Confirms manfaat handle internal-letter acronyms.

2. **Acc no class** harusnya naik paling banyak (sama pattern dengan naive).
   Banyak negative findings ada di RESULTS yang sebelumnya tidak ter-retrieve.

3. **Idx 30 (HBO/NF)**: kalau pred berubah dari `yes` → `no` (correct), itu validasi
   bahwa Schwartz-Hearst spesifik fix kasus internal-letter yang naive miss.

4. **Maybe class**: gain modest (~3-5pp). Bottleneck masih di prompt anti-maybe bias.

**Future work**: kombinasi Schwartz-Hearst + prompt engineering (P1 strict grounding)
mungkin bisa push akurasi >75%.
'''),
]


# ============================================================
# WRITE NOTEBOOKS
# ============================================================
nb3 = nb(nb3_cells)
nb4 = nb(nb4_cells)

p3 = HERE / '03_rebuild_chroma_sh.ipynb'
p4 = HERE / '04_baseline_openai_sh.ipynb'

with open(p3, 'w', encoding='utf-8') as f:
    json.dump(nb3, f, indent=1, ensure_ascii=False)
with open(p4, 'w', encoding='utf-8') as f:
    json.dump(nb4, f, indent=1, ensure_ascii=False)

print(f'Wrote: {p3.name}  ({len(nb3_cells)} cells)')
print(f'Wrote: {p4.name}  ({len(nb4_cells)} cells)')
