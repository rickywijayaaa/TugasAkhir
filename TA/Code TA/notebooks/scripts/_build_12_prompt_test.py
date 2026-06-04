"""Generator for `12_prompt_testing_sh.ipynb` — quick prompt-style test.

Tujuan: bandingkan style jawaban antara prompt baseline (vague paraphrase)
vs Option A "Evidence-First Citation". Run 50 sample stratified
(yes/no/maybe ada semua), pakai OpenRouter sebagai LLM gateway.

Notebook ini TIDAK overwrite output baseline yang sudah selesai re-run.
Output baru ke: `results/BM25_Expansion/sh_baseline_optionA_openrouter_phase1_answers.json`

Run dengan Python 3.11:
    "C:/Users/Ricky Wijaya/AppData/Local/Programs/Python/Python311/python.exe" _build_12_prompt_test.py
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "12_prompt_testing_sh.ipynb"


def md(src):  return {"cell_type":"markdown","metadata":{},"source":src.splitlines(keepends=True)}
def code(src):return {"cell_type":"code","metadata":{},"source":src.splitlines(keepends=True),
                     "outputs":[],"execution_count":None}


HEADER = md("""# 12 — Prompt Testing: Evidence-First Citation (OpenAI Direct)

**Tujuan**: test style prompt baru ("Option A — Evidence-First Citation") di
50 sample stratified (yes/no/maybe semua ada) untuk lihat apakah LLM
benar-benar menyitir context langsung dengan angka/quote spesifik.

**Pipeline**: identik dengan `04_baseline_openai_sh.ipynb` (SH BM25 + Chroma SH
+ RRF + GPT-4.1-mini). Yang berbeda HANYA:
1. **Prompt generation**: Evidence-First Citation (paksa quote dari context)
2. **Sample size**: 50 stratified (bukan 500)

Setting LLM provider, model, embed, retrieval = **sama persis** dengan notebook
04 → delta yang muncul nanti **murni dari prompt change**, bukan dari model
yang berbeda.

**Output**: `results/BM25_Expansion/sh_baseline_optionA_openai_phase1_answers.json`
(TIDAK overwrite hasil baseline yang sudah ada)

**Cost estimate**: ~$0.05 untuk 50 calls × ~600 token = 30K token
""")


CELL_IMPORTS = code('''import os, sys, json, pickle, time, re, warnings
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

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY belum di-set di environment (.env atau shell).")

print(f"Python: {sys.version.split()[0]} | OpenAI key: ...{api_key[-8:]}")
''')


CELL_CONFIG = code('''# === Konfigurasi (sama persis dengan notebook 04 baseline) ===
LLM_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"
TOP_K_BM25 = 50; TOP_K_DENSE = 50; TOP_K_RETRIEVAL = 5

DATASET_NAME = "qiaojin/PubMedQA"; DATASET_SUBSET = "pqa_labeled"
TOTAL_DATA = 500
N_TEST_SAMPLES = 50   # subset untuk testing
TEMPERATURE = 0.0; SEED = 42

# Stratified distribution untuk 50 sample
TARGET_PER_CLASS = {"yes": 20, "no": 20, "maybe": 10}
assert sum(TARGET_PER_CLASS.values()) == N_TEST_SAMPLES

HERE = Path(".").resolve()
NOTEBOOKS_DIR = HERE.parent if HERE.name == "BM25 Expansion" else HERE
PROJECT_ROOT  = NOTEBOOKS_DIR.parent
BM25_INDEX_PATH = NOTEBOOKS_DIR / "pubmedqa_bm25_sh.pkl"
CHROMA_DB_PATH  = NOTEBOOKS_DIR / "pubmedqa_chroma_sh"
RESULTS_DIR = PROJECT_ROOT / "results" / "BM25_Expansion"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_NAME = "sh_baseline_optionA_openai"
PHASE1_PATH = RESULTS_DIR / f"{CONFIG_NAME}_phase1_answers.json"

print("Konfigurasi:")
print(f"  LLM             : {LLM_MODEL}  (OpenAI direct, sama dengan notebook 04)")
print(f"  Embed           : {EMBED_MODEL}")
print(f"  BM25 index      : {BM25_INDEX_PATH.name}")
print(f"  Chroma path     : {CHROMA_DB_PATH.name}")
print(f"  N samples       : {N_TEST_SAMPLES}  (stratified yes:{TARGET_PER_CLASS['yes']} / no:{TARGET_PER_CLASS['no']} / maybe:{TARGET_PER_CLASS['maybe']})")
print(f"  Prompt variant  : OptionA_EvidenceFirstCitation")
print(f"  Output          : {PHASE1_PATH.name}")
print(f"  Existing baseline (for comparison): sh_baseline_openai_phase1_answers.json")

assert BM25_INDEX_PATH.exists(), "BM25 SH tidak ditemukan. Run build_sh_index.py."
assert CHROMA_DB_PATH.exists(),  "Chroma SH tidak ditemukan. Run notebook 03."
''')


CELL_LOAD = code('''def tokenize_bm25(text):
    return re.sub(r"[^a-zA-Z0-9\\s]", " ", text.lower()).split()

with open(BM25_INDEX_PATH, "rb") as f:
    saved = pickle.load(f)
bm25_index = saved["bm25"]; documents = saved["documents"]
print(f"Loaded BM25 SH: {len(documents)} chunks")

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
chroma_collection = chroma_client.get_collection(name="pubmedqa_docs_sh")
print(f"Loaded Chroma SH: {chroma_collection.count()} vectors")

ds = load_dataset(DATASET_NAME, DATASET_SUBSET)["train"]
pubmedqa_data = ds.select(range(TOTAL_DATA))
print(f"Loaded {len(pubmedqa_data)} samples (pool untuk stratified sampling)")
''')


CELL_STRATIFY = code('''# === Stratified Sampling 50 sample ===
import random
random.seed(SEED)

# Group sample indices by ground truth label
idx_by_label = {"yes": [], "no": [], "maybe": []}
for i in range(TOTAL_DATA):
    label = pubmedqa_data[i]["final_decision"]
    if label in idx_by_label:
        idx_by_label[label].append(i)

print("Distribusi label di full dataset (n=500):")
for lbl, idxs in idx_by_label.items():
    print(f"  {lbl:5s}: {len(idxs)} sample")

# Sampling deterministik per kelas
selected_indices = []
for lbl, target in TARGET_PER_CLASS.items():
    pool = idx_by_label[lbl]
    if len(pool) < target:
        print(f"WARNING: kelas {lbl} cuma punya {len(pool)} sample, perlu {target}")
        chosen = pool
    else:
        chosen = sorted(random.sample(pool, target))
    selected_indices.extend(chosen)

selected_indices.sort()
print(f"\\nTerpilih {len(selected_indices)} indices: {selected_indices[:10]}...")
print(f"\\nDistribusi terpilih:")
sel_labels = [pubmedqa_data[i]["final_decision"] for i in selected_indices]
for lbl in ("yes", "no", "maybe"):
    print(f"  {lbl:5s}: {sel_labels.count(lbl)} sample")
''')


CELL_OPENAI = code('''# === OpenAI client (direct, sama dengan notebook 04) ===
client = OpenAI(api_key=api_key)


def openai_generate(prompt, max_tokens=400, temperature=TEMPERATURE):
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


# Smoke test
_test = openai_generate("Reply with just the word OK", max_tokens=5)
print(f"OpenAI ready ({LLM_MODEL}): {_test!r}")
''')


CELL_RETRIEVE = code('''# === Retrieval Hybrid (BM25 SH + Chroma SH + RRF) ===
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
    bm25_results  = retrieve_bm25_raw(query, k=TOP_K_BM25)
    dense_results = retrieve_dense(query, k=TOP_K_DENSE)
    bm25_scores   = dict(bm25_results); dense_scores = dict(dense_results)
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

print("Retriever ready (SH BM25 + Chroma SH + RRF).")
''')


CELL_PROMPT = code(r'''# === OPSI A: Evidence-First Citation Prompt ===
GENERATION_PROMPT = (
    "You are a medical research assistant. "
    "Answer a biomedical yes/no/maybe question by directly citing the provided abstracts.\n\n"
    "Context from medical literature:\n{context}\n\n"
    "Question: {question}\n\n"
    "RESPONSE FORMAT (strict):\n"
    "1. Cite 1-2 specific findings from the context using this format:\n"
    "   - Abstract [N] states: \"<exact quote or close paraphrase of key finding>\"\n"
    "   - Include specific numbers, p-values, or statistics when present.\n"
    "2. Write ONE concluding sentence that maps the cited evidence to the question.\n"
    "3. End with EXACTLY one word on its own line: yes, no, or maybe.\n\n"
    "RULES:\n"
    "- Quote MUST be traceable to the context above.\n"
    "- Do NOT add external medical knowledge.\n"
    "- Do NOT use vague phrases like \"the study suggests\" without a direct citation.\n"
    "- For label choice:\n"
    "  * yes   : cited evidence supports the claim\n"
    "  * no    : cited evidence refutes the claim or shows no significant effect\n"
    "  * maybe : evidence is directly contradictory OR not relevant to the question\n"
    "  * Prefer yes/no over maybe when evidence leans in one direction.\n\n"
    "Answer:"
)


def generate_answer(query, retrieved):
    context = "\n\n".join(
        f"[{i}] ({r.document.section_label}): {r.document.text}"
        for i, r in enumerate(retrieved, 1)
    )
    return openai_generate(
        GENERATION_PROMPT.format(context=context, question=query),
        max_tokens=400, temperature=TEMPERATURE
    )


def extract_label(answer):
    """Find yes/no/maybe label at end of answer."""
    lines = [l.strip().lower() for l in answer.split("\n") if l.strip()]
    for line in reversed(lines[-5:]):
        word = re.sub(r"[^a-z]", "", line)
        if word in ("yes", "no", "maybe"):
            return word
    for label in ("yes", "no", "maybe"):
        if re.search(r"\b" + label + r"\b", answer.lower()):
            return label
    return "maybe"


# Smoke test on idx 30
test_i = 30
test_q = pubmedqa_data[test_i]["question"]
test_r = retrieve_hybrid(test_q)
test_ans = generate_answer(test_q, test_r)
print(f"--- Smoke test idx {test_i} (GT={pubmedqa_data[test_i]['final_decision']}) ---")
print(test_ans)
print(f"\nExtracted label: {extract_label(test_ans)}")
''')


CELL_RUN = code('''# === Run generation untuk 50 stratified sample ===
if PHASE1_PATH.exists():
    with open(PHASE1_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f).get("results", [])
    done_idxs = {r["idx"] for r in existing}
    print(f"Resume: {len(done_idxs)}/{len(selected_indices)} sudah selesai.")
    results = existing
else:
    results, done_idxs = [], set()
    print(f"Memulai generation: {len(selected_indices)} sampel.")

remaining = [i for i in selected_indices if i not in done_idxs]

if remaining:
    t0 = time.time()
    for i, idx in enumerate(remaining):
        s = pubmedqa_data[idx]
        q, gt, ref = s["question"], s["final_decision"], s["long_answer"]

        retrieved = retrieve_hybrid(q)
        answer = generate_answer(q, retrieved)
        predicted = extract_label(answer)

        results.append({
            "idx": idx, "pubid": str(s["pubid"]), "question": q,
            "ground_truth": gt, "predicted_label": predicted,
            "is_correct": predicted == gt, "answer": answer,
            "contexts":         [r.document.text for r in retrieved],
            "context_pubids":   [r.document.pubid for r in retrieved],
            "context_sections": [r.document.section_label for r in retrieved],
            "reference": ref,
            "retrieval_scores": [r.bm25_score for r in retrieved],
            "dense_scores":     [r.dense_score for r in retrieved],
            "rrf_scores":       [r.rrf_score for r in retrieved],
        })

        # Save tiap 5 sample
        if (i + 1) % 5 == 0 or i == len(remaining) - 1:
            with open(PHASE1_PATH, "w", encoding="utf-8") as f:
                json.dump({"config": CONFIG_NAME,
                           "llm_model": LLM_MODEL,
                           "embed_model": EMBED_MODEL,
                           "prompt_variant": "OptionA_EvidenceFirstCitation",
                           "timestamp": datetime.now().isoformat(),
                           "n_test_samples": len(selected_indices),
                           "target_per_class": TARGET_PER_CLASS,
                           "completed": len(results),
                           "results": results}, f, indent=2, ensure_ascii=False)
            done = len(results)
            acc = sum(r["is_correct"] for r in results) / done
            eta = (time.time() - t0) / (i + 1) * (len(remaining) - i - 1) / 60
            print(f"  [{done:2d}/{len(selected_indices)}] acc={acc:.1%} | ETA {eta:.1f} mnt")

print(f"\\nSelesai -> {PHASE1_PATH}")
''')


CELL_QUICK_STATS = code('''# === Hasil ringkas ===
with open(PHASE1_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)["results"]

n = len(results)
n_correct = sum(r["is_correct"] for r in results)
print(f"Prompt Testing — Option A Evidence-First ({n} sample stratified)")
print("=" * 65)
print(f"Overall accuracy: {n_correct}/{n} = {n_correct/n:.1%}")
print()
print("Per-class accuracy:")
for lbl in ("yes", "no", "maybe"):
    sub = [r for r in results if r["ground_truth"] == lbl]
    if sub:
        c = sum(r["is_correct"] for r in sub)
        print(f"  {lbl:5s}: {c:2d}/{len(sub):2d} = {c/len(sub):.1%}")
''')


CELL_COMPARE = code('''# === Side-by-side compare: Baseline (vague) vs Option A (citation) ===
# Load baseline answers untuk sample yang sama
baseline_path = RESULTS_DIR / "sh_baseline_openai_phase1_answers.json"
baseline = json.load(open(baseline_path, encoding="utf-8"))["results"] if baseline_path.exists() else []
baseline_by_idx = {r["idx"]: r for r in baseline}

# Pilih 1 contoh per kelas untuk preview
samples_per_class = {"yes": None, "no": None, "maybe": None}
for r in results:
    if samples_per_class[r["ground_truth"]] is None:
        samples_per_class[r["ground_truth"]] = r

print("=" * 100)
print("PERBANDINGAN GAYA JAWABAN — BASELINE (vague) vs OPTION A (citation)")
print("=" * 100)
for lbl, r in samples_per_class.items():
    if r is None: continue
    base = baseline_by_idx.get(r["idx"])
    print(f"\\n{'#'*60}")
    print(f"IDX {r['idx']} | GT={r['ground_truth']} | Q: {r['question'][:90]}")
    print('#'*60)
    print()
    print(f"--- BASELINE (gpt-4.1-mini, vague paraphrase) ---")
    print(f"Pred: {base['predicted_label'] if base else '?'} | Correct: {base['is_correct'] if base else '?'}")
    if base:
        print(base["answer"])
    print()
    print(f"--- OPTION A ({LLM_MODEL}, evidence-first citation) ---")
    print(f"Pred: {r['predicted_label']} | Correct: {r['is_correct']}")
    print(r["answer"])
    print()
''')


CELL_COMPARE_ALL = code('''# === Tampilkan SEMUA 50 jawaban Option A untuk review manual ===
print("=" * 100)
print(f"SEMUA {len(results)} JAWABAN OPTION A (untuk eye-balling)")
print("=" * 100)
for r in sorted(results, key=lambda x: (x["ground_truth"], x["idx"])):
    mark = "OK" if r["is_correct"] else "XX"
    print(f"\\n--- IDX {r['idx']} | GT={r['ground_truth']} | Pred={r['predicted_label']} [{mark}] ---")
    print(f"Q: {r['question']}")
    print(f"A:\\n{r['answer']}")
''')


def build_notebook():
    cells = [
        HEADER,
        CELL_IMPORTS,
        CELL_CONFIG,
        CELL_LOAD,
        CELL_STRATIFY,
        CELL_OPENAI,
        CELL_RETRIEVE,
        CELL_PROMPT,
        md("## Run Generation untuk 50 Sample (resumable, ~5-8 menit)"),
        CELL_RUN,
        md("## Hasil Ringkas"),
        CELL_QUICK_STATS,
        md("## Perbandingan Gaya Jawaban: Baseline vs Option A"),
        CELL_COMPARE,
        md("## Review Semua 50 Jawaban (eye-balling)"),
        CELL_COMPARE_ALL,
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {OUT.name}  ({len(cells)} cells)")


if __name__ == "__main__":
    build_notebook()
