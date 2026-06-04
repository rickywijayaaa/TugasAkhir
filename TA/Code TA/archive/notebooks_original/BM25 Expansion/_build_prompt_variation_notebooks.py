"""Generator for prompt-variation notebooks on top of SH BM25 + Chroma SH.

Produces 2 notebooks:
- 09_p1_strict_grounding_sh.ipynb  -> sh_p1_strict_openai_*
- 10_p3_fewshot_cot_sh.ipynb       -> sh_p3_fewshot_openai_*

Both use the EXACT same hybrid retrieval pipeline as 04_baseline_openai_sh.ipynb.
Only the GENERATION_PROMPT (and `max_tokens` for the fewshot variant) differ.

Run with the project's Python 3.11:
    "C:/Users/Ricky Wijaya/AppData/Local/Programs/Python/Python311/python.exe" \\
        _build_prompt_variation_notebooks.py
"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent

# ---------- Prompts ----------
P1_STRICT_GROUNDING = (
    'You are a medical research assistant evaluating biomedical evidence.\n\n'
    'GROUNDING RULES (strict):\n'
    '1. Use ONLY the information in the provided abstracts below. Do NOT use any '
    'external medical knowledge, common sense, or assumptions.\n'
    '2. If the abstracts do not contain enough information to answer, you MUST say so.\n'
    '3. Do not infer beyond what is explicitly stated.\n\n'
    'Context from medical literature:\n{context}\n\n'
    'Question: {question}\n\n'
    'Decision rubric (apply in order):\n'
    '- Answer "yes"   : the abstracts clearly and consistently support the claim in the question.\n'
    '- Answer "no"    : the abstracts clearly refute the claim, or report no significant effect.\n'
    '- Answer "maybe" : (a) findings are mixed or context-dependent (e.g. effect appears '
    'in subgroup A but not subgroup B), OR (b) the abstracts are inconclusive, OR '
    '(c) the abstracts do not directly address the question.\n\n'
    'Response format:\n'
    '- One short paragraph (2-3 sentences) citing the specific evidence from the abstracts.\n'
    '- Then a final line containing only one word: yes, no, or maybe.\n\n'
    'Answer:'
)

P3_FEWSHOT_COT = (
    'You are a medical research assistant. '
    'Answer biomedical yes/no/maybe questions using only the provided abstracts.\n\n'
    'Here are three examples of how to reason and answer:\n\n'
    '--- Example 1 ---\n'
    'Context: [1] (RESULTS) Daily aspirin use was associated with a 22% reduction in '
    'recurrent myocardial infarction (RR 0.78, 95% CI 0.71-0.85, p<0.001) in patients '
    'with prior MI over a 5-year follow-up.\n'
    'Question: Does aspirin reduce the risk of recurrent myocardial infarction?\n'
    'Reasoning: The abstract directly reports a statistically significant 22% reduction '
    'in recurrent MI with aspirin. The effect is consistent and clinically meaningful.\n'
    'Final answer:\n'
    'yes\n\n'
    '--- Example 2 ---\n'
    'Context: [1] (RESULTS) Vitamin C supplementation (1000 mg/day) did not significantly '
    'reduce the duration or severity of common cold symptoms compared to placebo (p=0.42).\n'
    'Question: Does vitamin C reduce the duration of common cold symptoms?\n'
    'Reasoning: The abstract explicitly reports no significant effect of vitamin C on '
    'cold duration. The p-value (0.42) indicates no statistically significant difference.\n'
    'Final answer:\n'
    'no\n\n'
    '--- Example 3 ---\n'
    'Context: [1] (RESULTS) In our cohort, statin therapy reduced cardiovascular events '
    'in patients aged 50-70 (HR 0.72) but showed no significant benefit in patients '
    'over 75 (HR 0.96, 95% CI 0.81-1.14).\n'
    'Question: Do statins reduce cardiovascular events in elderly patients?\n'
    'Reasoning: The evidence is subgroup-dependent. Statins help patients aged 50-70, '
    'but not those over 75. Since "elderly" can include both groups, the evidence is mixed.\n'
    'Final answer:\n'
    'maybe\n\n'
    '--- End of examples ---\n\n'
    'Now answer the following question using the same format:\n\n'
    'Context from medical literature:\n{context}\n\n'
    'Question: {question}\n\n'
    'Reasoning:'
)


# ---------- Notebook cells (shared) ----------
def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src.splitlines(keepends=True),
            "outputs": [], "execution_count": None}


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
    raise RuntimeError("OPENAI_API_KEY belum di-set di environment.")

print(f"Python: {sys.version.split()[0]} | NumPy: {np.__version__} | chromadb: {chromadb.__version__}")
''')


def cell_config(config_name, prompt_label):
    return code(f'''LLM_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"
TOP_K_BM25 = 50; TOP_K_DENSE = 50; TOP_K_RETRIEVAL = 5

DATASET_NAME = "qiaojin/PubMedQA"; DATASET_SUBSET = "pqa_labeled"
MAX_SAMPLES = 500
TEMPERATURE = 0.0; SEED = 42

HERE = Path(".").resolve()
NOTEBOOKS_DIR = HERE.parent if HERE.name == "BM25 Expansion" else HERE
PROJECT_ROOT = NOTEBOOKS_DIR.parent
BM25_INDEX_PATH = NOTEBOOKS_DIR / "pubmedqa_bm25_sh.pkl"
CHROMA_DB_PATH  = NOTEBOOKS_DIR / "pubmedqa_chroma_sh"
RESULTS_DIR = PROJECT_ROOT / "results" / "BM25_Expansion"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_NAME = "{config_name}"
PROMPT_LABEL = "{prompt_label}"
PHASE1_PATH = RESULTS_DIR / f"{{CONFIG_NAME}}_phase1_answers.json"
PHASE2_PATH = RESULTS_DIR / f"{{CONFIG_NAME}}_phase2_custom.json"

print("Konfigurasi:")
print(f"  LLM         : {{LLM_MODEL}}")
print(f"  Prompt      : {{PROMPT_LABEL}}")
print(f"  BM25 index  : {{BM25_INDEX_PATH.name}} (Schwartz-Hearst)")
print(f"  Chroma path : {{CHROMA_DB_PATH.name}} (Schwartz-Hearst)")
print(f"  Sampel      : {{MAX_SAMPLES}}")
print(f"  Output      : {{PHASE1_PATH.name}}, {{PHASE2_PATH.name}}")

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
pubmedqa_data = ds.select(range(MAX_SAMPLES))
print(f"Loaded {len(pubmedqa_data)} samples")
''')


CELL_OPENAI_CLIENT = code('''client = OpenAI(api_key=api_key)


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
''')


CELL_RETRIEVE = code('''def retrieve_dense(query, k=TOP_K_DENSE):
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

print("Retriever ready (BM25 SH + Chroma SH + RRF).")
''')


def cell_prompt(prompt_var, prompt_text, max_tokens):
    """Cell 6: GENERATION_PROMPT + generate_answer + extract_label + smoke test."""
    # Embed prompt text safely as a Python triple-quoted string
    safe = prompt_text.replace('"""', '\\"\\"\\"')
    src = f'''GENERATION_PROMPT = """{safe}"""

GEN_MAX_TOKENS = {max_tokens}


def generate_answer(query, retrieved):
    context = "\\n\\n".join(
        f"[{{i}}] ({{r.document.section_label}}): {{r.document.text}}"
        for i, r in enumerate(retrieved, 1)
    )
    return openai_generate(
        GENERATION_PROMPT.format(context=context, question=query),
        max_tokens=GEN_MAX_TOKENS, temperature=TEMPERATURE
    )


def extract_label(answer):
    """Extract yes/no/maybe label from the LLM answer.

    Strategy:
      1. Look at the last few lines for a single word answer (yes/no/maybe).
      2. Fallback to first whole-word match in the entire response.
      3. Default to "maybe" if no signal at all.
    """
    lines = [l.strip().lower() for l in answer.split("\\n") if l.strip()]
    for line in reversed(lines[-5:]):
        word = re.sub(r"[^a-z]", "", line)
        if word in ("yes", "no", "maybe"):
            return word
    for label in ("yes", "no", "maybe"):
        if re.search(r"\\b" + label + r"\\b", answer.lower()):
            return label
    return "maybe"


# Smoke test on idx 30
test_q = pubmedqa_data[30]["question"]
test_r = retrieve_hybrid(test_q)
test_ans = generate_answer(test_q, test_r)
print(f"--- Smoke test idx 30 (GT={{pubmedqa_data[30]['final_decision']}}) ---")
print(test_ans)
print(f"\\nExtracted label: {{extract_label(test_ans)}}")
'''
    return code(src)


CELL_PHASE1 = code('''if PHASE1_PATH.exists():
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
        answer    = generate_answer(q, retrieved)
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
            "dense_scores":     [r.dense_score for r in retrieved],
            "rrf_scores":       [r.rrf_score for r in retrieved],
        })

        if (i + 1) % 10 == 0 or i == MAX_SAMPLES - 1:
            with open(PHASE1_PATH, "w", encoding="utf-8") as f:
                json.dump({"config": CONFIG_NAME, "prompt_label": PROMPT_LABEL,
                           "llm_model": LLM_MODEL, "embed_model": EMBED_MODEL,
                           "timestamp": datetime.now().isoformat(),
                           "max_samples": MAX_SAMPLES, "completed": i+1,
                           "results": results_p1}, f, indent=2, ensure_ascii=False)
            done = i + 1
            acc = sum(r["is_correct"] for r in results_p1) / done
            eta = (time.time() - t0) / (i + 1 - start_from) * (MAX_SAMPLES - i - 1) / 60
            print(f"  [{done:3d}/{MAX_SAMPLES}] acc={acc:.1%} | ETA {eta:.1f} mnt")

print(f"\\nFase 1 selesai -> {PHASE1_PATH}")
''')


CELL_PHASE1_ANALYSIS = code('''with open(PHASE1_PATH, "r", encoding="utf-8") as f:
    results_p1 = json.load(f)["results"]

n = len(results_p1)
n_correct = sum(r["is_correct"] for r in results_p1)
print(f"{PROMPT_LABEL} ({n} sampel)")
print("=" * 60)
print(f"Label Accuracy : {n_correct}/{n} = {n_correct/n:.1%}")
print()
print("Per-label accuracy:")
for lbl in ["yes", "no", "maybe"]:
    sub = [r for r in results_p1 if r["ground_truth"] == lbl]
    if sub:
        c = sum(r["is_correct"] for r in sub)
        print(f"  {lbl:>5}: {c}/{len(sub)} = {c/len(sub):.1%}")

# Compare with SH baseline (default prompt)
print(f"\\n=== Comparison vs SH baseline (default prompt) ===")
configs = {
    "SH baseline (default prompt)": RESULTS_DIR / "sh_baseline_openai_phase1_answers.json",
    f"SH + {PROMPT_LABEL}":         PHASE1_PATH,
}
for name, path in configs.items():
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)["results"]
        a = sum(r["is_correct"] for r in d) / len(d)
        print(f"  {name:<45}: {a:.1%}  ({sum(r['is_correct'] for r in d)}/{len(d)})")
''')


CELL_EVALUATOR = code('''def _split_sentences(text):
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
        "faithfulness":      compute_faithfulness(ans, ctx),
        "context_recall":    compute_context_recall(ref, ctx),
        "answer_relevancy":  compute_answer_relevancy(q, ans),
        "context_precision": compute_context_precision(q, ctx, ref),
    }


print("Evaluator ready.")
''')


CELL_PHASE2 = code('''REQ = ["faithfulness", "context_recall", "answer_relevancy", "context_precision"]

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
                json.dump({"config": CONFIG_NAME, "prompt_label": PROMPT_LABEL,
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
''')


CELL_FINAL = code('''def summarize(p1_path, p2_path, label):
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


sh_base = summarize(
    RESULTS_DIR / "sh_baseline_openai_phase1_answers.json",
    RESULTS_DIR / "sh_baseline_openai_phase2_custom.json",
    "SH baseline (default prompt)"
)
sh_pv = summarize(PHASE1_PATH, PHASE2_PATH, f"SH + {PROMPT_LABEL}")

print("=" * 88)
print(f"COMPARISON: SH baseline vs SH + {PROMPT_LABEL}")
print("=" * 88)
print(f"{'Metric':<24} {'SH baseline':>14} {'SH + variation':>18} {'Δ (pp)':>10}")
print("-" * 88)
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
    if not (sh_base and sh_pv):
        continue
    delta = (sh_pv[key] - sh_base[key]) * 100
    print(f"  {name:<22} {sh_base[key]:>13.3f} {sh_pv[key]:>17.3f} {delta:>+9.2f}")
''')


def build(config_name, title, intro, prompt_text, prompt_label, max_tokens, out_path):
    cells = [
        md(f"# {title}\n\n{intro}\n"),
        CELL_IMPORTS,
        cell_config(config_name, prompt_label),
        CELL_LOAD,
        CELL_OPENAI_CLIENT,
        CELL_RETRIEVE,
        cell_prompt(config_name, prompt_text, max_tokens),
        md("## Phase 1 — Generate jawaban 500 sampel (resumable, ~15 menit)"),
        CELL_PHASE1,
        md("## Phase 1 — Quick analysis & comparison"),
        CELL_PHASE1_ANALYSIS,
        md("## Phase 2 — Custom evaluator 4 metrik (resumable, ~25 menit)"),
        CELL_EVALUATOR,
        CELL_PHASE2,
        md("## Final comparison vs SH baseline (default prompt)"),
        CELL_FINAL,
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path.name}  ({len(cells)} cells)")


if __name__ == "__main__":
    build(
        config_name="sh_p1_strict_openai",
        title="09 — Prompt Variation P1: Strict Grounding (SH BM25 + Chroma SH)",
        intro=(
            "Prompt variant: **P1 Strict Grounding** — explicit grounding rules + "
            "decision rubric yang lebih hati-hati. Tujuan: kurangi hallucination dan "
            "lihat apakah strict grounding mengurangi maybe-bias.\n\n"
            "Pipeline retrieval **identik** dengan `04_baseline_openai_sh.ipynb` "
            "(BM25 SH + Chroma SH + RRF). Yang berbeda **hanya prompt generation**."
        ),
        prompt_text=P1_STRICT_GROUNDING,
        prompt_label="P1 Strict Grounding",
        max_tokens=350,
        out_path=OUT_DIR / "09_p1_strict_grounding_sh.ipynb",
    )

    build(
        config_name="sh_p3_fewshot_openai",
        title="10 — Prompt Variation P3: Few-Shot CoT (SH BM25 + Chroma SH)",
        intro=(
            "Prompt variant: **P3 Few-Shot Chain-of-Thought** — 3 demo biomedical "
            "(yes/no/maybe) untuk mengajari LLM bagaimana melakukan reasoning lalu "
            "memutuskan label. Tujuan: tingkatkan akurasi maybe class & konsistensi "
            "reasoning.\n\n"
            "Pipeline retrieval **identik** dengan `04_baseline_openai_sh.ipynb` "
            "(BM25 SH + Chroma SH + RRF). Yang berbeda **hanya prompt generation**.\n\n"
            "> `max_tokens` dinaikkan ke 500 karena prompt ini meminta reasoning dulu "
            "sebelum jawaban final."
        ),
        prompt_text=P3_FEWSHOT_COT,
        prompt_label="P3 Few-Shot CoT",
        max_tokens=500,
        out_path=OUT_DIR / "10_p3_fewshot_cot_sh.ipynb",
    )

    print("\nAll 2 notebooks generated.")
