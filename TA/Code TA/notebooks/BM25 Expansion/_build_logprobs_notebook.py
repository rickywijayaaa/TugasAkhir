"""Generator for `11_logprobs_baseline_sh.ipynb`.

Same pipeline as `04_baseline_openai_sh.ipynb` (SH BM25 + Chroma SH + RRF + default
prompt), but with `logprobs=True` to capture per-token probability for the final
yes/no/maybe answer token. Adds:
- `confidence_dist`: {yes: p, no: p, maybe: p} per sample
- `confidence`: max(confidence_dist) — the model's belief in its chosen answer
- `entropy`: shannon entropy of confidence_dist (low = certain, high = uncertain)
- `top_alternative`: 2nd-highest label (useful for "is it close?" analysis)
- `margin`: top1 - top2 (low margin = LLM was torn)

Plus 4 analysis cells:
- Quick stats: avg confidence, accuracy stratified by confidence bucket
- Calibration plot (reliability diagram + ECE)
- Confidence-vs-correctness scatter
- Selective prediction curve (drop low-confidence samples)
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "11_logprobs_baseline_sh.ipynb"


def md(src):  return {"cell_type":"markdown","metadata":{},"source":src.splitlines(keepends=True)}
def code(src):return {"cell_type":"code","metadata":{},"source":src.splitlines(keepends=True),
                     "outputs":[],"execution_count":None}


HEADER = md("""# 11 — Logprobs Baseline (SH BM25 + Chroma SH + Default Prompt)

**Tujuan**: re-run baseline SH dengan `logprobs=True` untuk capture confidence
LLM (P(yes), P(no), P(maybe)) per sample. Dengan ini bisa dilakukan:

1. **Calibration analysis** — apakah confidence LLM well-calibrated?
   (e.g., samples dengan confidence 0.8 benar ~80%?)
2. **Selective prediction** — kalau buang N% sampel paling tidak yakin,
   berapa accuracy naik?
3. **Maybe-class drilldown** — apakah samples yang salah jadi "yes" sebenarnya
   confidence-nya rendah (LLM ragu)?
4. **Confidence-aware abstention** — flip ke `maybe` kalau gap top1-top2 kecil.

**Pipeline**: identik dengan `04_baseline_openai_sh.ipynb` — SH BM25 + Chroma SH
+ RRF + default prompt. Yang berubah hanya `logprobs=True` di OpenAI call dan
field tambahan di output JSON.

**Estimasi**: ~$0.50, ~15 menit (sama dengan baseline 04, logprobs gratis).
""")


CELL_IMPORTS = code('''import os, sys, json, pickle, time, re, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
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


CELL_CONFIG = code('''LLM_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"
TOP_K_BM25 = 50; TOP_K_DENSE = 50; TOP_K_RETRIEVAL = 5

DATASET_NAME = "qiaojin/PubMedQA"; DATASET_SUBSET = "pqa_labeled"
MAX_SAMPLES = 500
TEMPERATURE = 0.0; SEED = 42

HERE = Path(".").resolve()
NOTEBOOKS_DIR = HERE.parent if HERE.name == "BM25 Expansion" else HERE
PROJECT_ROOT  = NOTEBOOKS_DIR.parent
BM25_INDEX_PATH = NOTEBOOKS_DIR / "pubmedqa_bm25_sh.pkl"
CHROMA_DB_PATH  = NOTEBOOKS_DIR / "pubmedqa_chroma_sh"
RESULTS_DIR = PROJECT_ROOT / "results" / "BM25_Expansion"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_NAME = "sh_baseline_logprobs_openai"
PHASE1_PATH = RESULTS_DIR / f"{CONFIG_NAME}_phase1_answers.json"

print("Konfigurasi:")
print(f"  LLM         : {LLM_MODEL}  (with logprobs)")
print(f"  BM25 index  : {BM25_INDEX_PATH.name} (Schwartz-Hearst)")
print(f"  Chroma path : {CHROMA_DB_PATH.name} (Schwartz-Hearst)")
print(f"  Sampel      : {MAX_SAMPLES}")
print(f"  Output      : {PHASE1_PATH.name}")

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


CELL_OPENAI = code('''client = OpenAI(api_key=api_key)


def openai_generate_with_logprobs(prompt, max_tokens=300, temperature=TEMPERATURE):
    """Returns (text, content_tokens) where content_tokens is a list of
    ChatCompletionTokenLogprob objects with .token, .logprob, .top_logprobs."""
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens, seed=SEED,
                logprobs=True, top_logprobs=5,
            )
            text = resp.choices[0].message.content.strip()
            tokens = resp.choices[0].logprobs.content if resp.choices[0].logprobs else []
            return text, tokens
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


# smoke test
_test_text, _test_tokens = openai_generate_with_logprobs("Reply OK", max_tokens=5)
print(f"OpenAI ready: {_test_text!r}")
print(f"Tokens captured: {len(_test_tokens)} (first: {_test_tokens[0].token!r}, logprob={_test_tokens[0].logprob:.3f})")
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


CELL_PROMPT_AND_LABEL = code('''GENERATION_PROMPT = (
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

LABELS = ("yes", "no", "maybe")


def generate_answer_with_logprobs(query, retrieved):
    context = "\\n\\n".join(
        f"[{i}] ({r.document.section_label}): {r.document.text}"
        for i, r in enumerate(retrieved, 1)
    )
    text, tokens = openai_generate_with_logprobs(
        GENERATION_PROMPT.format(context=context, question=query),
        max_tokens=300, temperature=TEMPERATURE
    )
    return text, tokens


def extract_label_and_confidence(answer_text, content_tokens):
    """Find the final yes/no/maybe token in the LLM output and read its
    probability + top-5 alternatives. Returns:
      label, confidence_dist (dict), top_alternative, margin, entropy, source
    where `source` is 'token_logprobs' (logprobs valid) or 'fallback_regex'
    (could not find label token; fell back to text parsing without confidence).
    """
    # iterate token sequence from END to find first label token
    target_idx = None
    for i in range(len(content_tokens) - 1, -1, -1):
        word = content_tokens[i].token.strip().lower()
        # token may also include punctuation; strip
        clean = re.sub(r"[^a-z]", "", word)
        if clean in LABELS:
            target_idx = i
            break

    if target_idx is None:
        # fallback: parse text
        lines = [l.strip().lower() for l in answer_text.split("\\n") if l.strip()]
        for line in reversed(lines[-5:]):
            word = re.sub(r"[^a-z]", "", line)
            if word in LABELS:
                return word, None, None, None, None, "fallback_regex"
        for label in LABELS:
            if re.search(r"\\b" + label + r"\\b", answer_text.lower()):
                return label, None, None, None, None, "fallback_regex"
        return "maybe", None, None, None, None, "fallback_default"

    tok = content_tokens[target_idx]
    chosen_word = re.sub(r"[^a-z]", "", tok.token.strip().lower())

    # Build probability distribution over {yes, no, maybe}
    probs = {l: 0.0 for l in LABELS}
    probs[chosen_word] = math.exp(tok.logprob)
    for alt in tok.top_logprobs:
        w = re.sub(r"[^a-z]", "", alt.token.strip().lower())
        if w in LABELS:
            probs[w] = max(probs[w], math.exp(alt.logprob))

    # Normalize so probabilities sum to 1 over the 3 label space
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    # Stats
    sorted_labels = sorted(probs.items(), key=lambda x: -x[1])
    top1_label, top1_p = sorted_labels[0]
    top2_label, top2_p = sorted_labels[1]
    margin = top1_p - top2_p
    # Shannon entropy (base e), normalized by log(3) so it's in [0,1]
    eps = 1e-12
    entropy = -sum(p * math.log(p + eps) for p in probs.values()) / math.log(3)

    return top1_label, probs, top2_label, margin, entropy, "token_logprobs"


# Smoke test on idx 30
test_q = pubmedqa_data[30]["question"]
test_r = retrieve_hybrid(test_q)
test_text, test_tokens = generate_answer_with_logprobs(test_q, test_r)
label, dist, top2, margin, entropy, src = extract_label_and_confidence(test_text, test_tokens)

print(f"--- Smoke test idx 30 (GT={pubmedqa_data[30]['final_decision']}) ---")
print(test_text)
print()
print(f"Predicted label : {label}")
print(f"Source          : {src}")
print(f"Distribution    : {dist}")
_margin_s  = f"{margin:.3f}" if margin is not None else "n/a"
_entropy_s = f"{entropy:.3f}" if entropy is not None else "n/a"
print(f"Top alternative : {top2}  (margin = {_margin_s})")
print(f"Entropy (norm)  : {_entropy_s}  (0=certain, 1=max uncertainty)")
''')


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
        answer, tokens = generate_answer_with_logprobs(q, retrieved)
        predicted, conf_dist, top2, margin, entropy, src = extract_label_and_confidence(answer, tokens)

        results_p1.append({
            "idx": i, "pubid": str(s["pubid"]), "question": q,
            "ground_truth": gt, "predicted_label": predicted,
            "is_correct": predicted == gt,
            # logprob fields
            "confidence_dist": conf_dist,
            "confidence":      max(conf_dist.values()) if conf_dist else None,
            "top_alternative": top2,
            "margin":          margin,
            "entropy":         entropy,
            "logprob_source":  src,
            # standard fields
            "answer": answer,
            "contexts":        [r.document.text for r in retrieved],
            "context_pubids":  [r.document.pubid for r in retrieved],
            "context_sections":[r.document.section_label for r in retrieved],
            "reference": ref,
            "retrieval_scores":[r.bm25_score for r in retrieved],
            "dense_scores":    [r.dense_score for r in retrieved],
            "rrf_scores":      [r.rrf_score for r in retrieved],
        })

        if (i + 1) % 10 == 0 or i == MAX_SAMPLES - 1:
            with open(PHASE1_PATH, "w", encoding="utf-8") as f:
                json.dump({"config": CONFIG_NAME, "llm_model": LLM_MODEL,
                           "embed_model": EMBED_MODEL,
                           "timestamp": datetime.now().isoformat(),
                           "max_samples": MAX_SAMPLES, "completed": i+1,
                           "with_logprobs": True,
                           "results": results_p1}, f, indent=2, ensure_ascii=False)
            done = i + 1
            acc = sum(r["is_correct"] for r in results_p1) / done
            avg_conf = np.mean([r["confidence"] for r in results_p1 if r["confidence"] is not None])
            eta = (time.time() - t0) / (i + 1 - start_from) * (MAX_SAMPLES - i - 1) / 60
            print(f"  [{done:3d}/{MAX_SAMPLES}] acc={acc:.1%} | avg_conf={avg_conf:.3f} | ETA {eta:.1f} mnt")

print(f"\\nFase 1 selesai -> {PHASE1_PATH}")
''')


CELL_QUICK_STATS = code('''# === Quick stats: accuracy + confidence distribution ===
with open(PHASE1_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)["results"]

n = len(results)
n_correct = sum(r["is_correct"] for r in results)
print(f"SH Baseline + Logprobs ({n} sampel)")
print("=" * 60)
print(f"Overall accuracy : {n_correct}/{n} = {n_correct/n:.1%}")
print()

# Filter only samples with valid logprobs
with_lp = [r for r in results if r["confidence"] is not None]
print(f"Samples with valid logprobs : {len(with_lp)}/{n}")
print(f"Samples fell back to regex  : {n - len(with_lp)}")
print()

if with_lp:
    confs = np.array([r["confidence"] for r in with_lp])
    print("Confidence statistics:")
    print(f"  Mean              : {confs.mean():.3f}")
    print(f"  Median            : {np.median(confs):.3f}")
    print(f"  Std               : {confs.std():.3f}")
    print(f"  Min/Max           : {confs.min():.3f} / {confs.max():.3f}")
    print(f"  Pct >= 0.9        : {(confs >= 0.9).mean()*100:.1f}%")
    print(f"  Pct >= 0.7        : {(confs >= 0.7).mean()*100:.1f}%")
    print(f"  Pct <  0.5        : {(confs <  0.5).mean()*100:.1f}%")
    print()

    # Confidence per ground-truth class
    print("Mean confidence by ground truth class:")
    for lbl in LABELS:
        sub = [r for r in with_lp if r["ground_truth"] == lbl]
        if sub:
            c = np.mean([r["confidence"] for r in sub])
            print(f"  GT={lbl:5s}: avg_conf={c:.3f}  (n={len(sub)})")

    print()
    print("Mean confidence by predicted label:")
    for lbl in LABELS:
        sub = [r for r in with_lp if r["predicted_label"] == lbl]
        if sub:
            c = np.mean([r["confidence"] for r in sub])
            acc = sum(r["is_correct"] for r in sub) / len(sub)
            print(f"  Pred={lbl:5s}: avg_conf={c:.3f}  acc={acc:.1%}  (n={len(sub)})")
''')


CELL_CALIBRATION = code('''# === Calibration plot (reliability diagram) + Expected Calibration Error (ECE) ===
import matplotlib.pyplot as plt

with_lp = [r for r in results if r["confidence"] is not None]
confs   = np.array([r["confidence"]  for r in with_lp])
correct = np.array([r["is_correct"]  for r in with_lp]).astype(int)

# Bin samples by confidence
n_bins = 10
bin_edges = np.linspace(0.33, 1.0, n_bins + 1)  # confidence floor 0.33 = uniform over 3 classes
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

acc_per_bin = []
conf_per_bin = []
count_per_bin = []
for i in range(n_bins):
    in_bin = (confs >= bin_edges[i]) & (confs < bin_edges[i+1] if i < n_bins-1 else confs <= bin_edges[i+1])
    if in_bin.sum() > 0:
        acc_per_bin.append(correct[in_bin].mean())
        conf_per_bin.append(confs[in_bin].mean())
        count_per_bin.append(in_bin.sum())
    else:
        acc_per_bin.append(np.nan); conf_per_bin.append(bin_centers[i]); count_per_bin.append(0)

acc_per_bin   = np.array(acc_per_bin)
conf_per_bin  = np.array(conf_per_bin)
count_per_bin = np.array(count_per_bin)

# Expected Calibration Error (ECE)
total = count_per_bin.sum()
mask  = ~np.isnan(acc_per_bin)
ece = np.sum(count_per_bin[mask] / total * np.abs(acc_per_bin[mask] - conf_per_bin[mask]))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) Reliability diagram
ax = axes[0]
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration')
ax.bar(bin_centers, acc_per_bin, width=(bin_edges[1]-bin_edges[0])*0.9,
       alpha=0.7, color='#5B9BD5', edgecolor='black', label='Observed accuracy')
ax.plot(conf_per_bin[mask], acc_per_bin[mask], 'o-', color='#1B7F3A',
        markersize=8, linewidth=2, label='Acc vs confidence (binned)')
ax.set_xlabel('Confidence (max P over yes/no/maybe)')
ax.set_ylabel('Accuracy within bin')
ax.set_title(f'Reliability Diagram — ECE = {ece:.3f}', fontweight='bold')
ax.set_xlim(0.3, 1.0); ax.set_ylim(0, 1.0)
ax.legend(loc='lower right'); ax.grid(alpha=0.3)

# (b) Sample count per bin
ax = axes[1]
ax.bar(bin_centers, count_per_bin, width=(bin_edges[1]-bin_edges[0])*0.9,
       alpha=0.8, color='#A5A5A5', edgecolor='black')
for cx, cy in zip(bin_centers, count_per_bin):
    if cy > 0:
        ax.text(cx, cy + 5, str(int(cy)), ha='center', va='bottom', fontsize=9, fontweight='600')
ax.set_xlabel('Confidence bin')
ax.set_ylabel('# samples')
ax.set_title('Sample distribution across confidence bins', fontweight='bold')
ax.set_xlim(0.3, 1.0); ax.grid(axis='y', alpha=0.3)

plt.tight_layout(); plt.show()

print(f"Expected Calibration Error (ECE): {ece:.3f}")
print(f"  Lower = better calibrated. 0 = perfectly calibrated.")
print(f"  For reference: well-calibrated models typically < 0.05.")
''')


CELL_SCATTER = code('''# === Confidence-vs-correctness violin / boxplot ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) Violin: confidence distribution for correct vs wrong
ax = axes[0]
data_correct = [r["confidence"] for r in with_lp if r["is_correct"]]
data_wrong   = [r["confidence"] for r in with_lp if not r["is_correct"]]
parts = ax.violinplot([data_correct, data_wrong], showmedians=True, showmeans=False, widths=0.7)
for pc, color in zip(parts['bodies'], ['#1B7F3A', '#C00000']):
    pc.set_facecolor(color); pc.set_alpha(0.6); pc.set_edgecolor('black')
ax.set_xticks([1, 2])
ax.set_xticklabels([f'Correct\\n(n={len(data_correct)})', f'Wrong\\n(n={len(data_wrong)})'], fontsize=11)
ax.set_ylabel('Confidence (max P)')
ax.set_title('Confidence distribution: correct vs wrong predictions', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0.3, 1.05)

mean_c = np.mean(data_correct); mean_w = np.mean(data_wrong)
ax.text(1, 1.02, f'mean={mean_c:.3f}', ha='center', fontsize=10, fontweight='600', color='#1B7F3A')
ax.text(2, 1.02, f'mean={mean_w:.3f}', ha='center', fontsize=10, fontweight='600', color='#C00000')

# (b) Margin distribution: how torn was the LLM?
ax = axes[1]
margins = np.array([r["margin"] for r in with_lp])
ax.hist([np.array([r["margin"] for r in with_lp if r["is_correct"]]),
         np.array([r["margin"] for r in with_lp if not r["is_correct"]])],
        bins=20, color=['#1B7F3A', '#C00000'], alpha=0.7,
        label=['Correct','Wrong'], stacked=False, edgecolor='black')
ax.set_xlabel('Margin (top1 P − top2 P)')
ax.set_ylabel('# samples')
ax.set_title('Margin distribution — low margin = LLM was torn', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)

plt.tight_layout(); plt.show()

# Stats test: is mean confidence different between correct/wrong?
from scipy.stats import mannwhitneyu
stat, pval = mannwhitneyu(data_correct, data_wrong, alternative='greater')
print(f"Mann-Whitney U test (correct > wrong): U={stat:.0f}, p={pval:.2e}")
if pval < 0.05:
    print(f"  -> Confidence dari correct prediction SIGNIFIKAN lebih tinggi dari wrong (p<0.05)")
    print(f"     LLM punya weak signal: jawaban benar memang lebih confident.")
else:
    print(f"  -> Tidak signifikan. Confidence tidak diskriminatif.")
''')


CELL_SELECTIVE = code('''# === Selective prediction curve: drop low-confidence samples ===
sorted_results = sorted(with_lp, key=lambda r: -r["confidence"])  # most confident first

coverages, accuracies = [], []
for k in range(10, len(sorted_results)+1, 5):
    sub = sorted_results[:k]
    coverages.append(k / len(with_lp))
    accuracies.append(sum(r["is_correct"] for r in sub) / len(sub))

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(np.array(coverages)*100, np.array(accuracies)*100, '-', color='#1a1a1a', linewidth=2.5)
ax.axhline(sum(r["is_correct"] for r in with_lp)/len(with_lp)*100,
           color='#A5A5A5', linestyle='--', linewidth=1.5,
           label=f'Full coverage: {sum(r["is_correct"] for r in with_lp)/len(with_lp)*100:.1f}%')

# Mark interesting coverage points
for cov_target in [0.5, 0.7, 0.9]:
    k = int(cov_target * len(with_lp))
    sub = sorted_results[:k]
    acc = sum(r["is_correct"] for r in sub) / k
    ax.plot(cov_target*100, acc*100, 'o', color='#1B7F3A', markersize=10)
    ax.annotate(f'{acc*100:.1f}% @ {cov_target*100:.0f}% cov',
                xy=(cov_target*100, acc*100), xytext=(8, -15),
                textcoords='offset points', fontsize=10, fontweight='600',
                color='#1B7F3A')

ax.set_xlabel('Coverage (% samples kept, sorted by confidence)')
ax.set_ylabel('Accuracy on kept samples (%)')
ax.set_title('Selective Prediction Curve — drop least-confident first',
             fontweight='bold', pad=12)
ax.legend(loc='lower left'); ax.grid(alpha=0.3)
ax.set_xlim(0, 105); ax.set_ylim(60, 100)

plt.tight_layout(); plt.show()

# Print key thresholds
print()
print("Selective prediction trade-offs:")
print(f"  Coverage 100%  : accuracy = {accuracies[-1]*100:.1f}%  (full baseline)")
for cov_target in [0.9, 0.8, 0.7, 0.5]:
    k = int(cov_target * len(with_lp))
    if k > 0:
        sub = sorted_results[:k]
        acc = sum(r["is_correct"] for r in sub) / k
        threshold = sorted_results[k-1]["confidence"]
        print(f"  Coverage {cov_target*100:3.0f}%  : accuracy = {acc*100:.1f}%  (drop conf < {threshold:.3f})")
''')


CELL_ABSTENTION = code('''# === Confidence-aware abstention rule: flip low-margin to "maybe" ===
print("=" * 70)
print("CONFIDENCE-AWARE ABSTENTION — flip top1=yes/no with low margin to 'maybe'")
print("=" * 70)

for margin_threshold in [0.05, 0.10, 0.15, 0.20, 0.30]:
    flipped_correct = 0
    flipped_wrong   = 0
    unchanged_correct = 0
    unchanged_wrong   = 0
    n_flipped = 0
    for r in with_lp:
        new_pred = r["predicted_label"]
        if r["predicted_label"] in ("yes","no") and r["margin"] < margin_threshold:
            new_pred = "maybe"
            n_flipped += 1
        is_correct_new = (new_pred == r["ground_truth"])
        if new_pred != r["predicted_label"]:
            if is_correct_new: flipped_correct += 1
            else:              flipped_wrong   += 1
        else:
            if is_correct_new: unchanged_correct += 1
            else:              unchanged_wrong   += 1
    total = flipped_correct + flipped_wrong + unchanged_correct + unchanged_wrong
    new_acc = (flipped_correct + unchanged_correct) / total
    baseline_acc = sum(r["is_correct"] for r in with_lp) / len(with_lp)
    delta = (new_acc - baseline_acc) * 100
    print(f"  margin < {margin_threshold:.2f}: flip {n_flipped:3d} samples | "
          f"new_acc={new_acc*100:5.2f}% (Δ {delta:+.2f}pp) | "
          f"flips_helped={flipped_correct}, flips_hurt={flipped_wrong}")

print()
print("Interpretation:")
print("  Δ positif = abstention rule menaikkan accuracy (good).")
print("  Threshold optimal biasanya saat flips_helped > flips_hurt.")
''')


def build_notebook():
    cells = [
        HEADER,
        CELL_IMPORTS,
        CELL_CONFIG,
        CELL_LOAD,
        CELL_OPENAI,
        CELL_RETRIEVE,
        CELL_PROMPT_AND_LABEL,
        md("## Phase 1 — Generate jawaban 500 sampel dengan logprobs (resumable, ~15 menit)"),
        CELL_PHASE1,
        md("## Quick Stats — Confidence Distribution & Overall Accuracy"),
        CELL_QUICK_STATS,
        md("""## Calibration Analysis

Reliability diagram: bin samples by predicted confidence, lalu cek
apakah accuracy aktual di bin itu **match** dengan confidence-nya.

- **Diagonal** = perfectly calibrated (LLM bilang 0.8 yakin → memang benar ~80%)
- **Di atas diagonal** = under-confident (LLM under-estimate dirinya)
- **Di bawah diagonal** = over-confident (LLM over-estimate — issue umum)

**ECE (Expected Calibration Error)** = weighted mean gap. Lower = better.
"""),
        CELL_CALIBRATION,
        md("## Confidence vs Correctness — Apakah confidence diskriminatif?"),
        CELL_SCATTER,
        md("""## Selective Prediction — Drop low-confidence samples

Skenario medical: kalau LLM ragu, lebih baik tidak menjawab (abstain) daripada
keluarkan jawaban salah. Kurva ini menunjukkan **trade-off coverage vs accuracy**.
"""),
        CELL_SELECTIVE,
        md("""## Confidence-Aware Abstention — Mini-contribution untuk Bab IV

Aturan: kalau `top1` adalah `yes`/`no` tapi gap dengan `top2` < threshold,
**flip ke `maybe`** (LLM sebenarnya ragu). Test beberapa threshold dan lihat
apakah accuracy naik.
"""),
        CELL_ABSTENTION,
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
