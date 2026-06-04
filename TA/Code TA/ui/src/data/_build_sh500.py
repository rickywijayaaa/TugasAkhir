"""Extract 500 SH baseline results into a compact JSON for UI consumption.

Source: results/BM25_Expansion/sh_baseline_openai_phase1_answers.json
Output: ui/src/data/sh_baseline_500.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # .../Code TA
SRC = ROOT / "results" / "BM25_Expansion" / "sh_baseline_openai_phase1_answers.json"
OUT = HERE / "sh_baseline_500.json"

print(f"Source: {SRC}")
print(f"Output: {OUT}")

with open(SRC, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]
print(f"Loaded {len(results)} samples")

# Compact format — keep only what UI needs
compact = []
for r in results:
    compact.append({
        "idx": r["idx"],
        "pubid": r["pubid"],
        "question": r["question"],
        "ground_truth": r["ground_truth"],
        "predicted_label": r["predicted_label"],
        "is_correct": r["is_correct"],
        "answer": r["answer"],
        "reference": r["reference"],
        "contexts": r["contexts"],
        "context_pubids": r["context_pubids"],
        "context_sections": r["context_sections"],
        "retrieval_scores": [round(s, 3) for s in r.get("retrieval_scores", [])],
        "dense_scores":     [round(s, 4) for s in r.get("dense_scores", [])],
        "rrf_scores":       [round(s, 4) for s in r.get("rrf_scores", [])],
    })

# Stats
n_correct = sum(1 for r in compact if r["is_correct"])
by_label = {}
by_label_correct = {}
for r in compact:
    gt = r["ground_truth"]
    by_label[gt] = by_label.get(gt, 0) + 1
    if r["is_correct"]:
        by_label_correct[gt] = by_label_correct.get(gt, 0) + 1

stats = {
    "total": len(compact),
    "correct": n_correct,
    "wrong": len(compact) - n_correct,
    "accuracy": n_correct / len(compact),
    "per_class": {
        lbl: {
            "total": by_label.get(lbl, 0),
            "correct": by_label_correct.get(lbl, 0),
            "accuracy": by_label_correct.get(lbl, 0) / by_label.get(lbl, 1) if by_label.get(lbl) else 0,
        }
        for lbl in ("yes", "no", "maybe")
    },
    "config": "sh_baseline_openai",
    "model": "GPT-4.1-mini",
    "retriever": "BM25 SH + Chroma SH + RRF",
    "prompt": "Default (anti-maybe bias)",
}

output = {"stats": stats, "samples": compact}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=1)

size_kb = OUT.stat().st_size / 1024
print(f"\nWrote {len(compact)} samples to {OUT.name}  ({size_kb:.0f} KB)")
print(f"Accuracy: {n_correct}/{len(compact)} = {n_correct/len(compact)*100:.2f}%")
for lbl in ("yes", "no", "maybe"):
    p = stats["per_class"][lbl]
    print(f"  {lbl:5s}: {p['correct']}/{p['total']} = {p['accuracy']*100:.2f}%")
