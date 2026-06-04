"""Fix truncation bug in compute_faithfulness/recall/precision di semua notebook.

Bug: `c[:400]` truncate chunk ke 400 char, bikin judge kehilangan info terutama
untuk chunk SH expansion yang lebih panjang (53.6% > 400 char).

Fix: naikkan ke 1500 char (cukup untuk hampir semua chunk PubMedQA tanpa
explode token cost).

Juga fix `reference[:300]` → `reference[:800]` agar ground truth tidak ke-cut.

Usage:
    python _fix_faithfulness_truncation.py              # dry-run, list changes
    python _fix_faithfulness_truncation.py --apply      # apply fixes
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APPLY = "--apply" in sys.argv

# Target files (prioritize the ones that produced data we're comparing)
TARGETS = {
    "PRIORITAS 1 - SH Expansion (sumber bug paling parah)": [
        "BM25 Expansion/04_baseline_openai_sh.ipynb",
        "BM25 Expansion/05_qr_openai_sh.ipynb",
        "BM25 Expansion/06_cr_openai_sh.ipynb",
        "BM25 Expansion/07_qr_cr_openai_sh.ipynb",
    ],
    "PRIORITAS 2 - SH Prompt Variation (kalau sudah jalan Phase 2)": [
        "BM25 Expansion/09_p1_strict_grounding_sh.ipynb",
        "BM25 Expansion/10_p3_fewshot_cot_sh.ipynb",
    ],
    "PRIORITAS 3 - Original BM25 (untuk apples-to-apples comparison)": [
        "02.1 Baseline - OpenAI.ipynb",
        "03.1.1 QR - OpenAI - New Version.ipynb",
        "04.1 CR - OpenAI.ipynb",
        "05.1 QR+CR - OpenAI.ipynb",
    ],
    "PRIORITAS 4 - Generator scripts (sync biar regenerate tetap correct)": [
        "BM25 Expansion/_build_qr_cr_notebooks.py",
        "BM25 Expansion/_build_prompt_variation_notebooks.py",
        "BM25 Expansion/_build_sh_notebooks.py",
        "BM25 Expansion/_build_notebooks.py",
        "prompt_experiment/shared.py",
    ],
}

# Patterns to replace
SUBS = [
    (r"c\[:400\]",         "c[:1500]"),
    (r"reference\[:300\]", "reference[:800]"),
]


def fix_text(text):
    """Apply substitutions, return (new_text, n_changes)."""
    n = 0
    for old_pat, new_str in SUBS:
        new_text, count = re.subn(old_pat, new_str, text)
        text = new_text
        n += count
    return text, n


def fix_py(path):
    src = path.read_text(encoding="utf-8")
    new, n = fix_text(src)
    if n > 0 and APPLY:
        path.write_text(new, encoding="utf-8")
    return n


def fix_ipynb(path):
    raw = path.read_text(encoding="utf-8")
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [SKIP not valid JSON: {e}]")
        return 0
    total = 0
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        if isinstance(src, list):
            joined = "".join(src)
            new, n = fix_text(joined)
            if n > 0:
                cell["source"] = new.splitlines(keepends=True)
                total += n
        elif isinstance(src, str):
            new, n = fix_text(src)
            if n > 0:
                cell["source"] = new
                total += n
    if total > 0 and APPLY:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return total


def main():
    print(f"Mode: {'APPLY' if APPLY else 'DRY-RUN (tambahkan --apply untuk eksekusi)'}\n")
    grand_total = 0
    grand_files = 0
    for category, files in TARGETS.items():
        print(f"== {category} ==")
        for rel in files:
            path = HERE / rel
            if not path.exists():
                print(f"  [MISSING] {rel}")
                continue
            if path.suffix == ".ipynb":
                n = fix_ipynb(path)
            else:
                n = fix_py(path)
            status = "would change" if not APPLY else "CHANGED"
            print(f"  {n:>2d} replacements   {status:<14s}  {rel}")
            if n > 0:
                grand_total += n
                grand_files += 1
        print()
    print(f"\nTotal: {grand_total} replacements across {grand_files} files.")
    if not APPLY:
        print("Run again with --apply to actually modify files.")


if __name__ == "__main__":
    main()
