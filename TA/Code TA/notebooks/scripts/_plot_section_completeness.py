"""Plot section completeness chart for PubMedQA corpus (mirip reference image).

Source: notebooks/pubmedqa_bm25_sh.pkl (1706 chunks dari 500 paper)
Output: notebooks/BM25 Expansion/figures/section_completeness.png

Section dinormalisasi ke 8 canonical category (IMRAD-style structured abstract).
PubMedQA by design TIDAK include CONCLUSION (sengaja dihilangkan karena jawaban
yes/no/maybe sebenarnya ada di conclusion paper aslinya).
"""
import os, sys, pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Setup
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # Code TA
PICKLE_PATH = ROOT / "notebooks" / "pubmedqa_bm25_sh.pkl"
OUT_DIR = HERE / "figures"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "section_completeness.png"

# Also save to thesis image folder
THESIS_IMG_DIR = ROOT.parent / "Laporan TA" / "image"
THESIS_OUT_PATH = THESIS_IMG_DIR / "section_completeness.png"


class Document:
    def __init__(self, text="", **kwargs):
        self.text = text
        for k, v in kwargs.items():
            setattr(self, k, v)


import __main__
__main__.Document = Document


# Canonical section mapping
# Values can be a single canonical (str) OR a tuple of multiple (for compound sections)
SECTION_MAP = {
    # Background / Introduction
    "BACKGROUND": "background",
    "INTRODUCTION": "background",
    "CONTEXT": "background",
    "SUMMARY BACKGROUND DATA": "background",
    "SUMMARY OF BACKGROUND DATA": "background",
    "UNLABELLED": "background",
    "SUMMARY": "background",
    # Objective / Aim
    "OBJECTIVE": "objective",
    "OBJECTIVES": "objective",
    "AIM": "objective",
    "AIMS": "objective",
    "PURPOSE": "objective",
    "PURPOSES": "objective",
    "STUDY OBJECTIVE": "objective",
    "HYPOTHESIS": "objective",
    "RATIONALE AND OBJECTIVES": "objective",
    "BACKGROUND AND OBJECTIVES": "objective",
    "BACKGROUND AND OBJECTIVE": "objective",
    "BACKGROUND AND PURPOSE": "objective",
    "BACKGROUND AND AIMS": "objective",
    "BACKGROUND AND AIM": "objective",
    # Methods (design, data)
    "METHODS": "methods",
    "METHOD": "methods",
    "MATERIALS AND METHODS": "methods",
    "MATERIAL AND METHODS": "methods",
    "METHODS AND MATERIALS": "methods",
    "PATIENTS AND METHODS": "methods",
    "PATIENTS AND METHOD": "methods",
    "DESIGN": "methods",
    "STUDY DESIGN": "methods",
    "RESEARCH DESIGN AND METHODS": "methods",
    "DATA SOURCES": "methods",
    "DATA EXTRACTION": "methods",
    "STUDY SELECTION": "methods",
    "STUDY SELECTION AND ASSESSMENT": "methods",
    "STUDY SELECTION AND DATA EXTRACTION": "methods",
    "METHODS AND RESULTS": ("methods", "results"),
    "PROCEDURES": "methods",
    "PROCEDURE": "methods",
    # Patients / Subjects / Setting
    "PATIENTS": "patients_setting",
    "PARTICIPANTS": "patients_setting",
    "SUBJECTS": "patients_setting",
    "POPULATION": "patients_setting",
    "SETTING": "patients_setting",
    "SETTINGS": "patients_setting",
    "SAMPLE": "patients_setting",
    # Intervention
    "INTERVENTION": "intervention",
    "INTERVENTIONS": "intervention",
    "INTERVENTIONS AND OUTCOME MEASURES": "intervention",
    "EXPOSURE": "intervention",
    # Main Outcome Measures
    "MAIN OUTCOME MEASURES": "outcomes",
    "MAIN OUTCOME MEASURE": "outcomes",
    "MAIN OUTCOME": "outcomes",
    "OUTCOME": "outcomes",
    "OUTCOME MEASURES": "outcomes",
    "OUTCOMES MEASURED": "outcomes",
    "MEASUREMENTS": "outcomes",
    "MEASUREMENT": "outcomes",
    # Results
    "RESULTS": "results",
    "RESULT": "results",
    "MAIN RESULTS": "results",
    "FINDINGS": "results",
    "PRINCIPAL FINDINGS": "results",
    "MEASUREMENTS AND MAIN RESULTS": "results",
    # Discussion / Limitations
    "DISCUSSION": "discussion_limit",
    "LIMITATIONS": "discussion_limit",
    "KEY LIMITATIONS": "discussion_limit",
    "RESULTS AND LIMITATIONS": ("results", "discussion_limit"),
    "RESULTS AND DISCUSSION": ("results", "discussion_limit"),
    "CASE REPORTS": "discussion_limit",
    "CASE PRESENTATION": "discussion_limit",
    "CASE DESCRIPTION": "discussion_limit",
    "OBSERVATIONS": "discussion_limit",
    "PRESENTATION": "discussion_limit",
    "DESCRIPTION": "discussion_limit",
    # ---- Additional variants ----
    # Outcomes
    "MAIN OUTCOME MEASUREMENTS": "outcomes",
    "MEASUREMENT AND RESULTS": ("outcomes", "results"),
    "MEASUREMENTS AND RESULTS": ("outcomes", "results"),
    "MEASURES": "outcomes",
    "MAIN MEASURES": "outcomes",
    "ICU LOS": "outcomes",
    # Methods variants
    "SUBJECTS AND METHODS": "methods",
    "METHODOLOGY": "methods",
    "METHOD AND MATERIAL": "methods",
    "OBJECTIVE AND METHODS": "methods",
    "DESIGN, SETTING, AND PARTICIPANTS": "methods",
    "DESIGN, SETTING AND PATIENTS": "methods",
    "DESIGN AND SETTING": "methods",
    "DESIGN OF STUDY": "methods",
    "STUDY DESIGN AND METHODS": "methods",
    "RESEARCH DESIGN": "methods",
    "DATA SOURCE": "methods",
    "DATA SYNTHESIS": "methods",
    "PROBANDS AND METHODS": "methods",
    "CLINICAL TRIAL REGISTRATION": "methods",
    # Patients/setting variants
    "SETTING AND PARTICIPANTS": "patients_setting",
    "SETTINGS AND PARTICIPANTS": "patients_setting",
    "SETTING AND PATIENTS": "patients_setting",
    "PATIENTS AND PARTICIPANTS": "patients_setting",
    # Results variants
    "KEY RESULTS": "results",
    # Objective variants
    "OBJECT": "objective",
    "PRIMARY OBJECTIVE": "objective",
    "PURPOSE OF INVESTIGATION": "objective",
    "AIMS AND OBJECTIVES": "objective",
    "CONTEXT AND OBJECTIVES": "objective",
    "BACKGROUND AND AIM OF THE STUDY": "objective",
    "BACKGROUND AND STUDY OBJECTIVE": "objective",
    "INTRODUCTION AND HYPOTHESIS": "objective",
    "RATIONALE": "objective",
}

# Display label & order (top-to-bottom: paling lengkap → paling jarang)
CANONICAL_ORDER = [
    "results",           # RESULTS / FINDINGS
    "methods",           # METHODS / DESIGN
    "background",        # BACKGROUND / INTRODUCTION
    "objective",         # OBJECTIVE / AIM / PURPOSE
    "patients_setting",  # PATIENTS / PARTICIPANTS / SETTING
    "outcomes",          # MAIN OUTCOME MEASURES
    "intervention",      # INTERVENTION
    "discussion_limit",  # DISCUSSION / LIMITATIONS
]

DISPLAY_LABEL = {
    "results":          "Results / Findings",
    "methods":          "Methods / Design",
    "background":       "Background / Introduction",
    "objective":        "Objective / Aim / Purpose",
    "patients_setting": "Patients / Setting / Participants",
    "outcomes":         "Main Outcome Measures",
    "intervention":     "Intervention",
    "discussion_limit": "Discussion / Limitations",
}


def main():
    print(f"Loading {PICKLE_PATH}...")
    obj = pickle.load(open(PICKLE_PATH, "rb"))
    docs = obj["documents"]
    print(f"  {len(docs)} chunks loaded")

    # Group by pubid → set of canonical sections
    papers = defaultdict(set)
    raw_sections = defaultdict(int)
    unmapped = defaultdict(int)
    for d in docs:
        sec_raw = d.section_label.strip().upper() if d.section_label else ""
        raw_sections[sec_raw] += 1
        canon = SECTION_MAP.get(sec_raw)
        if canon is None:
            unmapped[sec_raw] += 1
            continue
        # Support compound sections (tuple) → assign multiple canonicals
        if isinstance(canon, tuple):
            for c in canon:
                papers[d.pubid].add(c)
        else:
            papers[d.pubid].add(canon)

    n_papers = len(papers)
    print(f"  {n_papers} unique papers")
    print(f"  {sum(unmapped.values())} chunks unmapped ({len(unmapped)} unique raw labels)")
    if unmapped:
        print("  Top unmapped:")
        for sec, cnt in sorted(unmapped.items(), key=lambda x: -x[1])[:10]:
            print(f"    {cnt:3d}  {sec!r}")

    # Count papers per canonical section
    section_counts = {}
    for canon in CANONICAL_ORDER:
        n_have = sum(1 for sections in papers.values() if canon in sections)
        pct = 100.0 * n_have / n_papers
        section_counts[canon] = (n_have, pct)

    # Sort by percentage descending
    sorted_canon = sorted(CANONICAL_ORDER, key=lambda c: -section_counts[c][1])

    # Print table
    print(f"\n{'Section':<40} {'Count':>8}  {'Pct':>8}")
    print("-" * 60)
    for canon in sorted_canon:
        n, pct = section_counts[canon]
        print(f"  {DISPLAY_LABEL[canon]:<38} {n:>8d}  {pct:>7.1f}%")

    # === Plot ===
    fig, ax = plt.subplots(figsize=(11, 5.5))

    labels = [DISPLAY_LABEL[c] for c in sorted_canon]
    counts = [section_counts[c][0] for c in sorted_canon]
    pcts   = [section_counts[c][1] for c in sorted_canon]

    # Single color for all bars
    BAR_COLOR = "#7B1E2A"   # dark red

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, pcts, color=BAR_COLOR, edgecolor="white", linewidth=0.5)

    # Labels on bars
    for bar, n, pct in zip(bars, counts, pcts):
        text = f"{pct:.1f}% ({n:,})"
        x_pos = bar.get_width() + 1.5
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, text,
                va="center", ha="left",
                fontsize=10, fontweight="500",
                color="#222")

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()  # top = highest %
    ax.set_xlim(0, 125)
    ax.set_xlabel("% Dokumen yang Memiliki Section", fontsize=11)
    ax.set_title("Kelengkapan Section per Dokumen — PubMedQA Corpus (n=500 papers)",
                 fontsize=13, fontweight="bold", pad=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888")
    ax.spines["bottom"].set_color("#888")
    ax.tick_params(axis="both", colors="#333")

    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.grid(axis="x", alpha=0.15, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    # Save to figures (workspace)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved (workspace): {OUT_PATH}")
    # Save to thesis image folder
    if THESIS_IMG_DIR.exists():
        plt.savefig(THESIS_OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved (thesis)   : {THESIS_OUT_PATH}")
    else:
        print(f"WARNING: thesis image folder not found at {THESIS_IMG_DIR}")


if __name__ == "__main__":
    main()
