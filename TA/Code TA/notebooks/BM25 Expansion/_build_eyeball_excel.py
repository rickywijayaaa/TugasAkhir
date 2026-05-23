"""Build a single Excel with all 500 samples x 4 variants for eye-balling labels & answers.

Sheet 1 'compare'  : wide layout, one row per idx, baseline/qr/cr/qr_cr columns side by side.
Sheet 2 'long'     : tall layout, one row per (idx, variant).
Sheet 3 'flips'    : only rows where predicted_label differs across variants.
Sheet 4 'summary'  : per-variant accuracy + per-class accuracy.
"""
import json
from pathlib import Path
import pandas as pd
from openpyxl.styles import PatternFill, Alignment, Font
from openpyxl.utils import get_column_letter

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "BM25_Expansion"
OUT_PATH    = Path(__file__).resolve().parent / "eyeball_500_4variants.xlsx"

FILES = {
    "baseline": "sh_baseline_openai_phase1_answers.json",
    "qr":       "sh_qr_openai_phase1_answers.json",
    "cr":       "sh_cr_openai_phase1_answers.json",
    "qr_cr":    "sh_qr_cr_openai_phase1_answers.json",
}

def load(name):
    return json.load(open(RESULTS_DIR / FILES[name], encoding="utf-8"))["results"]

print("Loading 4 result files...")
data = {k: load(k) for k in FILES}
N = len(data["baseline"])
print(f"  n={N}")

# --- Sheet 1: wide compare ---
rows = []
for i in range(N):
    base = data["baseline"][i]
    row = {
        "idx":          base["idx"],
        "pubid":        base["pubid"],
        "question":     base["question"],
        "ground_truth": base["ground_truth"],
        "reference":    base.get("reference", ""),
    }
    for v in ["baseline", "qr", "cr", "qr_cr"]:
        r = data[v][i]
        row[f"{v}_pred"]    = r["predicted_label"]
        row[f"{v}_correct"] = "TRUE" if r["is_correct"] else "FALSE"
        row[f"{v}_answer"]  = r["answer"]
    # transition flag
    preds = [data[v][i]["predicted_label"] for v in ["baseline","qr","cr","qr_cr"]]
    row["preds_unique"]  = len(set(preds))
    row["all_same_pred"] = "YES" if len(set(preds))==1 else "NO"
    # which variants got it right
    correct_set = [v for v in ["baseline","qr","cr","qr_cr"] if data[v][i]["is_correct"]]
    row["correct_variants"] = ",".join(correct_set) if correct_set else "(none)"
    row["n_correct_variants"] = len(correct_set)
    rows.append(row)
df_wide = pd.DataFrame(rows)

# --- Sheet 2: long ---
long_rows = []
for i in range(N):
    base = data["baseline"][i]
    for v in ["baseline","qr","cr","qr_cr"]:
        r = data[v][i]
        long_rows.append({
            "idx":          base["idx"],
            "pubid":        base["pubid"],
            "variant":      v,
            "question":     base["question"],
            "ground_truth": base["ground_truth"],
            "predicted":    r["predicted_label"],
            "is_correct":   "TRUE" if r["is_correct"] else "FALSE",
            "answer":       r["answer"],
            "reference":    base.get("reference",""),
        })
df_long = pd.DataFrame(long_rows)

# --- Sheet 3: flips (predictions differ across variants) ---
df_flips = df_wide[df_wide["preds_unique"] > 1].copy()
print(f"  Flip rows (preds differ across variants): {len(df_flips)}")

# --- Sheet 4: summary ---
sum_rows = []
for v in ["baseline","qr","cr","qr_cr"]:
    r = data[v]
    by = {}
    for x in r:
        gt = x["ground_truth"]
        by.setdefault(gt,[0,0]); by[gt][1]+=1; by[gt][0]+= 1 if x["is_correct"] else 0
    sum_rows.append({
        "variant":  v,
        "overall_acc": sum(1 for x in r if x["is_correct"])/len(r),
        "yes_acc":   by.get("yes",[0,1])[0]/by.get("yes",[0,1])[1] if "yes" in by else None,
        "no_acc":    by.get("no",[0,1])[0]/by.get("no",[0,1])[1]   if "no"  in by else None,
        "maybe_acc": by.get("maybe",[0,1])[0]/by.get("maybe",[0,1])[1] if "maybe" in by else None,
        "n_correct": sum(1 for x in r if x["is_correct"]),
        "n_total":   len(r),
    })
df_sum = pd.DataFrame(sum_rows)

# --- Write Excel ---
print(f"Writing {OUT_PATH.name}...")
with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as w:
    df_sum.to_excel(w, sheet_name="summary", index=False)
    df_wide.to_excel(w, sheet_name="compare", index=False)
    df_flips.to_excel(w, sheet_name="flips", index=False)
    df_long.to_excel(w, sheet_name="long", index=False)

# --- Format ---
from openpyxl import load_workbook
wb = load_workbook(OUT_PATH)

# colors
GREEN  = PatternFill("solid", fgColor="C6EFCE")
RED    = PatternFill("solid", fgColor="FFC7CE")
YELLOW = PatternFill("solid", fgColor="FFEB9C")
GREY   = PatternFill("solid", fgColor="EEEEEE")
HEADER = PatternFill("solid", fgColor="305496")

for sheet_name in ["compare", "flips", "long"]:
    ws = wb[sheet_name]
    # header style
    for cell in ws[1]:
        cell.fill = HEADER
        cell.font = Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "C2" if sheet_name != "long" else "B2"
    # column widths
    headers = [c.value for c in ws[1]]
    for j, h in enumerate(headers, start=1):
        col = get_column_letter(j)
        if h in ("question","reference"):
            ws.column_dimensions[col].width = 50
        elif h and h.endswith("_answer") or h == "answer":
            ws.column_dimensions[col].width = 60
        elif h in ("idx","pubid"):
            ws.column_dimensions[col].width = 10
        elif h in ("ground_truth","predicted","predicted_label","variant") or (h and h.endswith("_pred")):
            ws.column_dimensions[col].width = 12
        elif h and (h.endswith("_correct") or h == "is_correct"):
            ws.column_dimensions[col].width = 10
        else:
            ws.column_dimensions[col].width = 16

    # color cells based on correctness
    if sheet_name in ("compare", "flips"):
        # find indices of *_correct columns and *_pred columns
        col_idx = {h: i+1 for i, h in enumerate(headers)}
        for r in range(2, ws.max_row + 1):
            gt_cell = ws.cell(row=r, column=col_idx["ground_truth"])
            gt_cell.alignment = Alignment(horizontal="center")
            for v in ["baseline","qr","cr","qr_cr"]:
                ck = f"{v}_correct"
                pk = f"{v}_pred"
                if ck in col_idx:
                    cell = ws.cell(row=r, column=col_idx[ck])
                    cell.alignment = Alignment(horizontal="center")
                    cell.fill = GREEN if cell.value == "TRUE" else RED
                if pk in col_idx:
                    pcell = ws.cell(row=r, column=col_idx[pk])
                    pcell.alignment = Alignment(horizontal="center")
                    # highlight pred matching/mismatching ground truth
                    if pcell.value == gt_cell.value:
                        pcell.fill = GREEN
                    else:
                        pcell.fill = RED
            # all_same_pred
            if "all_same_pred" in col_idx:
                cell = ws.cell(row=r, column=col_idx["all_same_pred"])
                cell.alignment = Alignment(horizontal="center")
                if cell.value == "YES":
                    cell.fill = GREY
                else:
                    cell.fill = YELLOW
            # answer columns wrap
            for v in ["baseline","qr","cr","qr_cr"]:
                ak = f"{v}_answer"
                if ak in col_idx:
                    ws.cell(row=r, column=col_idx[ak]).alignment = Alignment(wrap_text=True, vertical="top")
            for h in ("question","reference"):
                if h in col_idx:
                    ws.cell(row=r, column=col_idx[h]).alignment = Alignment(wrap_text=True, vertical="top")
    else:  # long
        col_idx = {h: i+1 for i, h in enumerate(headers)}
        for r in range(2, ws.max_row + 1):
            cell = ws.cell(row=r, column=col_idx["is_correct"])
            cell.alignment = Alignment(horizontal="center")
            cell.fill = GREEN if cell.value == "TRUE" else RED
            ws.cell(row=r, column=col_idx["predicted"]).alignment = Alignment(horizontal="center")
            ws.cell(row=r, column=col_idx["ground_truth"]).alignment = Alignment(horizontal="center")
            ws.cell(row=r, column=col_idx["variant"]).alignment = Alignment(horizontal="center")
            for h in ("answer","question","reference"):
                if h in col_idx:
                    ws.cell(row=r, column=col_idx[h]).alignment = Alignment(wrap_text=True, vertical="top")

# summary sheet format
ws = wb["summary"]
for cell in ws[1]:
    cell.fill = HEADER
    cell.font = Font(bold=True, color="FFFFFF")
    cell.alignment = Alignment(horizontal="center")
for col_letter in ["B","C","D","E"]:
    for r in range(2, ws.max_row+1):
        c = ws[f"{col_letter}{r}"]
        if c.value is not None:
            c.number_format = "0.00%"
ws.column_dimensions["A"].width = 14
for col_letter in ["B","C","D","E","F","G"]:
    ws.column_dimensions[col_letter].width = 14

# Set row height for compare/flips/long for readability
for sheet_name in ["compare","flips","long"]:
    ws = wb[sheet_name]
    for r in range(2, ws.max_row+1):
        ws.row_dimensions[r].height = 90 if sheet_name != "long" else 70

wb.save(OUT_PATH)
print(f"Done: {OUT_PATH}")
print(f"  rows in compare : {len(df_wide)}")
print(f"  rows in flips   : {len(df_flips)}")
print(f"  rows in long    : {len(df_long)}")
