"""
Append Section I (MetaGen-Style Visualizations) ke notebook 07.
Run: py _append_metagen_section.py
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "07 - Visualisasi Perbandingan Konfigurasi.ipynb"

# ===========================================================
# Define the new cells (MetaGen-style)
# ===========================================================

def _to_lines(text):
    """Jupyter source format: list of strings, each ending with \\n except possibly the last."""
    lines = text.splitlines(keepends=True)
    return lines if lines else [""]

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _to_lines(text)}

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _to_lines(text),
    }


NEW_CELLS = []

# --- I. Header ---
NEW_CELLS.append(md("""---
# I. MetaGen-Style Visualizations & Tables

Mengikuti format visualisasi paper MetaGen Blended RAG (Sawarkar dkk. 2025) untuk Bab VI Evaluasi:
- **Figure 3 style**: progressive bar chart dengan label nilai di atas tiap bar + garis tren overlay
- **Figure 4 style**: line chart zoomed-in untuk highlight selisih kecil
- **Tabel ringkas**: format Before/After + komparasi state-of-the-art

Output:
- PNG files → `notebooks/figures/V_*.png` dan auto-copy ke `Laporan TA/image/`
- LaTeX tables → `notebooks/figures/V_tables.tex` (siap paste ke Bab VI)"""))

# --- I.0 Setup ---
NEW_CELLS.append(md("## I.0 Setup — MetaGen-style Matplotlib Config & Data Loading"))

NEW_CELLS.append(code("""import json
import shutil
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# === MetaGen-style configuration ===
# Pastel palette ala MetaGen Figure 3
METAGEN_COLORS = [
    '#A8C5DA',  # blue (lightest baseline)
    '#B7D5B0',  # green
    '#FFE6A8',  # yellow
    '#F5D08C',  # orange-pastel
    '#D8B4D8',  # purple
    '#F2B8C2',  # pink
    '#FFC9A8',  # peach
    '#B0D9B5',  # green (best)
]

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.axis': 'y',
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# === Paths ===
HERE = Path('.').resolve()
NB_DIR = HERE if HERE.name == 'notebooks' else HERE / 'notebooks'
RESULTS_DIR = NB_DIR.parent / 'results'
RESULTS_SH_DIR = RESULTS_DIR / 'BM25_Expansion'
RESULTS_BM25_DIR = RESULTS_DIR / 'BM25'
FIG_DIR = NB_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Auto-copy target ke Laporan TA/image/
LAPORAN_IMG = NB_DIR.parent.parent / 'Laporan TA' / 'image'
LAPORAN_IMG.mkdir(parents=True, exist_ok=True)

print(f'NB_DIR        : {NB_DIR}')
print(f'RESULTS_DIR   : {RESULTS_DIR}')
print(f'FIG_DIR       : {FIG_DIR}')
print(f'LAPORAN_IMG   : {LAPORAN_IMG}')


def save_fig(name, dpi=300):
    \"\"\"Save figure ke notebooks/figures/ dan auto-copy ke Laporan TA/image/.\"\"\"
    fp_nb = FIG_DIR / f'{name}.png'
    plt.savefig(fp_nb, dpi=dpi, bbox_inches='tight')
    fp_laporan = LAPORAN_IMG / f'{name}.png'
    shutil.copy(fp_nb, fp_laporan)
    print(f'  Saved: {fp_nb.name} -> {fp_laporan}')


# === Load aggregate data ===
def load_phase2_acc(path):
    \"\"\"Load phase2 JSON, return (accuracy, ragas_means_dict).\"\"\"
    with open(path) as f:
        d = json.load(f)
    r = d.get('results', [])
    if not r:
        return None, None
    acc = sum(1 for x in r if x.get('is_correct')) / len(r)
    mets = {}
    for k in ['faithfulness', 'context_recall', 'answer_relevancy', 'context_precision']:
        vals = [x.get(k) for x in r if isinstance(x.get(k), (int, float)) and x.get(k) == x.get(k)]
        mets[k] = sum(vals) / len(vals) if vals else 0.0
    return acc, mets

# OpenAI: BM25 only (no SH), Hybrid (no SH), Hybrid + SH
data_bm25 = {}      # BM25 only, no SH
data_hybrid = {}    # Hybrid, no SH
data_hybrid_sh = {} # Hybrid + SH

for cfg in ['baseline', 'qr', 'cr']:
    p = RESULTS_BM25_DIR / f'{cfg}_openai_phase2_custom.json'
    if p.exists():
        acc, mets = load_phase2_acc(p)
        data_bm25[cfg.upper()] = acc
# QR+CR for BM25 OpenAI -> combined_openai
p = RESULTS_BM25_DIR / 'combined_openai_phase2_custom.json'
if p.exists():
    acc, _ = load_phase2_acc(p)
    data_bm25['QR+CR'] = acc

for cfg in ['baseline', 'qr', 'cr', 'qr_cr']:
    p = RESULTS_DIR / f'{cfg}_openai_phase2_custom.json'
    if p.exists():
        acc, mets = load_phase2_acc(p)
        label = cfg.upper().replace('_', '+')
        data_hybrid[label] = acc

for cfg in ['sh_baseline', 'sh_qr', 'sh_cr', 'sh_qr_cr']:
    p = RESULTS_SH_DIR / f'{cfg}_openai_phase2_custom.json'
    if p.exists():
        acc, mets = load_phase2_acc(p)
        label = cfg.replace('sh_', '').upper().replace('_', '+')
        data_hybrid_sh[label] = acc

# Retrieval metrics
with open(NB_DIR / 'BM25 Expansion' / 'retrieval_metrics.json') as f:
    retr = json.load(f)

print()
print('BM25 only (no SH):', data_bm25)
print('Hybrid (no SH)   :', data_hybrid)
print('Hybrid + SH      :', data_hybrid_sh)
print('Retrieval metrics keys:', list(retr.keys()))"""))

# --- I.1 ---
NEW_CELLS.append(md("""## I.1 Retrieval Accuracy Progressive Bar Chart

Mengikuti **Figure 3 MetaGen**: progressive boost dari baseline ke konfigurasi terbaik, label nilai di atas tiap bar, garis tren overlay, palette pastel."""))

NEW_CELLS.append(code("""# === I.1 Retrieval Accuracy Progressive Boost (mirip MetaGen Figure 3) ===

# Konfigurasi progressive: BM25-only -> Hybrid no SH -> Hybrid+SH
# Untuk konfigurasi Baseline GPT-4.1-mini saja
labels = [
    'BM25-only\\n(no SH)',
    'Hybrid\\n(no SH)',
    'Hybrid + SH\\n(Baseline)',
    'Hybrid + SH\\n+ QR',
    'Hybrid + SH\\n+ CR',
    'Hybrid + SH\\n+ QR + CR',
]
values = [
    data_bm25.get('BASELINE', 0) * 100,
    data_hybrid.get('BASELINE', 0) * 100,
    data_hybrid_sh.get('BASELINE', 0) * 100,
    data_hybrid_sh.get('QR', 0) * 100,
    data_hybrid_sh.get('CR', 0) * 100,
    data_hybrid_sh.get('QR+CR', 0) * 100,
]

fig, ax = plt.subplots(figsize=(11, 5.5))

# Bar chart dengan pastel colors
bars = ax.bar(labels, values, color=METAGEN_COLORS[:len(labels)],
              edgecolor='#333', linewidth=0.7, width=0.72)

# Garis tren overlay (line) - di tengah bar
ax.plot(range(len(labels)), values, color='#2C3E50', marker='o',
        linewidth=2, markersize=7, zorder=10, markerfacecolor='white',
        markeredgewidth=2)

# Label nilai di atas tiap bar
for i, v in enumerate(values):
    ax.annotate(f'{v:.1f}%', xy=(i, v), xytext=(0, 8),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

ax.set_ylabel('Label Accuracy (%)', fontsize=12)
ax.set_title('PubMedQA — Label Accuracy Boost with Hybrid Retrieval & Schwartz-Hearst Expansion',
             fontsize=13, pad=15, fontweight='bold')

# Zoom y-axis
y_min = min(values) - 3
y_max = max(values) + 4
ax.set_ylim(y_min, y_max)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))

# Rotate x-labels sedikit
plt.setp(ax.get_xticklabels(), rotation=12, ha='right', fontsize=10)
plt.tight_layout()
save_fig('V1_progressive_boost')
plt.show()"""))

# --- I.2 ---
NEW_CELLS.append(md("""## I.2 Pengaruh Ekspansi Akronim — Line Chart Zoomed

Mengikuti **Figure 4 MetaGen**: line chart zoomed-in dengan 2 lines (no SH vs with SH), label nilai per titik."""))

NEW_CELLS.append(code("""# === I.2 Pengaruh SH: Line Chart ala MetaGen Figure 4 ===

configs = ['Baseline', '+ QR', '+ CR', '+ QR+CR']
no_sh = [data_hybrid.get('BASELINE',0)*100, data_hybrid.get('QR',0)*100,
         data_hybrid.get('CR',0)*100, data_hybrid.get('QR+CR',0)*100]
with_sh = [data_hybrid_sh.get('BASELINE',0)*100, data_hybrid_sh.get('QR',0)*100,
           data_hybrid_sh.get('CR',0)*100, data_hybrid_sh.get('QR+CR',0)*100]

fig, ax = plt.subplots(figsize=(10, 5.5))

# Two lines: with markers
ax.plot(configs, no_sh, marker='o', linewidth=2.5, markersize=10,
        color='#7DA7C8', label='Tanpa Ekspansi Akronim',
        markerfacecolor='white', markeredgewidth=2.5)
ax.plot(configs, with_sh, marker='s', linewidth=2.5, markersize=10,
        color='#2C3E50', label='Dengan Ekspansi Akronim (Schwartz-Hearst)',
        markerfacecolor='#5DAA68', markeredgewidth=2.5)

# Value labels per titik
for i, v in enumerate(no_sh):
    ax.annotate(f'{v:.1f}%', xy=(i, v), xytext=(0, -22),
                textcoords='offset points', ha='center',
                fontsize=10, color='#5577AA')
for i, v in enumerate(with_sh):
    ax.annotate(f'{v:.1f}%', xy=(i, v), xytext=(0, 12),
                textcoords='offset points', ha='center',
                fontsize=10, fontweight='bold', color='#2C5530')

# Delta annotation (selisih)
for i, (a, b) in enumerate(zip(no_sh, with_sh)):
    delta = b - a
    ax.annotate(f'+{delta:.1f}', xy=(i, (a+b)/2), xytext=(15, 0),
                textcoords='offset points', ha='left', va='center',
                fontsize=9, color='#2C7A2C', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                          edgecolor='#5DAA68', linewidth=0.8))

ax.set_ylabel('Label Accuracy (%)', fontsize=12)
ax.set_title('PubMedQA — Pengaruh Ekspansi Akronim Schwartz-Hearst pada GPT-4.1-mini',
             fontsize=13, pad=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)

y_min = min(no_sh) - 2
y_max = max(with_sh) + 3
ax.set_ylim(y_min, y_max)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))

plt.tight_layout()
save_fig('V2_sh_impact_line')
plt.show()"""))

# --- I.3 ---
NEW_CELLS.append(md("""## I.3 Cross-LLM Comparison — Grouped Bar with Value Labels

Bar chart 4 LLM × 4 konfigurasi mitigasi, dengan value labels MetaGen-style."""))

NEW_CELLS.append(code("""# === I.3 Cross-LLM Grouped Bar (mirip MetaGen style) ===

def load_xllm(prefix_dir, prefix_pattern):
    out = {}
    for cfg in ['baseline', 'qr', 'cr', 'qr_cr']:
        for model_key, fname in [('Llama 3.2', f'{cfg}_llama32'),
                                  ('Llama 3.3 70B', f'{cfg}_llama'),
                                  ('GPT-4.1-mini', f'{cfg}_openai'),
                                  ('Claude Haiku 4.5', f'{cfg}_claude')]:
            p = prefix_dir / f'{fname}_phase2_custom.json'
            if p.exists():
                acc, _ = load_phase2_acc(p)
                out.setdefault(model_key, {})[cfg.upper().replace('_','+')] = acc * 100
    return out

xllm = load_xllm(RESULTS_DIR, '')

# Plot
models = ['Llama 3.2', 'Claude Haiku 4.5', 'Llama 3.3 70B', 'GPT-4.1-mini']
configs = ['BASELINE', 'QR', 'CR', 'QR+CR']
model_colors = ['#F5B7B1', '#FAD7A0', '#A9CCE3', '#A2D5AB']

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(configs))
width = 0.20

for i, model in enumerate(models):
    vals = [xllm.get(model, {}).get(c, 0) for c in configs]
    bars = ax.bar(x + (i - 1.5) * width, vals, width,
                  label=model, color=model_colors[i],
                  edgecolor='#333', linewidth=0.7)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.annotate(f'{v:.1f}', xy=(bar.get_x() + bar.get_width()/2, v),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=11)
ax.set_ylabel('Label Accuracy (%)', fontsize=12)
ax.set_title('PubMedQA — Perbandingan Label Accuracy Empat LLM pada Empat Konfigurasi Mitigasi',
             fontsize=13, pad=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, framealpha=0.95, ncol=2)

ax.set_ylim(45, 78)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))

plt.tight_layout()
save_fig('V3_xllm_grouped_bar')
plt.show()"""))

# --- I.4 ---
NEW_CELLS.append(md("""## I.4 Retrieval IR Metrics — MetaGen Style

Bar chart untuk 3 metrik IR (Recall@5, MAP@5, nDCG@5) original vs SH expansion."""))

NEW_CELLS.append(code("""# === I.4 IR Retrieval Metrics Boost (mirip MetaGen Figure 3) ===

ir_metrics = ['Recall@5', 'MAP@5', 'nDCG@5']
original_vals = [
    retr['original']['aggregate']['recall@5'] * 100,
    retr['original']['aggregate']['map@5'] * 100,
    retr['original']['aggregate']['ndcg@5'] * 100,
]
sh_vals = [
    retr['v2_sh']['aggregate']['recall@5'] * 100,
    retr['v2_sh']['aggregate']['map@5'] * 100,
    retr['v2_sh']['aggregate']['ndcg@5'] * 100,
]

fig, ax = plt.subplots(figsize=(9, 5.5))

x = np.arange(len(ir_metrics))
width = 0.36

bars1 = ax.bar(x - width/2, original_vals, width, label='BM25 (tanpa Ekspansi)',
               color='#A8C5DA', edgecolor='#333', linewidth=0.7)
bars2 = ax.bar(x + width/2, sh_vals, width, label='BM25 + Ekspansi Akronim (SH)',
               color='#5DAA68', edgecolor='#333', linewidth=0.7)

for bars, vals in [(bars1, original_vals), (bars2, sh_vals)]:
    for bar, v in zip(bars, vals):
        ax.annotate(f'{v:.1f}%', xy=(bar.get_x() + bar.get_width()/2, v),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

# Delta arrows + labels
for i, (a, b) in enumerate(zip(original_vals, sh_vals)):
    delta = b - a
    ax.annotate(f'+{delta:.1f}', xy=(i, b + 1.5), xytext=(0, 8),
                textcoords='offset points', ha='center',
                fontsize=9, fontweight='bold', color='#2C7A2C',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#E8F5E9',
                          edgecolor='#5DAA68', linewidth=0.8))

ax.set_xticks(x)
ax.set_xticklabels(ir_metrics, fontsize=11)
ax.set_ylabel('Skor Metrik (%)', fontsize=12)
ax.set_title('PubMedQA — Peningkatan Metrik IR dengan Ekspansi Akronim Schwartz-Hearst',
             fontsize=13, pad=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)

ax.set_ylim(70, 96)
plt.tight_layout()
save_fig('V4_ir_metrics_boost')
plt.show()"""))

# --- I.5 Tables ---
NEW_CELLS.append(md("""## I.5 MetaGen-Style LaTeX Tables

Generate tabel siap-paste ke Bab VI mengikuti format MetaGen (kompak, before/after, nilai bold pada baris terbaik)."""))

NEW_CELLS.append(code("""# === I.5 Generate LaTeX Tables ala MetaGen ===

output_tex = []

# --- Tabel A: IR Metrics Before/After SH (mirip Tabel 2 MetaGen) ---
output_tex.append(r'''% Tabel VI.1 - IR Metrics Before/After Schwartz-Hearst
\\begin{table}[H]
\\centering
\\caption{Peningkatan Metrik Retrieval dengan dan Tanpa Ekspansi Akronim Schwartz-Hearst pada Korpus PubMedQA}
\\label{tbl:retrieval-metrics}
\\begin{tabular}{|l|c|c|c|}
\\hline
\\textbf{Konfigurasi BM25} & \\textbf{Recall@5} & \\textbf{MAP@5} & \\textbf{nDCG@5} \\\\ \\hline
Tanpa Ekspansi (\\textit{baseline}) & ''' + f"{original_vals[0]:.2f}\\%" + ' & ' + f"{original_vals[1]:.2f}\\%" + ' & ' + f"{original_vals[2]:.2f}\\%" + r''' \\\\ \\hline
\\textbf{Dengan Ekspansi (SH)} & \\textbf{''' + f"{sh_vals[0]:.2f}\\%" + r'''} & \\textbf{''' + f"{sh_vals[1]:.2f}\\%" + r'''} & \\textbf{''' + f"{sh_vals[2]:.2f}\\%" + r'''} \\\\ \\hline
\\textit{Peningkatan absolut} & ''' + f"+{sh_vals[0]-original_vals[0]:.2f}" + ' & ' + f"+{sh_vals[1]-original_vals[1]:.2f}" + ' & ' + f"+{sh_vals[2]-original_vals[2]:.2f}" + r''' \\\\ \\hline
\\end{tabular}
\\end{table}
''')

# --- Tabel B: BM25 vs Hybrid (progressive) ---
output_tex.append(r'''% Tabel VI.2 - BM25 vs Hybrid Progressive Boost
\\begin{table}[H]
\\centering
\\caption{Peningkatan \\textit{Label Accuracy} dari BM25 ke \\textit{Hybrid Retrieval} hingga Penambahan Ekspansi Akronim pada GPT-4.1-mini}
\\label{tbl:bm25-vs-hybrid-progressive}
\\begin{tabular}{|l|c|}
\\hline
\\textbf{Konfigurasi} & \\textbf{Label Accuracy (\\%)} \\\\ \\hline
BM25-only, tanpa Ekspansi (\\textit{baseline}) & ''' + f"{data_bm25.get('BASELINE',0)*100:.1f}" + r''' \\\\ \\hline
\\textit{Hybrid} (BM25 + \\textit{Dense}), tanpa Ekspansi & ''' + f"{data_hybrid.get('BASELINE',0)*100:.1f}" + r''' \\\\ \\hline
\\textit{Hybrid} + Ekspansi Akronim (SH) & ''' + f"{data_hybrid_sh.get('BASELINE',0)*100:.1f}" + r''' \\\\ \\hline
\\textit{Hybrid} + SH + QR & ''' + f"{data_hybrid_sh.get('QR',0)*100:.1f}" + r''' \\\\ \\hline
\\textit{Hybrid} + SH + CR & ''' + f"{data_hybrid_sh.get('CR',0)*100:.1f}" + r''' \\\\ \\hline
\\textbf{\\textit{Hybrid} + SH + QR + CR} & \\textbf{''' + f"{data_hybrid_sh.get('QR+CR',0)*100:.1f}" + r'''} \\\\ \\hline
\\end{tabular}
\\end{table}
''')

# --- Tabel C: BM25 vs Hybrid per Teknik ---
output_tex.append(r'''% Tabel VI.3 - BM25 vs Hybrid per Technique
\\begin{table}[H]
\\centering
\\caption{Perbandingan \\textit{Label Accuracy} antara BM25 Saja dan \\textit{Hybrid Retrieval} per Konfigurasi Mitigasi pada GPT-4.1-mini}
\\label{tbl:bm25-vs-hybrid}
\\begin{tabular}{|l|c|c|c|}
\\hline
\\textbf{Konfigurasi} & \\textbf{BM25 saja} & \\textbf{Hybrid} & \\textbf{\\textit{Delta}} \\\\ \\hline''')
for cfg, label in [('BASELINE','\\textit{Baseline}'),('QR','+ QR'),('CR','+ CR'),('QR+CR','+ QR + CR')]:
    bm = data_bm25.get(cfg, 0) * 100
    hy = data_hybrid.get(cfg, 0) * 100
    output_tex.append(f"{label} & {bm:.1f}\\% & {hy:.1f}\\% & +{hy-bm:.1f} \\\\\\\\ \\\\hline")
output_tex.append(r'''\\end{tabular}
\\end{table}
''')

# --- Tabel D: Cross-LLM ---
output_tex.append(r'''% Tabel VI.4 - Cross-LLM Comparison
\\begin{table}[H]
\\centering
\\caption{Perbandingan \\textit{Label Accuracy} Lintas-LLM pada Empat Konfigurasi Mitigasi (Mode \\textit{Hybrid}, Tanpa Ekspansi Akronim)}
\\label{tbl:xllm-comparison}
\\begin{tabular}{|l|c|c|c|c|}
\\hline
\\textbf{Konfigurasi} & \\textbf{Llama 3.2 3B} & \\textbf{Claude Haiku 4.5} & \\textbf{Llama 3.3 70B} & \\textbf{GPT-4.1-mini} \\\\ \\hline''')
for c in ['BASELINE', 'QR', 'CR', 'QR+CR']:
    label = '\\textit{Baseline}' if c == 'BASELINE' else f'+ {c}'
    row = f"{label}"
    for m in ['Llama 3.2', 'Claude Haiku 4.5', 'Llama 3.3 70B', 'GPT-4.1-mini']:
        v = xllm.get(m, {}).get(c, 0)
        row += f" & {v:.1f}\\%"
    row += r" \\\\ \\hline"
    output_tex.append(row)
# Average row
avg_row = '\\textbf{Rata-rata}'
for m in ['Llama 3.2', 'Claude Haiku 4.5', 'Llama 3.3 70B', 'GPT-4.1-mini']:
    vals = [xllm.get(m, {}).get(c, 0) for c in ['BASELINE','QR','CR','QR+CR'] if xllm.get(m, {}).get(c, 0) > 0]
    avg = sum(vals)/len(vals) if vals else 0
    avg_row += f" & \\textbf{{{avg:.1f}\\%}}"
avg_row += r" \\\\ \\hline"
output_tex.append(avg_row)
output_tex.append(r'''\\end{tabular}
\\end{table}
''')

# === Write & print ===
tex_file = FIG_DIR / 'V_tables_metagen_style.tex'
with open(tex_file, 'w', encoding='utf-8') as f:
    f.write('\\n'.join(output_tex))

print(f'LaTeX tables saved to: {tex_file}')
print()
print('=' * 80)
print('PREVIEW (paste ke Bab VI):')
print('=' * 80)
print('\\n'.join(output_tex)[:3000])"""))

# === Append to notebook ===
with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

original_count = len(nb['cells'])
nb['cells'].extend(NEW_CELLS)
print(f'Original cells: {original_count}')
print(f'Adding {len(NEW_CELLS)} new cells (Section I MetaGen-style)')
print(f'Final cells   : {len(nb["cells"])}')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'\nNotebook updated: {NB_PATH.name}')
