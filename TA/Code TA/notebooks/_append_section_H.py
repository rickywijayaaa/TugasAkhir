"""Append section H (Llama 3.2 Hybrid + cross-LLM comparison) to notebook 07.

Run once: python _append_section_H.py
"""
import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).parent / '07 - Visualisasi Perbandingan Konfigurasi.ipynb'


def md(source: str) -> dict:
    return {
        'cell_type': 'markdown',
        'id': str(uuid.uuid4())[:8],
        'metadata': {},
        'source': source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        'cell_type': 'code',
        'id': str(uuid.uuid4())[:8],
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': source.splitlines(keepends=True),
    }


# ============================================================
# CELLS
# ============================================================

H_INTRO = md('''---

# H. Llama 3.2 Hybrid + Perbandingan Cross-LLM

Setelah eksperimen Llama 3.2 dengan retriever **Hybrid (BM25 + Dense via RRF)** selesai untuk seluruh 4 method (Baseline, QR, CR, QR+CR), section ini fokus pada:

- **H.1** Analisis Llama 3.2 Hybrid standalone (label accuracy + per-label + 4 metrik RAGAS)
- **H.2** Perbandingan cross-LLM Hybrid: label accuracy & balanced accuracy (Llama 3.2 vs Llama 3.3 70B vs GPT-4.1-mini vs Claude Haiku 4.5)
- **H.3** Perbandingan cross-LLM Hybrid: 4 metrik RAGAS via heatmap
- **H.4** Deep-dive akurasi kelas "maybe" cross-LLM (kelas paling sulit untuk semua model)
- **H.5** Ringkasan temuan

Semua perbandingan **apple-to-apple** karena pakai retriever yang sama (Hybrid BM25+Dense+RRF). Variabel yang berubah: LLM dan technique.
''')

H0_LOAD = code('''# H.0 - Load Llama 3.2 Hybrid (4 method) + extend Claude jadi 4 method

# === Llama 3.2 Hybrid (4 method baru) ===
LLAMA32H_CONFIGS = OrderedDict([
    ('BL_Llama32H',    (RESULTS_DIR / 'baseline_llama32_phase1_answers.json',
                         RESULTS_DIR / 'baseline_llama32_phase2_custom.json',
                         'Llama 3.2', 'Hybrid', 'Baseline')),
    ('QR_Llama32H',    (RESULTS_DIR / 'qr_llama32_phase1_answers.json',
                         RESULTS_DIR / 'qr_llama32_phase2_custom.json',
                         'Llama 3.2', 'Hybrid', 'QR')),
    ('CR_Llama32H',    (RESULTS_DIR / 'cr_llama32_phase1_answers.json',
                         RESULTS_DIR / 'cr_llama32_phase2_custom.json',
                         'Llama 3.2', 'Hybrid', 'CR')),
    ('QRCR_Llama32H',  (RESULTS_DIR / 'qr_cr_llama32_phase1_answers.json',
                         RESULTS_DIR / 'qr_cr_llama32_phase2_custom.json',
                         'Llama 3.2', 'Hybrid', 'QR+CR')),
])

# === Claude full (3 method tambahan untuk lengkapi 4 method) ===
CLAUDE_EXTRA_CONFIGS = OrderedDict([
    ('QR_Claude',    (RESULTS_DIR / 'qr_claude_phase1_answers.json',
                       RESULTS_DIR / 'qr_claude_phase2_custom.json',
                       'Claude Haiku 4.5', 'Hybrid', 'QR')),
    ('CR_Claude',    (RESULTS_DIR / 'cr_claude_phase1_answers.json',
                       RESULTS_DIR / 'cr_claude_phase2_custom.json',
                       'Claude Haiku 4.5', 'Hybrid', 'CR')),
    ('QRCR_Claude',  (RESULTS_DIR / 'qr_cr_claude_phase1_answers.json',
                       RESULTS_DIR / 'qr_cr_claude_phase2_custom.json',
                       'Claude Haiku 4.5', 'Hybrid', 'QR+CR')),
])

extra_records = []
for cfg_dict in [LLAMA32H_CONFIGS, CLAUDE_EXTRA_CONFIGS]:
    for key, (p1, p2, model, retr, tech) in cfg_dict.items():
        try:
            data = load_config(p1, p2)
            extra_records.append({
                'key': key, 'model': model, 'retrieval': retr,
                'technique': tech, **data,
            })
        except Exception as e:
            print(f'  SKIP {key}: {e}')

df_extra = pd.DataFrame(extra_records)
for lbl in ['yes', 'no', 'maybe']:
    df_extra[f'{lbl}_correct']  = df_extra['per_label'].apply(lambda x: x[lbl]['correct'])
    df_extra[f'{lbl}_total']    = df_extra['per_label'].apply(lambda x: x[lbl]['total'])
    df_extra[f'{lbl}_accuracy'] = df_extra['per_label'].apply(lambda x: x[lbl]['accuracy'])

# Gabung ke df utama -> df_xllm (16 konfigurasi: 4 LLMs x 4 methods)
df_xllm = pd.concat([df, df_extra], ignore_index=True)
df_xllm['balanced_accuracy'] = df_xllm[['yes_accuracy', 'no_accuracy', 'maybe_accuracy']].mean(axis=1)

# Sorting konsisten
xllm_model_order = {'Llama 3.2': 0, 'Llama 3.3 70B': 1, 'GPT-4.1-mini': 2, 'Claude Haiku 4.5': 3}
xllm_tech_order  = {'Baseline': 0, 'QR': 1, 'CR': 2, 'QR+CR': 3}
df_xllm['_m'] = df_xllm['model'].map(xllm_model_order)
df_xllm['_t'] = df_xllm['technique'].map(xllm_tech_order)
df_xllm = df_xllm.sort_values(['_m', '_t']).drop(columns=['_m', '_t']).reset_index(drop=True)

# Subset Llama 3.2 untuk analisis standalone
df_llama32h = df_xllm[df_xllm['model'] == 'Llama 3.2'].reset_index(drop=True)

# Color map konsisten
LLM_COLORS = {
    'Llama 3.2'        : '#e67e22',  # oranye
    'Llama 3.3 70B'    : '#c0392b',  # merah
    'GPT-4.1-mini'     : '#1f4e79',  # biru tua
    'Claude Haiku 4.5' : '#117a65',  # hijau tua
}

print(f'Total konfigurasi cross-LLM Hybrid: {len(df_xllm)}')
print(df_xllm[['model', 'technique', 'accuracy', 'balanced_accuracy',
               'faithfulness', 'maybe_accuracy']].round(4).to_string(index=False))
''')

H1_INTRO = md('''## H.1 Llama 3.2 Hybrid Standalone

Visualisasi 4 method (Baseline, QR, CR, QR+CR) untuk Llama 3.2 dengan retriever Hybrid:
- **Figure H.1a**: Label Accuracy (overall + balanced) dan per-label accuracy (yes / no / maybe)
- **Figure H.1b**: 4 metrik RAGAS (Faithfulness, Context Recall, Answer Relevancy, Context Precision)

Tiga hal yang patut diperhatikan:
1. Akurasi kelas "maybe" sangat rendah di Baseline (anti-maybe bias)
2. QR memberi pergeseran prediksi paling besar
3. Faithfulness tetap rendah (~0.6) dibanding LLM lain (~0.9), refleksi kapabilitas model
''')

H1A_ACCURACY = code('''# H.1a - Llama 3.2 Hybrid: Overall + Balanced + Per-label Accuracy

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# === Subplot kiri: Overall + Balanced accuracy ===
ax = axes[0]
techniques = df_llama32h['technique'].tolist()
x = np.arange(len(techniques))
width = 0.36

bars_overall = ax.bar(x - width/2, df_llama32h['accuracy'] * 100, width,
                      label='Overall Accuracy', color='#e67e22', edgecolor='white')
bars_balanced = ax.bar(x + width/2, df_llama32h['balanced_accuracy'] * 100, width,
                       label='Balanced Accuracy', color='#d4ac0d', edgecolor='white')

for bars in [bars_overall, bars_balanced]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.6, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(techniques, fontsize=10.5, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Overall vs Balanced Accuracy\\n(Llama 3.2 Hybrid, n=500)',
             fontsize=11.5, fontweight='bold')
y_max = max(df_llama32h['accuracy'].max(), df_llama32h['balanced_accuracy'].max()) * 100
ax.set_ylim(0, y_max + 12)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(loc='upper left', fontsize=9.5)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# === Subplot kanan: Per-label accuracy ===
ax = axes[1]
width = 0.25
colors_lbl = {'yes': '#2f6b4f', 'no': '#b94a3b', 'maybe': '#d4ac0d'}

bars_y = ax.bar(x - width, df_llama32h['yes_accuracy']   * 100, width,
                label='yes',   color=colors_lbl['yes'])
bars_n = ax.bar(x,         df_llama32h['no_accuracy']    * 100, width,
                label='no',    color=colors_lbl['no'])
bars_m = ax.bar(x + width, df_llama32h['maybe_accuracy'] * 100, width,
                label='maybe', color=colors_lbl['maybe'])

for bars in [bars_y, bars_n, bars_m]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.0f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(techniques, fontsize=10.5, fontweight='bold')
ax.set_ylabel('Per-label Accuracy (%)', fontsize=11)
ax.set_title('Per-label Accuracy (yes / no / maybe)\\n(Llama 3.2 Hybrid, n=500)',
             fontsize=11.5, fontweight='bold')
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(loc='upper right', fontsize=9.5)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'H1a_llama32_hybrid_accuracy.png', dpi=200, bbox_inches='tight')
plt.show()
print(f'[Saved] {FIGURES_DIR}/H1a_llama32_hybrid_accuracy.png')
''')

H1B_RAGAS = code('''# H.1b - Llama 3.2 Hybrid: 4 RAGAS Metrics (grouped bar)

fig, ax = plt.subplots(figsize=(13, 5.5))

metrics = ['faithfulness', 'context_recall', 'answer_relevancy', 'context_precision']
metric_labels = ['Faithfulness', 'Context Recall', 'Answer Relevancy', 'Context Precision']
metric_colors = ['#1f4e79', '#117a65', '#b9770e', '#7d3c98']

n_techs   = len(df_llama32h)
n_metrics = len(metrics)
group_width = 0.84
bar_width   = group_width / n_metrics
x = np.arange(n_techs)

for i, (m, lbl, col) in enumerate(zip(metrics, metric_labels, metric_colors)):
    offset = (i - n_metrics/2 + 0.5) * bar_width
    bars = ax.bar(x + offset, df_llama32h[m] * 100, bar_width,
                  label=lbl, color=col, edgecolor='white', linewidth=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.7, f'{h:.1f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(df_llama32h['technique'].tolist(), fontsize=11, fontweight='bold')
ax.set_ylabel('Metric Value (%)', fontsize=11)
ax.set_title('Llama 3.2 Hybrid: 4 Metrik RAGAS per Method (n=500)',
             fontsize=12.5, fontweight='bold', pad=12)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(loc='lower right', fontsize=9.5, ncol=2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'H1b_llama32_hybrid_ragas.png', dpi=200, bbox_inches='tight')
plt.show()
print(f'[Saved] {FIGURES_DIR}/H1b_llama32_hybrid_ragas.png')
''')

H2_INTRO = md('''## H.2 Cross-LLM Hybrid: Label Accuracy

Bandingkan label accuracy 4 LLM (Llama 3.2, Llama 3.3 70B, GPT-4.1-mini, Claude Haiku 4.5) di 4 method (Baseline, QR, CR, QR+CR). Semua pakai retriever Hybrid.

- **Panel atas**: Overall Accuracy
- **Panel bawah**: Balanced Accuracy (rata-rata akurasi 3 kelas, lebih adil saat class imbalanced)

Perhatikan **Llama 3.2 (oranye) selalu di bawah** — model size effect. Tapi pola dampak QR/CR berbeda per LLM.
''')

H2_XLLM_ACC = code('''# H.2 - Cross-LLM Hybrid: Overall + Balanced Accuracy (2 panel)

fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

models = ['Llama 3.2', 'Llama 3.3 70B', 'GPT-4.1-mini', 'Claude Haiku 4.5']
techniques = ['Baseline', 'QR', 'CR', 'QR+CR']
n_models = len(models)
n_tech   = len(techniques)
group_width = 0.82
bar_width   = group_width / n_models
x = np.arange(n_tech)

for ax_idx, (metric, label) in enumerate(zip(['accuracy', 'balanced_accuracy'],
                                             ['Overall Accuracy', 'Balanced Accuracy'])):
    ax = axes[ax_idx]
    for i, m in enumerate(models):
        sub = df_xllm[df_xllm['model'] == m].set_index('technique').reindex(techniques)
        offset = (i - n_models/2 + 0.5) * bar_width
        vals   = sub[metric].fillna(0).values * 100
        bars   = ax.bar(x + offset, vals, bar_width,
                        label=m, color=LLM_COLORS[m], edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.6, f'{v:.1f}',
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(techniques, fontsize=11.5, fontweight='bold')
    ax.set_ylabel(f'{label} (%)', fontsize=11)
    ax.set_title(f'{label} per Method (4 LLMs, Hybrid Retriever, n=500)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 90)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    if ax_idx == 0:
        ax.legend(loc='upper right', fontsize=9.5, ncol=4, framealpha=0.92)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'H2_xllm_accuracy.png', dpi=200, bbox_inches='tight')
plt.show()
print(f'[Saved] {FIGURES_DIR}/H2_xllm_accuracy.png')
''')

H3_INTRO = md('''## H.3 Cross-LLM Hybrid: 4 Metrik RAGAS via Heatmap

Empat heatmap (satu per metrik RAGAS), masing-masing menampilkan **LLM (baris) x Method (kolom)**. Skala warna: hijau = lebih baik, merah = lebih buruk.

Perhatikan kontras:
- **Antar baris** menunjukkan kapabilitas LLM (Faithfulness Llama 3.2 jauh di bawah Claude/OpenAI)
- **Antar kolom** menunjukkan efektivitas method (CR cenderung memberi gain konsisten)
''')

H3_XLLM_RAGAS = code('''# H.3 - Cross-LLM Hybrid: 4 RAGAS Heatmaps (LLM x Method)

metrics = ['faithfulness', 'context_recall', 'answer_relevancy', 'context_precision']
metric_labels = ['Faithfulness', 'Context Recall', 'Answer Relevancy', 'Context Precision']
models = ['Llama 3.2', 'Llama 3.3 70B', 'GPT-4.1-mini', 'Claude Haiku 4.5']
techniques = ['Baseline', 'QR', 'CR', 'QR+CR']

fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

for ax, metric, label in zip(axes, metrics, metric_labels):
    pivot = df_xllm.pivot(index='model', columns='technique', values=metric)
    pivot = pivot.reindex(index=models, columns=techniques)
    mat = pivot.values

    im = ax.imshow(mat, cmap='RdYlGn', vmin=0.4, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(techniques)))
    ax.set_xticklabels(techniques, fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9.5)
    ax.set_title(label, fontsize=12, fontweight='bold', pad=10)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                color = 'white' if v < 0.55 or v > 0.88 else 'black'
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        color=color, fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle('Cross-LLM Hybrid: 4 Metrik RAGAS (LLM x Method, n=500)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'H3_xllm_ragas_heatmap.png', dpi=200, bbox_inches='tight')
plt.show()
print(f'[Saved] {FIGURES_DIR}/H3_xllm_ragas_heatmap.png')
''')

H4_INTRO = md('''## H.4 Deep-dive: Akurasi Kelas "Maybe" Cross-LLM

Kelas **maybe** adalah yang paling sulit untuk semua LLM. Section ini fokus khusus pada akurasi kelas maybe dan bagaimana method mitigasi (QR, CR, QR+CR) memengaruhinya per LLM.

Insight kritis: dampak QR / CR ke kelas maybe **berbeda drastis per LLM** — sinyal bahwa pemilihan method harus mempertimbangkan model yang dipakai.

- **Panel kiri**: Maybe accuracy absolut per LLM x Method
- **Panel kanan**: Delta maybe accuracy vs Baseline (per LLM) — menunjukkan compensation effect QR untuk model lemah
''')

H4_MAYBE = code('''# H.4 - Maybe-class accuracy cross-LLM (2 panel)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

models = ['Llama 3.2', 'Llama 3.3 70B', 'GPT-4.1-mini', 'Claude Haiku 4.5']
techniques = ['Baseline', 'QR', 'CR', 'QR+CR']

# === Subplot kiri: Maybe accuracy absolut grouped bar ===
ax = axes[0]
n_models = len(models); n_tech = len(techniques)
group_width = 0.82
bar_width   = group_width / n_models
x = np.arange(n_tech)

for i, m in enumerate(models):
    sub = df_xllm[df_xllm['model'] == m].set_index('technique').reindex(techniques)
    offset = (i - n_models/2 + 0.5) * bar_width
    vals   = sub['maybe_accuracy'].fillna(0).values * 100
    bars   = ax.bar(x + offset, vals, bar_width,
                    label=m, color=LLM_COLORS[m], edgecolor='white', linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.7, f'{v:.1f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels(techniques, fontsize=11, fontweight='bold')
ax.set_ylabel('Maybe Accuracy (%)', fontsize=11)
ax.set_title('Akurasi Kelas "Maybe" per Method (4 LLMs)',
             fontsize=12, fontweight='bold')
y_top = max(df_xllm['maybe_accuracy'].max() * 100, 10) + 8
ax.set_ylim(0, y_top)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(loc='upper left', fontsize=9, ncol=2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# === Subplot kanan: Delta maybe vs Baseline (per LLM) ===
ax = axes[1]
deltas = []
for m in models:
    sub = df_xllm[df_xllm['model'] == m].set_index('technique').reindex(techniques)
    if 'Baseline' not in sub.index or pd.isna(sub.loc['Baseline', 'maybe_accuracy']):
        continue
    base = sub.loc['Baseline', 'maybe_accuracy']
    for t in ['QR', 'CR', 'QR+CR']:
        if t in sub.index and not pd.isna(sub.loc[t, 'maybe_accuracy']):
            deltas.append({'model': m, 'technique': t,
                           'delta_pp': (sub.loc[t, 'maybe_accuracy'] - base) * 100})
df_delta = pd.DataFrame(deltas)

bar_w = 0.27
x_models = np.arange(len(models))
tech_colors = {'QR': '#3498db', 'CR': '#9b59b6', 'QR+CR': '#16a085'}

for i, t in enumerate(['QR', 'CR', 'QR+CR']):
    offset = (i - 1) * bar_w
    sub = df_delta[df_delta['technique'] == t].set_index('model').reindex(models)
    vals = sub['delta_pp'].fillna(0).values
    bars = ax.bar(x_models + offset, vals, bar_w,
                  label=t, color=tech_colors[t], edgecolor='white', linewidth=0.6)
    for bar, v in zip(bars, vals):
        if v != 0:
            va = 'bottom' if v > 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (0.7 if v > 0 else -0.7),
                    f'{v:+.1f}', ha='center', va=va, fontsize=8.5, fontweight='bold')

ax.axhline(y=0, color='black', linewidth=0.7)
ax.set_xticks(x_models); ax.set_xticklabels(models, fontsize=10, rotation=10)
ax.set_ylabel(r'$\\Delta$ Maybe Accuracy vs Baseline (pp)', fontsize=11)
ax.set_title('Dampak Mitigasi Method ke Kelas "Maybe" per LLM',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(loc='upper right', fontsize=9.5)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'H4_maybe_recall.png', dpi=200, bbox_inches='tight')
plt.show()
print(f'[Saved] {FIGURES_DIR}/H4_maybe_recall.png')
''')

H5_INTRO = md('''## H.5 Ringkasan Temuan

Cell di bawah memproduksi tabel ringkasan dan insight otomatis untuk seluruh 16 konfigurasi cross-LLM.
''')

H5_SUMMARY = code('''# H.5 - Auto-generated summary insights

print('=' * 80)
print('RINGKASAN TEMUAN H - Llama 3.2 Hybrid + Cross-LLM Comparison')
print('=' * 80)

# Llama 3.2 standalone
print('\\n[Llama 3.2 Hybrid Standalone]')
for _, r in df_llama32h.iterrows():
    print(f'  {r["technique"]:<10} acc={r["accuracy"]*100:>5.1f}%  '
          f'bal={r["balanced_accuracy"]*100:>5.1f}%  '
          f'maybe={r["maybe_accuracy"]*100:>5.1f}%  '
          f'faith={r["faithfulness"]*100:>5.1f}%')
best_l32 = df_llama32h.sort_values('balanced_accuracy', ascending=False).iloc[0]
print(f'  -> Best (balanced): {best_l32["technique"]} '
      f'({best_l32["balanced_accuracy"]*100:.1f}%)')

# Cross-LLM best per metric
print('\\n[Best Config per Metric (Cross-LLM Hybrid, 16 configs)]')
for metric, label in [('accuracy', 'Overall Accuracy'),
                      ('balanced_accuracy', 'Balanced Accuracy'),
                      ('faithfulness', 'Faithfulness'),
                      ('maybe_accuracy', 'Maybe Accuracy')]:
    best = df_xllm.sort_values(metric, ascending=False).iloc[0]
    print(f'  {label:<22}: {best["technique"]:>8} x {best["model"]:<18}'
          f' = {best[metric]*100:>5.1f}%')

# QR effect per LLM (compensation hypothesis check)
print('\\n[Delta Balanced Accuracy dari QR vs Baseline per LLM]')
print('  (Hipotesis: QR helps weaker models more)')
for m in ['Llama 3.2', 'Llama 3.3 70B', 'GPT-4.1-mini', 'Claude Haiku 4.5']:
    sub = df_xllm[df_xllm['model'] == m].set_index('technique')
    if 'Baseline' in sub.index and 'QR' in sub.index:
        d = (sub.loc['QR', 'balanced_accuracy'] - sub.loc['Baseline', 'balanced_accuracy']) * 100
        sign = 'helps' if d > 1 else ('hurts' if d < -1 else 'neutral')
        print(f'  {m:<18}: {d:+5.1f} pp  [{sign}]')

# CR effect per LLM
print('\\n[Delta Balanced Accuracy dari CR vs Baseline per LLM]')
for m in ['Llama 3.2', 'Llama 3.3 70B', 'GPT-4.1-mini', 'Claude Haiku 4.5']:
    sub = df_xllm[df_xllm['model'] == m].set_index('technique')
    if 'Baseline' in sub.index and 'CR' in sub.index:
        d = (sub.loc['CR', 'balanced_accuracy'] - sub.loc['Baseline', 'balanced_accuracy']) * 100
        sign = 'helps' if d > 1 else ('hurts' if d < -1 else 'neutral')
        print(f'  {m:<18}: {d:+5.1f} pp  [{sign}]')

print('\\n[Insight kritis untuk laporan]')
print('  1. Llama 3.2 baseline akurasi terendah di antara 4 LLM (model size effect).')
print('  2. Anti-maybe bias paling ekstrem di Llama 3.2 (maybe acc dekat 0%).')
print('  3. QR memberi gain terbesar di Llama 3.2 - "compensation mechanism" untuk model lemah.')
print('  4. CR konsisten bantu di Llama 3.3 / OpenAI / Claude tapi minimal di Llama 3.2.')
print('  5. Manfaat method mitigasi BERBANDING TERBALIK dengan kapabilitas LLM.')
''')

# ============================================================
# APPEND CELLS TO NOTEBOOK
# ============================================================
new_cells = [
    H_INTRO,
    H0_LOAD,
    H1_INTRO,
    H1A_ACCURACY,
    H1B_RAGAS,
    H2_INTRO,
    H2_XLLM_ACC,
    H3_INTRO,
    H3_XLLM_RAGAS,
    H4_INTRO,
    H4_MAYBE,
    H5_INTRO,
    H5_SUMMARY,
]

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cek apakah section H sudah ada (avoid duplicate)
existing = ''.join(
    ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    for c in nb['cells']
)
if '# H. Llama 3.2 Hybrid' in existing:
    print('Section H sudah ada di notebook. Skip append untuk hindari duplikat.')
    print('Hapus section H lama dulu kalau mau re-generate.')
else:
    nb['cells'].extend(new_cells)
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Appended {len(new_cells)} cells to {NB_PATH.name}')
    print(f'Total cells now: {len(nb["cells"])}')
