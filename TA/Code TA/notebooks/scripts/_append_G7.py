"""Append G.7 markdown + code cell to notebook 06."""
from pathlib import Path
import nbformat as nbf

NB_PATH = Path("06 - Visualisasi Perbandingan Konfigurasi.ipynb")

nb = nbf.read(NB_PATH, as_version=4)

# ─── Markdown cell: section header ──────────────────────────
md_source = """---

## G.7 Perbandingan Label Accuracy — Termasuk Llama 3.2 sebagai Baseline Historis

**Insight utama:** apakah Llama 3.2 (model lebih kecil, retriever BM25-only) signifikan tertinggal dari Llama 3.3 70B (Hybrid)?

Visualisasi ini memperluas chart sebelumnya dengan menambahkan **Llama 3.2** sebagai konfigurasi historis. Penting dicatat:

- **Llama 3.2** pakai retriever **BM25-only** (3 teknik tersedia: Baseline, QR, CR — tanpa QR+CR)
- **Llama 3.3 70B / GPT-4.1-mini / Claude Haiku 4.5** pakai retriever **Hybrid (BM25+Dense via RRF)**

Karena retriever berbeda, perbandingan vs Llama 3.2 bersifat **apple-to-orange** (model + retriever beda). Disclaimer ini wajib disebut saat menginterpretasi gap accuracy."""

# ─── Code cell: load Llama 3.2 + plot ───────────────────────
code_source = '''# G.7 — Tambah Llama 3.2 (BM25) ke perbandingan accuracy
# Llama 3.2 cuma punya 3 teknik di BM25 (Baseline, QR, CR — tanpa QR+CR)

LLAMA32_CONFIGS = OrderedDict([
    ('BL_Llama32', (BM25_DIR / 'baseline_phase1_answers.json',
                     BM25_DIR / 'baseline_phase2_custom.json',
                     'Llama 3.2', 'BM25', 'Baseline')),
    ('QR_Llama32', (BM25_DIR / 'qr_phase1_answers.json',
                     BM25_DIR / 'qr_phase2_custom.json',
                     'Llama 3.2', 'BM25', 'QR')),
    ('CR_Llama32', (BM25_DIR / 'cr_phase1_answers.json',
                     BM25_DIR / 'cr_phase2_custom.json',
                     'Llama 3.2', 'BM25', 'CR')),
])

llama32_records = []
for key, (p1, p2, model, retrieval, technique) in LLAMA32_CONFIGS.items():
    try:
        data = load_config(p1, p2)
        llama32_records.append({
            'key'       : key,
            'model'     : model,
            'retrieval' : retrieval,
            'technique' : technique,
            **data,
        })
    except Exception as e:
        print(f'  SKIP {key}: {e}')

df_llama32 = pd.DataFrame(llama32_records)

# Flatten per_label sama seperti df utama
for lbl in ['yes', 'no', 'maybe']:
    df_llama32[f'{lbl}_correct']  = df_llama32['per_label'].apply(lambda x: x[lbl]['correct'])
    df_llama32[f'{lbl}_total']    = df_llama32['per_label'].apply(lambda x: x[lbl]['total'])
    df_llama32[f'{lbl}_accuracy'] = df_llama32['per_label'].apply(lambda x: x[lbl]['accuracy'])

# Gabung df utama (9 Hybrid configs) + Llama 3.2 (3 BM25 configs)
df_g7 = pd.concat([df, df_llama32], ignore_index=True)

# Sort: Llama 3.2 -> Llama 3.3 -> OpenAI -> Claude
model_order_g7 = {'Llama 3.2': 0, 'Llama 3.3 70B': 1, 'GPT-4.1-mini': 2, 'Claude Haiku 4.5': 3}
tech_order_g7  = {'Baseline': 0, 'QR': 1, 'CR': 2, 'QR+CR': 3}
df_g7['_m']    = df_g7['model'].map(model_order_g7)
df_g7['_t']    = df_g7['technique'].map(tech_order_g7)
df_g7          = df_g7.sort_values(['_m', '_t']).drop(columns=['_m', '_t']).reset_index(drop=True)

print(f'Loaded {len(df_g7)} konfigurasi total ({len(df_llama32)} Llama 3.2 + {len(df)} existing)')
print(df_g7[['model', 'retrieval', 'technique', 'accuracy']].round(4).to_string(index=False))

# ============================================================
# Plot grouped bar (4 LLM x teknik tersedia)
# ============================================================
fig, ax = plt.subplots(figsize=(18, 6.8))

color_map = {
    'Llama 3.2'        : '#e67e22',  # oranye (Llama family, lebih muda)
    'Llama 3.3 70B'    : '#c0392b',  # merah
    'GPT-4.1-mini'     : '#1f4e79',  # biru tua
    'Claude Haiku 4.5' : '#117a65',  # hijau tua
}
models = ['Llama 3.2', 'Llama 3.3 70B', 'GPT-4.1-mini', 'Claude Haiku 4.5']

df_g7['x_label'] = df_g7['technique']
df_g7['color']   = df_g7['model'].map(color_map)

x_pos = np.arange(len(df_g7))
bars  = ax.bar(x_pos, df_g7['accuracy'] * 100, color=df_g7['color'],
               edgecolor='white', linewidth=1.2, width=0.72)

# Anotasi nilai akurasi + correct/n
for i, bar in enumerate(bars):
    acc    = df_g7.loc[i, 'accuracy']
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.7,
            f'{acc*100:.1f}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2, height - 3,
            f"{int(df_g7.loc[i, 'correct'])}/{int(df_g7.loc[i, 'n'])}",
            ha='center', va='top', fontsize=7.5, color='white')

ax.set_xticks(x_pos)
ax.set_xticklabels(df_g7['x_label'], fontsize=10, fontweight='bold')
ax.set_ylabel('Label Accuracy (%)', fontsize=11.5)
ax.set_title('G.7 Perbandingan Label Accuracy - 4 LLM x Teknik Mitigasi (n=500 PubMedQA)\\n'
             '(Llama 3.2 pakai BM25-only; Llama 3.3 / GPT-4.1-mini / Claude pakai Hybrid BM25+Dense)',
             fontsize=12.5, pad=14)
ax.set_ylim(0, 92)
ax.grid(axis='y', alpha=0.3)

# Pemisah vertikal antar group model
group_sizes = [(df_g7['model'] == m).sum() for m in models]
boundaries  = np.cumsum(group_sizes)[:-1]
for b in boundaries:
    ax.axvline(x=b - 0.5, color='gray', linestyle='--', alpha=0.6, linewidth=1)

# Label nama model di atas tiap group
group_y = 88
starts  = np.concatenate(([0], np.cumsum(group_sizes)[:-1]))
for m, start, size in zip(models, starts, group_sizes):
    if size == 0:
        continue
    # Tambah suffix retriever supaya jelas
    suffix = ' (BM25)' if m == 'Llama 3.2' else ' (Hybrid)'
    center = start + (size - 1) / 2
    ax.text(center, group_y, m + suffix,
            ha='center', fontsize=11, fontweight='bold', color=color_map[m],
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor=color_map[m], linewidth=1.4))

# Highlight bar TERBAIK
best_idx = df_g7['accuracy'].idxmax()
best_row = df_g7.loc[best_idx]
best_y   = best_row['accuracy'] * 100
bars[best_idx].set_edgecolor('#f39c12')
bars[best_idx].set_linewidth(2.4)

ax.annotate(r'$\\bigstar$ TERBAIK',
            xy=(best_idx, best_y + 0.8),
            xytext=(best_idx, best_y + 8),
            ha='center', fontsize=11.5, fontweight='bold', color='#b9770e',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef5e7',
                      edgecolor='#f39c12', linewidth=1.8),
            arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'G7_accuracy_4llm_with_llama32.png', dpi=200, bbox_inches='tight')
plt.show()

print(f'\\n[Saved] {FIGURES_DIR}/G7_accuracy_4llm_with_llama32.png')
print(f'Konfigurasi terbaik: {best_row["technique"]} x {best_row["model"]} ({best_row["retrieval"]}) = {best_row["accuracy"]*100:.1f}%')
print(f'\\nCatatan penting:')
print(f'  - Llama 3.2 pakai retriever BM25-only (3 teknik tersedia)')
print(f'  - Model lain pakai Hybrid BM25+Dense (4 teknik tersedia)')
print(f'  - Gap Llama 3.2 vs Llama 3.3 70B mencerminkan kombinasi: model size + retriever quality')
'''

# Append cells
nb['cells'].append(nbf.v4.new_markdown_cell(md_source))
nb['cells'].append(nbf.v4.new_code_cell(code_source))

with open(NB_PATH, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Saved: {NB_PATH}')
print(f'Total cells now: {len(nb["cells"])}')
