# Notebooks v2 — Struktur Rapih

Folder ini berisi notebook eksperimen TA "Mitigasi Halusinasi pada RAG-LLM" yang sudah ditata ulang per kelompok konfigurasi dan per model bahasa. Folder lama `notebooks/` (yang berantakan) masih ada selama transisi dan dapat dihapus setelah verifikasi.

## Struktur Folder

```
notebooks_v2/
├── 00_setup/                              ← persiapan data & index
│   ├── 01_dataset_exploration.ipynb
│   ├── 03_build_chroma_dengan_ekspansi.ipynb
│   └── 04_build_chroma_expanded_legacy.ipynb
│
├── 10_bm25/                               ← Kelompok 1: BM25 tanpa Ekspansi
│   ├── gpt4mini/  {baseline, qr, cr, qr_cr}.ipynb
│   ├── llama_3_3/ {baseline, qr, cr}.ipynb
│   └── llama_3_2/ {baseline, qr, cr, qr_cr}.ipynb
│
├── 20_hybrid_tanpa_ekspansi/              ← Kelompok 2: Hybrid + tanpa Ekspansi
│   ├── gpt4mini/
│   ├── claude/
│   ├── llama_3_3/
│   └── llama_3_2/
│
├── 30_hybrid_dengan_ekspansi/             ← Kelompok 3: Hybrid + dengan Ekspansi
│   └── gpt4mini/  (model lain bisa ditambah nanti)
│
├── 40_evaluation/                         ← evaluasi & visualisasi
│   ├── ragas_phase2_runner.ipynb          ← isi 4 RAGAS yang hilang (Hybrid no SH)
│   ├── retrieval_metrics_runner.ipynb     ← Recall@5, MAP@5, nDCG@5 untuk 12 config
│   └── visualizations.ipynb               ← notebook 07 lama (semua chart)
│
├── archive/                               ← legacy / out-of-scope
│   ├── 00_example_RAG.ipynb
│   ├── 06_1_AHD_OpenAI.ipynb              (AHD dikeluarkan dari scope)
│   ├── 09_VectorDB_Comparison.ipynb
│   ├── 10_Agentic_RAG_OpenAI.ipynb
│   ├── 11_Prompt_Variation_OpenAI.ipynb
│   └── legacy_*.ipynb                     (versi lama)
│
├── indexes/                               ← shared index files
│   ├── pubmedqa_bm25.pkl                  (asli, tanpa ekspansi)
│   ├── pubmedqa_bm25_ekspansi.pkl         (dengan ekspansi akronim)
│   ├── pubmedqa_chroma/                   (vektor asli)
│   ├── pubmedqa_chroma_ekspansi/          (vektor dengan ekspansi)
│   └── pubmedqa_chroma_llama/             (vektor Llama embedding)
│
└── scripts/                               ← helper modules & scripts
    ├── ragas_evaluator.py                 ← shared RAGAS metrics
    ├── retrieval_metrics.py               ← shared IR metrics
    ├── acronym_expander.py / _sh.py       ← ekspansi akronim
    ├── _build_*.py                        ← generator notebook
    ├── _patch_paths_gpt4mini.py           ← (sudah dijalankan)
    └── ...
```

## Path Resolution

Setiap notebook GPT-4.1-mini sudah pakai pattern walk-up otomatis:

```python
_HERE = Path('.').resolve()
PROJECT_ROOT = next((p for p in [_HERE] + list(_HERE.parents) if p.name == 'Code TA'), ...)
NOTEBOOKS_V2 = PROJECT_ROOT / 'notebooks_v2'
INDEXES_DIR  = NOTEBOOKS_V2 / 'indexes'
RESULTS_DIR  = PROJECT_ROOT / 'results_v2' / '<kelompok>'
```

Konsekuensi:
- Notebook portable — bisa dipindah lokasi, path auto-resolve
- Index disimpan **terpusat** di `notebooks_v2/indexes/` (tidak duplikat per notebook)
- Result disimpan **terpisah per kelompok** di `results_v2/<kelompok>/`

## Konvensi Nama

| Lama | Baru |
|---|---|
| `sh` (Schwartz-Hearst) | `ekspansi` (ekspansi akronim) |
| `BM25 Expansion/` | `30_hybrid_dengan_ekspansi/` |
| `pubmedqa_bm25_sh.pkl` | `pubmedqa_bm25_ekspansi.pkl` |
| `pubmedqa_chroma_sh/` | `pubmedqa_chroma_ekspansi/` |
| `*_openai` di nama config | hanya prefix kelompok saja (kelompok sudah jelas dari folder) |

## Cara Menambah Konfigurasi Model Lain

Untuk menambah Claude / Llama di Kelompok 3 (dengan ekspansi):
1. Copy notebook gpt4mini sebagai template
2. Ubah `LLM_MODEL` dan `EMBED_MODEL` di cell konfig
3. Sesuaikan CONFIG_NAME untuk output file
4. RESULTS_DIR sudah otomatis benar (per kelompok)

## Cara Mengisi Metrik yang Hilang

### A. Phase 2 RAGAS untuk Hybrid no Ekspansi (prioritas utama)

```bash
# Jalankan notebook 40_evaluation/ragas_phase2_runner.ipynb
# Akan loop 4 konfigurasi (baseline, qr, cr, qr_cr)
# Output: results_v2/20_hybrid_tanpa_ekspansi/{config}_phase2_custom.json
```

Estimasi: 2-4 jam (4 konfig × 500 sampel × ~6000 LLM calls)

### B. Retrieval Metrics untuk 12 Konfigurasi

```bash
# Jalankan notebook 40_evaluation/retrieval_metrics_runner.ipynb
# Cepat (no LLM, hanya hitungan numerik)
# Output: results_v2/<kelompok>/{config}_retrieval_metrics.json
```

Estimasi: < 1 menit

## Git Backup

Sebelum refactor, snapshot dibuat di tag `pre-refactor-2026-05-30`.
Untuk rollback total: `git reset --hard pre-refactor-2026-05-30`
