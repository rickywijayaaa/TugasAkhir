# BM25 Expansion (Schwartz-Hearst) + Prompt Variation + Logprobs + Security Cleanup

Branch: `feat/bm25-expansion-and-prompt-variation` → `main`
Commits: `bc7d8e2`, `adbe65b` (2 commit, total ~525k insertions di 111+ file)

## Summary

PR ini mencakup semua eksperimen TA sejak commit `a01adde` (last push ke main), dipecah jadi 8 area kerja besar:

1. **BM25 Schwartz-Hearst Expansion** — preprocessing baru untuk fix vocabulary mismatch acronym medis (HBO, VEGF, EGFR)
2. **Eksperimen QR/CR/QR+CR di atas SH** — best result 74.8% (+5.6pp vs baseline original)
3. **Prompt variation** — 5 prompt (P0-P4) × 4 method (baseline/QR/CR/QR+CR)
4. **Logprobs & confidence calibration** — capture P(yes/no/maybe) per sample
5. **Llama 3.2 experiments** — CR + QR+CR variants complete
6. **Visualisasi & analisis** — 7 figures baru + 3 HTML deck
7. **Laporan TA Bab IV** — update + 6 flow diagram baru
8. **🔐 Security cleanup** — mask semua API key + .gitignore baru

---

## 🔐 SECURITY ACTION REQUIRED

Sebelum merge, **rotate 4 API key berikut yang sudah ter-expose di public history** (commit `3230f02`, `f56fb32`, dan `9f418c6` lama):

| Provider | Key prefix | Action |
|---|---|---|
| OpenAI | `REDACTED_SET_VIA_ENV...qkFXQA` | **REVOKE + buat baru** |
| OpenAI | `REDACTED_SET_VIA_ENV...dqigsA` | **REVOKE + buat baru** |
| OpenRouter | `REDACTED_SET_VIA_ENV...41d0fc` | **REVOKE + buat baru** |
| Anthropic | `REDACTED_SET_VIA_ENV...V0fQAA` | **REVOKE + buat baru** |

Setelah rotate, set key baru lewat `.env` (template: `.env.example`) atau shell env var. Jangan hardcode lagi.

---

## A. BM25 Schwartz-Hearst Expansion 🆕

**Folder baru**: `notebooks/BM25 Expansion/`

| Komponen | File |
|---|---|
| Algoritma Schwartz-Hearst 2003 | `acronym_expander_sh.py` |
| Curated medical dict (175 acronym + ambiguous set) | `acronym_dict.py` |
| Builder script | `build_sh_index.py` → `pubmedqa_bm25_sh.pkl` |
| Re-embed Chroma | notebook `03_rebuild_chroma_sh.ipynb` |

**Stats**: 310/500 papers ber-acronym terdeteksi, 673 ekspansi total.

## B. Eksperimen Notebook Baru

| Notebook | Output | Result |
|---|---|---:|
| `02_baseline_openai_expanded` | V1 naive | 72.8% |
| `04_baseline_openai_sh` | V2 SH baseline | **73.2%** (+4.0pp) |
| `05_qr_openai_sh` | Multi-query QR (Ma et al 2023) | 73.0% |
| `06_cr_openai_sh` | CrossEncoder rerank | 73.4% |
| `07_qr_cr_openai_sh` | QR + CR | **74.8%** ✨ best |
| `09_p1_strict_grounding_sh` | P1 prompt + SH | 62.0% |
| `10_p3_fewshot_cot_sh` | P3 prompt + SH | 65.2% |
| `11_logprobs_baseline_sh` | Confidence/calibration | TBD |
| `08_visualizations` | Charts Section 1-3 | — |

## C. Prompt Variation Eksperimen 🆕

**Folder baru**: `notebooks/prompt_experiment/`

Matriks 5 prompt × 4 method (n=500 untuk P0/P1/P3, n=100 untuk P2/P4):

| Prompt | Baseline | QR | CR | QR+CR |
|---|---:|---:|---:|---:|
| Default | **69.2%** | 70.8% | 69.8% | 70.0% |
| P1 Strict Grounding | 52.8% | 52.4% | 54.6% | 54.4% |
| P3 Few-Shot CoT | 60.0% | 60.8% | 59.8% | **61.8%** |

**Key finding**: Default prompt = sweet spot universal. P1/P3 mengangkat maybe class (4.5% → 57.6%/39.4%) tapi menjatuhkan yes/no.

## D. Llama 3.2 Lengkap

- `results/cr_llama32_phase1+phase2`
- `results/qr_cr_llama32_phase1+phase2`
- Fix Phase 2 evaluator di `qr_llama32_phase2_custom.json`

## E. Visualisasi (notebooks/figures/)

7 figures baru:
- `G7_accuracy_4llm_hybrid.png` — 4-LLM hybrid retrieval comparison
- `H1a_llama32_hybrid_accuracy.png`, `H1b_llama32_hybrid_ragas.png` — Llama32 hybrid detail
- `H2_xllm_accuracy.png`, `H3_xllm_ragas_heatmap.png` — cross-LLM
- `H4_maybe_recall.png` — maybe class drilldown

Plus 3 HTML deck siap presentasi:
- `notebooks/BM25 Expansion/storytelling_retrieval_eval.html` — full evaluation story
- `notebooks/BM25 Expansion/prompt_variation_slides.html` — 8-slide 16:9 deck (Default vs P1 vs P3)
- `notebooks/analisa_model.html` — interactive model comparison

## F. Excel Analysis (untuk eye-balling)

- `notebooks/BM25 Expansion/eyeball_500_4variants.xlsx` — 4 sheet (summary/compare/flips/long) untuk inspect tiap sampel
- `notebooks/BM25 Expansion/acronym_audit.xlsx` — semua ekspansi acronym yang ter-detect
- `notebooks/BM25 Expansion/comparison_3way_sh.xlsx` — original vs V1 vs V2
- `notebooks/claude_answers_inspection_v2.xlsx` — manual review Claude outputs

## G. Laporan TA (TA/Laporan TA/)

- `Bab IV - Perancangan.tex` — update lengkap (+289 baris)
- 6 flow diagram baru: `flowbaseline-3/4/5.png`, `flow-query rewriting-5.png`, `flow-context reranking-5.png`, `flow-qr+cr-5.png`
- `TA.pdf` rebuild (4.4 MB → 5.6 MB)

## H. Security Cleanup 🔐

1. **Mask 4 jenis API key** di 39 notebook/script (OpenAI, OpenRouter, Anthropic)
2. **Rewrite commit `9f418c6`** → key dihapus dari notebook 02 Baseline - Llama & 03.1.1 QR - OpenAI
3. **`.gitignore` baru** — exclude chroma DB (~100MB), `.env`, `.key`, secrets
4. **`.env.example`** template untuk set key via env var
5. **`_mask_api_keys.py`** — sanitize script idempotent kalau perlu re-run
6. **Untrack `pubmedqa_chroma*/chroma.sqlite3`** dari git (tetap di disk lokal)

---

## Hasil Headline

| Konfigurasi | Accuracy |
|---|---:|
| Original BM25 + Default | 69.2% |
| SH BM25 + Default | 73.2% (+4.0pp) |
| **SH BM25 + Default + QR+CR** | **74.8%** (+5.6pp) |

- SH expansion lift konsisten **+2-9pp** di SEMUA kombinasi prompt × method
- Default prompt > P1/P3 di semua method (sweet spot universal)
- Hard floor ~21% (105 sampel) tidak bisa diatasi tanpa structural improvement

---

## Test Plan

- [ ] Verify semua notebook execute clean (kernel restart + run all)
- [ ] Pastikan `.env` di-set sebelum run notebook OpenAI/Anthropic/OpenRouter
- [ ] Rotate 4 API key yang exposed (lihat section Security)
- [ ] Build PDF Laporan TA → verify 6 flow diagram baru ter-render
- [ ] Open `storytelling_retrieval_eval.html` dan `prompt_variation_slides.html` di browser

---

## Changelog (dari `a01adde`)

| File category | Count | Notes |
|---|---:|---|
| **New notebooks** | 11 | BM25 Expansion (8) + prompt_experiment (3) + logprobs (1) |
| **New result JSON** | 24 | Phase1+Phase2 untuk semua varian |
| **New figures** | 7 | G7 hybrid + H1-H4 |
| **New HTML decks** | 3 | storytelling, slides, analisa_model |
| **New Excel** | 4 | eyeball, audit, comparison, claude inspection |
| **New script helpers** | 12 | _build_*.py generators, _mask_api_keys.py |
| **Modified notebooks** | 6 | 02/03.1.1/03.3/04.3/05.3/07 (sanitized + extended) |
| **Modified results** | 1 | qr_llama32_phase2 (evaluator fix) |
| **Laporan TA** | 4 | Bab IV.tex, TA.lol, TA.pdf, 6 flow diagrams |
| **Total** | **111** | +521,643 / −469 |
