# Catatan Sesi — 1 April 2026

**Topik:** Evaluasi RAG + Query Rewriting + Perbaikan Prompt Baseline
**Notebook aktif:** `02-RAG-QueryRewriting.ipynb`, `03-RAG-Baseline-Evaluation.ipynb`

---

## 1. Perbaikan dari Sesi 29 Maret: Prompt v2 (Fix "Maybe" Bias)

### Masalah Sebelumnya (v1)
Pada sesi 29 Maret, model selalu memprediksi `maybe` (431/500 = 86.2%), tidak pernah `no`.
Akurasi baseline v1: **22.0%** (110/500).

### Solusi — Prompt v2 (Tighter Maybe)
Prompt generation diperketat dengan instruksi eksplisit:
- `maybe` hanya boleh digunakan jika **konteks benar-benar bertentangan** (sebagian bilang yes, sebagian bilang no)
- Jika bukti condong ke satu arah — meski tidak 100% pasti — harus pilih `yes` atau `no`
- Ditambahkan klausul: *"Do NOT use maybe simply because the evidence is limited or not 100% certain"*

### Hasil Baseline v2
| Metrik | v1 (29 Mar) | v2 (1 Apr) | Delta |
|--------|-------------|------------|-------|
| Label Accuracy | 22.0% | **55.6%** | **+33.6pp** |
| Hallucination Rate | 78.0% | 44.4% | -33.6pp |
| Pred `yes` | 13.8% | 86.2% | +72.4pp |
| Pred `no` | 0.0% | 10.0% | +10pp |
| Pred `maybe` | 86.2% | 3.8% | -82.4pp |

> Perbaikan prompt sangat signifikan. File v1 disimpan ke `baseline_phase1_answers_v1_original_prompt.json` untuk dokumentasi.

---

## 2. Notebook 02 — Dibangun Ulang

### Perubahan Struktur
Notebook `02-RAG-QueryRewriting.ipynb` sebelumnya menggunakan FAISS dan tidak memiliki pipeline evaluasi.
Sesi ini notebook dibangun ulang mengikuti struktur `03-RAG-Baseline-Evaluation.ipynb`:

| Komponen | Sebelum (FAISS) | Sekarang (BM25 + QR) |
|----------|-----------------|----------------------|
| Retriever | FAISS (dense vector) | **BM25** (shared index dengan baseline) |
| Query Rewriting | Ada, tapi standalone | Terintegrasi di `retrieve_with_qr()` |
| Fase 1 (500 sampel) | Tidak ada | **Ada** — resume-able, simpan per 10 sampel |
| Fase 2 (RAGAS) | Tidak ada | **Ada** — resume-able |
| Perbandingan vs Baseline | Tidak ada | **Ada** — load `baseline_*.json`, hitung delta |
| Analisis QR | Tidak ada | **Ada** — % query berubah, inspeksi kualitatif |

### Implementasi Query Rewriting
```python
QUERY_REWRITE_PROMPT = (
    "Rewrite the following medical question to improve retrieval from PubMed.\n"
    "Rules: be more specific, expand abbreviations, add medical terminology.\n"
    "Output ONLY the rewritten question.\n\n"
    "Original question: {query}\n\nRewritten question:"
)

def rewrite_query(query: str) -> str:
    response = ollama.generate(model=LLM_MODEL, prompt=...,
                               options={'temperature': 0.3, 'seed': 42, 'num_predict': 150})
    rewritten = response['response'].strip()
    return rewritten if len(rewritten) >= 10 else query  # fallback ke original
```

Pipeline retrieval:
```python
def retrieve_with_qr(query, k=TOP_K_RETRIEVAL):
    rewritten = rewrite_query(query)          # LLM reformulasi
    tokens    = tokenize_bm25(rewritten)      # tokenisasi query baru
    scores    = bm25_index.get_scores(tokens) # BM25 search
    top_k     = np.argsort(scores)[::-1][:k]
    return results, rewritten                 # return rewritten untuk logging
```

File output: `results/qr_phase1_answers.json`, `results/qr_phase2_ragas.json`, `results/qr_results.csv`

---

## 3. Hasil Fase 1 — QR (500 sampel, selesai)

### Akurasi Label

| Metrik | Baseline (BM25) | BM25 + QR | Delta |
|--------|-----------------|-----------|-------|
| Label Accuracy | **55.6%** (278/500) | **50.0%** (250/500) | **-5.6pp** |
| Hallucination Rate | 44.4% | 50.0% | +5.6pp |
| Pred `yes` | 86.2% (431) | 81.2% (406) | -5pp |
| Pred `no` | 10.0% (50) | 13.2% (66) | +3.2pp |
| Pred `maybe` | 3.8% (19) | 5.6% (28) | +1.8pp |

### Confusion Matrix

**Baseline:**
```
         yes      no   maybe
  yes    249      16      10   (acc 90.5%)
   no    123      28       8   (acc 17.6%)
 maybe    59       6       1   (acc  1.5%)
```

**BM25 + QR:**
```
         yes      no   maybe
  yes    227      37      11   (acc 82.5%)
   no    123      21      15   (acc 13.2%)
 maybe    56       8       2   (acc  3.0%)
```

### Per-Sample Analysis (McNemar-style)

| Kategori | n | % |
|----------|---|---|
| Keduanya benar | 219 | 43.8% |
| Keduanya salah | 191 | 38.2% |
| Hanya Baseline benar | **59** | **11.8%** |
| Hanya QR benar | 31 | 6.2% |

QR memperbaiki 31 sampel, merusak 59 → **net loss −28 sampel**.

### BM25 Retrieval Score

| Konfigurasi | Avg BM25 Score |
|-------------|----------------|
| Baseline | 25.18 |
| BM25 + QR | **30.66** |
| Delta | **+5.48** (+21.7%) |

**Temuan paradoks:** QR berhasil meningkatkan BM25 retrieval score (dokumen lebih relevan secara keyword), tetapi accuracy label justru turun.

### Statistik Query Rewriting

- **100% dari 500 query diubah** oleh LLM (tidak ada yang dikembalikan verbatim)
- Contoh rewrite:

| Query Asli | Rewritten |
|-----------|-----------|
| *Do mitochondria play a role in remodelling lace plant leaves...* | *Do mitochondrial function and dynamics contribute to programmed cell death...* |
| *Landolt C and snellen e acuity: differences in strabismus amblyopia?* | *Landolt C and Snellen E acuity: a comparative analysis of visual field...* |
| *Can tailored interventions increase mammography use among HMO women?* | *Can targeted behavioral interventions incorporating patient education...* |

---

## 4. Interpretasi: Kenapa QR Lebih Buruk dari Baseline?

### Hipotesis 1 — Semantic Drift
LLM mereformulasi query dengan terminologi baru yang berbeda dari query asli. Karena generation menggunakan **query asli** (bukan rewritten), terjadi mismatch antara konteks yang diretrieval dan pertanyaan yang dijawab.

### Hipotesis 2 — BM25 Tidak Cocok untuk QR
Query rewriting lebih efektif untuk **dense retrieval** (semantic similarity). Pada BM25, query yang lebih panjang/verbose justru bisa mendilusi keyword penting dan menurunkan signal.

### Hipotesis 3 — Model Masih Over-predict "yes"
Kedua konfigurasi masih sangat biased ke `yes` (81–86%). QR tidak mengubah perilaku generation secara fundamental — yang berubah hanya konteks yang diretrieval, dan ini tidak cukup.

---

## 5. Status RAGAS

| Konfigurasi | Fase 2 Status | n valid |
|-------------|---------------|---------|
| Baseline | Partial — **55/500** sampel | faith=24, ans_rel=13, ctx_prec=9, ctx_rec=51 |
| QR | **Belum dijalankan** | — |

RAGAS baseline masih banyak NaN. Perlu investigasi apakah karena timeout atau format output.

Angka sementara Baseline RAGAS (tidak representatif):
| Metrik | Mean | n valid |
|--------|------|---------|
| Faithfulness | 0.7410 | 24 |
| Answer Relevancy | 0.4237 | 13 |
| Context Precision | 1.0000 | 9 |
| Context Recall | 0.6643 | 51 |

---

## 6. File yang Dibuat/Diperbarui Sesi Ini

| File | Status | Keterangan |
|------|--------|------------|
| `notebooks/02-RAG-QueryRewriting.ipynb` | **Dibangun ulang** | Struktur identik baseline, BM25+QR, evaluasi lengkap |
| `results/qr_phase1_answers.json` | **Baru** | 500 sampel QR, termasuk `rewritten_query` per record |
| `results/baseline_phase1_answers_v1_original_prompt.json` | **Baru** | Arsip hasil v1 prompt (22% accuracy) untuk dokumentasi |
| `results/baseline_phase2_ragas.json` | Partial | 55/500 sampel |

---

## 7. Hal yang Perlu Dilakukan Selanjutnya

### Priority Tinggi
- [ ] **Jalankan Fase 2 RAGAS untuk QR** — bisa semalam, hasilnya kritis untuk perbandingan
- [ ] **Selesaikan Fase 2 RAGAS Baseline** — 445 sampel tersisa (resume dari yang sudah ada)
- [ ] **Investigasi NaN RAGAS** — cek apakah ada pola pada sampel yang menghasilkan NaN (panjang konteks? format jawaban?)

### Priority Sedang
- [ ] **Analisis kualitatif degradasi QR** — inspeksi 59 sampel yang baseline benar tapi QR salah
- [ ] **Coba QR dengan dense retrieval** — apakah QR lebih efektif jika retriever-nya FAISS/semantic?
- [ ] **Implementasi Context Reranking (CR)** — notebook `04-RAG-ContextReranking-Evaluation.ipynb`

### Pertanyaan Terbuka
1. **Apakah QR tetap akan dimasukkan sebagai teknik mitigasi di skripsi** meskipun hasilnya lebih buruk? Temuan negatif tetap valid dan bisa menjadi kontribusi (menunjukkan batasan QR pada BM25).
2. **Apakah akan ditambahkan ablation study** — QR dengan FAISS vs QR dengan BM25?
3. **Berapa sampel minimum untuk RAGAS yang representatif?** Saat ini terlalu banyak NaN.
