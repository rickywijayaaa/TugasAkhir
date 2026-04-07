# RANGKUMAN EVALUASI RAG: BASELINE vs QUERY REWRITING
## Dokumen Persiapan Bimbingan Tugas Akhir

**Tanggal:** 2 April 2026
**Topik:** Evaluasi Retrieval-Augmented Generation (RAG) pada Dataset PubMedQA

---

## 1. KONFIGURASI EKSPERIMEN

| Parameter | Nilai |
|-----------|-------|
| Dataset | PubMedQA (pqa_labeled) |
| Jumlah Sampel | 500 |
| Retriever | BM25 (rank_bm25) |
| Top-K Retrieval | 5 dokumen |
| LLM Generator | Llama 3.2 (via Ollama) |
| Embedding | nomic-embed-text (hanya untuk evaluasi RAGAS) |
| Temperature | 0.0 (deterministik) |

### Konfigurasi yang Dibandingkan:
1. **Baseline**: BM25 tanpa modifikasi query
2. **Query Rewriting (QR)**: BM25 dengan reformulasi query menggunakan LLM

---

## 2. DATASET: PubMedQA

### 2.1 Apa itu PubMedQA?

PubMedQA adalah dataset benchmark untuk **Biomedical Research Question Answering** yang dikembangkan dari artikel-artikel PubMed. Dataset ini dirancang untuk mengevaluasi kemampuan model dalam menjawab pertanyaan riset biomedis dengan jawaban **yes/no/maybe**.

**Paper:** Jin et al., "PubMedQA: A Dataset for Biomedical Research Question Answering" (EMNLP 2019)

### 2.2 Struktur Dataset PubMedQA

PubMedQA terdiri dari **3 subset** dengan karakteristik berbeda:

| Subset | Nama Lengkap | Ukuran | Label | Kegunaan |
|--------|--------------|--------|-------|----------|
| **PQA-L** | pqa_labeled | **1,000** | Expert-annotated (yes/no/maybe) | Evaluasi & benchmark |
| **PQA-U** | pqa_unlabeled | **61,200** | Tidak ada | Training (pseudo-labeling) |
| **PQA-A** | pqa_artificial | **211,269** | Auto-generated (yes/no saja) | Pretraining |

**Total keseluruhan: ~273,500 QA instances**

### 2.3 Detail PQA-L (pqa_labeled) - Dataset yang Digunakan

PQA-L adalah subset dengan **1,000 sampel yang dilabeli oleh expert** (dokter/peneliti biomedis).

```
PQA-L (1,000 sampel)
│
├── Test Set: 500 sampel
│   └── Digunakan untuk benchmark resmi (comparable dengan paper lain)
│
└── Train/Validation Set: 500 sampel
    ├── Training: 450 sampel
    └── Validation: 50 sampel
    └── (10-fold cross validation)
```

### 2.4 Distribusi Label di PQA-L

| Label | Jumlah (approx) | Persentase |
|-------|-----------------|------------|
| **yes** | ~550 | **~55%** |
| **no** | ~320 | **~32%** |
| **maybe** | ~130 | **~13%** |

**Catatan:** Dataset ini **imbalanced** - mayoritas label adalah "yes". Ini menjelaskan mengapa model cenderung bias memprediksi "yes".

### 2.5 Struktur Setiap Instance

Setiap sampel PubMedQA berisi:

| Field | Deskripsi | Contoh |
|-------|-----------|--------|
| **pubid** | PubMed ID artikel | "21645374" |
| **question** | Judul artikel yang diubah menjadi pertanyaan | "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?" |
| **context** | Abstrak artikel (TANPA conclusion) | "Background: ... Methods: ... Results: ..." |
| **long_answer** | Conclusion dari abstrak (jawaban panjang) | "Results depicted mitochondrial dynamics..." |
| **final_decision** | Label yes/no/maybe | "yes" |

### 2.6 Kenapa Ada Pilihan 500 vs 1000 Sampel?

| Pilihan | Penjelasan | Kapan Digunakan |
|---------|------------|-----------------|
| **500 sampel** | Test set resmi PubMedQA | Evaluasi yang comparable dengan paper lain |
| **1000 sampel** | Seluruh pqa_labeled (train + test) | Evaluasi dengan sample size lebih besar untuk validitas statistik |

**Dalam eksperimen ini:**
- Menggunakan **500 sampel** (test set resmi)
- Dapat ditingkatkan ke **1000 sampel** untuk evaluasi final thesis

### 2.7 Cara Dataset Digunakan dalam Eksperimen

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALUR PENGGUNAAN DATASET                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PubMedQA (500 sampel)                                          │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────┐                                            │
│  │ Setiap Sampel:  │                                            │
│  │ - question      │──────┐                                     │
│  │ - context       │      │                                     │
│  │ - long_answer   │      │                                     │
│  │ - final_decision│      │                                     │
│  └─────────────────┘      │                                     │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────┐                │
│  │           BM25 INDEX (1,706 dokumen)         │                │
│  │  (Dibangun dari context semua sampel)        │                │
│  │  Setiap abstrak dipecah per section          │                │
│  └─────────────────────────────────────────────┘                │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────┐                │
│  │              RETRIEVAL (Top-5)               │                │
│  │  Query (question) → BM25 → 5 dokumen         │                │
│  └─────────────────────────────────────────────┘                │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────┐                │
│  │           GENERATION (LLM)                   │                │
│  │  Context + Question → Llama 3.2 → Answer     │                │
│  └─────────────────────────────────────────────┘                │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────┐                │
│  │              EVALUASI                        │                │
│  │  Predicted Label vs Ground Truth             │                │
│  │  (final_decision)                            │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.8 Sumber Dataset

- **Homepage:** https://pubmedqa.github.io/
- **Hugging Face:** https://huggingface.co/datasets/qiaojin/PubMedQA
- **GitHub:** https://github.com/pubmedqa/pubmedqa
- **Paper:** https://aclanthology.org/D19-1259/

---

## 3. HASIL EVALUASI: LABEL ACCURACY

| Konfigurasi | Label Accuracy | Hallucination Rate |
|-------------|----------------|-------------------|
| **Baseline (BM25)** | **55.6%** (278/500) | 44.4% |
| **BM25 + Query Rewriting** | **50.0%** (250/500) | 50.0% |
| **Delta** | **-5.6%** | +5.6% |

### Temuan Utama:
- Query Rewriting **MENURUNKAN** akurasi sebesar 5.6 poin persentase
- Hallucination rate meningkat dari 44.4% menjadi 50.0%

---

## 4. HASIL EVALUASI: RETRIEVAL SCORES

### 4.1 Statistik Retrieval Score

| Metrik | Baseline (BM25) | QR (BM25+QR) | Delta |
|--------|-----------------|--------------|-------|
| **Top-1 Score Mean** | 42.54 | 44.88 | **+2.34** |
| **Top-1 Score Median** | 40.58 | 42.01 | +1.43 |
| **Top-1 Score Std Dev** | 17.78 | 17.13 | -0.65 |
| **Avg Top-5 Mean** | 25.18 | 30.66 | **+5.48** |
| **Avg Top-5 Median** | 24.47 | 29.85 | +5.38 |

### 4.2 Perbandingan Head-to-Head (per sampel)

| Kondisi | Jumlah | Persentase |
|---------|--------|------------|
| QR score lebih tinggi | 282/500 | **56.4%** |
| Baseline score lebih tinggi | 210/500 | 42.0% |
| Score sama | 8/500 | 1.6% |

### 4.3 Temuan Paradoks: Score Lebih Tinggi ≠ Akurasi Lebih Baik

| Kuartil Top-1 Score | Akurasi Baseline | Akurasi QR | Delta |
|---------------------|------------------|------------|-------|
| Q1 (rendah) | 56.8% | 48.0% | -8.8% |
| Q2 | 53.6% | 50.4% | -3.2% |
| Q3 | 58.4% | 55.2% | -3.2% |
| Q4 (tinggi) | 53.6% | 46.4% | -7.2% |

**Insight Penting:**
- Query Rewriting menghasilkan retrieval score yang lebih tinggi di 56.4% kasus
- NAMUN, akurasi justru menurun di SEMUA kuartil
- **Score BM25 yang lebih tinggi TIDAK menjamin dokumen yang lebih relevan**

---

## 5. DISTRIBUSI PREDIKSI (CONFUSION MATRIX)

### 5.1 Baseline

```
GT\Pred     yes      no   maybe
    yes     249      16      10    (recall: 90.5%)
     no     123      28       8    (recall: 17.6%)
  maybe      59       6       1    (recall: 1.5%)
```

### 5.2 Query Rewriting

```
GT\Pred     yes      no   maybe
    yes     227      37      11    (recall: 82.5%)
     no     123      21      15    (recall: 13.2%)
  maybe      56       8       2    (recall: 3.0%)
```

### 5.3 Insight:
- Model sangat **bias ke prediksi "yes"** (~80-86% prediksi)
- Ground truth: 55% yes, 32% no, 13% maybe
- Recall untuk "no" dan "maybe" sangat rendah (<20%)

---

## 6. CARA KERJA BM25 (BEST MATCHING 25)

### 6.1 Apa itu BM25?
BM25 adalah algoritma **keyword-based retrieval** yang menghitung relevansi dokumen berdasarkan kecocokan kata kunci. Merupakan pengembangan dari TF-IDF dengan normalisasi panjang dokumen.

### 6.2 Formula BM25

```
Score(D, Q) = Σ IDF(qi) × [f(qi,D) × (k1+1)] / [f(qi,D) + k1 × (1 - b + b × |D|/avgdl)]
```

Dimana:
- **D** = Dokumen yang dinilai
- **Q** = Query (pertanyaan pengguna)
- **qi** = Kata ke-i dalam query
- **f(qi,D)** = Frekuensi kata qi dalam dokumen D (Term Frequency)
- **|D|** = Panjang dokumen D (jumlah kata)
- **avgdl** = Rata-rata panjang dokumen dalam corpus
- **k1** = Parameter saturasi TF (default: 1.2-2.0)
- **b** = Parameter normalisasi panjang dokumen (default: 0.75)

### 6.3 Komponen Score BM25

#### A. IDF (Inverse Document Frequency)
```
IDF(qi) = log[(N - n(qi) + 0.5) / (n(qi) + 0.5)]
```
- **N** = Total jumlah dokumen dalam corpus
- **n(qi)** = Jumlah dokumen yang mengandung kata qi
- **Fungsi**: Memberikan bobot lebih tinggi pada kata yang jarang muncul

**Contoh:**
- Kata "aspirin" muncul di 5 dari 1706 dokumen → IDF tinggi (~5.5)
- Kata "the" muncul di 1500 dari 1706 dokumen → IDF rendah (~0.1)

#### B. Term Frequency (TF) dengan Saturasi
```
TF_saturated = [f(qi,D) × (k1+1)] / [f(qi,D) + k1]
```
- Mencegah dominasi kata yang muncul sangat sering
- Saturasi membatasi kontribusi TF maksimal

#### C. Normalisasi Panjang Dokumen
```
Normalization = 1 - b + b × (|D| / avgdl)
```
- Dokumen panjang tidak mendominasi hasil
- b=0.75 berarti 75% penalti untuk dokumen panjang

### 6.4 Proses Retrieval BM25

```
1. TOKENISASI QUERY
   Input: "Does aspirin reduce myocardial infarction risk?"
   Output: ['does', 'aspirin', 'reduce', 'myocardial', 'infarction', 'risk']

2. HITUNG SCORE SETIAP DOKUMEN
   Untuk setiap dokumen dalam corpus:
   - Hitung IDF untuk setiap kata query
   - Hitung TF saturated untuk setiap kata query
   - Terapkan normalisasi panjang dokumen
   - Jumlahkan semua kontribusi kata

3. RANKING
   - Urutkan dokumen berdasarkan score (descending)
   - Ambil Top-K dokumen (K=5 dalam eksperimen)
```

### 6.5 Interpretasi Score BM25

| Range Score | Interpretasi |
|-------------|--------------|
| > 50 | Sangat relevan (banyak kata query cocok dengan IDF tinggi) |
| 30-50 | Cukup relevan |
| 15-30 | Relevan rendah |
| < 15 | Kurang relevan |

**Catatan:** Score BM25 tidak dinormalisasi ke range [0,1]. Nilainya bergantung pada:
- Jumlah kata query yang cocok
- IDF kata-kata tersebut
- Frekuensi dalam dokumen
- Panjang dokumen

---

## 7. CARA KERJA QUERY REWRITING

### 7.1 Tujuan Query Rewriting
Reformulasi query asli agar lebih kaya terminologi medis untuk meningkatkan kualitas retrieval BM25.

### 7.2 Prompt yang Digunakan

```
You are a query rewriting assistant for a biomedical question-answering system.
Rewrite the following medical question to improve retrieval from a PubMed research database.

Rules:
1. Be more specific and add relevant medical/scientific terminology.
2. Expand abbreviations (e.g. "MI" -> "myocardial infarction").
3. Preserve the original yes/no/maybe answerable intent.
4. Output ONLY the rewritten question, no explanations.
```

### 7.3 Contoh Query Rewriting

| Query Asli | Query Rewritten |
|------------|-----------------|
| Does aspirin reduce risk of MI? | Does low-dose aspirin therapy significantly decrease the incidence of recurrent myocardial infarction in patients with established coronary artery disease? |
| Is vitamin D effective for COVID-19? | Is vitamin D supplementation effective in reducing mortality and morbidity in patients with COVID-19? |
| Can exercise prevent T2DM? | Can regular aerobic exercise interventions effectively reduce or prevent the onset of type 2 diabetes mellitus in adults? |

### 7.4 Mengapa Score Meningkat tapi Akurasi Menurun?

**Hipotesis:**

1. **Over-specification**
   - Query yang terlalu spesifik menambah kata-kata yang tidak ada di dokumen relevan
   - Contoh: "low-dose", "recurrent", "established coronary artery disease" mungkin tidak ada di abstrak yang sebenarnya relevan

2. **Keyword Drift**
   - LLM menambahkan terminologi yang berbeda dari yang digunakan dalam dokumen
   - BM25 hanya mencocokkan kata secara eksak, tidak memahami sinonim

3. **Semantic Gap**
   - Query rewritten lebih panjang = lebih banyak kata = score akumulatif lebih tinggi
   - TAPI kata-kata tambahan mungkin tidak relevan dengan dokumen yang benar-benar menjawab pertanyaan

4. **Dataset Mismatch**
   - Dokumen PubMedQA menggunakan terminologi tertentu
   - Query rewriting mungkin menggunakan terminologi yang berbeda meskipun bermakna sama

---

## 8. KESIMPULAN DAN REKOMENDASI

### 8.1 Kesimpulan

1. **Query Rewriting TIDAK efektif** untuk meningkatkan akurasi pada sistem RAG dengan BM25 retriever
2. **Score retrieval yang lebih tinggi tidak menjamin dokumen yang lebih relevan** untuk menjawab pertanyaan
3. **Model sangat bias** ke prediksi "yes" (over-optimistic)
4. **BM25 memiliki keterbatasan** karena hanya mencocokkan kata secara eksak (keyword matching)

### 8.2 Rekomendasi untuk Perbaikan

| Teknik | Deskripsi | Potensi Perbaikan |
|--------|-----------|-------------------|
| **Dense Retrieval** | Gunakan embedding (DPR, Contriever) untuk semantic matching | Mengatasi vocabulary mismatch |
| **Hybrid Retrieval** | Kombinasi BM25 + Dense Retrieval | Memanfaatkan kelebihan keduanya |
| **Cross-Encoder Reranking** | Rerank hasil BM25 dengan cross-encoder | Meningkatkan presisi top-K |
| **Prompt Engineering** | Perbaiki prompt generasi untuk mengurangi bias "yes" | Meningkatkan akurasi prediksi |
| **RAG Fusion** | Gunakan multiple queries dan gabungkan hasil | Meningkatkan recall |

### 8.3 Pertanyaan untuk Diskusi dengan Dosen

1. Apakah perlu menjalankan evaluasi dengan 1000 sampel untuk validitas statistik?
2. Apakah sebaiknya mengganti retriever dari BM25 ke dense retrieval?
3. Bagaimana cara mengurangi bias prediksi "yes"?
4. Apakah Query Rewriting lebih cocok untuk retriever berbasis semantic?
5. Apakah perlu menambahkan evaluasi RAGAS (Faithfulness, Context Recall, dll)?

---

## 9. LAMPIRAN: DATA MENTAH

### 9.1 Statistik Lengkap Retrieval Score

```
BASELINE (BM25):
- Top-1 Score: Mean=42.54, Median=40.58, Std=17.78, Min=9.90, Max=118.29
- Avg Top-5:   Mean=25.18, Median=24.47

QUERY REWRITING (BM25+QR):
- Top-1 Score: Mean=44.88, Median=42.01, Std=17.13, Min=10.36, Max=124.58
- Avg Top-5:   Mean=30.66, Median=29.85
```

### 9.2 Per-Label Accuracy

| Label | Baseline | Query Rewriting | Delta |
|-------|----------|-----------------|-------|
| yes | 90.5% (249/275) | 82.5% (227/275) | -8.0% |
| no | 17.6% (28/159) | 13.2% (21/159) | -4.4% |
| maybe | 1.5% (1/66) | 3.0% (2/66) | +1.5% |

---

*Dokumen ini dibuat untuk keperluan bimbingan Tugas Akhir.*
