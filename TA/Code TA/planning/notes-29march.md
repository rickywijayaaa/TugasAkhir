# Catatan Sesi — 29 Maret 2026

**Topik:** Evaluasi Baseline RAG + Diskusi Teknis
**Notebook aktif:** `03-RAG-Baseline-Evaluation.ipynb`

---

## 1. Perubahan Teknis: FAISS → BM25

### Alasan Penggantian

Retriever awal menggunakan FAISS (dense vector search). Setelah diskusi, diganti ke **BM25 (rank-bm25)** dengan alasan berikut:

| Aspek | FAISS | BM25 |
|---|---|---|
| Mekanisme | Kemiripan vektor (semantic) | Kecocokan token (lexical) |
| Cocok untuk PubMedQA | Kurang — terminologi medis spesifik sering tidak tertangkap embedding generik | Lebih baik — exact match pada istilah medis (gene names, drug names, dsb.) |
| Setup | Butuh embedding model + indexing berat | Langsung dari teks, ringan |
| Relevansi ke tahap berikutnya | Tidak ada kaitan langsung | BM25 → CrossEncoder reranking = alur natural untuk +CR |
| Reproducibility | Bergantung pada model embedding yang digunakan | Deterministik |

### Implementasi BM25

```python
# Tokenisasi
def tokenize_bm25(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

# Build index
tokenized_corpus = [tokenize_bm25(doc) for doc in corpus_texts]
bm25_index = BM25Okapi(tokenized_corpus)

# Retrieve
tokens = tokenize_bm25(query)
scores = bm25_index.get_scores(tokens)
top_k_idx = np.argsort(scores)[::-1][:k]
```

File indeks disimpan ke `data/pubmedqa_bm25.pkl` untuk efisiensi re-run.

---

## 2. Hasil Phase 1 — Generation (500 sampel)

### Setup
- Dataset: PubMedQA `pqa_labeled`, 500 sampel pertama
- Retriever: BM25, top-10 → top-3 konteks dikirim ke LLM
- LLM: Ollama (lokal)
- Timestamp: 2026-03-29T01:40:58

### Akurasi Label

| Metrik | Nilai |
|---|---|
| **Overall Accuracy** | **22.0%** (110/500) |
| Hallucination/Unknown Rate | **0.0%** — label extractor berjalan sempurna |

### Breakdown Per Label Ground Truth

| Ground Truth | Jumlah | Benar | Akurasi |
|---|---|---|---|
| `yes` | 275 (55%) | 53 | 19.3% |
| `no` | 159 (31.8%) | 0 | **0.0%** |
| `maybe` | 66 (13.2%) | 57 | 86.4% |

### Distribusi Prediksi LLM

| Prediksi | Jumlah | Persentase |
|---|---|---|
| `maybe` | 431 | **86.2%** |
| `yes` | 69 | 13.8% |
| `no` | 0 | **0%** |

### Confusion Matrix (ringkasan)

| | GT=yes | GT=no | GT=maybe |
|---|---|---|---|
| pred=yes | 53 | 7 | 9 |
| pred=no | 0 | 0 | 0 |
| pred=maybe | 222 | 152 | 57 |

### Interpretasi
- Model **tidak pernah memprediksi `no`** dari 500 sampel → bias kritis
- 222/275 entri berlabel `yes` diprediksi `maybe` → model tidak berani berkomitmen
- Hanya entri `maybe` yang berhasil diprediksi dengan baik (86.4%), dan ini bersifat "kebetulan" karena model memang default ke `maybe`
- **Root cause**: konteks BM25 tidak cukup kuat/spesifik untuk mendorong LLM membuat keputusan tegas yes/no

---

## 3. Hasil Phase 2 — RAGAS (50 sampel)

> ⚠️ Hanya 50 sampel yang berhasil dievaluasi RAGAS. Perlu investigasi.

| Metrik | Mean | Valid/50 | NaN |
|---|---|---|---|
| Context Recall | 0.6783 | 47/50 | 3 |
| Faithfulness | 0.7515 | 23/50 | 27 |
| Answer Relevancy | 0.4591 | 12/50 | 38 |
| Context Precision | 1.0000 | 8/50 | 42 |

Catatan:
- **Context Recall 0.68** — paling reliable (47/50 valid). BM25 berhasil mengambil ~68% informasi yang relevan. Retrieval cukup baik di level coverage.
- **Context Precision = 1.0** dari hanya 8 sampel — tidak representatif, bisa jadi artefak RAGAS.
- **Faithfulness 0.75** — saat jawaban ada, cukup grounded ke konteks. Tapi karena semua jawaban "maybe", ini terbatas maknanya.
- **Answer Relevancy 0.46** — rendah. Jawaban `maybe` memang tidak relevan terhadap pertanyaan spesifik.
- Banyak NaN → kemungkinan timeout masih terjadi di sebagian metrik meski sudah `max_workers=1`.

---

## 4. Hal yang Perlu Didiskusikan / Ditingkatkan

### A. Masalah Kritis (Priority: Tinggi)

#### A1. Prompt Generation — "Maybe" Bias
Model selalu memilih `maybe` sebagai default hedge. Opsi perbaikan yang perlu didiskusikan:

1. **Ubah instruksi prompt** — perkuat dengan contoh (few-shot), atau tambahkan frasa seperti "Berikan jawaban definitif berdasarkan konteks. Gunakan `maybe` HANYA jika konteks benar-benar ambigu."
2. **Tambahkan chain-of-thought** — minta model jelaskan reasoning sebelum memberi label, agar lebih terdorong memilih yes/no
3. **Evaluasi model lain** — apakah model yang digunakan memang cenderung hedging? Coba model berbeda (mis. llama3.2 vs llama3.1)

#### A2. RAGAS Phase 2 — Banyak NaN
50 sampel valid, banyak NaN. Perlu cek:
- Apakah ada error timeout yang masih lolos meski `max_workers=1`?
- Apakah ada perbedaan format antara `answer` yang dihasilkan dan yang diharapkan RAGAS?
- Apakah `answer_relevancy` memerlukan embedding yang berjalan lambat dan sering timeout?

### B. Perbaikan Sebelum Full Evaluation (Priority: Sedang)

#### B1. Ubah MAX_SAMPLES dari 500 ke 1000
Notebook saat ini hanya mengevaluasi 500 sampel. Untuk laporan akhir harus diubah ke 1000.
- Lokasi: Cell konfigurasi di `03-RAG-Baseline-Evaluation.ipynb`
- Variabel: `MAX_SAMPLES = 500` → `MAX_SAMPLES = 1000`

#### B2. Investigasi dan Perbaiki NaN di RAGAS
Sebelum re-run penuh, pastikan RAGAS berjalan dengan bersih di semua 1000 sampel.

#### B3. Simpan Baseline Results Final sebagai CSV
Merge phase1 + phase2 ke `results/baseline_results.csv` agar mudah dibandingkan dengan konfigurasi +QR, +CR, dsb.

### C. Langkah Berikutnya — Implementasi +QR

Notebook `04-RAG-QueryRewriting-Evaluation.ipynb` perlu dibuat dengan:
- Semua komponen baseline (BM25 retrieval + generation)
- Tambahan: fungsi `rewrite_query(question)` yang memanggil LLM sebelum retrieval
- Prompt rewriting perlu dirancang — diskusikan formatnya
- Simpan juga `rewritten_query` di output untuk analisis

### D. Pertanyaan Terbuka

1. **Apakah perlu perbaiki prompt generation DULU sebelum lanjut ke +QR?**
   Jika tidak diperbaiki, semua konfigurasi mungkin akan tetap bias "maybe" dan sulit dibedakan.

2. **Metrik tambahan apa yang relevan selain RAGAS?**
   Saat ini: accuracy, hallucination rate, 4 RAGAS metrics. Apakah perlu F1-score per label?

3. **Apakah 500 sampel cukup untuk demonstrasi sementara?**
   Atau langsung evaluasi 1000 sampel dari awal untuk semua konfigurasi?

4. **Bagaimana desain Active Hallucination Detection?**
   Ini masih placeholder. Opsi: NLI-based (misal DeBERTa), self-consistency, atau LLM-as-judge.

---

## 5. File yang Sudah Dibuat

| File | Keterangan |
|---|---|
| `notebooks/03-RAG-Baseline-Evaluation.ipynb` | Notebook baseline (BM25 + RAGAS), 500 sampel |
| `results/baseline_phase1_answers.json` | 500 jawaban LLM + label prediksi |
| `results/baseline_phase2_ragas.json` | 50 sampel RAGAS scores |
| `data/pubmedqa_bm25.pkl` | Indeks BM25 tersimpan |
| `Laporan TA/Bab IV - Perancangan.tex` | Template Bab IV LaTeX (lengkap) |
| `Laporan TA/Bab V - Implementasi.tex` | Template Bab V LaTeX (placeholder) |
| `Laporan TA/Bab VI - Evaluasi.tex` | Template Bab VI LaTeX (placeholder + rumus RAGAS) |
