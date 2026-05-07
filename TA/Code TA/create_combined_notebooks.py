"""
Script untuk generate 2 notebook baru:
- 05-RAG-QR-CR.ipynb: Combined QR + CR dengan Llama 3.2
- 05.1-RAG-QR-CR-OpenAI.ipynb: Combined QR + CR dengan GPT-4.1-mini

Pipeline: original query -> QR rewrite -> BM25 top-20 (rewritten) -> CrossEncoder rerank top-5 -> generate (original question)
Evaluasi: custom zero-NaN 4 metrik (faithfulness, context_recall, answer_relevancy, context_precision)
"""

import json
import uuid
from pathlib import Path

NB_DIR = Path("C:/Users/Ricky Wijaya/Documents/STI/Semester 8/TA/Code TA/notebooks")


# ============================================================
# HELPER
# ============================================================

def new_id():
    return uuid.uuid4().hex[:8]


def mdcell(content, cell_id=None):
    return {
        "cell_type": "markdown",
        "id": cell_id or new_id(),
        "metadata": {},
        "source": [line + '\n' for line in content.split('\n')[:-1]] + [content.split('\n')[-1]]
    }


def codecell(content, cell_id=None):
    return {
        "cell_type": "code",
        "id": cell_id or new_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + '\n' for line in content.split('\n')[:-1]] + [content.split('\n')[-1]]
    }


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


# ============================================================
# SHARED CELL CONTENTS
# ============================================================

CELL_DATACLASS = '''@dataclass
class Document:
    text         : str
    pubid        : str
    question     : str
    section_label: str
    answer       : str
    decision     : str


@dataclass
class RetrievalResult:
    document       : Document
    score          : float           # BM25 score
    reranker_score : float = 0.0     # CrossEncoder score (diisi setelah rerank)


def tokenize_bm25(text: str) -> List[str]:
    """Tokenizer untuk BM25: hapus tanda baca, lowercase, split spasi."""
    return re.sub(r'[^a-zA-Z0-9\\s]', ' ', text.lower()).split()


sample_text = 'Does aspirin (75mg) reduce myocardial infarction risk?'
print(f'Tokenisasi BM25: {tokenize_bm25(sample_text)}')
print('Data classes dan tokenizer siap.')'''


CELL_DATASET = '''def load_pubmedqa(subset=DATASET_SUBSET, max_samples=MAX_SAMPLES):
    print(f'Memuat PubMedQA ({subset})...')
    dataset = load_dataset(DATASET_NAME, subset, trust_remote_code=True)
    data    = dataset['train']
    if max_samples and len(data) > max_samples:
        data = data.select(range(max_samples))
    print(f'Dimuat {len(data)} sampel')
    return data


def prepare_documents(data) -> List[Document]:
    docs = []
    for item in data:
        pubid = str(item['pubid'])
        for ctx, label in zip(item['context']['contexts'], item['context']['labels']):
            docs.append(Document(
                text=ctx.strip(), pubid=pubid,
                question=item['question'], section_label=label,
                answer=item['long_answer'], decision=item['final_decision']
            ))
    print(f'Total potongan dokumen: {len(docs)}')
    return docs


def load_or_build_bm25(data) -> Tuple[BM25Okapi, List[Document]]:
    """Muat BM25 index dari file jika ada, atau bangun dari scratch."""
    if BM25_INDEX_PATH.exists():
        print(f'Memuat BM25 index dari {BM25_INDEX_PATH}...')
        with open(BM25_INDEX_PATH, 'rb') as f:
            saved = pickle.load(f)
        print(f'Dimuat: {len(saved["documents"])} dokumen')
        return saved['bm25'], saved['documents']
    else:
        print('Membangun BM25 index baru...')
        documents = prepare_documents(data)
        tokenized = [tokenize_bm25(d.text) for d in documents]
        bm25      = BM25Okapi(tokenized)
        with open(BM25_INDEX_PATH, 'wb') as f:
            pickle.dump({'bm25': bm25, 'documents': documents}, f)
        print(f'Index disimpan ke {BM25_INDEX_PATH}')
        return bm25, documents


# Load dataset + BM25 index (shared dengan notebook lain)
_full_data            = load_dataset(DATASET_NAME, DATASET_SUBSET, trust_remote_code=True)['train']
bm25_index, documents = load_or_build_bm25(_full_data.select(range(500)))
pubmedqa_data         = _full_data.select(range(MAX_SAMPLES))
print(f'\\nEvaluasi akan menggunakan {len(pubmedqa_data)} sampel pertama.')'''


CELL_LOAD_RERANKER = '''print(f'Memuat CrossEncoder: {RERANKER_MODEL}...')
print('(Download ~85MB sekali, lalu di-cache)')
t0 = time.time()
cross_encoder = CrossEncoder(RERANKER_MODEL)
print(f'CrossEncoder siap dalam {time.time()-t0:.1f} detik')

# Smoke test
_pairs = [
    ('Does aspirin prevent heart attacks?', 'Aspirin reduces platelet aggregation and is used in cardiovascular prevention.'),
    ('Does aspirin prevent heart attacks?', 'Weather patterns affect agricultural yields in tropical regions.'),
]
_scores = cross_encoder.predict(_pairs)
print(f'\\nSmoke test CrossEncoder:')
print(f'  Relevan       : {_scores[0]:.4f}')
print(f'  Tidak relevan : {_scores[1]:.4f}')
assert _scores[0] > _scores[1], 'CrossEncoder gagal membedakan relevan vs tidak!'
print('Reranker berfungsi dengan benar.')'''


CELL_EXTRACT_LABEL = '''def extract_label(answer: str) -> str:
    """Ekstrak prediksi yes/no/maybe dari teks jawaban."""
    lines = [l.strip().lower() for l in answer.split('\\n') if l.strip()]
    for line in reversed(lines[-3:]):
        word = re.sub(r'[^a-z]', '', line)
        if word in ('yes', 'no', 'maybe'):
            return word
    for label in ('yes', 'no', 'maybe'):
        if re.search(r'\\b' + label + r'\\b', answer.lower()):
            return label
    return 'maybe'


cases = [
    ('Strong evidence.\\nyes', 'yes'),
    ('No effect found.\\nno',  'no'),
    ('Mixed results.\\nmaybe', 'maybe'),
    ('Verdict: yes.',          'yes'),
    ('Totally unclear.',       'maybe'),
]
all_ok = all(extract_label(txt) == exp for txt, exp in cases)
print(f'Unit test extract_label: {"PASS" if all_ok else "FAIL"}')
print(f'Label dari test answer: {extract_label(test_ans)!r}')'''


GENERATION_PROMPT_BODY = '''GENERATION_PROMPT = (
    'You are a medical research assistant. '
    'Answer a biomedical yes/no/maybe question based solely on the provided scientific abstracts.\\n\\n'
    'Context from medical literature:\\n{context}\\n\\n'
    'Question: {question}\\n\\n'
    'Instructions:\\n'
    '- Carefully read the context and assess whether it supports or refutes the question.\\n'
    '- Provide a brief explanation (2-3 sentences) using ONLY the information above.\\n'
    '- End your response with EXACTLY ONE of these words on its own line: yes, no, or maybe.\\n'
    '  - yes   : the evidence supports the hypothesis, even if not perfectly conclusive\\n'
    '  - no    : the evidence refutes or does not support the hypothesis\\n'
    '  - maybe : ONLY if the evidence is directly contradictory (some findings say yes,\\n'
    '            others say no), or if the context contains no relevant information at all\\n'
    '- IMPORTANT: If the evidence leans in one direction, even partially, choose yes or no.\\n'
    '  Do NOT use maybe simply because the evidence is limited or not 100%% certain.\\n\\n'
    'Answer:'
)'''


QUERY_REWRITE_PROMPT_BODY = '''QUERY_REWRITE_PROMPT = (
    'You are a query rewriting assistant for a biomedical question-answering system.\\n'
    'Rewrite the following medical question to improve retrieval from a PubMed research database.\\n\\n'
    'Rules:\\n'
    '1. Be more specific and add relevant medical/scientific terminology.\\n'
    '2. Expand abbreviations (e.g. "MI" -> "myocardial infarction").\\n'
    '3. Preserve the original yes/no/maybe answerable intent.\\n'
    '4. Output ONLY the rewritten question, no explanations.\\n\\n'
    'Original question: {query}\\n\\n'
    'Rewritten question:'
)'''


# ============================================================
# NOTEBOOK 05: Llama 3.2 (Ollama)
# ============================================================

def build_notebook_05_llama():
    cells = []

    # Title
    cells.append(mdcell('''# 05 — RAG Combined (Query Rewriting + Context Reranking)

Notebook ini mengevaluasi kombinasi dua teknik mitigasi halusinasi:
1. **Query Rewriting** (QR) — reformulasi pertanyaan dengan LLM sebelum retrieval
2. **Context Reranking** (CR) — CrossEncoder menilai relevansi kandidat dari BM25

**Pipeline:**
```
original query
  -> QR (rewrite via LLM)
  -> BM25 retrieve top-20 (pakai rewritten query)
  -> CrossEncoder rerank top-5 (pakai rewritten query)
  -> LLM generate (pakai ORIGINAL question)
```

**Model:** `llama3.2` (3B parameter) via Ollama
**Evaluator:** Custom zero-NaN 4 metrik (faithfulness, context_recall, answer_relevancy, context_precision)'''))

    # Install
    cells.append(codecell('''# Install dependencies (jalankan sekali saja)
# !pip install ollama rank-bm25 sentence-transformers datasets'''))

    # Imports
    cells.append(codecell('''import os, sys, json, pickle, time, re, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime

from rank_bm25 import BM25Okapi
import ollama
from datasets import load_dataset

try:
    from sentence_transformers import CrossEncoder
    print('sentence-transformers tersedia.')
except ImportError:
    print('sentence-transformers belum terinstall!')
    print('Jalankan: pip install sentence-transformers')
    raise

warnings.filterwarnings('ignore')
print('Semua library berhasil diimpor!')
print(f'Python: {sys.version.split()[0]} | NumPy: {np.__version__}')'''))

    # Config
    cells.append(codecell('''# ============================================================
# KONFIGURASI
# ============================================================
LLM_MODEL        = 'llama3.2'
RERANKER_MODEL   = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

TOP_K_CANDIDATES = 20   # BM25 ambil 20 kandidat
TOP_K_RETRIEVAL  = 5    # Setelah rerank, ambil top-5

DATASET_NAME   = 'qiaojin/PubMedQA'
DATASET_SUBSET = 'pqa_labeled'
MAX_SAMPLES    = 500

TEMPERATURE = 0.0
SEED        = 42

NOTEBOOK_DIR    = Path('.')
BM25_INDEX_PATH = NOTEBOOK_DIR / 'pubmedqa_bm25.pkl'
RESULTS_DIR     = Path('../results')
RESULTS_DIR.mkdir(exist_ok=True)

CONFIG_NAME        = 'combined'
PHASE1_PATH        = RESULTS_DIR / f'{CONFIG_NAME}_phase1_answers.json'
PHASE2_CUSTOM_PATH = RESULTS_DIR / f'{CONFIG_NAME}_phase2_custom.json'
FINAL_CSV_PATH     = RESULTS_DIR / f'{CONFIG_NAME}_results.csv'

print('Konfigurasi:')
print(f'  LLM          : {LLM_MODEL} (via Ollama)')
print(f'  Reranker     : {RERANKER_MODEL}')
print(f'  Retriever    : BM25 top-{TOP_K_CANDIDATES} -> CrossEncoder -> top-{TOP_K_RETRIEVAL}')
print(f'  Query Rewrite: ENABLED (pakai {LLM_MODEL})')
print(f'  Sampel       : {MAX_SAMPLES}')
print(f'  Config       : {CONFIG_NAME}')
print(f'  Output P1    : {PHASE1_PATH}')
print(f'  Output P2    : {PHASE2_CUSTOM_PATH}')'''))

    # Dataclass
    cells.append(codecell(CELL_DATACLASS))

    # Dataset
    cells.append(codecell(CELL_DATASET))

    # Reranker
    cells.append(codecell(CELL_LOAD_RERANKER))

    # QR - Llama version
    cells.append(codecell(QUERY_REWRITE_PROMPT_BODY + '''


def rewrite_query(query: str) -> str:
    """Reformulasi query pakai LLM. Fallback ke query asli kalau gagal / terlalu pendek."""
    prompt = QUERY_REWRITE_PROMPT.format(query=query)
    try:
        response = ollama.generate(
            model=LLM_MODEL, prompt=prompt,
            options={'temperature': 0.3, 'seed': SEED, 'num_predict': 150}
        )
        rewritten = response['response'].strip().replace('\\n', ' ')
        return rewritten if len(rewritten) >= 10 else query
    except Exception as e:
        print(f'  [QR Error] {e} -- pakai query asli')
        return query


# Test
print('Contoh Query Rewriting:')
print('=' * 70)
for q in ['Does aspirin reduce the risk of MI?',
          'Can exercise prevent T2DM?']:
    rw = rewrite_query(q)
    print(f'\\nAsli    : {q}')
    print(f'Rewrite : {rw}')'''))

    # Combined retrieve
    cells.append(codecell('''def retrieve_with_qr_cr(
    query: str,
    k_candidates: int = TOP_K_CANDIDATES,
    k_final: int = TOP_K_RETRIEVAL
) -> Tuple[List[RetrievalResult], str]:
    """
    Pipeline gabungan: QR + BM25 + CR.

    Urutan:
      1. QR: reformulasi query asli -> rewritten
      2. BM25: ambil top-k_candidates pakai rewritten
      3. CrossEncoder: rerank pakai rewritten, ambil top-k_final

    Returns:
      (retrieved_docs, rewritten_query)
      retrieved_docs punya atribut .score (BM25) dan .reranker_score
    """
    # 1. Query Rewriting
    rewritten = rewrite_query(query)

    # 2. BM25 retrieval pakai rewritten
    tokens     = tokenize_bm25(rewritten)
    scores     = bm25_index.get_scores(tokens)
    top_cands  = np.argsort(scores)[::-1][:k_candidates]
    candidates = [
        RetrievalResult(document=documents[i], score=float(scores[i]))
        for i in top_cands
    ]

    # 3. CrossEncoder reranking pakai rewritten
    pairs           = [(rewritten, r.document.text) for r in candidates]
    reranker_scores = cross_encoder.predict(pairs)
    for r, rs in zip(candidates, reranker_scores):
        r.reranker_score = float(rs)

    # Sort by reranker score
    reranked = sorted(candidates, key=lambda r: r.reranker_score, reverse=True)
    return reranked[:k_final], rewritten


# Test
test_q = 'Does aspirin reduce the risk of myocardial infarction?'
test_r, test_rw = retrieve_with_qr_cr(test_q)
print(f'Query asli : {test_q}')
print(f'Rewrite    : {test_rw}')
print(f'\\nTop-{TOP_K_RETRIEVAL} dokumen (QR -> BM25 top-{TOP_K_CANDIDATES} -> CR):')
for i, r in enumerate(test_r, 1):
    print(f'  [{i}] BM25={r.score:.2f} | Reranker={r.reranker_score:.4f} | {r.document.section_label} | {r.document.text[:70]}...')'''))

    # Generate
    cells.append(codecell(GENERATION_PROMPT_BODY + '''


def generate_answer(query: str, retrieved: List[RetrievalResult]) -> str:
    """Generate jawaban. PENTING: pakai ORIGINAL query, bukan rewritten."""
    context = '\\n\\n'.join(
        f'[{i}] ({r.document.section_label}): {r.document.text}'
        for i, r in enumerate(retrieved, 1)
    )
    response = ollama.generate(
        model=LLM_MODEL,
        prompt=GENERATION_PROMPT.format(context=context, question=query),
        options={'temperature': TEMPERATURE, 'seed': SEED, 'num_predict': 300}
    )
    return response['response'].strip()


# Test
test_ans = generate_answer(test_q, test_r)
print('Output generation:')
print('-' * 60)
print(test_ans)
print('-' * 60)'''))

    # Extract label
    cells.append(codecell(CELL_EXTRACT_LABEL))

    # Custom Eval - Llama version (4 metrics)
    cells.append(codecell(CUSTOM_EVAL_LLAMA))

    # Demo
    cells.append(mdcell('## Demo — 5 Sampel Pertama'))
    cells.append(codecell('''DEMO_SIZE    = 5
demo_results = []

print(f'DEMO: {DEMO_SIZE} sampel pertama ({LLM_MODEL}, QR+CR)')
print('=' * 65)

for i in range(DEMO_SIZE):
    s         = pubmedqa_data[i]
    q         = s['question']
    gt        = s['final_decision']

    retrieved, rewritten = retrieve_with_qr_cr(q)
    answer               = generate_answer(q, retrieved)
    predicted            = extract_label(answer)
    correct              = predicted == gt

    demo_results.append({
        'idx': i, 'question': q, 'rewritten_query': rewritten,
        'ground_truth': gt, 'predicted_label': predicted,
        'is_correct': correct, 'answer': answer,
    })

    verdict = 'BENAR' if correct else 'SALAH'
    print(f'\\n[{i+1}/{DEMO_SIZE}] {q[:75]}...')
    print(f'  Rewritten: {rewritten[:80]}...')
    print(f'  GT={gt} | Pred={predicted} | {verdict}')
    print(f'  Jawaban  : {answer[:120]}...')

n_ok = sum(r['is_correct'] for r in demo_results)
print(f'\\n{"="*65}')
print(f'Demo Accuracy: {n_ok}/{DEMO_SIZE} = {n_ok/DEMO_SIZE:.1%}')'''))

    # Phase 1
    cells.append(mdcell('''## Phase 1 — Generate Jawaban (500 Sampel)

Estimasi waktu Llama 3.2 lokal: ~5-8 jam untuk 500 sampel (QR + CR menambah overhead).
Resume otomatis jika interrupted.'''))
    cells.append(codecell('''if PHASE1_PATH.exists():
    with open(PHASE1_PATH, 'r', encoding='utf-8') as f:
        phase1_results = json.load(f)['results']
    start_from = len(phase1_results)
    print(f'Resume Fase 1: {start_from}/{MAX_SAMPLES} sudah selesai.')
else:
    phase1_results, start_from = [], 0
    print(f'Memulai Fase 1: {MAX_SAMPLES} sampel (QR + BM25 top-{TOP_K_CANDIDATES} -> CR -> top-{TOP_K_RETRIEVAL}).')

if start_from < MAX_SAMPLES:
    print(f'Memproses {MAX_SAMPLES - start_from} sampel tersisa...\\n')
    t_start = time.time()

    for i in range(start_from, MAX_SAMPLES):
        s          = pubmedqa_data[i]
        q, gt, ref = s['question'], s['final_decision'], s['long_answer']

        retrieved, rewritten = retrieve_with_qr_cr(q)
        answer               = generate_answer(q, retrieved)
        predicted            = extract_label(answer)

        phase1_results.append({
            'idx'             : i,
            'pubid'           : str(s['pubid']),
            'question'        : q,
            'rewritten_query' : rewritten,
            'ground_truth'    : gt,
            'predicted_label' : predicted,
            'is_correct'      : predicted == gt,
            'answer'          : answer,
            'contexts'        : [r.document.text for r in retrieved],
            'reference'       : ref,
            'retrieval_scores': [r.score for r in retrieved],
            'reranker_scores' : [r.reranker_score for r in retrieved],
        })

        if (i + 1) % 10 == 0 or i == MAX_SAMPLES - 1:
            with open(PHASE1_PATH, 'w', encoding='utf-8') as f:
                json.dump({'config': CONFIG_NAME,
                           'llm_model': LLM_MODEL,
                           'timestamp': datetime.now().isoformat(),
                           'max_samples': MAX_SAMPLES, 'completed': i+1,
                           'results': phase1_results}, f, indent=2, ensure_ascii=False)
            done = i + 1
            acc  = sum(r['is_correct'] for r in phase1_results) / done
            eta  = (time.time()-t_start) / done * (MAX_SAMPLES-done) / 60
            print(f'  [{done:3d}/{MAX_SAMPLES}] Akurasi: {acc:.1%} | pred={predicted}, gt={gt} | ETA {eta:.1f} mnt')

    print(f'\\nFase 1 selesai! Disimpan ke {PHASE1_PATH}')
else:
    print(f'Fase 1 sudah selesai ({MAX_SAMPLES} sampel).')'''))

    # Phase 1 analysis
    cells.append(mdcell('## Analisis Phase 1'))
    cells.append(codecell(PHASE1_ANALYSIS))

    # Phase 2
    cells.append(mdcell('''## Phase 2 — Custom Evaluator 4 Metrik (500 Sampel)

**Metrik:**
1. Faithfulness — kalimat jawaban didukung konteks?
2. Context Recall — fakta reference tercakup konteks?
3. Answer Relevancy — kalimat jawaban relevan dengan pertanyaan?
4. Context Precision — konteks relevan di rank atas? (Average Precision)

Estimasi waktu: ~1-2 jam untuk 500 sampel.
Smart resume: deteksi sampel dengan metrik lengkap dan hanya upgrade yang perlu.'''))
    cells.append(codecell(PHASE2_LOOP))

    # Summary
    cells.append(mdcell('## Summary — Hasil Akhir'))
    cells.append(codecell(SUMMARY_CELL_LLAMA))

    return make_notebook(cells)


# ============================================================
# SHARED: Custom Evaluator (Llama version)
# ============================================================

CUSTOM_EVAL_LLAMA = '''def _split_sentences(text: str) -> List[str]:
    """Pecah teks menjadi kalimat. Filter kalimat terlalu pendek (<15 char)."""
    parts = re.split(r'(?<=[.!?])\\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) >= 15]


def _llm_yes_no(prompt: str) -> bool:
    """Tanya LLM ya/tidak. Return True=yes, False=no. Fallback False jika gagal."""
    try:
        resp = ollama.generate(
            model=LLM_MODEL, prompt=prompt,
            options={'temperature': 0, 'seed': SEED, 'num_predict': 10}
        )
        return 'yes' in resp['response'].strip().lower()[:15]
    except Exception:
        return False


def compute_faithfulness(answer: str, contexts: List[str]) -> float:
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0
    ctx_text = '\\n'.join(f'[{i+1}] {c[:400]}' for i, c in enumerate(contexts))
    prompt_tmpl = (
        'Context:\\n{ctx}\\n\\n'
        'Statement: {sent}\\n\\n'
        'Is this statement directly supported by the context above? '
        'Answer with only "yes" or "no".'
    )
    supported = sum(1 for s in sentences if _llm_yes_no(prompt_tmpl.format(ctx=ctx_text, sent=s)))
    return supported / len(sentences)


def compute_context_recall(reference: str, contexts: List[str]) -> float:
    sentences = _split_sentences(reference)
    if not sentences:
        return 0.0
    ctx_text = '\\n'.join(f'[{i+1}] {c[:400]}' for i, c in enumerate(contexts))
    prompt_tmpl = (
        'Context:\\n{ctx}\\n\\n'
        'Statement: {sent}\\n\\n'
        'Is this statement supported by the context above? '
        'Answer with only "yes" or "no".'
    )
    covered = sum(1 for s in sentences if _llm_yes_no(prompt_tmpl.format(ctx=ctx_text, sent=s)))
    return covered / len(sentences)


def compute_answer_relevancy(question: str, answer: str) -> float:
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0
    prompt_tmpl = (
        'Question: {question}\\n\\n'
        'Statement: {sent}\\n\\n'
        'Is this statement relevant to answering the question above? '
        'Answer with only "yes" or "no".'
    )
    relevant = sum(1 for s in sentences if _llm_yes_no(prompt_tmpl.format(question=question, sent=s)))
    return relevant / len(sentences)


def compute_context_precision(question: str, contexts: List[str], reference: str) -> float:
    if not contexts:
        return 0.0
    prompt_tmpl = (
        'Question: {question}\\n\\n'
        'Ground truth answer: {reference}\\n\\n'
        'Retrieved context: {ctx}\\n\\n'
        'Does this context contain information useful for correctly answering '
        'the question based on the ground truth? Answer with only "yes" or "no".'
    )
    relevance = []
    for ctx in contexts:
        is_rel = _llm_yes_no(prompt_tmpl.format(
            question=question, reference=reference[:300], ctx=ctx[:400]
        ))
        relevance.append(1 if is_rel else 0)
    total_relevant = sum(relevance)
    if total_relevant == 0:
        return 0.0
    precision_sum = 0.0
    relevant_count = 0
    for k, rel in enumerate(relevance):
        if rel:
            relevant_count += 1
            precision_sum += relevant_count / (k + 1)
    return precision_sum / total_relevant


def evaluate_custom(question: str, answer: str,
                    contexts: List[str], reference: str) -> Dict:
    return {
        'faithfulness'      : compute_faithfulness(answer, contexts),
        'context_recall'    : compute_context_recall(reference, contexts),
        'answer_relevancy'  : compute_answer_relevancy(question, answer),
        'context_precision' : compute_context_precision(question, contexts, reference),
    }


# Smoke test
_ctx = ['Aspirin reduces blood clotting and is used for heart attack prevention.']
_ans = 'Aspirin helps prevent heart attacks. It works by reducing clotting.'
_ref = 'Aspirin is used for heart attack prevention by reducing blood clotting.'
_q   = 'Does aspirin prevent heart attacks?'
_r   = evaluate_custom(_q, _ans, _ctx, _ref)
print('Smoke test evaluate_custom (4 metrik):')
print(f'  faithfulness      = {_r["faithfulness"]:.3f}')
print(f'  context_recall    = {_r["context_recall"]:.3f}')
print(f'  answer_relevancy  = {_r["answer_relevancy"]:.3f}')
print(f'  context_precision = {_r["context_precision"]:.3f}')
print('Zero-NaN evaluator siap (4 metrik).')'''


# ============================================================
# SHARED: Custom Evaluator (OpenAI version)
# ============================================================

CUSTOM_EVAL_OPENAI = CUSTOM_EVAL_LLAMA.replace(
    '''def _llm_yes_no(prompt: str) -> bool:
    """Tanya LLM ya/tidak. Return True=yes, False=no. Fallback False jika gagal."""
    try:
        resp = ollama.generate(
            model=LLM_MODEL, prompt=prompt,
            options={'temperature': 0, 'seed': SEED, 'num_predict': 10}
        )
        return 'yes' in resp['response'].strip().lower()[:15]
    except Exception:
        return False''',
    '''def _llm_yes_no(prompt: str) -> bool:
    """Tanya LLM ya/tidak via OpenAI. Return True=yes, False=no. Fallback False jika gagal."""
    try:
        resp = openai_generate(prompt, max_tokens=10, temperature=0.0)
        return 'yes' in resp.lower()[:15]
    except Exception:
        return False'''
)


# ============================================================
# SHARED: Phase 1 Analysis
# ============================================================

PHASE1_ANALYSIS = '''with open(PHASE1_PATH, 'r', encoding='utf-8') as f:
    results_p1 = json.load(f)['results']

n         = len(results_p1)
n_correct = sum(r['is_correct'] for r in results_p1)
gts       = [r['ground_truth']    for r in results_p1]
preds     = [r['predicted_label'] for r in results_p1]

print(f'ANALISIS PHASE 1 — {n} sampel ({CONFIG_NAME})')
print('=' * 55)
print(f'Label Accuracy    : {n_correct}/{n} = {n_correct/n:.1%}')
print(f'Hallucination Rate: {(n-n_correct)/n:.1%}\\n')

print(f'  {"Label":<8} | {"Ground Truth":>12} | {"Prediksi":>10}')
print(f'  {"-"*40}')
for lbl in ['yes','no','maybe']:
    g, p = gts.count(lbl), preds.count(lbl)
    print(f'  {lbl:<8} | {g:>6} ({g/n:.0%})    | {p:>6} ({p/n:.0%})')

print('\\nConfusion Matrix (baris=GT, kolom=Pred):')
lbls = ['yes','no','maybe']
print('  ' + f'{"GT/Pred":>8}' + ''.join(f'{l:>8}' for l in lbls))
for gt_l in lbls:
    row = f'  {gt_l:>8}'
    for pr_l in lbls:
        cnt = sum(1 for r in results_p1 if r['ground_truth']==gt_l and r['predicted_label']==pr_l)
        row += f'{cnt:>8}'
    print(row)

# Rata-rata skor BM25 dan Reranker
avg_bm25     = np.mean([np.mean(r['retrieval_scores']) for r in results_p1])
avg_reranker = np.mean([np.mean(r['reranker_scores'])  for r in results_p1])
print(f'\\nRata-rata skor BM25     : {avg_bm25:.4f}')
print(f'Rata-rata skor Reranker : {avg_reranker:.4f}')'''


# ============================================================
# SHARED: Phase 2 Loop (smart resume)
# ============================================================

PHASE2_LOOP = '''MAX_CUSTOM_SAMPLES = 500
REQUIRED_METRICS = ['faithfulness', 'context_recall', 'answer_relevancy', 'context_precision']

with open(PHASE1_PATH, 'r', encoding='utf-8') as f:
    p1_custom = json.load(f)['results'][:MAX_CUSTOM_SAMPLES]

if PHASE2_CUSTOM_PATH.exists():
    with open(PHASE2_CUSTOM_PATH, 'r', encoding='utf-8') as f:
        p2_custom = json.load(f)['results']
    done_custom = {r['idx'] for r in p2_custom if all(m in r for m in REQUIRED_METRICS)}
    needs_upgrade = [r for r in p2_custom if not all(m in r for m in REQUIRED_METRICS)]
    print(f'Resume: {len(done_custom)}/{MAX_CUSTOM_SAMPLES} selesai dengan 4 metrik.')
    if needs_upgrade:
        print(f'Perlu upgrade: {len(needs_upgrade)} sampel.')
else:
    p2_custom, done_custom, needs_upgrade = [], set(), []
    print(f'Mulai: {MAX_CUSTOM_SAMPLES} sampel (custom zero-NaN, 4 metrik).')

# Tahap 1: Upgrade
if needs_upgrade:
    print(f'\\nTahap 1: Upgrade {len(needs_upgrade)} sampel...')
    t_up = time.time()
    p1_lookup = {r['idx']: r for r in p1_custom}
    for i, r in enumerate(needs_upgrade):
        src = p1_lookup[r['idx']]
        if 'answer_relevancy' not in r:
            r['answer_relevancy'] = compute_answer_relevancy(src['question'], src['answer'])
        if 'context_precision' not in r:
            r['context_precision'] = compute_context_precision(src['question'], src['contexts'], src['reference'])
        done_custom.add(r['idx'])
        if (i + 1) % 5 == 0 or i == len(needs_upgrade) - 1:
            with open(PHASE2_CUSTOM_PATH, 'w', encoding='utf-8') as f:
                json.dump({
                    'config': CONFIG_NAME, 'timestamp': datetime.now().isoformat(),
                    'max_samples': MAX_CUSTOM_SAMPLES,
                    'metrics': REQUIRED_METRICS,
                    'evaluator': 'custom_zero_nan_4metrics',
                    'results': p2_custom
                }, f, indent=2, ensure_ascii=False)
            done  = i + 1
            eta   = (time.time()-t_up)/done*(len(needs_upgrade)-done)/60 if done < len(needs_upgrade) else 0
            print(f'  upgrade [{done:3d}/{len(needs_upgrade)}] | ETA {eta:.1f} mnt')

# Tahap 2: Evaluasi sampel baru
remaining = [r for r in p1_custom if r['idx'] not in done_custom]
print(f'\\nTahap 2: Evaluasi {len(remaining)} sampel baru...\\n')

t0 = time.time()
for i, r in enumerate(remaining):
    scores = evaluate_custom(r['question'], r['answer'], r['contexts'], r['reference'])
    p2_custom.append({
        'idx': r['idx'], 'ground_truth': r['ground_truth'],
        'predicted_label': r['predicted_label'], 'is_correct': r['is_correct'],
        **scores
    })
    if (i + 1) % 5 == 0 or i == len(remaining) - 1:
        with open(PHASE2_CUSTOM_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                'config': CONFIG_NAME, 'timestamp': datetime.now().isoformat(),
                'max_samples': MAX_CUSTOM_SAMPLES,
                'metrics': REQUIRED_METRICS,
                'evaluator': 'custom_zero_nan_4metrics',
                'results': p2_custom
            }, f, indent=2, ensure_ascii=False)
        done  = i + 1
        total = len(remaining)
        eta   = (time.time()-t0)/done*(total-done)/60 if done < total else 0
        avg_f  = sum(x['faithfulness']      for x in p2_custom) / len(p2_custom)
        avg_cr = sum(x['context_recall']    for x in p2_custom) / len(p2_custom)
        avg_ar = sum(x['answer_relevancy']  for x in p2_custom) / len(p2_custom)
        avg_cp = sum(x['context_precision'] for x in p2_custom) / len(p2_custom)
        print(f'  [{done:3d}/{total}] idx={r["idx"]} | '
              f'f={scores["faithfulness"]:.2f} cr={scores["context_recall"]:.2f} '
              f'ar={scores["answer_relevancy"]:.2f} cp={scores["context_precision"]:.2f} | '
              f'avg: f={avg_f:.3f} cr={avg_cr:.3f} ar={avg_ar:.3f} cp={avg_cp:.3f} | ETA {eta:.1f}m')

print(f'\\nSelesai! -> {PHASE2_CUSTOM_PATH}')'''


# ============================================================
# SHARED: Summary
# ============================================================

def summary_cell(model_label):
    template = '''with open(PHASE2_CUSTOM_PATH, 'r', encoding='utf-8') as f:
    p2 = json.load(f)['results']

n      = len(p2)
acc    = sum(r['is_correct']        for r in p2) / n
avg_f  = sum(r['faithfulness']      for r in p2) / n
avg_cr = sum(r['context_recall']    for r in p2) / n
avg_ar = sum(r['answer_relevancy']  for r in p2) / n
avg_cp = sum(r['context_precision'] for r in p2) / n

print('=' * 65)
print(f'  {CONFIG_NAME.upper()} (QR + CR) — {n} sampel')
print(f'  LLM: {LLM_MODEL}')
print('=' * 65)
print(f'  Label Accuracy     : {acc:.1%}')
print(f'  Hallucination Rate : {1-acc:.1%}')
print(f'  Faithfulness       : {avg_f:.4f}')
print(f'  Context Recall     : {avg_cr:.4f}')
print(f'  Answer Relevancy   : {avg_ar:.4f}')
print(f'  Context Precision  : {avg_cp:.4f}')
print(f'  NaN count          : 0')
print('=' * 65)

# Per-label
print('\\nPer-label accuracy:')
for lbl in ['yes','no','maybe']:
    sub = [r for r in p2 if r['ground_truth'] == lbl]
    if sub:
        lbl_acc = sum(r['is_correct']        for r in sub) / len(sub)
        lbl_f   = sum(r['faithfulness']      for r in sub) / len(sub)
        lbl_cr  = sum(r['context_recall']    for r in sub) / len(sub)
        lbl_ar  = sum(r['answer_relevancy']  for r in sub) / len(sub)
        lbl_cp  = sum(r['context_precision'] for r in sub) / len(sub)
        print(f'  {lbl:>5}: acc={lbl_acc:.1%} (n={len(sub)}) | '
              f'faith={lbl_f:.3f} cr={lbl_cr:.3f} ar={lbl_ar:.3f} cp={lbl_cp:.3f}')

print(f'\\nBaris tabel skripsi (__MODEL__):')
print(f'  | Combined QR+CR (__MODEL__) | {acc:.3f} | {1-acc:.3f} | '
      f'{avg_f:.3f} | {avg_cr:.3f} | {avg_ar:.3f} | {avg_cp:.3f} |')'''
    return template.replace('__MODEL__', model_label)


SUMMARY_CELL_LLAMA  = summary_cell('Llama 3.2')
SUMMARY_CELL_OPENAI = summary_cell('GPT-4.1-mini')


# ============================================================
# NOTEBOOK 05.1: OpenAI GPT-4.1-mini
# ============================================================

def build_notebook_051_openai():
    cells = []

    cells.append(mdcell('''# 05.1 — RAG Combined (Query Rewriting + Context Reranking) dengan OpenAI

Notebook ini mengevaluasi kombinasi dua teknik mitigasi halusinasi dengan model OpenAI.

**Pipeline:**
```
original query
  -> QR (rewrite via OpenAI)
  -> BM25 retrieve top-20 (pakai rewritten query)
  -> CrossEncoder rerank top-5 (pakai rewritten query)
  -> OpenAI generate (pakai ORIGINAL question)
```

**Model:** `gpt-4.1-mini` via OpenAI API
**Evaluator:** Custom zero-NaN 4 metrik (faithfulness, context_recall, answer_relevancy, context_precision)'''))

    cells.append(codecell('''# Install dependencies (jalankan sekali saja)
# !pip install openai rank-bm25 sentence-transformers datasets'''))

    cells.append(codecell('''import os, sys, json, pickle, time, re, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime

from openai import OpenAI
from rank_bm25 import BM25Okapi
from datasets import load_dataset

try:
    from sentence_transformers import CrossEncoder
    print('sentence-transformers tersedia.')
except ImportError:
    print('sentence-transformers belum terinstall!')
    print('Jalankan: pip install sentence-transformers')
    raise

warnings.filterwarnings('ignore')
print('Semua library berhasil diimpor!')
print(f'Python: {sys.version.split()[0]} | NumPy: {np.__version__}')'''))

    cells.append(codecell('''# ============================================================
# KONFIGURASI
# ============================================================
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YOUR_API_KEY_HERE')

LLM_MODEL      = 'gpt-4.1-mini'  # Model OpenAI
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

TOP_K_CANDIDATES = 20
TOP_K_RETRIEVAL  = 5

DATASET_NAME   = 'qiaojin/PubMedQA'
DATASET_SUBSET = 'pqa_labeled'
MAX_SAMPLES    = 500

TEMPERATURE = 0.0
SEED        = 42

NOTEBOOK_DIR    = Path('.')
BM25_INDEX_PATH = NOTEBOOK_DIR / 'pubmedqa_bm25.pkl'
RESULTS_DIR     = Path('../results')
RESULTS_DIR.mkdir(exist_ok=True)

CONFIG_NAME        = 'combined_openai'
PHASE1_PATH        = RESULTS_DIR / f'{CONFIG_NAME}_phase1_answers.json'
PHASE2_CUSTOM_PATH = RESULTS_DIR / f'{CONFIG_NAME}_phase2_custom.json'
FINAL_CSV_PATH     = RESULTS_DIR / f'{CONFIG_NAME}_results.csv'

print('Konfigurasi:')
print(f'  LLM          : {LLM_MODEL} (via OpenAI API)')
print(f'  Reranker     : {RERANKER_MODEL}')
print(f'  Retriever    : BM25 top-{TOP_K_CANDIDATES} -> CrossEncoder -> top-{TOP_K_RETRIEVAL}')
print(f'  Query Rewrite: ENABLED (pakai {LLM_MODEL})')
print(f'  Sampel       : {MAX_SAMPLES}')
print(f'  Config       : {CONFIG_NAME}')
print(f'  Output P1    : {PHASE1_PATH}')
print(f'  Output P2    : {PHASE2_CUSTOM_PATH}')
print()
if 'YOUR_API_KEY' in OPENAI_API_KEY:
    print('  OPENAI_API_KEY belum diisi! Set env var atau isi di cell ini.')
else:
    print(f'  OPENAI_API_KEY: {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}')'''))

    cells.append(codecell(CELL_DATACLASS))
    cells.append(codecell(CELL_DATASET))
    cells.append(codecell(CELL_LOAD_RERANKER))

    # OpenAI client setup
    cells.append(codecell('''# ============================================================
# Setup OpenAI Client
# ============================================================
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def openai_generate(prompt: str, max_tokens: int = 300, temperature: float = TEMPERATURE) -> str:
    """Wrapper OpenAI API dengan retry otomatis."""
    for attempt in range(5):
        try:
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                seed=SEED,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                wait = (attempt + 1) * 10
                print(f'  [Rate limit] Tunggu {wait}s... (attempt {attempt+1}/5)')
                time.sleep(wait)
            elif '500' in err or '502' in err or '503' in err:
                wait = (attempt + 1) * 5
                print(f'  [Server error] Tunggu {wait}s...')
                time.sleep(wait)
            else:
                print(f'  [OpenAI Error] {type(e).__name__}: {err[:100]}')
                raise
    raise RuntimeError('OpenAI API gagal setelah 5 percobaan.')


# Smoke test
print('Testing OpenAI API...')
_test = openai_generate('Reply with exactly: OK', max_tokens=5)
print(f'Response: {_test!r}')
print('OpenAI client siap!')'''))

    # QR - OpenAI version
    cells.append(codecell(QUERY_REWRITE_PROMPT_BODY + '''


def rewrite_query(query: str) -> str:
    """Reformulasi query pakai OpenAI. Fallback ke query asli kalau gagal."""
    prompt = QUERY_REWRITE_PROMPT.format(query=query)
    try:
        rewritten = openai_generate(prompt, max_tokens=150, temperature=0.3)
        rewritten = rewritten.strip().replace('\\n', ' ')
        return rewritten if len(rewritten) >= 10 else query
    except Exception as e:
        print(f'  [QR Error] {e} -- pakai query asli')
        return query


# Test
print('Contoh Query Rewriting:')
print('=' * 70)
for q in ['Does aspirin reduce the risk of MI?',
          'Can exercise prevent T2DM?']:
    rw = rewrite_query(q)
    print(f'\\nAsli    : {q}')
    print(f'Rewrite : {rw}')'''))

    # Combined retrieve (same as Llama - uses cross_encoder + BM25, both local)
    cells.append(codecell('''def retrieve_with_qr_cr(
    query: str,
    k_candidates: int = TOP_K_CANDIDATES,
    k_final: int = TOP_K_RETRIEVAL
) -> Tuple[List[RetrievalResult], str]:
    """
    Pipeline gabungan: QR + BM25 + CR.

    1. QR: rewrite via OpenAI
    2. BM25: top-k_candidates pakai rewritten
    3. CrossEncoder: rerank pakai rewritten, top-k_final
    """
    rewritten = rewrite_query(query)

    tokens     = tokenize_bm25(rewritten)
    scores     = bm25_index.get_scores(tokens)
    top_cands  = np.argsort(scores)[::-1][:k_candidates]
    candidates = [
        RetrievalResult(document=documents[i], score=float(scores[i]))
        for i in top_cands
    ]

    pairs           = [(rewritten, r.document.text) for r in candidates]
    reranker_scores = cross_encoder.predict(pairs)
    for r, rs in zip(candidates, reranker_scores):
        r.reranker_score = float(rs)

    reranked = sorted(candidates, key=lambda r: r.reranker_score, reverse=True)
    return reranked[:k_final], rewritten


# Test
test_q = 'Does aspirin reduce the risk of myocardial infarction?'
test_r, test_rw = retrieve_with_qr_cr(test_q)
print(f'Query asli : {test_q}')
print(f'Rewrite    : {test_rw}')
print(f'\\nTop-{TOP_K_RETRIEVAL} dokumen (QR -> BM25 top-{TOP_K_CANDIDATES} -> CR):')
for i, r in enumerate(test_r, 1):
    print(f'  [{i}] BM25={r.score:.2f} | Reranker={r.reranker_score:.4f} | {r.document.section_label} | {r.document.text[:70]}...')'''))

    # Generate - OpenAI
    cells.append(codecell(GENERATION_PROMPT_BODY + '''


def generate_answer(query: str, retrieved: List[RetrievalResult]) -> str:
    """Generate jawaban via OpenAI. PENTING: pakai ORIGINAL query, bukan rewritten."""
    context = '\\n\\n'.join(
        f'[{i}] ({r.document.section_label}): {r.document.text}'
        for i, r in enumerate(retrieved, 1)
    )
    return openai_generate(
        GENERATION_PROMPT.format(context=context, question=query),
        max_tokens=300,
        temperature=TEMPERATURE
    )


# Test
test_ans = generate_answer(test_q, test_r)
print('Output generation:')
print('-' * 60)
print(test_ans)
print('-' * 60)'''))

    cells.append(codecell(CELL_EXTRACT_LABEL))
    cells.append(codecell(CUSTOM_EVAL_OPENAI))

    # Demo
    cells.append(mdcell('## Demo — 5 Sampel Pertama'))
    cells.append(codecell('''DEMO_SIZE    = 5
demo_results = []

print(f'DEMO: {DEMO_SIZE} sampel pertama ({LLM_MODEL}, QR+CR)')
print('=' * 65)

for i in range(DEMO_SIZE):
    s         = pubmedqa_data[i]
    q         = s['question']
    gt        = s['final_decision']

    retrieved, rewritten = retrieve_with_qr_cr(q)
    answer               = generate_answer(q, retrieved)
    predicted            = extract_label(answer)
    correct              = predicted == gt

    demo_results.append({
        'idx': i, 'question': q, 'rewritten_query': rewritten,
        'ground_truth': gt, 'predicted_label': predicted,
        'is_correct': correct, 'answer': answer,
    })

    verdict = 'BENAR' if correct else 'SALAH'
    print(f'\\n[{i+1}/{DEMO_SIZE}] {q[:75]}...')
    print(f'  Rewritten: {rewritten[:80]}...')
    print(f'  GT={gt} | Pred={predicted} | {verdict}')
    print(f'  Jawaban  : {answer[:120]}...')

n_ok = sum(r['is_correct'] for r in demo_results)
print(f'\\n{"="*65}')
print(f'Demo Accuracy: {n_ok}/{DEMO_SIZE} = {n_ok/DEMO_SIZE:.1%}')'''))

    # Phase 1
    cells.append(mdcell('''## Phase 1 — Generate Jawaban (500 Sampel)

Estimasi waktu GPT-4.1-mini: ~10-20 menit untuk 500 sampel (QR tambah overhead).
Resume otomatis jika interrupted.'''))
    cells.append(codecell('''if PHASE1_PATH.exists():
    with open(PHASE1_PATH, 'r', encoding='utf-8') as f:
        phase1_results = json.load(f)['results']
    start_from = len(phase1_results)
    print(f'Resume Fase 1: {start_from}/{MAX_SAMPLES} sudah selesai.')
else:
    phase1_results, start_from = [], 0
    print(f'Memulai Fase 1: {MAX_SAMPLES} sampel (QR + BM25 top-{TOP_K_CANDIDATES} -> CR -> top-{TOP_K_RETRIEVAL}).')

if start_from < MAX_SAMPLES:
    print(f'Memproses {MAX_SAMPLES - start_from} sampel tersisa...\\n')
    t_start = time.time()

    for i in range(start_from, MAX_SAMPLES):
        s          = pubmedqa_data[i]
        q, gt, ref = s['question'], s['final_decision'], s['long_answer']

        retrieved, rewritten = retrieve_with_qr_cr(q)
        answer               = generate_answer(q, retrieved)
        predicted            = extract_label(answer)

        phase1_results.append({
            'idx'             : i,
            'pubid'           : str(s['pubid']),
            'question'        : q,
            'rewritten_query' : rewritten,
            'ground_truth'    : gt,
            'predicted_label' : predicted,
            'is_correct'      : predicted == gt,
            'answer'          : answer,
            'contexts'        : [r.document.text for r in retrieved],
            'reference'       : ref,
            'retrieval_scores': [r.score for r in retrieved],
            'reranker_scores' : [r.reranker_score for r in retrieved],
        })

        if (i + 1) % 10 == 0 or i == MAX_SAMPLES - 1:
            with open(PHASE1_PATH, 'w', encoding='utf-8') as f:
                json.dump({'config': CONFIG_NAME,
                           'llm_model': LLM_MODEL,
                           'timestamp': datetime.now().isoformat(),
                           'max_samples': MAX_SAMPLES, 'completed': i+1,
                           'results': phase1_results}, f, indent=2, ensure_ascii=False)
            done = i + 1
            acc  = sum(r['is_correct'] for r in phase1_results) / done
            eta  = (time.time()-t_start) / done * (MAX_SAMPLES-done) / 60
            print(f'  [{done:3d}/{MAX_SAMPLES}] Akurasi: {acc:.1%} | pred={predicted}, gt={gt} | ETA {eta:.1f} mnt')

    print(f'\\nFase 1 selesai! Disimpan ke {PHASE1_PATH}')
else:
    print(f'Fase 1 sudah selesai ({MAX_SAMPLES} sampel).')'''))

    cells.append(mdcell('## Analisis Phase 1'))
    cells.append(codecell(PHASE1_ANALYSIS))

    cells.append(mdcell('''## Phase 2 — Custom Evaluator 4 Metrik (500 Sampel)

**Metrik:** Faithfulness, Context Recall, Answer Relevancy, Context Precision.
Estimasi waktu: ~15-30 menit untuk 500 sampel (dengan OpenAI).
Smart resume: deteksi sampel dengan metrik lengkap dan hanya upgrade yang perlu.'''))
    cells.append(codecell(PHASE2_LOOP))

    cells.append(mdcell('## Summary — Hasil Akhir'))
    cells.append(codecell(SUMMARY_CELL_OPENAI))

    return make_notebook(cells)


# ============================================================
# WRITE FILES
# ============================================================

def main():
    # Llama notebook
    nb_05 = build_notebook_05_llama()
    out_05 = NB_DIR / '05-RAG-QR-CR.ipynb'
    with open(out_05, 'w', encoding='utf-8') as f:
        json.dump(nb_05, f, indent=1, ensure_ascii=False)
    print(f'Saved: {out_05} ({len(nb_05["cells"])} cells)')

    # OpenAI notebook
    nb_051 = build_notebook_051_openai()
    out_051 = NB_DIR / '05.1-RAG-QR-CR-OpenAI.ipynb'
    with open(out_051, 'w', encoding='utf-8') as f:
        json.dump(nb_051, f, indent=1, ensure_ascii=False)
    print(f'Saved: {out_051} ({len(nb_051["cells"])} cells)')

    # Verify
    print('\nVerification:')
    for path in [out_05, out_051]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f'  {path.name}: VALID ({len(data["cells"])} cells)')
        except Exception as e:
            print(f'  {path.name}: CORRUPT! {e}')


if __name__ == '__main__':
    main()
