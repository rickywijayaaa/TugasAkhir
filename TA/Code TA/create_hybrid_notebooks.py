"""
Script untuk generate 2 notebook baru:
- 06-RAG-Hybrid-OpenAI.ipynb: BM25 + Dense (OpenAI embeddings) via RRF
- 07-RAG-Hybrid-CR-OpenAI.ipynb: Hybrid + CrossEncoder reranking

Pipeline Notebook 06:
    query -> BM25 top-50 + Dense top-50 -> RRF fusion -> top-5 -> generate

Pipeline Notebook 07:
    query -> BM25 top-50 + Dense top-50 -> RRF fusion -> top-20
          -> CrossEncoder rerank -> top-5 -> generate

Evaluator: custom zero-NaN 4 metrik (sama dengan notebook lain)
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
    lines = content.split('\n')
    return {
        "cell_type": "markdown",
        "id": cell_id or new_id(),
        "metadata": {},
        "source": [line + '\n' for line in lines[:-1]] + [lines[-1]]
    }


def codecell(content, cell_id=None):
    lines = content.split('\n')
    return {
        "cell_type": "code",
        "id": cell_id or new_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + '\n' for line in lines[:-1]] + [lines[-1]]
    }


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"}
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
    score          : float           # BM25 score (atau RRF score)
    doc_id         : int = -1        # index di documents list
    bm25_score     : float = 0.0
    dense_score    : float = 0.0
    rrf_score      : float = 0.0
    reranker_score : float = 0.0


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


_full_data            = load_dataset(DATASET_NAME, DATASET_SUBSET, trust_remote_code=True)['train']
bm25_index, documents = load_or_build_bm25(_full_data.select(range(500)))
pubmedqa_data         = _full_data.select(range(MAX_SAMPLES))
print(f'\\nEvaluasi akan menggunakan {len(pubmedqa_data)} sampel pertama.')'''


CELL_OPENAI_SETUP = '''# ============================================================
# Setup OpenAI Client
# ============================================================
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def openai_generate(prompt: str, max_tokens: int = 300, temperature: float = TEMPERATURE) -> str:
    """Wrapper OpenAI chat completion dengan retry."""
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
                print(f'  [Rate limit] Tunggu {wait}s...')
                time.sleep(wait)
            elif '500' in err or '502' in err or '503' in err:
                wait = (attempt + 1) * 5
                print(f'  [Server error] Tunggu {wait}s...')
                time.sleep(wait)
            else:
                print(f'  [OpenAI Error] {type(e).__name__}: {err[:100]}')
                raise
    raise RuntimeError('OpenAI API gagal setelah 5 percobaan.')


def openai_embed(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """Wrapper OpenAI embeddings dengan retry. Accepts list of texts, returns list of vectors."""
    for attempt in range(5):
        try:
            response = openai_client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in response.data]
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                wait = (attempt + 1) * 10
                print(f'  [Rate limit] Tunggu {wait}s...')
                time.sleep(wait)
            elif '500' in err or '502' in err or '503' in err:
                wait = (attempt + 1) * 5
                print(f'  [Server error] Tunggu {wait}s...')
                time.sleep(wait)
            else:
                print(f'  [Embed Error] {type(e).__name__}: {err[:100]}')
                raise
    raise RuntimeError('OpenAI embeddings gagal setelah 5 percobaan.')


# Smoke test
print('Testing OpenAI API...')
_test = openai_generate('Reply with exactly: OK', max_tokens=5)
print(f'  Chat response: {_test!r}')
_emb = openai_embed(['aspirin reduces heart attack risk'])
print(f'  Embed dim: {len(_emb[0])} (expected: 1536 for text-embedding-3-small)')
print('OpenAI client siap!')'''


CELL_CHROMA_INDEX = '''# ============================================================
# Build / Load ChromaDB index dengan OpenAI embeddings
# Embed sekali, di-cache di CHROMA_DB_PATH
# ============================================================

def build_or_load_chroma_index(documents: List[Document]) -> chromadb.Collection:
    """
    Build ChromaDB persistent collection dengan OpenAI embeddings.
    Kalau sudah ada dan count-nya match, load saja.
    """
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection_name = 'pubmedqa_docs'

    try:
        collection = chroma_client.get_collection(name=collection_name)
        count = collection.count()
        if count == len(documents):
            print(f'Load Chroma collection "{collection_name}" ({count} dokumen) dari {CHROMA_DB_PATH}')
            return collection
        else:
            print(f'Count mismatch: chroma={count}, documents={len(documents)}. Rebuild...')
            chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    print(f'Build Chroma collection ({len(documents)} dokumen)...')
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={'hnsw:space': 'cosine'}
    )

    # Batch embedding
    BATCH = 100
    t0 = time.time()
    for start in range(0, len(documents), BATCH):
        batch_docs = documents[start:start + BATCH]
        batch_texts = [d.text[:8000] for d in batch_docs]  # truncate untuk safety
        batch_ids   = [str(start + i) for i in range(len(batch_docs))]
        batch_meta  = [{'pubid': d.pubid, 'section': d.section_label} for d in batch_docs]

        embeddings = openai_embed(batch_texts)
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_meta,
            embeddings=embeddings,
        )
        done = start + len(batch_docs)
        eta  = (time.time()-t0)/done*(len(documents)-done)/60 if done < len(documents) else 0
        print(f'  [{done}/{len(documents)}] embedded | ETA {eta:.1f} mnt')

    print(f'Chroma index dibangun dalam {(time.time()-t0)/60:.1f} menit')
    return collection


chroma_collection = build_or_load_chroma_index(documents)
print(f'\\nChroma index siap: {chroma_collection.count()} dokumen')'''


CELL_DENSE_RRF = '''# ============================================================
# Dense retrieval + RRF fusion
# ============================================================

def retrieve_dense(query: str, k: int = 50) -> List[Tuple[int, float]]:
    """
    Dense retrieval via ChromaDB.
    Returns: list of (doc_id, distance_score) sorted by relevance.
    """
    qvec = openai_embed([query])[0]
    results = chroma_collection.query(
        query_embeddings=[qvec],
        n_results=k,
        include=['distances']
    )
    doc_ids   = [int(i) for i in results['ids'][0]]
    distances = results['distances'][0]  # cosine distance (lower = better)
    # Convert to similarity score (1 - distance) for consistency
    scores = [1.0 - d for d in distances]
    return list(zip(doc_ids, scores))


def retrieve_bm25_raw(query: str, k: int = 50) -> List[Tuple[int, float]]:
    """BM25 retrieval, return (doc_id, bm25_score) sorted desc."""
    tokens = tokenize_bm25(query)
    scores = bm25_index.get_scores(tokens)
    top_k  = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_k]


def reciprocal_rank_fusion(
    rank_lists: List[List[int]],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion.
    rank_lists: list of rank lists, each berisi doc_ids terurut.
    k: konstanta RRF (default 60 dari literatur).
    Returns: list of (doc_id, rrf_score) sorted desc.
    """
    scores = {}
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def retrieve_hybrid(
    query: str,
    k_bm25: int = TOP_K_BM25,
    k_dense: int = TOP_K_DENSE,
    k_final: int = TOP_K_RETRIEVAL
) -> List[RetrievalResult]:
    """
    Hybrid retrieval: BM25 top-k_bm25 + Dense top-k_dense, merged via RRF.
    Return top-k_final by RRF score.
    """
    bm25_results  = retrieve_bm25_raw(query, k=k_bm25)
    dense_results = retrieve_dense(query, k=k_dense)

    # Bangun lookup untuk skor asli
    bm25_scores  = dict(bm25_results)
    dense_scores = dict(dense_results)

    # RRF fusion
    rank_lists = [
        [doc_id for doc_id, _ in bm25_results],
        [doc_id for doc_id, _ in dense_results],
    ]
    fused = reciprocal_rank_fusion(rank_lists, k=60)

    # Build RetrievalResult dengan semua skor
    results = []
    for doc_id, rrf_score in fused[:k_final]:
        results.append(RetrievalResult(
            document=documents[doc_id],
            score=rrf_score,  # score utama = RRF
            doc_id=doc_id,
            bm25_score=bm25_scores.get(doc_id, 0.0),
            dense_score=dense_scores.get(doc_id, 0.0),
            rrf_score=rrf_score,
        ))
    return results


# Test retrieval
test_q = 'Does aspirin reduce the risk of myocardial infarction?'
test_r = retrieve_hybrid(test_q)
print(f'Query: {test_q}')
print(f'\\nTop-{TOP_K_RETRIEVAL} dokumen (Hybrid BM25+Dense via RRF):')
for i, r in enumerate(test_r, 1):
    print(f'  [{i}] RRF={r.rrf_score:.4f} | BM25={r.bm25_score:.2f} | Dense={r.dense_score:.3f} '
          f'| {r.document.section_label} | {r.document.text[:70]}...')'''


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


CELL_GENERATE = GENERATION_PROMPT_BODY + '''


def generate_answer(query: str, retrieved: List[RetrievalResult]) -> str:
    """Generate jawaban via OpenAI pakai original query."""
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
print('-' * 60)'''


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


CUSTOM_EVAL = '''# ============================================================
# Custom Zero-NaN Evaluator — 4 metrik via OpenAI
# ============================================================

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) >= 15]


def _llm_yes_no(prompt: str) -> bool:
    try:
        resp = openai_generate(prompt, max_tokens=10, temperature=0.0)
        return 'yes' in resp.lower()[:15]
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
_r   = evaluate_custom('Does aspirin prevent heart attacks?', _ans, _ctx, _ref)
print('Smoke test (4 metrik):')
for k, v in _r.items():
    print(f'  {k} = {v:.3f}')
print('Zero-NaN evaluator siap.')'''


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


def summary_cell(config_label):
    template = '''with open(PHASE2_CUSTOM_PATH, 'r', encoding='utf-8') as f:
    p2 = json.load(f)['results']

n      = len(p2)
acc    = sum(r['is_correct']        for r in p2) / n
avg_f  = sum(r['faithfulness']      for r in p2) / n
avg_cr = sum(r['context_recall']    for r in p2) / n
avg_ar = sum(r['answer_relevancy']  for r in p2) / n
avg_cp = sum(r['context_precision'] for r in p2) / n

print('=' * 65)
print(f'  {CONFIG_NAME.upper()} (__LABEL__) - {n} sampel')
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

print('\\nPer-label accuracy:')
for lbl in ['yes','no','maybe']:
    sub = [r for r in p2 if r['ground_truth'] == lbl]
    if sub:
        lbl_acc = sum(r['is_correct'] for r in sub) / len(sub)
        print(f'  {lbl:>5}: {sum(r["is_correct"] for r in sub)}/{len(sub)} = {lbl_acc:.1%}')

# Perbandingan dengan konfigurasi OpenAI sebelumnya
print('\\nPerbandingan dengan konfigurasi OpenAI sebelumnya:')
for prev_config, prev_path in [
    ('Baseline OpenAI', '../results/baseline_openai_phase2_custom.json'),
    ('QR OpenAI',       '../results/qr_openai_phase2_custom.json'),
    ('CR OpenAI',       '../results/cr_openai_phase2_custom.json'),
]:
    try:
        with open(prev_path, 'r', encoding='utf-8') as f:
            prev = json.load(f)['results']
        p_acc = sum(r['is_correct'] for r in prev) / len(prev)
        p_f   = sum(r['faithfulness'] for r in prev) / len(prev)
        delta = (acc - p_acc) * 100
        print(f'  {prev_config:<20}: acc={p_acc:.1%} faith={p_f:.4f} | delta acc = {delta:+.1f}%')
    except FileNotFoundError:
        print(f'  {prev_config:<20}: file tidak ditemukan')

print(f'\\nBaris tabel skripsi:')
print(f'  | __LABEL__ | {acc:.3f} | {1-acc:.3f} | '
      f'{avg_f:.3f} | {avg_cr:.3f} | {avg_ar:.3f} | {avg_cp:.3f} |')'''
    return template.replace('__LABEL__', config_label)


# ============================================================
# NOTEBOOK 06: Hybrid (BM25 + Dense via RRF)
# ============================================================

def build_notebook_06():
    cells = []

    cells.append(mdcell('''# 06 — RAG Hybrid Retrieval (BM25 + Dense via RRF)

Notebook ini mengevaluasi konfigurasi **Hybrid Retrieval** yang menggabungkan BM25 (sparse) dan Dense retrieval (OpenAI embeddings) menggunakan Reciprocal Rank Fusion (RRF).

**Pipeline:**
```
original query
  -> BM25 top-50 ranking
  -> Dense top-50 ranking (OpenAI text-embedding-3-small)
  -> RRF fusion
  -> top-5
  -> OpenAI generate
```

**Latar belakang:**
Berdasarkan Zhang & Zhang (2025), salah satu sumber halusinasi RAG adalah masalah retriever. BM25 unggul pada exact entity matching (istilah medis spesifik) tetapi lemah pada semantic matching (parafrasa, sinonim). Dense retrieval kebalikannya. Hybrid retrieval menggabungkan keduanya.

**Model:**
- Generator: `gpt-4.1-mini` via OpenAI API
- Embedder: `text-embedding-3-small` (1536 dim) via OpenAI API
- Vector DB: chromadb (persistent, cosine distance)

**Evaluator:** Custom zero-NaN 4 metrik'''))

    cells.append(codecell('''# Install dependencies (jalankan sekali saja)
# !pip install openai rank-bm25 chromadb datasets'''))

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
import chromadb

warnings.filterwarnings('ignore')
print('Semua library berhasil diimpor!')
print(f'Python: {sys.version.split()[0]} | NumPy: {np.__version__} | chromadb: {chromadb.__version__}')'''))

    cells.append(codecell('''# ============================================================
# KONFIGURASI
# ============================================================
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YOUR_API_KEY_HERE')

LLM_MODEL   = 'gpt-4.1-mini'
EMBED_MODEL = 'text-embedding-3-small'  # 1536 dim, $0.02 / 1M token

TOP_K_BM25      = 50   # BM25 ambil top-50
TOP_K_DENSE     = 50   # Dense ambil top-50
TOP_K_RETRIEVAL = 5    # Final top-5 setelah RRF

DATASET_NAME   = 'qiaojin/PubMedQA'
DATASET_SUBSET = 'pqa_labeled'
MAX_SAMPLES    = 500

TEMPERATURE = 0.0
SEED        = 42

NOTEBOOK_DIR    = Path('.')
BM25_INDEX_PATH = NOTEBOOK_DIR / 'pubmedqa_bm25.pkl'
CHROMA_DB_PATH  = NOTEBOOK_DIR / 'pubmedqa_chroma'
RESULTS_DIR     = Path('../results')
RESULTS_DIR.mkdir(exist_ok=True)

CONFIG_NAME        = 'hybrid_openai'
PHASE1_PATH        = RESULTS_DIR / f'{CONFIG_NAME}_phase1_answers.json'
PHASE2_CUSTOM_PATH = RESULTS_DIR / f'{CONFIG_NAME}_phase2_custom.json'

print('Konfigurasi:')
print(f'  LLM          : {LLM_MODEL} (via OpenAI)')
print(f'  Embedder     : {EMBED_MODEL}')
print(f'  Retriever    : BM25 top-{TOP_K_BM25} + Dense top-{TOP_K_DENSE} -> RRF -> top-{TOP_K_RETRIEVAL}')
print(f'  Vector DB    : chromadb (path: {CHROMA_DB_PATH})')
print(f'  Sampel       : {MAX_SAMPLES}')
print(f'  Config       : {CONFIG_NAME}')
print()
if 'YOUR_API_KEY' in OPENAI_API_KEY:
    print('  OPENAI_API_KEY belum diisi! Set env var OPENAI_API_KEY atau isi di cell ini.')
else:
    print(f'  OPENAI_API_KEY: {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}')'''))

    cells.append(codecell(CELL_DATACLASS))
    cells.append(codecell(CELL_DATASET))
    cells.append(codecell(CELL_OPENAI_SETUP))
    cells.append(codecell(CELL_CHROMA_INDEX))
    cells.append(codecell(CELL_DENSE_RRF))
    cells.append(codecell(CELL_GENERATE))
    cells.append(codecell(CELL_EXTRACT_LABEL))
    cells.append(codecell(CUSTOM_EVAL))

    # Demo
    cells.append(mdcell('## Demo — 5 Sampel Pertama'))
    cells.append(codecell('''DEMO_SIZE = 5

print(f'DEMO: {DEMO_SIZE} sampel pertama (Hybrid BM25+Dense via RRF, {LLM_MODEL})')
print('=' * 65)

for i in range(DEMO_SIZE):
    s  = pubmedqa_data[i]
    q  = s['question']
    gt = s['final_decision']

    retrieved = retrieve_hybrid(q)
    answer    = generate_answer(q, retrieved)
    predicted = extract_label(answer)
    correct   = predicted == gt

    verdict = 'BENAR' if correct else 'SALAH'
    print(f'\\n[{i+1}/{DEMO_SIZE}] {q[:75]}...')
    print(f'  Top-1 RRF={retrieved[0].rrf_score:.4f} (BM25={retrieved[0].bm25_score:.2f}, Dense={retrieved[0].dense_score:.3f})')
    print(f'  GT={gt} | Pred={predicted} | {verdict}')
    print(f'  Jawaban: {answer[:120]}...')'''))

    # Phase 1
    cells.append(mdcell('''## Phase 1 — Generate Jawaban (500 Sampel)

Estimasi waktu: ~15-20 menit dengan GPT-4.1-mini. Resume otomatis.'''))
    cells.append(codecell('''if PHASE1_PATH.exists():
    with open(PHASE1_PATH, 'r', encoding='utf-8') as f:
        phase1_results = json.load(f)['results']
    start_from = len(phase1_results)
    print(f'Resume Fase 1: {start_from}/{MAX_SAMPLES} sudah selesai.')
else:
    phase1_results, start_from = [], 0
    print(f'Memulai Fase 1: {MAX_SAMPLES} sampel (Hybrid BM25+Dense).')

if start_from < MAX_SAMPLES:
    print(f'Memproses {MAX_SAMPLES - start_from} sampel tersisa...\\n')
    t_start = time.time()

    for i in range(start_from, MAX_SAMPLES):
        s          = pubmedqa_data[i]
        q, gt, ref = s['question'], s['final_decision'], s['long_answer']

        retrieved = retrieve_hybrid(q)
        answer    = generate_answer(q, retrieved)
        predicted = extract_label(answer)

        phase1_results.append({
            'idx'             : i,
            'pubid'           : str(s['pubid']),
            'question'        : q,
            'ground_truth'    : gt,
            'predicted_label' : predicted,
            'is_correct'      : predicted == gt,
            'answer'          : answer,
            'contexts'        : [r.document.text for r in retrieved],
            'reference'       : ref,
            'retrieval_scores': [r.bm25_score  for r in retrieved],
            'dense_scores'    : [r.dense_score for r in retrieved],
            'rrf_scores'      : [r.rrf_score   for r in retrieved],
        })

        if (i + 1) % 10 == 0 or i == MAX_SAMPLES - 1:
            with open(PHASE1_PATH, 'w', encoding='utf-8') as f:
                json.dump({'config': CONFIG_NAME,
                           'llm_model': LLM_MODEL,
                           'embed_model': EMBED_MODEL,
                           'timestamp': datetime.now().isoformat(),
                           'max_samples': MAX_SAMPLES, 'completed': i+1,
                           'results': phase1_results}, f, indent=2, ensure_ascii=False)
            done = i + 1
            acc  = sum(r['is_correct'] for r in phase1_results) / done
            eta  = (time.time()-t_start) / done * (MAX_SAMPLES-done) / 60
            print(f'  [{done:3d}/{MAX_SAMPLES}] Akurasi: {acc:.1%} | pred={predicted}, gt={gt} | ETA {eta:.1f} mnt')

    print(f'\\nFase 1 selesai! -> {PHASE1_PATH}')
else:
    print(f'Fase 1 sudah selesai ({MAX_SAMPLES} sampel).')'''))

    # Phase 1 analysis
    cells.append(mdcell('## Analisis Phase 1'))
    cells.append(codecell('''with open(PHASE1_PATH, 'r', encoding='utf-8') as f:
    results_p1 = json.load(f)['results']

n         = len(results_p1)
n_correct = sum(r['is_correct'] for r in results_p1)

print(f'ANALISIS PHASE 1 - {n} sampel ({CONFIG_NAME})')
print('=' * 55)
print(f'Label Accuracy    : {n_correct}/{n} = {n_correct/n:.1%}')
print(f'Hallucination Rate: {(n-n_correct)/n:.1%}\\n')

# Per-label
print('Per-label accuracy:')
for lbl in ['yes','no','maybe']:
    sub = [r for r in results_p1 if r['ground_truth'] == lbl]
    if sub:
        c = sum(r['is_correct'] for r in sub)
        print(f'  {lbl:>5}: {c}/{len(sub)} = {c/len(sub):.1%}')

# Prediction distribution
preds = [r['predicted_label'] for r in results_p1]
print(f'\\nDistribusi prediksi:')
for lbl in ['yes','no','maybe']:
    print(f'  {lbl:>5}: {preds.count(lbl)} ({preds.count(lbl)/n:.0%})')

# Hybrid-specific: overlap BM25 vs Dense
overlap_ratios = []
for r in results_p1:
    # top-5 doc ids would need to be re-computed; instead compare scores
    # cek apakah rank tertinggi BM25 dan Dense overlap di top-5 final
    pass

print(f'\\nRata-rata skor BM25  : {np.mean([np.mean(r["retrieval_scores"]) for r in results_p1]):.4f}')
print(f'Rata-rata skor Dense : {np.mean([np.mean(r["dense_scores"])     for r in results_p1]):.4f}')
print(f'Rata-rata skor RRF   : {np.mean([np.mean(r["rrf_scores"])       for r in results_p1]):.4f}')'''))

    # Phase 2
    cells.append(mdcell('''## Phase 2 — Custom Evaluator 4 Metrik (500 Sampel)

Estimasi waktu: ~20-30 menit.'''))
    cells.append(codecell(PHASE2_LOOP))

    # Summary
    cells.append(mdcell('## Summary — Hasil Akhir'))
    cells.append(codecell(summary_cell('Hybrid OpenAI')))

    return make_notebook(cells)


# ============================================================
# NOTEBOOK 07: Hybrid + CrossEncoder Reranking
# ============================================================

def build_notebook_07():
    cells = []

    cells.append(mdcell('''# 07 — RAG Hybrid + Context Reranking (BM25 + Dense + CrossEncoder)

Notebook ini menggabungkan **Hybrid Retrieval** (BM25 + Dense via RRF) dengan **Context Reranking** (CrossEncoder) untuk menghasilkan konfigurasi paling komprehensif.

**Pipeline:**
```
original query
  -> BM25 top-50 + Dense top-50 (OpenAI)
  -> RRF fusion -> top-20
  -> CrossEncoder rerank (ms-marco-MiniLM-L-6-v2)
  -> top-5
  -> OpenAI generate
```

**Rationale:**
- Hybrid memastikan kandidat berkualitas tinggi (lexical + semantic).
- CrossEncoder rerank memilih 5 konteks paling relevan dari 20 kandidat.
- Kombinasi ini adalah rekomendasi umum di literatur RAG produksi.

**Model:**
- Generator: `gpt-4.1-mini` via OpenAI API
- Embedder: `text-embedding-3-small`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (lokal, CPU)

**Evaluator:** Custom zero-NaN 4 metrik'''))

    cells.append(codecell('''# Install dependencies (jalankan sekali saja)
# !pip install openai rank-bm25 chromadb sentence-transformers datasets'''))

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
import chromadb

try:
    from sentence_transformers import CrossEncoder
    print('sentence-transformers tersedia.')
except ImportError:
    print('sentence-transformers belum terinstall! Jalankan: pip install sentence-transformers')
    raise

warnings.filterwarnings('ignore')
print('Semua library berhasil diimpor!')
print(f'Python: {sys.version.split()[0]} | chromadb: {chromadb.__version__}')'''))

    cells.append(codecell('''# ============================================================
# KONFIGURASI
# ============================================================
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YOUR_API_KEY_HERE')

LLM_MODEL      = 'gpt-4.1-mini'
EMBED_MODEL    = 'text-embedding-3-small'
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

TOP_K_BM25       = 50    # BM25 ambil 50
TOP_K_DENSE      = 50    # Dense ambil 50
TOP_K_CANDIDATES = 20    # Setelah RRF, ambil 20 kandidat
TOP_K_RETRIEVAL  = 5     # Setelah rerank, ambil top-5

DATASET_NAME   = 'qiaojin/PubMedQA'
DATASET_SUBSET = 'pqa_labeled'
MAX_SAMPLES    = 500

TEMPERATURE = 0.0
SEED        = 42

NOTEBOOK_DIR    = Path('.')
BM25_INDEX_PATH = NOTEBOOK_DIR / 'pubmedqa_bm25.pkl'
CHROMA_DB_PATH  = NOTEBOOK_DIR / 'pubmedqa_chroma'
RESULTS_DIR     = Path('../results')
RESULTS_DIR.mkdir(exist_ok=True)

CONFIG_NAME        = 'hybrid_cr_openai'
PHASE1_PATH        = RESULTS_DIR / f'{CONFIG_NAME}_phase1_answers.json'
PHASE2_CUSTOM_PATH = RESULTS_DIR / f'{CONFIG_NAME}_phase2_custom.json'

print('Konfigurasi:')
print(f'  LLM          : {LLM_MODEL}')
print(f'  Embedder     : {EMBED_MODEL}')
print(f'  Reranker     : {RERANKER_MODEL}')
print(f'  Retriever    : BM25 top-{TOP_K_BM25} + Dense top-{TOP_K_DENSE}')
print(f'                 -> RRF -> top-{TOP_K_CANDIDATES}')
print(f'                 -> CrossEncoder -> top-{TOP_K_RETRIEVAL}')
print(f'  Sampel       : {MAX_SAMPLES}')
print(f'  Config       : {CONFIG_NAME}')
print()
if 'YOUR_API_KEY' in OPENAI_API_KEY:
    print('  OPENAI_API_KEY belum diisi!')
else:
    print(f'  OPENAI_API_KEY: {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}')'''))

    cells.append(codecell(CELL_DATACLASS))
    cells.append(codecell(CELL_DATASET))
    cells.append(codecell(CELL_OPENAI_SETUP))
    cells.append(codecell(CELL_CHROMA_INDEX))

    # Load CrossEncoder
    cells.append(codecell('''# ============================================================
# Load CrossEncoder
# ============================================================
print(f'Memuat CrossEncoder: {RERANKER_MODEL}...')
t0 = time.time()
cross_encoder = CrossEncoder(RERANKER_MODEL)
print(f'CrossEncoder siap dalam {time.time()-t0:.1f} detik')

_pairs = [
    ('Does aspirin prevent heart attacks?', 'Aspirin reduces platelet aggregation.'),
    ('Does aspirin prevent heart attacks?', 'Weather patterns affect agriculture.'),
]
_scores = cross_encoder.predict(_pairs)
print(f'Smoke test: relevan={_scores[0]:.3f}, tidak relevan={_scores[1]:.3f}')
assert _scores[0] > _scores[1]
print('Reranker berfungsi.')'''))

    # Hybrid + rerank retrieval
    cells.append(codecell('''# ============================================================
# Dense retrieval + RRF + CrossEncoder rerank
# ============================================================

def retrieve_dense(query: str, k: int = 50) -> List[Tuple[int, float]]:
    qvec = openai_embed([query])[0]
    results = chroma_collection.query(query_embeddings=[qvec], n_results=k, include=['distances'])
    doc_ids   = [int(i) for i in results['ids'][0]]
    distances = results['distances'][0]
    scores    = [1.0 - d for d in distances]
    return list(zip(doc_ids, scores))


def retrieve_bm25_raw(query: str, k: int = 50) -> List[Tuple[int, float]]:
    tokens = tokenize_bm25(query)
    scores = bm25_index.get_scores(tokens)
    top_k  = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_k]


def reciprocal_rank_fusion(
    rank_lists: List[List[int]], k: int = 60
) -> List[Tuple[int, float]]:
    scores = {}
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def retrieve_hybrid_rerank(
    query: str,
    k_bm25: int = TOP_K_BM25,
    k_dense: int = TOP_K_DENSE,
    k_candidates: int = TOP_K_CANDIDATES,
    k_final: int = TOP_K_RETRIEVAL
) -> List[RetrievalResult]:
    """
    Pipeline lengkap:
      1. BM25 top-k_bm25 + Dense top-k_dense
      2. RRF fusion -> top-k_candidates
      3. CrossEncoder rerank -> top-k_final
    """
    bm25_results  = retrieve_bm25_raw(query, k=k_bm25)
    dense_results = retrieve_dense(query, k=k_dense)

    bm25_scores  = dict(bm25_results)
    dense_scores = dict(dense_results)

    # RRF
    rank_lists = [
        [doc_id for doc_id, _ in bm25_results],
        [doc_id for doc_id, _ in dense_results],
    ]
    fused = reciprocal_rank_fusion(rank_lists, k=60)
    top_candidates = fused[:k_candidates]

    # Build candidates
    candidates = []
    for doc_id, rrf_score in top_candidates:
        candidates.append(RetrievalResult(
            document=documents[doc_id],
            score=rrf_score,
            doc_id=doc_id,
            bm25_score=bm25_scores.get(doc_id, 0.0),
            dense_score=dense_scores.get(doc_id, 0.0),
            rrf_score=rrf_score,
        ))

    # CrossEncoder rerank
    pairs = [(query, r.document.text) for r in candidates]
    reranker_scores = cross_encoder.predict(pairs)
    for r, rs in zip(candidates, reranker_scores):
        r.reranker_score = float(rs)

    reranked = sorted(candidates, key=lambda r: r.reranker_score, reverse=True)
    return reranked[:k_final]


# Test
test_q = 'Does aspirin reduce the risk of myocardial infarction?'
test_r = retrieve_hybrid_rerank(test_q)
print(f'Query: {test_q}')
print(f'\\nTop-{TOP_K_RETRIEVAL} dokumen (Hybrid + CR):')
for i, r in enumerate(test_r, 1):
    print(f'  [{i}] Rerank={r.reranker_score:.3f} | RRF={r.rrf_score:.4f} '
          f'| BM25={r.bm25_score:.2f} | Dense={r.dense_score:.3f} | {r.document.text[:60]}...')'''))

    # Generate — wrapper yang pakai retrieve_hybrid_rerank
    cells.append(codecell(GENERATION_PROMPT_BODY + '''


def generate_answer(query: str, retrieved: List[RetrievalResult]) -> str:
    context = '\\n\\n'.join(
        f'[{i}] ({r.document.section_label}): {r.document.text}'
        for i, r in enumerate(retrieved, 1)
    )
    return openai_generate(
        GENERATION_PROMPT.format(context=context, question=query),
        max_tokens=300,
        temperature=TEMPERATURE
    )


test_ans = generate_answer(test_q, test_r)
print('Output generation:')
print('-' * 60)
print(test_ans)
print('-' * 60)'''))

    cells.append(codecell(CELL_EXTRACT_LABEL))
    cells.append(codecell(CUSTOM_EVAL))

    # Demo
    cells.append(mdcell('## Demo — 5 Sampel Pertama'))
    cells.append(codecell('''DEMO_SIZE = 5

print(f'DEMO: {DEMO_SIZE} sampel pertama (Hybrid + CR, {LLM_MODEL})')
print('=' * 65)

for i in range(DEMO_SIZE):
    s  = pubmedqa_data[i]
    q  = s['question']
    gt = s['final_decision']

    retrieved = retrieve_hybrid_rerank(q)
    answer    = generate_answer(q, retrieved)
    predicted = extract_label(answer)
    correct   = predicted == gt

    verdict = 'BENAR' if correct else 'SALAH'
    print(f'\\n[{i+1}/{DEMO_SIZE}] {q[:75]}...')
    print(f'  Top-1 Rerank={retrieved[0].reranker_score:.3f} (RRF={retrieved[0].rrf_score:.4f})')
    print(f'  GT={gt} | Pred={predicted} | {verdict}')'''))

    # Phase 1
    cells.append(mdcell('''## Phase 1 — Generate Jawaban (500 Sampel)

Estimasi waktu: ~20-25 menit dengan GPT-4.1-mini. Resume otomatis.'''))
    cells.append(codecell('''if PHASE1_PATH.exists():
    with open(PHASE1_PATH, 'r', encoding='utf-8') as f:
        phase1_results = json.load(f)['results']
    start_from = len(phase1_results)
    print(f'Resume Fase 1: {start_from}/{MAX_SAMPLES} sudah selesai.')
else:
    phase1_results, start_from = [], 0
    print(f'Memulai Fase 1: {MAX_SAMPLES} sampel (Hybrid + CR).')

if start_from < MAX_SAMPLES:
    print(f'Memproses {MAX_SAMPLES - start_from} sampel tersisa...\\n')
    t_start = time.time()

    for i in range(start_from, MAX_SAMPLES):
        s          = pubmedqa_data[i]
        q, gt, ref = s['question'], s['final_decision'], s['long_answer']

        retrieved = retrieve_hybrid_rerank(q)
        answer    = generate_answer(q, retrieved)
        predicted = extract_label(answer)

        phase1_results.append({
            'idx'             : i,
            'pubid'           : str(s['pubid']),
            'question'        : q,
            'ground_truth'    : gt,
            'predicted_label' : predicted,
            'is_correct'      : predicted == gt,
            'answer'          : answer,
            'contexts'        : [r.document.text for r in retrieved],
            'reference'       : ref,
            'retrieval_scores': [r.bm25_score     for r in retrieved],
            'dense_scores'    : [r.dense_score    for r in retrieved],
            'rrf_scores'      : [r.rrf_score      for r in retrieved],
            'reranker_scores' : [r.reranker_score for r in retrieved],
        })

        if (i + 1) % 10 == 0 or i == MAX_SAMPLES - 1:
            with open(PHASE1_PATH, 'w', encoding='utf-8') as f:
                json.dump({'config': CONFIG_NAME,
                           'llm_model': LLM_MODEL,
                           'embed_model': EMBED_MODEL,
                           'reranker_model': RERANKER_MODEL,
                           'timestamp': datetime.now().isoformat(),
                           'max_samples': MAX_SAMPLES, 'completed': i+1,
                           'results': phase1_results}, f, indent=2, ensure_ascii=False)
            done = i + 1
            acc  = sum(r['is_correct'] for r in phase1_results) / done
            eta  = (time.time()-t_start) / done * (MAX_SAMPLES-done) / 60
            print(f'  [{done:3d}/{MAX_SAMPLES}] Akurasi: {acc:.1%} | pred={predicted}, gt={gt} | ETA {eta:.1f} mnt')

    print(f'\\nFase 1 selesai! -> {PHASE1_PATH}')
else:
    print(f'Fase 1 sudah selesai ({MAX_SAMPLES} sampel).')'''))

    cells.append(mdcell('## Analisis Phase 1'))
    cells.append(codecell('''with open(PHASE1_PATH, 'r', encoding='utf-8') as f:
    results_p1 = json.load(f)['results']

n         = len(results_p1)
n_correct = sum(r['is_correct'] for r in results_p1)

print(f'ANALISIS PHASE 1 - {n} sampel ({CONFIG_NAME})')
print('=' * 55)
print(f'Label Accuracy    : {n_correct}/{n} = {n_correct/n:.1%}')

print('\\nPer-label accuracy:')
for lbl in ['yes','no','maybe']:
    sub = [r for r in results_p1 if r['ground_truth'] == lbl]
    if sub:
        c = sum(r['is_correct'] for r in sub)
        print(f'  {lbl:>5}: {c}/{len(sub)} = {c/len(sub):.1%}')

preds = [r['predicted_label'] for r in results_p1]
print(f'\\nDistribusi prediksi:')
for lbl in ['yes','no','maybe']:
    print(f'  {lbl:>5}: {preds.count(lbl)} ({preds.count(lbl)/n:.0%})')

print(f'\\nRata-rata skor:')
print(f'  BM25     : {np.mean([np.mean(r["retrieval_scores"]) for r in results_p1]):.4f}')
print(f'  Dense    : {np.mean([np.mean(r["dense_scores"])     for r in results_p1]):.4f}')
print(f'  RRF      : {np.mean([np.mean(r["rrf_scores"])       for r in results_p1]):.4f}')
print(f'  Reranker : {np.mean([np.mean(r["reranker_scores"])  for r in results_p1]):.4f}')'''))

    cells.append(mdcell('''## Phase 2 — Custom Evaluator 4 Metrik (500 Sampel)

Estimasi waktu: ~20-30 menit.'''))
    cells.append(codecell(PHASE2_LOOP))

    cells.append(mdcell('## Summary — Hasil Akhir'))
    cells.append(codecell(summary_cell('Hybrid + CR OpenAI')))

    return make_notebook(cells)


# ============================================================
# MAIN
# ============================================================

def main():
    nb_06 = build_notebook_06()
    out_06 = NB_DIR / '06-RAG-Hybrid-OpenAI.ipynb'
    with open(out_06, 'w', encoding='utf-8') as f:
        json.dump(nb_06, f, indent=1, ensure_ascii=False)
    print(f'Saved: {out_06} ({len(nb_06["cells"])} cells)')

    nb_07 = build_notebook_07()
    out_07 = NB_DIR / '07-RAG-Hybrid-CR-OpenAI.ipynb'
    with open(out_07, 'w', encoding='utf-8') as f:
        json.dump(nb_07, f, indent=1, ensure_ascii=False)
    print(f'Saved: {out_07} ({len(nb_07["cells"])} cells)')

    # Verify
    import ast
    print('\nVerification:')
    for path in [out_06, out_07]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Syntax check
            all_ok = True
            for i, cell in enumerate(data['cells']):
                if cell['cell_type'] != 'code':
                    continue
                src = ''.join(cell['source'])
                if src.strip().startswith('#') and 'pip' in src and len(src) < 200:
                    continue
                try:
                    ast.parse(src)
                except SyntaxError as e:
                    all_ok = False
                    print(f'  SYNTAX ERROR in {path.name} cell [{i}]: {e.msg} at line {e.lineno}')
            status = 'VALID' if all_ok else 'SYNTAX ERROR'
            print(f'  {path.name}: {status} ({len(data["cells"])} cells)')
        except Exception as e:
            print(f'  {path.name}: CORRUPT! {e}')


if __name__ == '__main__':
    main()
