"""Shared utilities for prompt-variation experiments on PubMedQA.

Reuses the BM25 index and ChromaDB collection built by the main OpenAI notebooks
(02.1 Baseline, 03.1 QR, 04.1 CR, 05.1 QR+CR). Paths are pointed at the main
notebooks/ folder so no re-indexing is needed.

Exposes 4 RAG-method runners:
    - run_baseline(query)   -> List[RetrievalResult]
    - run_qr(query)         -> (List[RetrievalResult], rewritten_query)
    - run_cr(query)         -> List[RetrievalResult]
    - run_qr_cr(query)      -> (List[RetrievalResult], rewritten_query)

And:
    - generate_answer(query, retrieved, prompt_template, max_tokens)
    - extract_label(answer)
    - evaluate_custom(question, answer, contexts, reference)
    - run_phase1(method, prompt_template, max_tokens, max_samples, out_path)
    - run_phase2(phase1_path, out_path, max_samples)
"""

import os, json, pickle, time, re, warnings
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path
from datetime import datetime

from openai import OpenAI
from rank_bm25 import BM25Okapi
from datasets import load_dataset
import chromadb

warnings.filterwarnings('ignore')


# ============================================================
# CONFIG
# ============================================================
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YOUR_OPENAI_KEY_HERE')

LLM_MODEL   = 'gpt-4.1-mini'
EMBED_MODEL = 'text-embedding-3-small'  # 1536 dim

TOP_K_BM25       = 50
TOP_K_DENSE      = 50
TOP_K_RETRIEVAL  = 5     # default for baseline / QR
TOP_K_RERANKER   = 20    # CR candidates before reranking
RERANKER_MODEL   = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

DATASET_NAME    = 'qiaojin/PubMedQA'
DATASET_SUBSET  = 'pqa_labeled'

TEMPERATURE = 0.0
SEED        = 42

# Paths - auto-detect notebooks/ folder (where pubmedqa_bm25.pkl lives)
# Works whether prompt_experiment is INSIDE notebooks/ or a SIBLING of it.
PROMPT_EXP_DIR  = Path(__file__).resolve().parent


def _find_notebooks_dir() -> Path:
    """Locate the notebooks/ folder that contains pubmedqa_bm25.pkl."""
    candidates = [
        PROMPT_EXP_DIR.parent,                       # inside notebooks/ (current setup)
        PROMPT_EXP_DIR.parent / 'notebooks',         # sibling of notebooks/
        PROMPT_EXP_DIR / 'notebooks',                # notebooks/ as subdir (unlikely)
    ]
    for c in candidates:
        if (c / 'pubmedqa_bm25.pkl').exists():
            return c.resolve()
    # Fallback: most likely candidate so the error message points somewhere sensible
    return PROMPT_EXP_DIR.parent.resolve()


NOTEBOOKS_DIR    = _find_notebooks_dir()
# Project root = parent of notebooks/. Used for results/ folder.
PROJECT_ROOT     = NOTEBOOKS_DIR.parent if NOTEBOOKS_DIR.name == 'notebooks' else PROMPT_EXP_DIR.parent.parent
BM25_INDEX_PATH  = NOTEBOOKS_DIR / 'pubmedqa_bm25.pkl'
CHROMA_DB_PATH   = NOTEBOOKS_DIR / 'pubmedqa_chroma'   # OpenAI embeddings (1536 dim)
RESULTS_DIR      = PROJECT_ROOT / 'results' / 'prompt_experiment'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
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
    score          : float
    doc_id         : int = -1
    bm25_score     : float = 0.0
    dense_score    : float = 0.0
    rrf_score      : float = 0.0
    reranker_score : float = 0.0


def tokenize_bm25(text: str) -> List[str]:
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower()).split()


# ============================================================
# OPENAI CLIENT
# ============================================================
_client: Optional[OpenAI] = None


def openai_client() -> OpenAI:
    global _client, OPENAI_API_KEY
    if _client is None:
        # Re-read env var at call time (so setting it AFTER import still works)
        key = os.environ.get('OPENAI_API_KEY', OPENAI_API_KEY)
        if not key or key.startswith('YOUR_'):
            raise RuntimeError(
                'OPENAI_API_KEY not set. Choose one of:\n'
                '  1) In notebook cell BEFORE import shared:\n'
                "     import os; # os.environ["OPENAI_API_KEY"] = "<REDACTED — set via shell env or .env file>"\n"
                '  2) Set Windows env var permanently (then restart Jupyter)\n'
                '  3) Set shared.OPENAI_API_KEY = "sk-..." after import'
            )
        OPENAI_API_KEY = key
        _client = OpenAI(api_key=key)
    return _client


def openai_generate(prompt: str, max_tokens: int = 300, temperature: float = TEMPERATURE) -> str:
    for attempt in range(5):
        try:
            response = openai_client().chat.completions.create(
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
                time.sleep((attempt + 1) * 10)
            elif '500' in err or '502' in err or '503' in err:
                time.sleep((attempt + 1) * 5)
            else:
                raise
    raise RuntimeError('OpenAI API failed after 5 attempts.')


def openai_embed(texts: List[str]) -> List[List[float]]:
    for attempt in range(5):
        try:
            response = openai_client().embeddings.create(model=EMBED_MODEL, input=texts)
            return [d.embedding for d in response.data]
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                time.sleep((attempt + 1) * 10)
            elif '500' in err or '502' in err or '503' in err:
                time.sleep((attempt + 1) * 5)
            else:
                raise
    raise RuntimeError('OpenAI embeddings failed after 5 attempts.')


# ============================================================
# LOAD DATA + INDEXES (called once)
# ============================================================
_pubmedqa_data = None
_bm25_index    = None
_documents     = None
_chroma_coll   = None
_cross_encoder = None


def load_everything(max_samples: int = 500):
    """Load PubMedQA, BM25, Chroma, and CrossEncoder. Idempotent."""
    global _pubmedqa_data, _bm25_index, _documents, _chroma_coll, _cross_encoder

    if _bm25_index is None:
        full = load_dataset(DATASET_NAME, DATASET_SUBSET)['train']
        _pubmedqa_data = full.select(range(max_samples))
        print(f'Loaded PubMedQA: {len(_pubmedqa_data)} samples')

        if not BM25_INDEX_PATH.exists():
            raise FileNotFoundError(
                f'BM25 index not found at {BM25_INDEX_PATH}. '
                f'Run the main baseline notebook (02.1) first to build it.'
            )
        # The pickle was built by the main notebook where Document / RetrievalResult
        # were defined in __main__. Inject our shared.* classes into __main__ so
        # pickle.load can resolve them.
        import __main__
        if not hasattr(__main__, 'Document'):
            __main__.Document = Document
        if not hasattr(__main__, 'RetrievalResult'):
            __main__.RetrievalResult = RetrievalResult

        with open(BM25_INDEX_PATH, 'rb') as f:
            saved = pickle.load(f)
        _bm25_index = saved['bm25']
        _documents  = saved['documents']
        print(f'Loaded BM25 index: {len(_documents)} document chunks')

    if _chroma_coll is None:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        _chroma_coll  = chroma_client.get_collection(name='pubmedqa_docs')
        print(f'Loaded Chroma collection: {_chroma_coll.count()} vectors')

    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
        print(f'Loaded CrossEncoder: {RERANKER_MODEL}')

    return _pubmedqa_data, _bm25_index, _documents, _chroma_coll, _cross_encoder


# ============================================================
# RETRIEVAL PRIMITIVES
# ============================================================
def retrieve_bm25_raw(query: str, k: int = TOP_K_BM25) -> List[Tuple[int, float]]:
    tokens = tokenize_bm25(query)
    scores = _bm25_index.get_scores(tokens)
    top_k  = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_k]


def retrieve_dense(query: str, k: int = TOP_K_DENSE) -> List[Tuple[int, float]]:
    qvec    = openai_embed([query])[0]
    results = _chroma_coll.query(query_embeddings=[qvec], n_results=k, include=['distances'])
    doc_ids   = [int(i) for i in results['ids'][0]]
    distances = results['distances'][0]
    scores    = [1.0 - d for d in distances]
    return list(zip(doc_ids, scores))


def reciprocal_rank_fusion(rank_lists: List[List[int]], k: int = 60) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for rl in rank_lists:
        for rank, doc_id in enumerate(rl):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_retrieve(query: str, k_final: int) -> List[RetrievalResult]:
    """BM25 + Dense merged via RRF, return top-k_final."""
    bm25_results  = retrieve_bm25_raw(query, k=TOP_K_BM25)
    dense_results = retrieve_dense(query, k=TOP_K_DENSE)
    bm25_scores   = dict(bm25_results)
    dense_scores  = dict(dense_results)
    fused = reciprocal_rank_fusion([
        [d for d, _ in bm25_results],
        [d for d, _ in dense_results],
    ])
    out = []
    for doc_id, rrf_score in fused[:k_final]:
        out.append(RetrievalResult(
            document=_documents[doc_id], score=rrf_score, doc_id=doc_id,
            bm25_score=bm25_scores.get(doc_id, 0.0),
            dense_score=dense_scores.get(doc_id, 0.0),
            rrf_score=rrf_score,
        ))
    return out


# ============================================================
# QUERY REWRITING (for QR / QR+CR)
# ============================================================
QUERY_REWRITE_PROMPT = (
    'You are a query rewriting assistant for a biomedical question-answering system.\n'
    'Rewrite the following medical question to improve retrieval from a PubMed research database.\n\n'
    'Rules:\n'
    '1. Be more specific and add relevant medical/scientific terminology.\n'
    '2. Expand abbreviations (e.g. "MI" -> "myocardial infarction").\n'
    '3. Preserve the original yes/no/maybe answerable intent.\n'
    '4. Output ONLY the rewritten question, no explanations.\n\n'
    'Original question: {query}\n\n'
    'Rewritten question:'
)


def rewrite_query(query: str) -> str:
    try:
        rw = openai_generate(QUERY_REWRITE_PROMPT.format(query=query),
                             max_tokens=150, temperature=0.3)
        rw = rw.strip().replace('\n', ' ')
        return rw if len(rw) >= 10 else query
    except Exception:
        return query


# ============================================================
# CONTEXT RERANKING (for CR / QR+CR)
# ============================================================
def rerank_with_cross_encoder(query: str, candidates: List[RetrievalResult],
                              k_final: int) -> List[RetrievalResult]:
    pairs  = [(query, r.document.text) for r in candidates]
    scores = _cross_encoder.predict(pairs)
    for r, s in zip(candidates, scores):
        r.reranker_score = float(s)
    return sorted(candidates, key=lambda r: -r.reranker_score)[:k_final]


# ============================================================
# 4 RAG METHOD RUNNERS
# ============================================================
def run_baseline(query: str) -> Tuple[List[RetrievalResult], str]:
    """Method: baseline. Returns (retrieved_chunks, query_used_for_retrieval)."""
    return hybrid_retrieve(query, k_final=TOP_K_RETRIEVAL), query


def run_qr(query: str) -> Tuple[List[RetrievalResult], str]:
    """Method: query rewriting. Returns (retrieved_chunks, rewritten_query)."""
    rewritten = rewrite_query(query)
    return hybrid_retrieve(rewritten, k_final=TOP_K_RETRIEVAL), rewritten


def run_cr(query: str) -> Tuple[List[RetrievalResult], str]:
    """Method: context reranking. Returns (reranked_top5_chunks, original_query)."""
    candidates = hybrid_retrieve(query, k_final=TOP_K_RERANKER)
    reranked   = rerank_with_cross_encoder(query, candidates, k_final=TOP_K_RETRIEVAL)
    return reranked, query


def run_qr_cr(query: str) -> Tuple[List[RetrievalResult], str]:
    """Method: QR + CR. Returns (reranked_top5_chunks, rewritten_query)."""
    rewritten  = rewrite_query(query)
    candidates = hybrid_retrieve(rewritten, k_final=TOP_K_RERANKER)
    reranked   = rerank_with_cross_encoder(rewritten, candidates, k_final=TOP_K_RETRIEVAL)
    return reranked, rewritten


METHOD_RUNNERS: Dict[str, Callable] = {
    'baseline' : run_baseline,
    'qr'       : run_qr,
    'cr'       : run_cr,
    'qr_cr'    : run_qr_cr,
}


# ============================================================
# GENERATION + LABEL EXTRACTION
# ============================================================
def generate_answer(query: str, retrieved: List[RetrievalResult],
                    prompt_template: str, max_tokens: int = 300) -> str:
    context = '\n\n'.join(
        f'[{i}] ({r.document.section_label}): {r.document.text}'
        for i, r in enumerate(retrieved, 1)
    )
    full_prompt = prompt_template.format(context=context, question=query)
    return openai_generate(full_prompt, max_tokens=max_tokens, temperature=TEMPERATURE)


def extract_label(answer: str) -> str:
    """Extract yes/no/maybe from response text. Robust to multi-line output."""
    lines = [l.strip().lower() for l in answer.split('\n') if l.strip()]
    for line in reversed(lines[-3:]):
        word = re.sub(r'[^a-z]', '', line)
        if word in ('yes', 'no', 'maybe'):
            return word
    for label in ('yes', 'no', 'maybe'):
        if re.search(r'\b' + label + r'\b', answer.lower()):
            return label
    return 'maybe'


# ============================================================
# CUSTOM ZERO-NAN EVALUATOR (same as main notebooks)
# ============================================================
def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) >= 15]


def _llm_yes_no(prompt: str) -> bool:
    try:
        resp = openai_generate(prompt, max_tokens=10, temperature=0.0)
        return 'yes' in resp.lower()[:15]
    except Exception:
        return False


def compute_faithfulness(answer: str, contexts: List[str]) -> float:
    sents = _split_sentences(answer)
    if not sents: return 0.0
    ctx_text = '\n'.join(f'[{i+1}] {c[:400]}' for i, c in enumerate(contexts))
    tmpl = ('Context:\n{ctx}\n\nStatement: {sent}\n\n'
            'Is this statement directly supported by the context above? '
            'Answer with only "yes" or "no".')
    return sum(_llm_yes_no(tmpl.format(ctx=ctx_text, sent=s)) for s in sents) / len(sents)


def compute_context_recall(reference: str, contexts: List[str]) -> float:
    sents = _split_sentences(reference)
    if not sents: return 0.0
    ctx_text = '\n'.join(f'[{i+1}] {c[:400]}' for i, c in enumerate(contexts))
    tmpl = ('Context:\n{ctx}\n\nStatement: {sent}\n\n'
            'Is this statement supported by the context above? '
            'Answer with only "yes" or "no".')
    return sum(_llm_yes_no(tmpl.format(ctx=ctx_text, sent=s)) for s in sents) / len(sents)


def compute_answer_relevancy(question: str, answer: str) -> float:
    sents = _split_sentences(answer)
    if not sents: return 0.0
    tmpl = ('Question: {q}\n\nStatement: {sent}\n\n'
            'Is this statement relevant to answering the question above? '
            'Answer with only "yes" or "no".')
    return sum(_llm_yes_no(tmpl.format(q=question, sent=s)) for s in sents) / len(sents)


def compute_context_precision(question: str, contexts: List[str], reference: str) -> float:
    if not contexts: return 0.0
    tmpl = ('Question: {q}\n\nGround truth answer: {ref}\n\nRetrieved context: {ctx}\n\n'
            'Does this context contain information useful for correctly answering '
            'the question based on the ground truth? Answer with only "yes" or "no".')
    relevance = [1 if _llm_yes_no(tmpl.format(q=question, ref=reference[:300], ctx=c[:400])) else 0
                 for c in contexts]
    total_rel = sum(relevance)
    if total_rel == 0: return 0.0
    prec_sum, rel_count = 0.0, 0
    for k, rel in enumerate(relevance):
        if rel:
            rel_count += 1
            prec_sum  += rel_count / (k + 1)
    return prec_sum / total_rel


def evaluate_custom(question: str, answer: str, contexts: List[str], reference: str) -> Dict:
    return {
        'faithfulness'     : compute_faithfulness(answer, contexts),
        'context_recall'   : compute_context_recall(reference, contexts),
        'answer_relevancy' : compute_answer_relevancy(question, answer),
        'context_precision': compute_context_precision(question, contexts, reference),
    }


# ============================================================
# PHASE 1 (generation) and PHASE 2 (evaluation) RUNNERS
# ============================================================
def make_config_name(prompt_id: str, method: str) -> str:
    """e.g. P2_zeroshot_cot + qr_cr -> 'P2_zeroshot_cot__qr_cr_openai'."""
    return f'{prompt_id}__{method}_openai'


def run_phase1(prompt_id: str, prompt_template: str, max_tokens: int,
               method: str, max_samples: int = 100, verbose_every: int = 10):
    """Run Phase 1 (generation) for one (prompt, method) combo. Resumable."""
    load_everything(max_samples=max_samples)
    config = make_config_name(prompt_id, method)
    out    = RESULTS_DIR / f'{config}_phase1.json'
    runner = METHOD_RUNNERS[method]

    if out.exists():
        with open(out, 'r', encoding='utf-8') as f:
            results = json.load(f)['results']
        start = len(results)
        print(f'  Resume {config}: {start}/{max_samples} done.')
    else:
        results, start = [], 0
        print(f'  Start  {config}: 0/{max_samples}')

    if start >= max_samples:
        print(f'  Already complete.')
        return out

    t0 = time.time()
    for i in range(start, max_samples):
        s = _pubmedqa_data[i]
        q, gt, ref = s['question'], s['final_decision'], s['long_answer']

        retrieved, used_q = runner(q)
        answer    = generate_answer(q, retrieved, prompt_template, max_tokens=max_tokens)
        predicted = extract_label(answer)

        results.append({
            'idx': i, 'pubid': str(s['pubid']),
            'question': q, 'query_used': used_q,
            'ground_truth': gt, 'predicted_label': predicted,
            'is_correct': predicted == gt,
            'answer': answer,
            'contexts': [r.document.text for r in retrieved],
            'reference': ref,
            'rrf_scores'     : [r.rrf_score      for r in retrieved],
            'bm25_scores'    : [r.bm25_score     for r in retrieved],
            'dense_scores'   : [r.dense_score    for r in retrieved],
            'reranker_scores': [r.reranker_score for r in retrieved],
        })

        if (i + 1) % verbose_every == 0 or i == max_samples - 1:
            with open(out, 'w', encoding='utf-8') as f:
                json.dump({
                    'prompt_id': prompt_id, 'method': method, 'llm_model': LLM_MODEL,
                    'timestamp': datetime.now().isoformat(),
                    'max_samples': max_samples, 'completed': i + 1,
                    'results': results,
                }, f, indent=2, ensure_ascii=False)
            acc = sum(r['is_correct'] for r in results) / len(results)
            eta = (time.time() - t0) / (i + 1 - start) * (max_samples - i - 1) / 60
            print(f'    [{i+1:3d}/{max_samples}] acc={acc:.1%} | ETA {eta:.1f}m')
    return out


def run_phase2(phase1_path: Path, max_samples: int = 100, verbose_every: int = 10):
    """Run Phase 2 (4-metric evaluation) for one phase1 result file. Resumable."""
    load_everything(max_samples=max_samples)
    with open(phase1_path, 'r', encoding='utf-8') as f:
        p1 = json.load(f)
    p1_results = p1['results'][:max_samples]

    out = phase1_path.with_name(phase1_path.stem.replace('_phase1', '_phase2') + '.json')
    REQ = ['faithfulness', 'context_recall', 'answer_relevancy', 'context_precision']

    if out.exists():
        with open(out, 'r', encoding='utf-8') as f:
            p2 = json.load(f)['results']
        done = {r['idx'] for r in p2 if all(m in r for m in REQ)}
    else:
        p2, done = [], set()

    remaining = [r for r in p1_results if r['idx'] not in done]
    print(f'  Phase2 {phase1_path.stem}: {len(done)}/{max_samples} done, {len(remaining)} remaining.')
    if not remaining:
        return out

    t0 = time.time()
    for i, r in enumerate(remaining):
        scores = evaluate_custom(r['question'], r['answer'], r['contexts'], r['reference'])
        p2.append({
            'idx': r['idx'], 'ground_truth': r['ground_truth'],
            'predicted_label': r['predicted_label'], 'is_correct': r['is_correct'],
            **scores,
        })
        if (i + 1) % verbose_every == 0 or i == len(remaining) - 1:
            with open(out, 'w', encoding='utf-8') as f:
                json.dump({
                    'prompt_id': p1.get('prompt_id'), 'method': p1.get('method'),
                    'timestamp': datetime.now().isoformat(),
                    'max_samples': max_samples,
                    'evaluator': 'custom_zero_nan_4metrics',
                    'metrics': REQ, 'results': p2,
                }, f, indent=2, ensure_ascii=False)
            eta = (time.time() - t0) / (i + 1) * (len(remaining) - i - 1) / 60
            avg = {m: sum(x[m] for x in p2) / len(p2) for m in REQ}
            print(f'    [{i+1:3d}/{len(remaining)}] '
                  f'f={avg["faithfulness"]:.3f} cr={avg["context_recall"]:.3f} '
                  f'ar={avg["answer_relevancy"]:.3f} cp={avg["context_precision"]:.3f} '
                  f'| ETA {eta:.1f}m')
    return out


# ============================================================
# SUMMARY HELPERS
# ============================================================
def print_summary(phase2_path: Path):
    with open(phase2_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    r = data['results']
    n = len(r)
    if n == 0:
        print(f'  {phase2_path.stem}: empty')
        return
    acc  = sum(x['is_correct'] for x in r) / n
    f_   = sum(x['faithfulness'] for x in r) / n
    cr_  = sum(x['context_recall'] for x in r) / n
    ar_  = sum(x['answer_relevancy'] for x in r) / n
    cp_  = sum(x['context_precision'] for x in r) / n
    print(f'  {phase2_path.stem:<45} | n={n} | '
          f'acc={acc:.3f} f={f_:.3f} cr={cr_:.3f} ar={ar_:.3f} cp={cp_:.3f}')
    # per-label
    for lbl in ['yes', 'no', 'maybe']:
        sub = [x for x in r if x['ground_truth'] == lbl]
        if sub:
            sub_acc = sum(x['is_correct'] for x in sub) / len(sub)
            print(f'      {lbl:>5}: {sum(x["is_correct"] for x in sub)}/{len(sub)} = {sub_acc:.1%}')
