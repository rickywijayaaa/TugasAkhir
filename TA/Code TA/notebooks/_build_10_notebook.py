"""Generate 09-Agentic-RAG-OpenAI.ipynb programmatically."""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "09-Agentic-RAG-OpenAI.ipynb"
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ───────────────────────── INTRO ─────────────────────────
md("""# 09 — Agentic RAG dengan Hybrid Search (OpenAI)

Notebook ini mengimplementasi **Agentic RAG** sebagai eksperimen pendukung untuk membandingkan paradigma static pipeline (notebook 02–05) vs agentic dynamic pipeline.

## Konsep Agentic RAG

Berbeda dengan static pipeline yang menjalankan urutan tetap (Query → QR → Retriever → CR → LLM), pendekatan agentic membuat retrieval dan komponen mitigasi **menjadi tools yang dipanggil LLM secara otonom**:

```
Question → LLM Agent (loop)
              ├─→ rewrite_query  (decide kalau query ambigu)
              ├─→ hybrid_search  (BM25 + Dense via RRF)
              ├─→ rerank_context (CrossEncoder rerank)
              └─→ submit_answer  (yes/no/maybe + reasoning)
```

LLM decide sendiri tool mana yang dipanggil dan dalam urutan apa.

## Tools yang di-expose

1. **rewrite_query**: rewrite kueri medis untuk expand abbreviation & istilah ilmiah
2. **hybrid_search**: BM25 top-50 + Dense top-50 → RRF → top-K
3. **rerank_context**: CrossEncoder rerank kandidat
4. **submit_answer**: jawaban final yes/no/maybe + explanation

## Konfigurasi

- **LLM Agent**: gpt-4.1-mini (function calling native)
- **Max iterations**: 5 (rewrite + search + rerank + buffer + answer)
- **Stop conditions**: submit_answer dipanggil, atau max_iter tercapai
- **Fallback**: extract label dari last assistant message (pakai final disambiguation call), TIDAK force "maybe"

## Output

- `results/agentic_openai_phase1_answers.json` — Phase 1 jawaban + tool call log
- `results/agentic_openai_phase2_custom.json` — Phase 2 RAGAS 4-metric
- `figures/I_agentic_vs_static.png` — visualisasi perbandingan
""")

# ───────────────────────── CELL 1: imports & config ─────────────────────────
md("## Cell 1 — Imports & Konfigurasi")
code("""import os, sys, json, pickle, time, re, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime

from openai import OpenAI
from rank_bm25 import BM25Okapi
import chromadb

warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI
# ============================================================
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YOUR_OPENAI_KEY_HERE')

LLM_MODEL   = 'gpt-4.1-mini'
EMBED_MODEL = 'text-embedding-3-small'

# Hybrid retrieval params (sama dengan notebook 02.1)
TOP_K_BM25       = 50
TOP_K_DENSE      = 50
TOP_K_RETRIEVAL  = 5

# Agentic params
MAX_ITERATIONS   = 5
RERANK_TOP_N     = 20  # ambil 20 kandidat untuk rerank, pilih top-5

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

CONFIG_NAME        = 'agentic_openai'
PHASE1_PATH        = RESULTS_DIR / f'{CONFIG_NAME}_phase1_answers.json'
PHASE2_CUSTOM_PATH = RESULTS_DIR / f'{CONFIG_NAME}_phase2_custom.json'

print('Konfigurasi:')
print(f'  LLM Agent       : {LLM_MODEL} (native function calling)')
print(f'  Embedder        : {EMBED_MODEL}')
print(f'  Max iterations  : {MAX_ITERATIONS}')
print(f'  Hybrid retrieval: BM25 top-{TOP_K_BM25} + Dense top-{TOP_K_DENSE} -> RRF -> top-{TOP_K_RETRIEVAL}')
print(f'  Reranker        : CrossEncoder ms-marco-MiniLM-L-6-v2 (lazy load)')
print(f'  Sampel          : {MAX_SAMPLES}')
print(f'  Output          : {PHASE1_PATH}')
print()
if 'YOUR_API_KEY' in OPENAI_API_KEY:
    print('  OPENAI_API_KEY belum diset!')
else:
    print(f'  OPENAI_API_KEY  : {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}')""")

# ───────────────────────── CELL 2: data classes & loaders ─────────────────────────
md("## Cell 2 — Data Classes & Load BM25/Chroma")
code("""@dataclass
class Document:
    text         : str
    pubid        : str
    question     : str
    section_label: str
    answer       : str
    decision     : str

def tokenize_bm25(text: str) -> List[str]:
    return re.sub(r'[^a-zA-Z0-9\\s]', ' ', text.lower()).split()

# Load BM25 index
print('Loading BM25 index...')
with open(BM25_INDEX_PATH, 'rb') as f:
    bm25_data = pickle.load(f)
bm25_index = bm25_data['bm25']
documents  = bm25_data['documents']
print(f'  BM25 loaded: {len(documents)} dokumen')

# Load Chroma collection
print('Loading Chroma collection...')
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
chroma_collection = chroma_client.get_collection(name='pubmedqa_docs')
print(f'  Chroma loaded: {chroma_collection.count()} dokumen')

# Load PubMedQA test data
print('Loading PubMedQA test set...')
import csv
PUBMEDQA_CSV = Path('../data/pubmedqa_pqa_labeled.csv')
test_samples = []
with open(PUBMEDQA_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        test_samples.append({
            'idx'           : len(test_samples),
            'pubid'         : row['pubid'],
            'question'      : row['question'],
            'final_decision': row['final_decision'],
            'long_answer'   : row['long_answer'],
        })
        if len(test_samples) >= MAX_SAMPLES:
            break
print(f'  Loaded {len(test_samples)} test samples')""")

# ───────────────────────── CELL 3: openai client + retrieval helpers ─────────────────────────
md("## Cell 3 — OpenAI Client & Retrieval Helpers (reuse dari notebook 02.1)")
code("""openai_client = OpenAI(api_key=OPENAI_API_KEY)

def openai_embed(texts: List[str]) -> List[List[float]]:
    for attempt in range(5):
        try:
            r = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [d.embedding for d in r.data]
        except Exception as e:
            if attempt == 4: raise
            time.sleep((attempt + 1) * 5)
    raise RuntimeError('embed gagal')

# ============================================================
# Retrieval primitives — wrap fungsi yang sudah ada
# ============================================================
def retrieve_dense(query: str, k: int) -> List[Tuple[int, float]]:
    qvec = openai_embed([query])[0]
    r = chroma_collection.query(query_embeddings=[qvec], n_results=k, include=['distances'])
    doc_ids   = [int(i) for i in r['ids'][0]]
    distances = r['distances'][0]
    return [(d, 1.0 - dist) for d, dist in zip(doc_ids, distances)]

def retrieve_bm25_raw(query: str, k: int) -> List[Tuple[int, float]]:
    tokens = tokenize_bm25(query)
    scores = bm25_index.get_scores(tokens)
    top    = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top]

def reciprocal_rank_fusion(rank_lists: List[List[int]], k: int = 60) -> List[Tuple[int, float]]:
    s = {}
    for rl in rank_lists:
        for rank, did in enumerate(rl):
            s[did] = s.get(did, 0.0) + 1.0 / (k + rank + 1)
    return sorted(s.items(), key=lambda x: -x[1])

def hybrid_retrieve(query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
    \"\"\"Return list of dict per dokumen: {doc_id, text, section, bm25, dense, rrf}.\"\"\"
    bm25_pairs  = retrieve_bm25_raw(query, k=TOP_K_BM25)
    dense_pairs = retrieve_dense(query,    k=TOP_K_DENSE)
    bm25_scores  = dict(bm25_pairs)
    dense_scores = dict(dense_pairs)
    fused = reciprocal_rank_fusion([
        [d for d, _ in bm25_pairs],
        [d for d, _ in dense_pairs],
    ])
    out = []
    for doc_id, rrf_score in fused[:top_k]:
        d = documents[doc_id]
        out.append({
            'doc_id' : doc_id,
            'text'   : d.text,
            'section': d.section_label,
            'bm25'   : bm25_scores.get(doc_id, 0.0),
            'dense'  : dense_scores.get(doc_id, 0.0),
            'rrf'    : rrf_score,
        })
    return out

# Smoke test
test_q = 'Does aspirin reduce myocardial infarction risk?'
test_r = hybrid_retrieve(test_q, top_k=3)
print(f'Smoke test query: {test_q}')
for i, r in enumerate(test_r, 1):
    print(f'  [{i}] doc_id={r["doc_id"]} rrf={r["rrf"]:.4f} | {r["section"]} | {r["text"][:70]}...')""")

# ───────────────────────── CELL 4: cross-encoder reranker ─────────────────────────
md("## Cell 4 — Lazy Load CrossEncoder Reranker")
code("""# Lazy load CrossEncoder hanya saat dipanggil pertama kali
_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print('  Loading CrossEncoder ms-marco-MiniLM-L-6-v2...')
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        print('  CrossEncoder ready')
    return _cross_encoder

def rerank_documents(query: str, candidates: List[Dict], top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
    \"\"\"Rerank candidate documents using CrossEncoder, return top_k.\"\"\"
    if not candidates:
        return []
    ce = get_cross_encoder()
    pairs  = [(query, c['text'][:1500]) for c in candidates]
    scores = ce.predict(pairs)
    for c, s in zip(candidates, scores):
        c['reranker_score'] = float(s)
    return sorted(candidates, key=lambda x: -x['reranker_score'])[:top_k]

print('CrossEncoder helper ready (lazy load saat dipanggil)')""")

# ───────────────────────── CELL 5: tools definition ─────────────────────────
md("""## Cell 5 — Definisi 4 Tools untuk Agent

Setiap tool dideklarasikan dalam OpenAI function calling schema. Agent dapat memanggil tool ini sesuai keputusannya.""")
code("""TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'rewrite_query',
            'description': (
                'Rewrite the medical question to expand abbreviations and add relevant scientific '
                'terminology for better retrieval. Use this when the original question is vague, '
                'too short, or contains medical abbreviations like MI, CAD, T2DM. '
                'Do NOT use if the question is already specific and well-formed.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'original_query': {
                        'type': 'string',
                        'description': 'The original user question to be rewritten',
                    },
                },
                'required': ['original_query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'hybrid_search',
            'description': (
                'Search the PubMedQA biomedical corpus using hybrid BM25 (sparse keyword) + '
                'Dense (semantic) retrieval merged via Reciprocal Rank Fusion. '
                'Returns top-K most relevant abstract sections with BM25, dense, and RRF scores.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'The query to search (use rewritten query if available)',
                    },
                    'top_k': {
                        'type': 'integer',
                        'description': 'Number of top documents to return (5-20 recommended)',
                        'default': 5,
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'rerank_context',
            'description': (
                'Rerank previously retrieved candidate documents using a CrossEncoder model that '
                'jointly evaluates (query, document) relevance. Use this when you have many '
                'candidates from hybrid_search and want to filter to the most relevant ones. '
                'Pass doc_ids from a previous hybrid_search call.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'The query to rerank against',
                    },
                    'doc_ids': {
                        'type': 'array',
                        'items': {'type': 'integer'},
                        'description': 'List of doc_ids from previous hybrid_search to rerank',
                    },
                    'top_k': {
                        'type': 'integer',
                        'description': 'Number of top documents to return after reranking',
                        'default': 5,
                    },
                },
                'required': ['query', 'doc_ids'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'submit_answer',
            'description': (
                'Submit the final yes/no/maybe answer to the medical question. Call this ONLY when '
                'you have gathered enough evidence from retrieved documents. Provide a brief '
                'explanation grounded in the retrieved context.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'answer': {
                        'type': 'string',
                        'enum': ['yes', 'no', 'maybe'],
                        'description': (
                            'yes: evidence supports the hypothesis, '
                            'no: evidence refutes the hypothesis, '
                            'maybe: evidence is directly contradictory or no relevant info'
                        ),
                    },
                    'explanation': {
                        'type': 'string',
                        'description': '2-3 sentence explanation citing the retrieved evidence',
                    },
                },
                'required': ['answer', 'explanation'],
            },
        },
    },
]

print(f'Defined {len(TOOLS)} tools:')
for t in TOOLS:
    print(f'  - {t["function"]["name"]}: {t["function"]["description"][:80]}...')""")

# ───────────────────────── CELL 6: tool execution ─────────────────────────
md("## Cell 6 — Tool Dispatcher & Helper Functions")
code("""# State global per-agent-run untuk track candidate cache (untuk rerank)
class AgentState:
    \"\"\"State per agent invocation: simpan kandidat untuk rerank, log tool calls.\"\"\"
    def __init__(self):
        self.candidates_cache : Dict[int, Dict] = {}  # doc_id -> doc info
        self.tool_log         : List[Dict] = []
        self.iterations       : int = 0

def llm_rewrite_query(original: str) -> str:
    \"\"\"Internal LLM call untuk rewrite query (separate dari main agent loop).\"\"\"
    prompt = (
        'You are a query rewriting assistant for a biomedical QA system.\\n'
        'Rewrite the medical question to improve retrieval from PubMed abstracts.\\n\\n'
        'Rules:\\n'
        '1. Expand medical abbreviations (e.g., MI -> myocardial infarction)\\n'
        '2. Add specific scientific terminology where appropriate\\n'
        '3. Preserve the original yes/no/maybe answerable intent\\n'
        '4. Output ONLY the rewritten question, no explanations\\n\\n'
        f'Question: {original}\\n\\nRewritten:'
    )
    for attempt in range(3):
        try:
            r = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0, max_tokens=120, seed=SEED,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2: raise
            time.sleep((attempt + 1) * 5)
    return original

def execute_tool(name: str, args: Dict[str, Any], state: AgentState) -> Any:
    \"\"\"Dispatch tool call. Returns JSON-serializable result for tool message.\"\"\"
    if name == 'rewrite_query':
        rewritten = llm_rewrite_query(args['original_query'])
        return {'rewritten_query': rewritten}

    elif name == 'hybrid_search':
        q     = args['query']
        top_k = int(args.get('top_k', 5))
        top_k = max(1, min(top_k, 20))  # clamp
        results = hybrid_retrieve(q, top_k=top_k)
        # Cache kandidat untuk rerank
        for r in results:
            state.candidates_cache[r['doc_id']] = r
        # Kembalikan ringkas (preview text) ke LLM agar token efisien
        return [
            {
                'doc_id'        : r['doc_id'],
                'section'       : r['section'],
                'text_preview'  : r['text'][:300],
                'bm25_score'    : round(r['bm25'], 2),
                'dense_score'   : round(r['dense'], 3),
                'rrf_score'     : round(r['rrf'], 4),
            }
            for r in results
        ]

    elif name == 'rerank_context':
        q       = args['query']
        doc_ids = args['doc_ids']
        top_k   = int(args.get('top_k', 5))
        # Ambil dari cache; kalau ada doc_id yang belum tercache, fallback skip
        candidates = [state.candidates_cache[did] for did in doc_ids if did in state.candidates_cache]
        if not candidates:
            return {'error': 'No cached candidates. Call hybrid_search first.'}
        reranked = rerank_documents(q, candidates, top_k=top_k)
        return [
            {
                'doc_id'         : r['doc_id'],
                'section'        : r['section'],
                'text_preview'   : r['text'][:300],
                'reranker_score' : round(r.get('reranker_score', 0.0), 4),
            }
            for r in reranked
        ]

    elif name == 'submit_answer':
        # Sentinel — caller akan detect dan stop loop
        return {'final': True, 'answer': args['answer'], 'explanation': args['explanation']}

    return {'error': f'Unknown tool: {name}'}

print('Tool dispatcher ready')""")

# ───────────────────────── CELL 7: agent loop ─────────────────────────
md("## Cell 7 — Agent Loop (max 5 iterasi + fallback parsing)")
code("""SYSTEM_PROMPT = (
    'You are a medical research assistant answering yes/no/maybe questions from PubMedQA. '
    'You have access to tools for retrieving and reranking biomedical abstracts.\\n\\n'
    'Strategy:\\n'
    '1. If the question is vague or has medical abbreviations, consider calling rewrite_query first.\\n'
    '2. Call hybrid_search with the (possibly rewritten) query to get candidates.\\n'
    '3. If hybrid_search returns many noisy candidates, you may call rerank_context.\\n'
    '4. Once you have sufficient evidence, call submit_answer with yes/no/maybe.\\n\\n'
    'Important:\\n'
    '- Base your answer ONLY on retrieved abstracts, NOT prior knowledge.\\n'
    '- Use "maybe" ONLY if evidence is directly contradictory OR no relevant info found.\\n'
    '- If evidence leans even partially in one direction, choose yes or no.\\n'
    '- Be efficient. You have a maximum of {max_iter} tool call iterations.'
)

def parse_label_from_text(text: str) -> Optional[str]:
    \"\"\"Strict label parser dari teks. Return None kalau tidak ditemukan.\"\"\"
    if not text:
        return None
    lines = [l.strip().lower() for l in text.split('\\n') if l.strip()]
    # Cek baris terakhir dulu
    for line in reversed(lines[-3:]):
        word = re.sub(r'[^a-z]', '', line)
        if word in ('yes', 'no', 'maybe'):
            return word
    # Cek seluruh text untuk standalone word
    for label in ('yes', 'no', 'maybe'):
        if re.search(r'\\b' + label + r'\\b', text.lower()):
            return label
    return None

def force_disambiguation_call(messages: List[Dict]) -> Tuple[str, str]:
    \"\"\"Final LLM call untuk minta jawaban definitif. Return (label, raw_text).\"\"\"
    follow_up = list(messages) + [{
        'role': 'user',
        'content': (
            'Based on all the evidence you have gathered above, what is your final answer to the '
            'original question? Respond with EXACTLY ONE word: yes, no, or maybe. No explanation.'
        ),
    }]
    for attempt in range(3):
        try:
            r = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=follow_up,
                temperature=0.0, max_tokens=5, seed=SEED,
            )
            text = r.choices[0].message.content.strip()
            word = re.sub(r'[^a-z]', '', text.lower())
            if word in ('yes', 'no', 'maybe'):
                return word, text
            # Kalau response aneh, parse longgar
            for label in ('yes', 'no', 'maybe'):
                if label in text.lower():
                    return label, text
            return 'maybe', text  # absolute fallback (sangat jarang)
        except Exception as e:
            if attempt == 2: raise
            time.sleep((attempt + 1) * 5)
    return 'maybe', ''

def run_agent(question: str, max_iter: int = MAX_ITERATIONS, verbose: bool = False) -> Dict:
    \"\"\"
    Run agentic RAG untuk satu question.
    Return dict: answer, explanation, contexts, iterations, tool_log, fallback_used
    \"\"\"
    state = AgentState()
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT.format(max_iter=max_iter)},
        {'role': 'user',   'content': question},
    ]

    final_answer      = None
    final_explanation = None
    final_contexts    = []   # list dict (top context yang dipakai untuk submit)
    fallback_used     = None

    for it in range(max_iter):
        state.iterations = it + 1
        try:
            resp = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice='auto',
                temperature=TEMPERATURE,
                seed=SEED,
            )
        except Exception as e:
            if verbose: print(f'  [iter {it+1}] LLM error: {e}')
            time.sleep(5)
            try:
                resp = openai_client.chat.completions.create(
                    model=LLM_MODEL, messages=messages, tools=TOOLS,
                    tool_choice='auto', temperature=TEMPERATURE, seed=SEED,
                )
            except Exception:
                raise

        msg = resp.choices[0].message

        # Kalau tidak ada tool_calls, agent return text answer langsung
        if not msg.tool_calls:
            text = msg.content or ''
            label = parse_label_from_text(text)
            if label:
                final_answer = label
                final_explanation = text
                fallback_used = 'no_tool_text_parse'
            else:
                # Force disambiguation call
                messages.append({'role': 'assistant', 'content': text})
                label, raw = force_disambiguation_call(messages)
                final_answer = label
                final_explanation = text + f'\\n[FORCED: {raw}]'
                fallback_used = 'force_disambiguation'
            break

        # Append assistant message dengan tool_calls
        messages.append({
            'role': 'assistant',
            'content': msg.content,
            'tool_calls': [
                {
                    'id': tc.id,
                    'type': 'function',
                    'function': {'name': tc.function.name, 'arguments': tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute tiap tool_call
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            if verbose:
                print(f'  [iter {it+1}] tool={name} args={list(args.keys())}')

            result = execute_tool(name, args, state)
            state.tool_log.append({'iter': it + 1, 'name': name, 'args_keys': list(args.keys())})

            # Kalau submit_answer, set final
            if name == 'submit_answer':
                final_answer = args.get('answer', 'maybe')
                final_explanation = args.get('explanation', '')
                # Take top contexts dari cache (yang paling sering di-reference)
                # Untuk simplicity ambil semua kandidat di cache
                final_contexts = list(state.candidates_cache.values())[:TOP_K_RETRIEVAL]
                fallback_used = None
                # Tetap append tool message untuk completeness
                messages.append({
                    'role': 'tool', 'tool_call_id': tc.id,
                    'content': json.dumps(result),
                })
                return {
                    'answer'         : final_answer,
                    'explanation'    : final_explanation,
                    'contexts'       : final_contexts,
                    'iterations'     : state.iterations,
                    'tool_log'       : state.tool_log,
                    'fallback_used'  : fallback_used,
                    'iter_overflow'  : False,
                }

            messages.append({
                'role': 'tool', 'tool_call_id': tc.id,
                'content': json.dumps(result),
            })

        # Sudah submit_answer? sudah return di atas

    # Max iter tercapai tanpa submit_answer
    # Strategy: extract dari last assistant message, fallback ke disambiguation call
    last_text = ''
    for m in reversed(messages):
        if m.get('role') == 'assistant' and m.get('content'):
            last_text = m['content']
            break

    label = parse_label_from_text(last_text)
    if label:
        final_answer      = label
        final_explanation = last_text
        fallback_used     = 'iter_overflow_text_parse'
    else:
        label, raw = force_disambiguation_call(messages)
        final_answer      = label
        final_explanation = (last_text or '') + f'\\n[FORCED: {raw}]'
        fallback_used     = 'iter_overflow_force_disambiguation'

    final_contexts = list(state.candidates_cache.values())[:TOP_K_RETRIEVAL]
    return {
        'answer'         : final_answer,
        'explanation'    : final_explanation,
        'contexts'       : final_contexts,
        'iterations'     : state.iterations,
        'tool_log'       : state.tool_log,
        'fallback_used'  : fallback_used,
        'iter_overflow'  : True,
    }

print(f'Agent loop ready (max_iter={MAX_ITERATIONS})')""")

# ───────────────────────── CELL 8: smoke test ─────────────────────────
md("## Cell 8 — Smoke Test (3 sampel)")
code("""print(f'SMOKE TEST: 3 sampel pertama (verbose mode)\\n' + '='*65)
for i in range(3):
    s = test_samples[i]
    print(f'\\n[{i+1}/3] Q: {s["question"][:80]}...')
    print(f'      GT: {s["final_decision"]}')

    t0 = time.perf_counter()
    result = run_agent(s['question'], verbose=True)
    elapsed = time.perf_counter() - t0

    correct = result['answer'] == s['final_decision']
    print(f'      Pred: {result["answer"]} ({"BENAR" if correct else "SALAH"})')
    print(f'      Iterations: {result["iterations"]}/{MAX_ITERATIONS}')
    print(f'      Tool calls: {[t["name"] for t in result["tool_log"]]}')
    print(f'      Fallback: {result["fallback_used"]}')
    print(f'      Time: {elapsed:.1f}s')""")

# ───────────────────────── CELL 9: phase 1 full run ─────────────────────────
md("""## Cell 9 — Phase 1: Full Run 500 Sampel (Resume-able)

Estimasi: ~30-45 menit, ~$3 OpenAI cost.""")
code("""# Resume kalau ada
if PHASE1_PATH.exists():
    with open(PHASE1_PATH, 'r', encoding='utf-8') as f:
        phase1_results = json.load(f)['results']
    start_from = len(phase1_results)
    print(f'Resume Phase 1: {start_from}/{MAX_SAMPLES} sudah selesai.')
else:
    phase1_results, start_from = [], 0
    print(f'Memulai Phase 1: {MAX_SAMPLES} sampel agentic.')

if start_from < MAX_SAMPLES:
    print(f'Memproses {MAX_SAMPLES - start_from} sampel...\\n')
    t_start = time.time()

    for i in range(start_from, MAX_SAMPLES):
        s = test_samples[i]
        try:
            result = run_agent(s['question'], verbose=False)
        except Exception as e:
            print(f'  [{i}] ERROR: {type(e).__name__}: {str(e)[:120]}')
            # Save progress + raise
            with open(PHASE1_PATH, 'w', encoding='utf-8') as f:
                json.dump({
                    'config': CONFIG_NAME, 'llm_model': LLM_MODEL,
                    'embed_model': EMBED_MODEL, 'max_iter': MAX_ITERATIONS,
                    'timestamp': datetime.now().isoformat(),
                    'max_samples': MAX_SAMPLES, 'completed': len(phase1_results),
                    'results': phase1_results,
                }, f, indent=2, ensure_ascii=False)
            raise

        contexts_text = [c['text'] for c in result['contexts']]

        phase1_results.append({
            'idx'            : i,
            'pubid'          : str(s['pubid']),
            'question'       : s['question'],
            'ground_truth'   : s['final_decision'],
            'predicted_label': result['answer'],
            'is_correct'     : result['answer'] == s['final_decision'],
            'answer'         : result['explanation'],
            'contexts'       : contexts_text,
            'reference'      : s['long_answer'],
            'iterations'     : result['iterations'],
            'tool_log'       : result['tool_log'],
            'fallback_used'  : result['fallback_used'],
            'iter_overflow'  : result['iter_overflow'],
        })

        if (i + 1) % 10 == 0 or i == MAX_SAMPLES - 1:
            with open(PHASE1_PATH, 'w', encoding='utf-8') as f:
                json.dump({
                    'config': CONFIG_NAME, 'llm_model': LLM_MODEL,
                    'embed_model': EMBED_MODEL, 'max_iter': MAX_ITERATIONS,
                    'timestamp': datetime.now().isoformat(),
                    'max_samples': MAX_SAMPLES, 'completed': i + 1,
                    'results': phase1_results,
                }, f, indent=2, ensure_ascii=False)
            done = i + 1
            acc  = sum(r['is_correct'] for r in phase1_results) / done
            avg_iter = np.mean([r['iterations'] for r in phase1_results])
            n_overflow = sum(1 for r in phase1_results if r['iter_overflow'])
            eta  = (time.time() - t_start) / (done - start_from) * (MAX_SAMPLES - done) / 60
            print(f'  [{done:3d}/{MAX_SAMPLES}] acc={acc:.1%} avg_iter={avg_iter:.1f} '
                  f'overflow={n_overflow} pred={result["answer"]} gt={s["final_decision"]} ETA={eta:.1f}m')

    print(f'\\nPhase 1 selesai! -> {PHASE1_PATH}')
else:
    print(f'Phase 1 sudah selesai ({MAX_SAMPLES} sampel).')""")

# ───────────────────────── CELL 10: phase 1 analysis ─────────────────────────
md("## Cell 10 — Phase 1 Analysis")
code("""with open(PHASE1_PATH, 'r', encoding='utf-8') as f:
    p1 = json.load(f)['results']

n         = len(p1)
n_correct = sum(r['is_correct'] for r in p1)

print(f'AGENTIC RAG OPENAI - {n} sampel')
print('=' * 60)
print(f'Label Accuracy    : {n_correct}/{n} = {n_correct/n:.1%}')
print(f'Hallucination Rate: {(n-n_correct)/n:.1%}\\n')

# Per-label
print('Per-label accuracy:')
for lbl in ['yes', 'no', 'maybe']:
    sub = [r for r in p1 if r['ground_truth'] == lbl]
    if sub:
        c = sum(r['is_correct'] for r in sub)
        print(f'  {lbl:>5}: {c}/{len(sub)} = {c/len(sub):.1%}')

# Distribusi prediksi
preds = [r['predicted_label'] for r in p1]
print(f'\\nDistribusi prediksi:')
for lbl in ['yes', 'no', 'maybe']:
    print(f'  {lbl:>5}: {preds.count(lbl)} ({preds.count(lbl)/n:.0%})')

# === Agentic-specific: iteration & tool usage ===
print(f'\\nAgentic statistics:')
iters = [r['iterations'] for r in p1]
print(f'  Avg iterations: {np.mean(iters):.2f} (min={min(iters)}, max={max(iters)})')
print(f'  Iter distribution: {pd.Series(iters).value_counts().sort_index().to_dict()}')

n_overflow = sum(1 for r in p1 if r['iter_overflow'])
print(f'  Iter overflow (max_iter reached): {n_overflow}/{n} ({n_overflow/n:.1%})')

# Tool usage frequency
tool_freq = {}
for r in p1:
    for t in r['tool_log']:
        tool_freq[t['name']] = tool_freq.get(t['name'], 0) + 1
print(f'\\n  Tool call frequency:')
for name, count in sorted(tool_freq.items(), key=lambda x: -x[1]):
    avg_per_q = count / n
    print(f'    {name:<20}: {count:5d} total ({avg_per_q:.2f}/query)')

# Fallback usage
fb_freq = {}
for r in p1:
    fb = r['fallback_used'] or 'submit_answer (normal)'
    fb_freq[fb] = fb_freq.get(fb, 0) + 1
print(f'\\n  Answer extraction:')
for fb, count in sorted(fb_freq.items(), key=lambda x: -x[1]):
    print(f'    {fb:<40}: {count:4d} ({count/n:.1%})')""")

# ───────────────────────── CELL 11: phase 2 RAGAS ─────────────────────────
md("""## Cell 11 — Phase 2: Custom RAGAS (4 metrik, zero-NaN evaluator)

Reuse evaluator dari notebook 02.1. Estimasi: ~30 menit, ~$2 OpenAI cost.""")
code("""# ============================================================
# Custom Zero-NaN Evaluator — 4 metrik (sama dengan notebook 02.1)
# ============================================================
def openai_generate(prompt: str, max_tokens: int = 300, temperature: float = TEMPERATURE) -> str:
    for attempt in range(5):
        try:
            r = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=temperature, max_tokens=max_tokens, seed=SEED,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 4: raise
            time.sleep((attempt + 1) * 5)
    raise RuntimeError('llm gagal')

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
    if not sentences: return 0.0
    ctx_text = '\\n'.join(f'[{i+1}] {c[:400]}' for i, c in enumerate(contexts))
    p = ('Context:\\n{ctx}\\n\\nStatement: {sent}\\n\\n'
         'Is this statement directly supported by the context above? Answer with only "yes" or "no".')
    supp = sum(1 for s in sentences if _llm_yes_no(p.format(ctx=ctx_text, sent=s)))
    return supp / len(sentences)

def compute_context_recall(reference: str, contexts: List[str]) -> float:
    sentences = _split_sentences(reference)
    if not sentences: return 0.0
    ctx_text = '\\n'.join(f'[{i+1}] {c[:400]}' for i, c in enumerate(contexts))
    p = ('Context:\\n{ctx}\\n\\nStatement: {sent}\\n\\n'
         'Is this statement supported by the context above? Answer with only "yes" or "no".')
    cov = sum(1 for s in sentences if _llm_yes_no(p.format(ctx=ctx_text, sent=s)))
    return cov / len(sentences)

def compute_answer_relevancy(question: str, answer: str) -> float:
    sentences = _split_sentences(answer)
    if not sentences: return 0.0
    p = ('Question: {q}\\n\\nStatement: {s}\\n\\n'
         'Is this statement relevant to answering the question above? Answer with only "yes" or "no".')
    rel = sum(1 for s in sentences if _llm_yes_no(p.format(q=question, s=s)))
    return rel / len(sentences)

def compute_context_precision(question: str, contexts: List[str], reference: str) -> float:
    if not contexts: return 0.0
    p = ('Question: {q}\\n\\nGround truth answer: {r}\\n\\nRetrieved context: {ctx}\\n\\n'
         'Does this context contain information useful for correctly answering the question '
         'based on the ground truth? Answer with only "yes" or "no".')
    relevance = []
    for ctx in contexts:
        is_rel = _llm_yes_no(p.format(q=question, r=reference[:300], ctx=ctx[:400]))
        relevance.append(1 if is_rel else 0)
    total_rel = sum(relevance)
    if total_rel == 0: return 0.0
    psum, rcount = 0.0, 0
    for k, rel in enumerate(relevance):
        if rel:
            rcount += 1
            psum += rcount / (k + 1)
    return psum / total_rel

def evaluate_custom(question, answer, contexts, reference):
    return {
        'faithfulness'      : compute_faithfulness(answer, contexts),
        'context_recall'    : compute_context_recall(reference, contexts),
        'answer_relevancy'  : compute_answer_relevancy(question, answer),
        'context_precision' : compute_context_precision(question, contexts, reference),
    }

# ============================================================
# Run Phase 2 dengan resume
# ============================================================
MAX_CUSTOM = 500
REQUIRED = ['faithfulness', 'context_recall', 'answer_relevancy', 'context_precision']

with open(PHASE1_PATH, 'r', encoding='utf-8') as f:
    p1_data = json.load(f)['results'][:MAX_CUSTOM]

if PHASE2_CUSTOM_PATH.exists():
    with open(PHASE2_CUSTOM_PATH, 'r', encoding='utf-8') as f:
        p2 = json.load(f)['results']
    done = {r['idx'] for r in p2 if all(m in r for m in REQUIRED)}
    print(f'Resume Phase 2: {len(done)}/{MAX_CUSTOM} sudah selesai dengan 4 metrik.')
else:
    p2, done = [], set()
    print(f'Mulai Phase 2: {MAX_CUSTOM} sampel.')

remaining = [r for r in p1_data if r['idx'] not in done]
print(f'Mengevaluasi {len(remaining)} sampel baru...\\n')

t0 = time.time()
for i, r in enumerate(remaining):
    scores = evaluate_custom(r['question'], r['answer'], r['contexts'], r['reference'])
    p2.append({
        'idx': r['idx'], 'ground_truth': r['ground_truth'],
        'predicted_label': r['predicted_label'], 'is_correct': r['is_correct'],
        **scores
    })
    if (i + 1) % 5 == 0 or i == len(remaining) - 1:
        with open(PHASE2_CUSTOM_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                'config': CONFIG_NAME, 'timestamp': datetime.now().isoformat(),
                'max_samples': MAX_CUSTOM, 'metrics': REQUIRED,
                'evaluator': 'custom_zero_nan_4metrics',
                'results': p2
            }, f, indent=2, ensure_ascii=False)
        d = i + 1
        eta = (time.time() - t0) / d * (len(remaining) - d) / 60 if d < len(remaining) else 0
        avg_f  = np.mean([x['faithfulness']      for x in p2])
        avg_cr = np.mean([x['context_recall']    for x in p2])
        avg_ar = np.mean([x['answer_relevancy']  for x in p2])
        avg_cp = np.mean([x['context_precision'] for x in p2])
        print(f'  [{d:3d}/{len(remaining)}] idx={r["idx"]} | '
              f'f={scores["faithfulness"]:.2f} cr={scores["context_recall"]:.2f} '
              f'ar={scores["answer_relevancy"]:.2f} cp={scores["context_precision"]:.2f} | '
              f'avg: f={avg_f:.3f} cr={avg_cr:.3f} ar={avg_ar:.3f} cp={avg_cp:.3f} | ETA={eta:.1f}m')

print(f'\\nPhase 2 selesai! -> {PHASE2_CUSTOM_PATH}')""")

# ───────────────────────── CELL 12: summary ─────────────────────────
md("## Cell 12 — Summary & Comparison vs Static OpenAI")
code("""with open(PHASE2_CUSTOM_PATH, 'r', encoding='utf-8') as f:
    p2 = json.load(f)['results']

n      = len(p2)
acc    = sum(r['is_correct']        for r in p2) / n
avg_f  = np.mean([r['faithfulness']      for r in p2])
avg_cr = np.mean([r['context_recall']    for r in p2])
avg_ar = np.mean([r['answer_relevancy']  for r in p2])
avg_cp = np.mean([r['context_precision'] for r in p2])

print('=' * 65)
print(f'  AGENTIC OPENAI - {n} sampel')
print(f'  LLM Agent: {LLM_MODEL} | Max iter: {MAX_ITERATIONS}')
print('=' * 65)
print(f'  Label Accuracy     : {acc:.1%}')
print(f'  Hallucination Rate : {1-acc:.1%}')
print(f'  Faithfulness       : {avg_f:.4f}')
print(f'  Context Recall     : {avg_cr:.4f}')
print(f'  Answer Relevancy   : {avg_ar:.4f}')
print(f'  Context Precision  : {avg_cp:.4f}')
print('=' * 65)

# Per-label
print('\\nPer-label accuracy:')
for lbl in ['yes', 'no', 'maybe']:
    sub = [r for r in p2 if r['ground_truth'] == lbl]
    if sub:
        c = sum(r['is_correct'] for r in sub)
        print(f'  {lbl:>5}: {c}/{len(sub)} = {c/len(sub):.1%}')

# Comparison vs static OpenAI configs
print('\\n=== Perbandingan vs Static OpenAI ===')
for prev_name, prev_path in [
    ('Baseline OpenAI',  '../results/baseline_openai_phase2_custom.json'),
    ('QR OpenAI',        '../results/qr_openai_phase2_custom.json'),
    ('CR OpenAI',        '../results/cr_openai_phase2_custom.json'),
    ('QR+CR OpenAI',     '../results/qr_cr_openai_phase2_custom.json'),
    ('Hybrid OpenAI',    '../results/hybrid_openai_phase2_custom.json'),
]:
    try:
        with open(prev_path, 'r', encoding='utf-8') as f:
            prev = json.load(f)['results']
        p_acc = sum(r['is_correct']        for r in prev) / len(prev)
        p_f   = np.mean([r['faithfulness']      for r in prev])
        p_cr  = np.mean([r['context_recall']    for r in prev])
        p_ar  = np.mean([r['answer_relevancy']  for r in prev])
        p_cp  = np.mean([r['context_precision'] for r in prev])
        delta = (acc - p_acc) * 100
        print(f'  {prev_name:<18}: acc={p_acc:.1%} f={p_f:.3f} cr={p_cr:.3f} '
              f'ar={p_ar:.3f} cp={p_cp:.3f} | delta acc = {delta:+.1f}%')
    except FileNotFoundError:
        print(f'  {prev_name:<18}: file tidak ditemukan')

print(f'\\nBaris untuk tabel skripsi:')
print(f'  | Agentic OpenAI | {acc:.3f} | {1-acc:.3f} | '
      f'{avg_f:.3f} | {avg_cr:.3f} | {avg_ar:.3f} | {avg_cp:.3f} |')""")

# ───────────────────────── CELL 13: visualization ─────────────────────────
md("## Cell 13 — Visualisasi Agentic vs Static")
code("""import matplotlib.pyplot as plt

# Load semua phase2 results
configs_data = {}
for name, path in [
    ('Baseline',     '../results/baseline_openai_phase2_custom.json'),
    ('QR',           '../results/qr_openai_phase2_custom.json'),
    ('CR',           '../results/cr_openai_phase2_custom.json'),
    ('QR+CR',        '../results/qr_cr_openai_phase2_custom.json'),
    ('Hybrid',       '../results/hybrid_openai_phase2_custom.json'),
    ('Agentic',      str(PHASE2_CUSTOM_PATH)),
]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)['results']
        configs_data[name] = {
            'acc'  : sum(r['is_correct']        for r in data) / len(data),
            'f'    : np.mean([r['faithfulness']      for r in data]),
            'cr'   : np.mean([r['context_recall']    for r in data]),
            'ar'   : np.mean([r['answer_relevancy']  for r in data]),
            'cp'   : np.mean([r['context_precision'] for r in data]),
        }
    except FileNotFoundError:
        pass

names    = list(configs_data.keys())
accs     = [configs_data[n]['acc']*100 for n in names]
f_scores = [configs_data[n]['f']  for n in names]
cr_scores= [configs_data[n]['cr'] for n in names]

# Color: Agentic emas, lainnya biru tua
colors = ['#1f4e79' if n != 'Agentic' else '#f39c12' for n in names]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Agentic OpenAI vs Static OpenAI Configurations\\n(500 sampel PubMedQA, gpt-4.1-mini)',
             fontsize=14, fontweight='bold', y=1.00)

# Panel 1: Label Accuracy
ax = axes[0, 0]
bars = ax.bar(names, accs, color=colors, edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Label Accuracy (%)')
ax.set_title('Label Accuracy')
ax.set_ylim(60, max(accs) + 4)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Panel 2: Iter distribution untuk Agentic
ax = axes[0, 1]
with open(PHASE1_PATH, 'r', encoding='utf-8') as f:
    p1_agentic = json.load(f)['results']
iters_agentic = [r['iterations'] for r in p1_agentic]
ax.hist(iters_agentic, bins=range(1, MAX_ITERATIONS + 2),
        color='#f39c12', edgecolor='white', linewidth=1.5, align='left', rwidth=0.8)
ax.set_xlabel('Number of iterations')
ax.set_ylabel('Count')
ax.set_title(f'Agentic Iterations Distribution (avg={np.mean(iters_agentic):.1f})')
ax.set_xticks(range(1, MAX_ITERATIONS + 1))
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Panel 3: Faithfulness comparison
ax = axes[1, 0]
bars = ax.bar(names, f_scores, color=colors, edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, f_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Faithfulness')
ax.set_title('Faithfulness Score')
ax.set_ylim(0.5, max(f_scores) + 0.05)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Panel 4: Tool usage Agentic
ax = axes[1, 1]
tool_counts = {}
for r in p1_agentic:
    for t in r['tool_log']:
        tool_counts[t['name']] = tool_counts.get(t['name'], 0) + 1
tnames  = list(tool_counts.keys())
tcounts = [tool_counts[n] / len(p1_agentic) for n in tnames]  # avg per query
bars = ax.bar(tnames, tcounts, color='#f39c12', edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, tcounts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Avg calls per query')
ax.set_title('Agent Tool Usage (avg per query)')
ax.set_xticks(range(len(tnames)))
ax.set_xticklabels(tnames, rotation=20, ha='right')
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
FIGURES_DIR = NOTEBOOK_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)
out_fig = FIGURES_DIR / 'I_agentic_vs_static.png'
plt.savefig(out_fig, dpi=200, bbox_inches='tight')
plt.show()
print(f'\\nSaved: {out_fig}')""")

# ───────────────────────── SAVE ─────────────────────────
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    },
    'language_info': {'name': 'python', 'version': '3.11'},
}

with open(OUT, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Saved: {OUT}')
print(f'Total cells: {len(cells)}')
