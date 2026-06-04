"""Generate 08-VectorDB-Comparison.ipynb programmatically."""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "08-VectorDB-Comparison.ipynb"
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ───────────────────────── INTRO ─────────────────────────
md("""# 08 — Vector DB Comparison: ChromaDB vs Qdrant vs Weaviate

Notebook ini membandingkan tiga vector database sebagai backend dense retriever pada pipeline RAG TA — semua memakai embedding cache yang sama (1.706 chunk PubMedQA, OpenAI `text-embedding-3-small`, 1536-dim).

## Tujuan

1. **Phase A — Retrieval-level**: Recall@k vs brute-force oracle, indexing time, query latency.
2. **Phase B — End-to-end**: Label accuracy (yes/no/maybe) untuk pipeline Hybrid (BM25 + Dense via RRF) pada 50 sampel PubMedQA, dengan Dense backend di-swap antar DB. Plus Jaccard similarity antar top-5 set.

## Hipotesis

- Recall@5 ketiga DB akan **hampir identik** karena algoritma HNSW dan parameter sama.
- Latency bisa berbeda — Qdrant biasanya tercepat (Rust core), Chroma dan Weaviate setara.
- End-to-end label accuracy akan **identik atau ±1 sampel** karena top-k sama.

## Strategi reuse

- **Embedding dokumen**: load dari `pubmedqa_chroma/` cache (no OpenAI cost).
- **Embedding query**: embed sekali untuk 100 query, reuse untuk ketiga DB.
- **Pipeline Hybrid**: reuse `tokenize_bm25`, RRF, dan struktur dari `02.1 Baseline - OpenAI.ipynb`.

## Fallback

Kalau Weaviate embedded gagal di Windows, notebook auto-skip Weaviate dan lanjut dengan 2 DB saja.
""")

# ───────────────────────── CELL 1: install ─────────────────────────
md("## Cell 1 — Install dependencies (skip kalau sudah ada)")
code("""# Install qdrant-client dan weaviate-client kalau belum ada.
# Run sekali, comment out setelahnya.
# !pip install qdrant-client>=1.7 weaviate-client>=4.4 tenacity --quiet""")

# ───────────────────────── CELL 2: imports & config ─────────────────────────
md("## Cell 2 — Imports & Konfigurasi")
code("""import os, sys, json, pickle, time, re, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
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
EMBED_MODEL = 'text-embedding-3-small'  # 1536 dim

# Phase A: retrieval benchmark
N_TEST_QUERIES = 100   # query untuk recall@k & latency
TOP_K          = 5
TOP_K_BROAD    = 20

# Phase B: end-to-end label accuracy
N_END_TO_END   = 50    # sampel untuk label accuracy (per DB)
TOP_K_BM25     = 50
TOP_K_DENSE    = 50
TOP_K_FINAL    = 5

TEMPERATURE = 0.0
SEED        = 42

NOTEBOOK_DIR    = Path('.')
BM25_INDEX_PATH = NOTEBOOK_DIR / 'pubmedqa_bm25.pkl'
CHROMA_DB_PATH  = NOTEBOOK_DIR / 'pubmedqa_chroma'
RESULTS_DIR     = Path('../results')
FIGURES_DIR     = NOTEBOOK_DIR / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

OUTPUT_JSON = RESULTS_DIR / 'vectordb_comparison.json'
OUTPUT_FIG  = FIGURES_DIR / 'H_vectordb_comparison.png'

print('Konfigurasi:')
print(f'  Embedder     : {EMBED_MODEL} (cached, no re-embed)')
print(f'  Phase A      : {N_TEST_QUERIES} query, top-{TOP_K} & top-{TOP_K_BROAD}')
print(f'  Phase B      : {N_END_TO_END} sampel end-to-end (Hybrid + GPT-4.1-mini)')
print(f'  Output       : {OUTPUT_JSON}')
print()
if 'YOUR_API_KEY' in OPENAI_API_KEY:
    print('  OPENAI_API_KEY belum diset. Set env var dulu sebelum lanjut.')
else:
    print(f'  OPENAI_API_KEY: {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}')""")

# ───────────────────────── CELL 3: load embeddings ─────────────────────────
md("""## Cell 3 — Load embeddings dari ChromaDB cache

Kita load 1.706 vektor yang sudah di-embed sebelumnya supaya tidak perlu re-embed (hemat ~$0.03 OpenAI cost dan ~30 detik).""")
code("""# Load embedding cache dari ChromaDB persistent
chroma_cache_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
cache_collection    = chroma_cache_client.get_collection(name='pubmedqa_docs')

# Ambil semua dokumen + embedding + metadata
print('Loading embeddings dari Chroma cache...')
cache_data = cache_collection.get(include=['embeddings', 'documents', 'metadatas'])

doc_ids       = [int(i) for i in cache_data['ids']]
doc_texts     = cache_data['documents']
doc_metas     = cache_data['metadatas']
doc_embeds_np = np.array(cache_data['embeddings'], dtype=np.float32)

# Sort by id supaya konsisten
order         = np.argsort(doc_ids)
doc_ids       = [doc_ids[i] for i in order]
doc_texts     = [doc_texts[i] for i in order]
doc_metas     = [doc_metas[i] for i in order]
doc_embeds_np = doc_embeds_np[order]

# Normalize untuk cosine similarity (penting untuk brute-force baseline)
norms = np.linalg.norm(doc_embeds_np, axis=1, keepdims=True)
doc_embeds_normed = doc_embeds_np / np.clip(norms, 1e-10, None)

N_DOCS, EMB_DIM = doc_embeds_np.shape
print(f'  Total dokumen     : {N_DOCS}')
print(f'  Dimensi embedding : {EMB_DIM}')
print(f'  Sample text       : {doc_texts[0][:80]}...')
print(f'  Sample meta       : {doc_metas[0]}')""")

# ───────────────────────── CELL 4: ChromaDB fresh in-memory ─────────────────────────
md("""## Cell 4 — Build ChromaDB fresh (in-memory, fair comparison)

Untuk fairness, kita rebuild Chroma in-memory (bukan pakai persistent yang sudah jadi). Tetap pakai embedding yang sama.""")
code("""# Build Chroma in-memory client untuk indexing time fairness
print('Building ChromaDB in-memory...')
chroma_mem_client = chromadb.EphemeralClient()  # in-memory, tidak persist
try:
    chroma_mem_client.delete_collection(name='vectordb_bench')
except Exception:
    pass

t0 = time.perf_counter()
chroma_mem = chroma_mem_client.create_collection(
    name='vectordb_bench',
    metadata={'hnsw:space': 'cosine'}
)
# Add in batches
BATCH = 200
for start in range(0, N_DOCS, BATCH):
    end = min(start + BATCH, N_DOCS)
    chroma_mem.add(
        ids=[str(i) for i in doc_ids[start:end]],
        embeddings=doc_embeds_np[start:end].tolist(),
        documents=doc_texts[start:end],
        metadatas=doc_metas[start:end],
    )
chroma_index_time = time.perf_counter() - t0
print(f'  ChromaDB indexed   : {chroma_mem.count()} dokumen')
print(f'  Indexing time      : {chroma_index_time:.2f} detik')

# Smoke test query
_q_emb = doc_embeds_np[0].tolist()
_r = chroma_mem.query(query_embeddings=[_q_emb], n_results=3, include=['distances'])
print(f'  Smoke query top-3  : ids={_r["ids"][0]} dist={[f"{d:.3f}" for d in _r["distances"][0]]}')""")

# ───────────────────────── CELL 5: Qdrant ─────────────────────────
md("""## Cell 5 — Build Qdrant in-memory

`QdrantClient(":memory:")` jalan tanpa Docker. Data hilang setelah restart kernel — itu OK karena kita rebuild dari embedding cache.""")
code("""QDRANT_AVAILABLE = False
qdrant_index_time = None
qdrant_client = None
qdrant_collection_name = 'vectordb_bench'

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    print('Building Qdrant in-memory...')
    qdrant_client = QdrantClient(':memory:')

    t0 = time.perf_counter()
    if qdrant_client.collection_exists(qdrant_collection_name):
        qdrant_client.delete_collection(qdrant_collection_name)
    qdrant_client.create_collection(
        collection_name=qdrant_collection_name,
        vectors_config=VectorParams(size=EMB_DIM, distance=Distance.COSINE),
    )
    # Upsert points
    points = [
        PointStruct(
            id=int(doc_ids[i]),
            vector=doc_embeds_np[i].tolist(),
            payload={'text': doc_texts[i], **(doc_metas[i] or {})},
        )
        for i in range(N_DOCS)
    ]
    # Upsert in batches biar tidak timeout
    QBATCH = 200
    for start in range(0, len(points), QBATCH):
        qdrant_client.upsert(
            collection_name=qdrant_collection_name,
            points=points[start:start + QBATCH],
        )
    qdrant_index_time = time.perf_counter() - t0

    count = qdrant_client.count(collection_name=qdrant_collection_name).count
    print(f'  Qdrant indexed     : {count} dokumen')
    print(f'  Indexing time      : {qdrant_index_time:.2f} detik')

    # Smoke test (query_points — search() deprecated in qdrant-client >=1.10)
    _resp = qdrant_client.query_points(
        collection_name=qdrant_collection_name,
        query=doc_embeds_np[0].tolist(),
        limit=3,
    )
    _hits = _resp.points
    print(f'  Smoke query top-3  : ids={[h.id for h in _hits]} '
          f'scores={[f"{h.score:.3f}" for h in _hits]}')
    QDRANT_AVAILABLE = True

except Exception as e:
    print(f'  Qdrant gagal: {type(e).__name__}: {e}')
    print('  Fallback: skip Qdrant di cell-cell selanjutnya.')""")

# ───────────────────────── CELL 6: Weaviate ─────────────────────────
md("""## Cell 6 — Build Weaviate embedded (Windows mungkin gagal)

`weaviate.connect_to_embedded()` auto-download binary. Kalau gagal di Windows, kita skip Weaviate.""")
code("""WEAVIATE_AVAILABLE = False
weaviate_index_time = None
weaviate_client = None
weaviate_collection = None
weaviate_collection_name = 'VectordbBench'  # Weaviate butuh CamelCase

try:
    import weaviate
    from weaviate.embedded import EmbeddedOptions
    from weaviate.classes.config import Configure, Property, DataType, VectorDistances
    import weaviate.classes as wvc

    print('Connecting to Weaviate embedded (auto-download binary)...')
    weaviate_client = weaviate.connect_to_embedded(
        version='1.24.10',
        port=8079,
        grpc_port=50050,
    )

    t0 = time.perf_counter()
    # Drop if exists
    if weaviate_client.collections.exists(weaviate_collection_name):
        weaviate_client.collections.delete(weaviate_collection_name)

    weaviate_collection = weaviate_client.collections.create(
        name=weaviate_collection_name,
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
        ),
        properties=[
            Property(name='text', data_type=DataType.TEXT),
            Property(name='pubid', data_type=DataType.TEXT),
            Property(name='section', data_type=DataType.TEXT),
        ],
    )
    # Batch insert
    with weaviate_collection.batch.fixed_size(batch_size=100) as batch:
        for i in range(N_DOCS):
            meta = doc_metas[i] or {}
            batch.add_object(
                properties={
                    'text': doc_texts[i],
                    'pubid': str(meta.get('pubid', '')),
                    'section': str(meta.get('section', '')),
                },
                vector=doc_embeds_np[i].tolist(),
                uuid=None,  # auto
            )
    weaviate_index_time = time.perf_counter() - t0

    count = weaviate_collection.aggregate.over_all(total_count=True).total_count
    print(f'  Weaviate indexed   : {count} dokumen')
    print(f'  Indexing time      : {weaviate_index_time:.2f} detik')

    # Smoke test
    _resp = weaviate_collection.query.near_vector(
        near_vector=doc_embeds_np[0].tolist(),
        limit=3,
        return_metadata=wvc.query.MetadataQuery(distance=True),
    )
    _ids_text = [(o.properties['text'][:30], o.metadata.distance) for o in _resp.objects]
    print(f'  Smoke query top-3  : {_ids_text}')
    WEAVIATE_AVAILABLE = True

except Exception as e:
    print(f'  Weaviate gagal: {type(e).__name__}: {str(e)[:200]}')
    print('  Fallback: skip Weaviate di cell-cell selanjutnya.')
    if weaviate_client is not None:
        try:
            weaviate_client.close()
        except Exception:
            pass
        weaviate_client = None""")

# ───────────────────────── CELL 7: query embeddings + brute force ─────────────────────────
md("""## Cell 7 — Embed query test + brute-force oracle

100 query test diambil dari PubMedQA. Kita embed sekali (cost ~$0.001), reuse untuk ketiga DB. Brute-force pakai numpy = ground truth top-k.""")
code("""# Load PubMedQA questions (untuk query test)
import csv
PUBMEDQA_CSV = Path('../data/pubmedqa_pqa_labeled.csv')
test_questions = []
test_pubids    = []
test_gts       = []
test_long_ans  = []

with open(PUBMEDQA_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        test_questions.append(row['question'])
        test_pubids.append(row['pubid'])
        test_gts.append(row['final_decision'])
        test_long_ans.append(row['long_answer'])
        if len(test_questions) >= max(N_TEST_QUERIES, N_END_TO_END):
            break

print(f'Loaded {len(test_questions)} questions from PubMedQA')

# OpenAI client untuk embed query
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def embed_query_batch(texts: List[str], batch_size: int = 50) -> np.ndarray:
    \"\"\"Embed list of queries via OpenAI, return np.ndarray (n, dim).\"\"\"
    out = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        for attempt in range(5):
            try:
                resp = openai_client.embeddings.create(model=EMBED_MODEL, input=batch)
                out.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = (attempt + 1) * 5
                print(f'  Retry embed batch ({type(e).__name__}): tunggu {wait}s')
                time.sleep(wait)
    return np.array(out, dtype=np.float32)

print(f'\\nEmbedding {N_TEST_QUERIES} query test...')
t0 = time.perf_counter()
query_embeds = embed_query_batch(test_questions[:N_TEST_QUERIES])
print(f'  Embedded in {time.perf_counter() - t0:.1f} detik')
print(f'  Shape: {query_embeds.shape}')

# Normalize
qnorms = np.linalg.norm(query_embeds, axis=1, keepdims=True)
query_embeds_normed = query_embeds / np.clip(qnorms, 1e-10, None)

# ============================================================
# Brute-force oracle: cosine similarity 100 x 1.706
# ============================================================
print('\\nComputing brute-force oracle (cosine similarity)...')
t0 = time.perf_counter()
sim_matrix = query_embeds_normed @ doc_embeds_normed.T  # (n_query, n_docs)
# Top-K untuk tiap query
oracle_top_k       = np.argsort(-sim_matrix, axis=1)[:, :TOP_K]
oracle_top_k_broad = np.argsort(-sim_matrix, axis=1)[:, :TOP_K_BROAD]
print(f'  Oracle computed in {time.perf_counter() - t0:.2f} detik')
print(f'  Oracle top-{TOP_K} shape: {oracle_top_k.shape}')""")

# ───────────────────────── CELL 8: query each DB ─────────────────────────
md("""## Cell 8 — Query tiap DB, ukur latency & top-K

Untuk tiap query, ambil top-5 dan top-20 dari tiap DB sambil ukur latency per query.""")
code("""# Helper: query Chroma → returns top-K ids
def query_chroma(qvec: np.ndarray, k: int) -> List[int]:
    r = chroma_mem.query(query_embeddings=[qvec.tolist()], n_results=k, include=['distances'])
    return [int(i) for i in r['ids'][0]]

def query_qdrant(qvec: np.ndarray, k: int) -> List[int]:
    resp = qdrant_client.query_points(
        collection_name=qdrant_collection_name,
        query=qvec.tolist(),
        limit=k,
    )
    return [int(p.id) for p in resp.points]

def query_weaviate(qvec: np.ndarray, k: int) -> List[int]:
    # Weaviate UUID auto-generated, tidak match doc_id. Kita match via text instead.
    # Untuk fairness, kita simpan mapping uuid → doc_id saat insert.
    # Karena belum, fallback: query lalu match text ke doc_texts.
    resp = weaviate_collection.query.near_vector(
        near_vector=qvec.tolist(),
        limit=k,
        return_properties=['text'],
    )
    text_to_id = {doc_texts[i]: doc_ids[i] for i in range(N_DOCS)}
    return [text_to_id.get(o.properties['text'], -1) for o in resp.objects]

# ============================================================
# Run queries, ukur latency
# ============================================================
def benchmark_db(name: str, query_fn, ks: List[int]) -> Dict:
    \"\"\"Untuk tiap query, ukur latency saat ambil top-max(ks). Lalu derive top-k untuk k lain dari hasilnya.\"\"\"
    max_k = max(ks)
    latencies = []
    top_k_results = {k: [] for k in ks}

    for i in range(N_TEST_QUERIES):
        qvec = query_embeds[i]
        t0 = time.perf_counter()
        ids = query_fn(qvec, max_k)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms
        # Slice ke ks lain
        for k in ks:
            top_k_results[k].append(ids[:k])

    return {
        'latencies_ms': latencies,
        'mean_ms'     : float(np.mean(latencies)),
        'p50_ms'      : float(np.percentile(latencies, 50)),
        'p95_ms'      : float(np.percentile(latencies, 95)),
        'top_k'       : {k: top_k_results[k] for k in ks},
    }

ks = [TOP_K, TOP_K_BROAD]
db_results = {}

print(f'Benchmarking ChromaDB ({N_TEST_QUERIES} queries, top-{max(ks)})...')
db_results['chromadb'] = benchmark_db('chromadb', query_chroma, ks)
print(f'  mean={db_results["chromadb"]["mean_ms"]:.2f}ms  '
      f'p95={db_results["chromadb"]["p95_ms"]:.2f}ms')

if QDRANT_AVAILABLE:
    print(f'\\nBenchmarking Qdrant ({N_TEST_QUERIES} queries)...')
    db_results['qdrant'] = benchmark_db('qdrant', query_qdrant, ks)
    print(f'  mean={db_results["qdrant"]["mean_ms"]:.2f}ms  '
          f'p95={db_results["qdrant"]["p95_ms"]:.2f}ms')

if WEAVIATE_AVAILABLE:
    print(f'\\nBenchmarking Weaviate ({N_TEST_QUERIES} queries)...')
    db_results['weaviate'] = benchmark_db('weaviate', query_weaviate, ks)
    print(f'  mean={db_results["weaviate"]["mean_ms"]:.2f}ms  '
          f'p95={db_results["weaviate"]["p95_ms"]:.2f}ms')""")

# ───────────────────────── CELL 9: recall + jaccard vs oracle ─────────────────────────
md("""## Cell 9 — Hitung Recall@k vs Brute-force Oracle

Recall@k = |DB_top_k ∩ oracle_top_k| / k. Sanity check: oracle vs oracle harus 1.0.""")
code("""def recall_at_k(predicted: List[List[int]], oracle: np.ndarray, k: int) -> float:
    \"\"\"Mean recall@k across queries.\"\"\"
    recalls = []
    for i in range(len(predicted)):
        pred_set   = set(predicted[i][:k])
        oracle_set = set(oracle[i, :k].tolist())
        if oracle_set:
            recalls.append(len(pred_set & oracle_set) / len(oracle_set))
    return float(np.mean(recalls))

def jaccard_at_k(predicted: List[List[int]], oracle: np.ndarray, k: int) -> float:
    \"\"\"Mean Jaccard similarity @ k.\"\"\"
    jaccs = []
    for i in range(len(predicted)):
        a = set(predicted[i][:k])
        b = set(oracle[i, :k].tolist())
        union = a | b
        if union:
            jaccs.append(len(a & b) / len(union))
    return float(np.mean(jaccs))

# Sanity: oracle vs oracle
oracle_self_recall = recall_at_k(oracle_top_k.tolist(), oracle_top_k, TOP_K)
print(f'Sanity check: oracle vs oracle recall@{TOP_K} = {oracle_self_recall:.4f} (expect 1.0)')

# Compute metrics per DB
metric_summary = {}
for db_name, res in db_results.items():
    metric_summary[db_name] = {
        f'recall@{TOP_K}'        : recall_at_k(res['top_k'][TOP_K], oracle_top_k, TOP_K),
        f'recall@{TOP_K_BROAD}'  : recall_at_k(res['top_k'][TOP_K_BROAD], oracle_top_k_broad, TOP_K_BROAD),
        f'jaccard@{TOP_K}'       : jaccard_at_k(res['top_k'][TOP_K], oracle_top_k, TOP_K),
        f'jaccard@{TOP_K_BROAD}' : jaccard_at_k(res['top_k'][TOP_K_BROAD], oracle_top_k_broad, TOP_K_BROAD),
        'mean_latency_ms'        : res['mean_ms'],
        'p95_latency_ms'         : res['p95_ms'],
    }

# Print as table
import pandas as pd
df_metrics = pd.DataFrame(metric_summary).T
df_metrics_display = df_metrics.copy()
for col in df_metrics_display.columns:
    if 'recall' in col or 'jaccard' in col:
        df_metrics_display[col] = df_metrics_display[col].apply(lambda x: f'{x:.4f}')
    else:
        df_metrics_display[col] = df_metrics_display[col].apply(lambda x: f'{x:.2f} ms')
print('\\n=== Phase A: Retrieval-level metrics ===\\n')
print(df_metrics_display.to_string())""")

# ───────────────────────── CELL 10: Phase B end-to-end ─────────────────────────
md("""## Cell 10 — Phase B: End-to-end Label Accuracy

Run pipeline Hybrid penuh (BM25 + Dense via RRF → top-5 → GPT-4.1-mini → yes/no/maybe) untuk **50 sampel** dengan Dense backend di-swap antar DB.

Cost: ~$0.30 OpenAI total (50 sampel × 3 DB × ~$0.002/call). Estimasi waktu ~5 menit.""")
code("""# ============================================================
# Load BM25 index (reuse dari notebook lain)
# ============================================================
# Penting: pickle butuh class 'Document' ter-define di module __main__
# karena pubmedqa_bm25.pkl di-save dari notebook lain dengan dataclass ini.
@dataclass
class Document:
    text         : str
    pubid        : str
    question     : str
    section_label: str
    answer       : str
    decision     : str

print('Loading BM25 index...')
with open(BM25_INDEX_PATH, 'rb') as f:
    bm25_data = pickle.load(f)
bm25_index   = bm25_data['bm25']
bm25_docs    = bm25_data['documents']  # List[Document]
print(f'  BM25 index: {len(bm25_docs)} dokumen')

def tokenize_bm25(text: str) -> List[str]:
    return re.sub(r'[^a-zA-Z0-9\\s]', ' ', text.lower()).split()

def bm25_top_k(query: str, k: int) -> List[Tuple[int, float]]:
    tokens = tokenize_bm25(query)
    scores = bm25_index.get_scores(tokens)
    top    = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top]

def rrf(rank_lists: List[List[int]], k: int = 60) -> List[Tuple[int, float]]:
    s = {}
    for rl in rank_lists:
        for rank, did in enumerate(rl):
            s[did] = s.get(did, 0.0) + 1.0 / (k + rank + 1)
    return sorted(s.items(), key=lambda x: -x[1])

GENERATION_PROMPT = (
    'You are a medical research assistant. '
    'Answer a biomedical yes/no/maybe question based solely on the provided scientific abstracts.\\n\\n'
    'Context from medical literature:\\n{context}\\n\\n'
    'Question: {question}\\n\\n'
    'Instructions:\\n'
    '- Carefully read the context and assess whether it supports or refutes the question.\\n'
    '- Provide a brief explanation (2-3 sentences) using ONLY the information above.\\n'
    '- End your response with EXACTLY ONE of these words on its own line: yes, no, or maybe.\\n'
    '  - yes   : the evidence supports the hypothesis\\n'
    '  - no    : the evidence refutes or does not support the hypothesis\\n'
    '  - maybe : ONLY if the evidence is directly contradictory, or if the context contains no relevant information at all\\n'
    '- IMPORTANT: If the evidence leans in one direction, even partially, choose yes or no.\\n\\n'
    'Answer:'
)

def llm_generate(prompt: str, max_tokens: int = 300) -> str:
    for attempt in range(5):
        try:
            r = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
                seed=SEED,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep((attempt + 1) * 5)
    raise RuntimeError('LLM gagal')

def extract_label(answer: str) -> str:
    lines = [l.strip().lower() for l in answer.split('\\n') if l.strip()]
    for line in reversed(lines[-3:]):
        word = re.sub(r'[^a-z]', '', line)
        if word in ('yes', 'no', 'maybe'):
            return word
    for label in ('yes', 'no', 'maybe'):
        if re.search(r'\\b' + label + r'\\b', answer.lower()):
            return label
    return 'maybe'

# ============================================================
# Hybrid retrieval dengan dense backend swappable
# ============================================================
def hybrid_topk(query: str, qvec: np.ndarray, dense_query_fn) -> List[int]:
    bm25_pairs  = bm25_top_k(query, TOP_K_BM25)
    dense_ids   = dense_query_fn(qvec, TOP_K_DENSE)
    fused = rrf([
        [d for d, _ in bm25_pairs],
        dense_ids,
    ])
    return [d for d, _ in fused[:TOP_K_FINAL]]

# ============================================================
# Run end-to-end untuk N_END_TO_END sampel x setiap DB
# ============================================================
end_to_end = {}  # {db_name: {'preds': [...], 'gts': [...], 'top5_sets': [set, ...]}}
db_query_fns = {}
if 'chromadb' in db_results:
    db_query_fns['chromadb'] = query_chroma
if QDRANT_AVAILABLE:
    db_query_fns['qdrant'] = query_qdrant
if WEAVIATE_AVAILABLE:
    db_query_fns['weaviate'] = query_weaviate

# Embed query untuk N_END_TO_END kalau N_END_TO_END > N_TEST_QUERIES
e2e_questions = test_questions[:N_END_TO_END]
e2e_gts       = test_gts[:N_END_TO_END]
if N_END_TO_END <= N_TEST_QUERIES:
    e2e_qvecs = query_embeds[:N_END_TO_END]
else:
    e2e_qvecs = embed_query_batch(e2e_questions)

for db_name, qfn in db_query_fns.items():
    print(f'\\n=== Running end-to-end: {db_name} ({N_END_TO_END} sampel) ===')
    preds, top5_sets = [], []
    t0 = time.perf_counter()
    for i in range(N_END_TO_END):
        q    = e2e_questions[i]
        qvec = e2e_qvecs[i]
        top5_ids = hybrid_topk(q, qvec, qfn)
        top5_sets.append(set(top5_ids))
        # Build context from BM25 docs (since they have full Document object)
        ctx_text = '\\n\\n'.join(
            f'[{j+1}] ({bm25_docs[did].section_label}): {bm25_docs[did].text}'
            for j, did in enumerate(top5_ids)
        )
        ans  = llm_generate(GENERATION_PROMPT.format(context=ctx_text, question=q))
        pred = extract_label(ans)
        preds.append(pred)
        if (i + 1) % 10 == 0:
            acc = sum(p == g for p, g in zip(preds, e2e_gts[:i+1])) / (i + 1)
            eta = (time.perf_counter() - t0) / (i + 1) * (N_END_TO_END - i - 1)
            print(f'  [{i+1}/{N_END_TO_END}] acc={acc:.1%} pred={pred} gt={e2e_gts[i]} ETA={eta:.0f}s')

    accuracy = sum(p == g for p, g in zip(preds, e2e_gts)) / N_END_TO_END
    end_to_end[db_name] = {
        'preds'     : preds,
        'gts'       : e2e_gts,
        'top5_sets' : top5_sets,
        'accuracy'  : accuracy,
    }
    print(f'  -> Final accuracy: {accuracy:.1%}')""")

# ───────────────────────── CELL 11: Jaccard between DBs ─────────────────────────
md("""## Cell 11 — Jaccard Similarity antar DB (top-5 sets)

Apakah top-5 yang dikembalikan tiap DB identik? Kalau Jaccard ≈ 1.0, end-to-end accuracy juga akan identik.""")
code("""# Pairwise Jaccard antar DB
db_names_e2e = list(end_to_end.keys())
jaccard_pairs = {}
for i, a in enumerate(db_names_e2e):
    for b in db_names_e2e[i+1:]:
        jaccs = []
        for set_a, set_b in zip(end_to_end[a]['top5_sets'], end_to_end[b]['top5_sets']):
            union = set_a | set_b
            if union:
                jaccs.append(len(set_a & set_b) / len(union))
        jaccard_pairs[f'{a}_vs_{b}'] = float(np.mean(jaccs))

print('Pairwise Jaccard (top-5 sets, mean over 50 queries):')
for pair, j in jaccard_pairs.items():
    print(f'  {pair:<28}: {j:.4f}')

# Update metric_summary dengan accuracy + tambahan
for db_name in db_names_e2e:
    metric_summary[db_name]['e2e_label_accuracy'] = end_to_end[db_name]['accuracy']

# Save raw results
output_payload = {
    'config': {
        'embed_model'    : EMBED_MODEL,
        'llm_model'      : LLM_MODEL,
        'n_docs'         : N_DOCS,
        'n_test_queries' : N_TEST_QUERIES,
        'n_end_to_end'   : N_END_TO_END,
        'top_k'          : TOP_K,
        'top_k_broad'    : TOP_K_BROAD,
        'timestamp'      : datetime.now().isoformat(),
    },
    'indexing_time_sec': {
        'chromadb' : float(chroma_index_time),
        'qdrant'   : float(qdrant_index_time)   if qdrant_index_time   is not None else None,
        'weaviate' : float(weaviate_index_time) if weaviate_index_time is not None else None,
    },
    'phase_a_metrics' : metric_summary,
    'phase_b_e2e'     : {
        db: {'accuracy': float(end_to_end[db]['accuracy']),
             'preds'   : end_to_end[db]['preds'],
             'gts'     : end_to_end[db]['gts']}
        for db in db_names_e2e
    },
    'jaccard_pairwise': jaccard_pairs,
    'oracle_self_recall_check': float(oracle_self_recall),
}

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output_payload, f, indent=2, ensure_ascii=False)
print(f'\\nSaved: {OUTPUT_JSON}')""")

# ───────────────────────── CELL 12: visualization ─────────────────────────
md("""## Cell 12 — Visualisasi & Ringkasan""")
code("""import matplotlib.pyplot as plt

# Susun data untuk plot
db_names = list(metric_summary.keys())
db_palette = {'chromadb': '#1f4e79', 'qdrant': '#c0392b', 'weaviate': '#117a65'}
colors = [db_palette.get(d, '#888') for d in db_names]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Vector DB Comparison — ChromaDB vs Qdrant vs Weaviate\\n(1.706 PubMedQA chunks, OpenAI text-embedding-3-small)',
             fontsize=14, fontweight='bold', y=1.00)

# Panel 1: Indexing time
ax = axes[0, 0]
idx_times = [output_payload['indexing_time_sec'].get(d) or 0 for d in db_names]
bars = ax.bar(db_names, idx_times, color=colors, edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, idx_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(idx_times) * 0.02,
            f'{v:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Indexing time (detik)')
ax.set_title('Indexing Time (1.706 dokumen)')
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel 2: Latency (boxplot)
ax = axes[0, 1]
latency_data = [db_results[d]['latencies_ms'] for d in db_names]
bp = ax.boxplot(latency_data, labels=db_names, patch_artist=True,
                boxprops=dict(linewidth=1.2), medianprops=dict(color='black', linewidth=1.5))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('Query latency (ms)')
ax.set_title(f'Query Latency ({N_TEST_QUERIES} queries, top-{TOP_K_BROAD})')
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel 3: Recall@5
ax = axes[1, 0]
recalls5  = [metric_summary[d][f'recall@{TOP_K}']       for d in db_names]
recalls20 = [metric_summary[d][f'recall@{TOP_K_BROAD}'] for d in db_names]
x = np.arange(len(db_names))
w = 0.36
b1 = ax.bar(x - w/2, recalls5,  w, label=f'Recall@{TOP_K}',
            color=colors, edgecolor='white', linewidth=1.5)
b2 = ax.bar(x + w/2, recalls20, w, label=f'Recall@{TOP_K_BROAD}',
            color=colors, edgecolor='white', linewidth=1.5, alpha=0.55)
for bars, vals in [(b1, recalls5), (b2, recalls20)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(db_names)
ax.set_ylabel('Recall vs Brute-force Oracle')
ax.set_title('Retrieval Recall@k (vs cosine brute-force oracle)')
ax.set_ylim(0, 1.1)
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel 4: End-to-end label accuracy
ax = axes[1, 1]
e2e_dbs   = [d for d in db_names if 'e2e_label_accuracy' in metric_summary[d]]
e2e_accs  = [metric_summary[d]['e2e_label_accuracy'] * 100 for d in e2e_dbs]
e2e_cols  = [db_palette.get(d, '#888') for d in e2e_dbs]
bars = ax.bar(e2e_dbs, e2e_accs, color=e2e_cols, edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, e2e_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Label Accuracy (%)')
ax.set_title(f'End-to-end Label Accuracy ({N_END_TO_END} sampel, Hybrid + GPT-4.1-mini)')
ax.set_ylim(0, max(e2e_accs) * 1.25 if e2e_accs else 100)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=200, bbox_inches='tight')
plt.show()
print(f'\\nSaved figure: {OUTPUT_FIG}')

# ============================================================
# Tabel ringkasan markdown
# ============================================================
print('\\n' + '=' * 80)
print('RINGKASAN — Vector DB Comparison')
print('=' * 80)
print(f'\\n{"DB":<10} {"Index(s)":>10} {"Lat μ(ms)":>10} {"Lat p95(ms)":>12} '
      f'{"R@5":>8} {"R@20":>8} {"E2E Acc":>10}')
print('-' * 80)
for d in db_names:
    m = metric_summary[d]
    idx_t = output_payload['indexing_time_sec'].get(d) or 0
    e2e = m.get('e2e_label_accuracy')
    e2e_str = f'{e2e*100:.1f}%' if e2e is not None else '—'
    print(f'{d:<10} {idx_t:>10.2f} {m["mean_latency_ms"]:>10.2f} {m["p95_latency_ms"]:>12.2f} '
          f'{m[f"recall@{TOP_K}"]:>8.4f} {m[f"recall@{TOP_K_BROAD}"]:>8.4f} {e2e_str:>10}')

print('\\nJaccard pairwise (top-5):')
for pair, j in jaccard_pairs.items():
    print(f'  {pair:<28}: {j:.4f}')

print('\\n=== Insight ===')
recall_key = f'recall@{TOP_K}'
recall_str = ', '.join(f'{d}={metric_summary[d][recall_key]:.3f}' for d in db_names)
print(f'1. Recall@{TOP_K} ketiga DB: {recall_str}')

fastest_db = min(db_names, key=lambda d: metric_summary[d]['mean_latency_ms'])
print(f'2. Latency tercepat: {fastest_db}')

if e2e_accs:
    spread = max(e2e_accs) - min(e2e_accs)
    print(f'3. End-to-end accuracy spread: {spread:.1f}pp '
          f'(kalau < 2pp, swap DB tidak signifikan)')
else:
    print('3. End-to-end accuracy: tidak ada data (Phase B di-skip)')""")

# ───────────────────────── CELL 13: Phase B Extended 500 sampel ─────────────────────────
md("""## Cell 13 — Phase B Extended: 500 Sampel dengan Resume

Run pipeline Hybrid penuh untuk **500 sampel** (apple-to-apple dengan baseline `02.1 Hybrid OpenAI`). Resume-able: kalau interrupt di tengah, re-run cell akan lanjut dari sampel terakhir.

**Cost:** ~$3 OpenAI total (500 × 3 DB). **Waktu:** 50–75 menit.

**Output:** `results/vectordb_e2e_full/{db_name}_500.json` per DB + `results/vectordb_comparison_500.json` summary.""")
code("""# ============================================================
# Phase B Extended — 500 sampel dengan resume
# ============================================================
N_E2E_FULL = 500

E2E_FULL_DIR = RESULTS_DIR / 'vectordb_e2e_full'
E2E_FULL_DIR.mkdir(exist_ok=True)

# Reload test data dari CSV dengan limit 500 (bukan dari test_questions yang dibatasi 100)
PUBMEDQA_CSV_FULL = Path('../data/pubmedqa_pqa_labeled.csv')
needed_questions, needed_gts, needed_pubids = [], [], []
with open(PUBMEDQA_CSV_FULL, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        needed_questions.append(row['question'])
        needed_gts.append(row['final_decision'])
        needed_pubids.append(row['pubid'])
        if len(needed_questions) >= N_E2E_FULL:
            break

print(f'Loaded {len(needed_questions)} sampel dari CSV (target {N_E2E_FULL})')
assert len(needed_questions) >= N_E2E_FULL, f'CSV cuma punya {len(needed_questions)} sampel'

# Embed semua 500 query (sekali, ~$0.005)
if 'full_query_embeds' not in globals() or len(full_query_embeds) < N_E2E_FULL:
    print(f'Embedding {N_E2E_FULL} query test...')
    t0 = time.perf_counter()
    full_query_embeds = embed_query_batch(needed_questions)
    print(f'  Done in {time.perf_counter()-t0:.1f}s, shape={full_query_embeds.shape}')
else:
    print(f'Reuse {len(full_query_embeds)} embedded queries')

end_to_end_full = {}

for db_name, qfn in db_query_fns.items():
    out_path = E2E_FULL_DIR / f'{db_name}_500.json'

    # Resume kalau ada
    if out_path.exists():
        with open(out_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        results_list = saved['results']
        start_from   = len(results_list)
        print(f'\\n[{db_name}] Resume dari {start_from}/{N_E2E_FULL}')
    else:
        results_list, start_from = [], 0
        print(f'\\n[{db_name}] Mulai 500 sampel dari awal')

    if start_from >= N_E2E_FULL:
        print(f'  Sudah selesai. Skip.')
    else:
        t0 = time.perf_counter()
        for i in range(start_from, N_E2E_FULL):
            q    = needed_questions[i]
            qvec = full_query_embeds[i]
            gt   = needed_gts[i]

            try:
                top5_ids = hybrid_topk(q, qvec, qfn)
                ctx_text = '\\n\\n'.join(
                    f'[{j+1}] ({bm25_docs[did].section_label}): {bm25_docs[did].text}'
                    for j, did in enumerate(top5_ids)
                )
                ans  = llm_generate(GENERATION_PROMPT.format(context=ctx_text, question=q))
                pred = extract_label(ans)
            except Exception as e:
                print(f'  [{i}] ERROR: {type(e).__name__}: {str(e)[:120]}')
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump({'db': db_name, 'completed': len(results_list),
                               'results': results_list}, f, indent=2, ensure_ascii=False)
                raise

            results_list.append({
                'idx'        : i,
                'pubid'      : needed_pubids[i],
                'question'   : q,
                'gt'         : gt,
                'pred'       : pred,
                'is_correct' : pred == gt,
                'top5_ids'   : top5_ids,
            })

            # Save tiap 10 sampel
            if (i + 1) % 10 == 0 or i == N_E2E_FULL - 1:
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump({'db': db_name, 'completed': len(results_list),
                               'results': results_list}, f, indent=2, ensure_ascii=False)
                done = i + 1
                acc  = sum(r['is_correct'] for r in results_list) / done
                eta  = (time.perf_counter()-t0)/(done-start_from)*(N_E2E_FULL-done)/60
                print(f'  [{done:3d}/{N_E2E_FULL}] acc={acc:.1%} pred={pred} gt={gt} ETA={eta:.1f}m')

    # Hitung final acc + per-label + jaccard sets
    accuracy = sum(r['is_correct'] for r in results_list) / len(results_list)
    end_to_end_full[db_name] = {
        'accuracy'  : accuracy,
        'preds'     : [r['pred']     for r in results_list],
        'gts'       : [r['gt']       for r in results_list],
        'top5_sets' : [set(r['top5_ids']) for r in results_list],
        'per_label' : {
            lbl: sum(1 for r in results_list if r['gt'] == lbl and r['is_correct']) /
                 max(1, sum(1 for r in results_list if r['gt'] == lbl))
            for lbl in ['yes', 'no', 'maybe']
        },
    }
    print(f'  >> {db_name} final accuracy: {accuracy:.1%}')

# ============================================================
# Pairwise Jaccard untuk 500 sampel
# ============================================================
print('\\n' + '='*60)
print(f'PHASE B EXTENDED — 500 sampel')
print('='*60)

db_names_full = list(end_to_end_full.keys())
print(f'\\n{"DB":<12} {"Acc":>8} {"yes":>8} {"no":>8} {"maybe":>8}')
print('-'*50)
for d in db_names_full:
    e = end_to_end_full[d]
    print(f'{d:<12} {e["accuracy"]*100:>7.1f}% '
          f'{e["per_label"]["yes"]*100:>7.1f}% '
          f'{e["per_label"]["no"]*100:>7.1f}% '
          f'{e["per_label"]["maybe"]*100:>7.1f}%')

print('\\nPairwise Jaccard top-5 (500 query):')
jaccard_500 = {}
for i, a in enumerate(db_names_full):
    for b in db_names_full[i+1:]:
        jaccs = [len(sa & sb) / max(1, len(sa | sb))
                 for sa, sb in zip(end_to_end_full[a]['top5_sets'],
                                    end_to_end_full[b]['top5_sets'])]
        j = float(np.mean(jaccs))
        jaccard_500[f'{a}_vs_{b}'] = j
        print(f'  {a:<10} vs {b:<10}: {j:.4f}')

accs_500 = [end_to_end_full[d]['accuracy']*100 for d in db_names_full]
if len(accs_500) >= 2:
    print(f'\\nAccuracy spread: {max(accs_500)-min(accs_500):.2f} pp')
    print('(< 2pp = swap DB tidak signifikan untuk skala TA ini)')

# Save merged summary
with open(RESULTS_DIR / 'vectordb_comparison_500.json', 'w', encoding='utf-8') as f:
    json.dump({
        'n_samples'       : N_E2E_FULL,
        'timestamp'       : datetime.now().isoformat(),
        'accuracy_per_db' : {d: end_to_end_full[d]['accuracy'] for d in db_names_full},
        'per_label_per_db': {d: end_to_end_full[d]['per_label'] for d in db_names_full},
        'jaccard_pairwise': jaccard_500,
    }, f, indent=2, ensure_ascii=False)
print(f'\\nSaved: {RESULTS_DIR / "vectordb_comparison_500.json"}')""")

# ───────────────────────── CELL 14: cleanup ─────────────────────────
md("""## Cell 14 — Cleanup""")
code("""# Tutup koneksi Weaviate kalau aktif
if WEAVIATE_AVAILABLE and weaviate_client is not None:
    try:
        weaviate_client.close()
        print('Weaviate client closed')
    except Exception:
        pass
print('Done.')""")

# ───────────────────────── SAVE ─────────────────────────
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    },
    'language_info': {
        'name': 'python',
        'version': '3.11',
    },
}

with open(OUT, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Saved: {OUT}')
print(f'Total cells: {len(cells)}')
