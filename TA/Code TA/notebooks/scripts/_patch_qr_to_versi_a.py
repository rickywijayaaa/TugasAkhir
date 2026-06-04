"""
Patch Kel.3 notebooks (30_hybrid_dengan_ekspansi/gpt4mini/qr.ipynb dan qr_cr.ipynb)
dari Versi B (multi-query, paper Ma dkk. 2023) ke Versi A (single-query).

Yang diubah:
1. QUERY_REWRITE_PROMPT → Versi A
2. parse_rewriter_output → dihapus (tidak perlu)
3. rewrite_query → return str (bukan list)
4. retrieve_hybrid → loop multi-query → single-query
5. Smoke test cell → sesuaikan dengan API baru
6. Hapus konstanta MAX_REWRITE_QUERIES

Run: py _patch_qr_to_versi_a.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
    'notebooks_root_placeholder',  # actual targets at runtime
]

# Versi A QR prompt + rewrite_query function (copied dari Kel.2 untuk konsistensi)
QR_VERSI_A_CELL = '''# ============================================================
# Query Rewriting (Versi A — single-query rewriting)
# ============================================================
# Pendekatan: kueri pengguna ditulis ulang menjadi SATU kueri yang lebih
# spesifik dan kaya istilah biomedis. Berbeda dengan pendekatan multi-query
# decomposition, versi ini hanya menghasilkan satu kueri agar pipeline tetap
# sederhana dan reproducibility tinggi.

QUERY_REWRITE_PROMPT = (
    'You are a query rewriting assistant for a biomedical question-answering system.\\n'
    'Rewrite the following medical question to improve retrieval from a PubMed research database.\\n\\n'
    'Rules:\\n'
    '1. Be more specific and add relevant medical/scientific terminology.\\n'
    '2. Expand abbreviations (e.g. "MI" -> "myocardial infarction").\\n'
    '3. Preserve the original yes/no/maybe answerable intent.\\n'
    '4. Output ONLY the rewritten question, no explanations.\\n\\n'
    'Original question: {query}\\n\\n'
    'Rewritten question:'
)


def rewrite_query(query: str) -> str:
    """Reformulasi query untuk meningkatkan kualitas retrieval (single-query)."""
    prompt = QUERY_REWRITE_PROMPT.format(query=query)
    try:
        rewritten = openai_generate(prompt, max_tokens=150, temperature=0.3)
        rewritten = rewritten.strip().replace('\\n', ' ')
        return rewritten if len(rewritten) >= 10 else query
    except Exception as e:
        print(f'  [QR Error] {e} -- pakai query asli')
        return query


# Smoke test
print('Contoh Query Rewriting (single-query):')
for q in ['Does aspirin reduce the risk of MI?', 'Can exercise prevent T2DM?']:
    rw = rewrite_query(q)
    print(f'\\nAsli    : {q}')
    print(f'Rewrite : {rw}')
'''


def patch_qr_notebook(nb_path: Path):
    """Patch notebook: replace QR cell + retrieve_hybrid usage."""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    qr_cell_idx = None
    retrieve_cell_idx = None
    smoke_test_idx = None

    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        if 'QUERY_REWRITE_PROMPT' in src and qr_cell_idx is None:
            qr_cell_idx = i
        if 'def retrieve_hybrid' in src and 'rewritten_queries' in src:
            retrieve_cell_idx = i
        if 'rewritten_queries' in src and 'Smoke test' in src and retrieve_cell_idx is None:
            smoke_test_idx = i

    if qr_cell_idx is None:
        print(f'  WARN: QR cell not found in {nb_path.name}')
        return False

    # Step 1: Replace QR cell content
    nb['cells'][qr_cell_idx]['source'] = QR_VERSI_A_CELL.splitlines(keepends=True)

    # Step 2: Modify retrieve_hybrid (if found) — change multi-query loop to single
    if retrieve_cell_idx is not None:
        old_src = ''.join(nb['cells'][retrieve_cell_idx]['source'])

        # New retrieve_hybrid that uses single rewritten query
        new_retrieve = '''def retrieve_dense(query, k=TOP_K_DENSE):
    qvec = openai_embed([query])[0]
    res = chroma_collection.query(query_embeddings=[qvec], n_results=k, include=["distances"])
    doc_ids = [int(i) for i in res["ids"][0]]
    distances = res["distances"][0]
    scores = [1.0 - d for d in distances]
    return list(zip(doc_ids, scores))


def retrieve_bm25_raw(query, k=TOP_K_BM25):
    tokens = tokenize_bm25(query)
    scores = bm25_index.get_scores(tokens)
    top_k = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_k]


def reciprocal_rank_fusion(rank_lists, k=60):
    scores = {}
    for rl in rank_lists:
        for rank, doc_id in enumerate(rl):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def retrieve_hybrid(query, k_final=TOP_K_RETRIEVAL):
    """Single-query QR + Hybrid retrieval.
    Query asli ditulis ulang menjadi SATU kueri baru, lalu dipakai untuk
    BM25 + Dense, hasilnya digabung via RRF."""
    rewritten = rewrite_query(query)

    bm25_results = retrieve_bm25_raw(rewritten, k=TOP_K_BM25)
    dense_results = retrieve_dense(rewritten, k=TOP_K_DENSE)

    bm25_score_lookup = {d: s for d, s in bm25_results}
    dense_score_lookup = {d: s for d, s in dense_results}

    fused = reciprocal_rank_fusion([
        [d for d, _ in bm25_results],
        [d for d, _ in dense_results],
    ])

    out = []
    for doc_id, rrf in fused[:k_final]:
        out.append(RetrievalResult(
            document=documents[doc_id], score=rrf, doc_id=doc_id,
            bm25_score=bm25_score_lookup.get(doc_id, 0.0),
            dense_score=dense_score_lookup.get(doc_id, 0.0), rrf_score=rrf,
        ))
    return out, rewritten


# Smoke test
test_q = pubmedqa_data[0]["question"]
test_r, test_rw = retrieve_hybrid(test_q)
print(f"Query   : {test_q}")
print(f"Rewrite : {test_rw}")
print(f"\\nTop-{TOP_K_RETRIEVAL} chunks:")
for i, r in enumerate(test_r, 1):
    print(f"  [{i}] RRF={r.rrf_score:.4f} | BM25={r.bm25_score:6.2f} | Dense={r.dense_score:.3f} | "
          f"({r.document.section_label[:25]}) pubid={r.document.pubid}")
'''
        nb['cells'][retrieve_cell_idx]['source'] = new_retrieve.splitlines(keepends=True)

    # Step 3: Patch the eval loop cell to use single rewritten (not list)
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        if 'rewritten_queries' in src and i != qr_cell_idx and i != retrieve_cell_idx:
            # Replace 'rewritten_queries' references with 'rewritten' (single string)
            new_src = src.replace('rewritten_queries', 'rewritten')
            # Also fix any list-iteration patterns that no longer apply
            nb['cells'][i]['source'] = new_src.splitlines(keepends=True)

    # Step 4: Remove MAX_REWRITE_QUERIES constant (not needed)
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        if 'MAX_REWRITE_QUERIES' in src:
            new_src = '\n'.join(
                line for line in src.split('\n')
                if 'MAX_REWRITE_QUERIES' not in line
            )
            nb['cells'][i]['source'] = new_src.splitlines(keepends=True)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return True


# Target notebooks
targets = [
    ROOT / '30_hybrid_dengan_ekspansi' / 'gpt4mini' / 'qr.ipynb',
    ROOT / '30_hybrid_dengan_ekspansi' / 'gpt4mini' / 'qr_cr.ipynb',
]

for nb_path in targets:
    if not nb_path.exists():
        print(f'MISSING: {nb_path}')
        continue
    ok = patch_qr_notebook(nb_path)
    status = 'PATCHED' if ok else 'WARN'
    print(f'{status}: {nb_path.relative_to(ROOT)}')

print('\nSelesai. Verifikasi: buka kedua notebook & cek bahwa QR sekarang single-query.')
