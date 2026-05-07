"""
Clean generator for 6 new Hybrid variant notebooks.

All notebooks share:
- BM25 index: pubmedqa_bm25.pkl
- ChromaDB: pubmedqa_chroma_openai/ or pubmedqa_chroma_llama/ (separate per embedder)
- RRF fusion logic
- 4-metric custom evaluator

Variants differ in:
- LLM: Llama (Ollama) or OpenAI
- Technique: Baseline (no QR no CR), QR, CR, QR+CR
"""

import json
import copy
from pathlib import Path

NB_DIR = Path('C:/Users/Ricky Wijaya/Documents/STI/Semester 8/TA/Code TA/notebooks')

# Templates
BASELINE_OPENAI_PATH = NB_DIR / '02.1 Baseline - OpenAI.ipynb'
CR_OPENAI_PATH = NB_DIR / '04.1 CR - OpenAI.ipynb'


# ================================================================
# SHARED BLOCKS
# ================================================================

QR_PROMPT = '''QUERY_REWRITE_PROMPT = (
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


# ================================================================
# HELPERS
# ================================================================

def load_notebook(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(nb: dict, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)


def set_cell_source(cell: dict, content: str):
    lines = content.split('\n')
    cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    cell['outputs'] = []
    cell['execution_count'] = None


def get_cell_source(cell: dict) -> str:
    return ''.join(cell.get('source', []))


def find_cell_idx(nb: dict, predicate) -> int:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and predicate(get_cell_source(cell)):
            return i
    return -1


def replace_in_cells(nb: dict, old: str, new: str):
    """Replace substring in all code cells."""
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = get_cell_source(cell)
            if old in src:
                set_cell_source(cell, src.replace(old, new))


# ================================================================
# TRANSFORMATIONS
# ================================================================

def transform_openai_to_llama(nb: dict) -> dict:
    """Convert OpenAI-based hybrid notebook to Llama (Ollama) based."""
    nb = copy.deepcopy(nb)

    # 1. Find imports cell, add ollama
    imp_idx = find_cell_idx(nb, lambda s: 'from openai import OpenAI' in s and 'chromadb' in s)
    if imp_idx >= 0:
        src = get_cell_source(nb['cells'][imp_idx])
        # Replace OpenAI import with Ollama import
        src = src.replace('from openai import OpenAI', 'import ollama')
        set_cell_source(nb['cells'][imp_idx], src)

    # 2. Find config cell, modify
    cfg_idx = find_cell_idx(nb, lambda s: 'OPENAI_API_KEY' in s and 'CONFIG_NAME' in s and 'TOP_K_BM25' in s)
    if cfg_idx >= 0:
        src = get_cell_source(nb['cells'][cfg_idx])
        # Remove OpenAI API key line and related comment
        src = src.replace(
            "OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YOUR_API_KEY_HERE')",
            "# Llama + Ollama: tidak perlu API key eksternal"
        )
        # Change LLM_MODEL
        src = src.replace(
            "LLM_MODEL   = 'gpt-4.1-mini'",
            "LLM_MODEL   = 'llama3.2'  # Llama 3.2 via Ollama"
        )
        # Change embed model to Ollama nomic-embed-text
        src = src.replace(
            "EMBED_MODEL = 'text-embedding-3-small'  # 1536 dim, $0.02 / 1M token",
            "EMBED_MODEL = 'nomic-embed-text'  # Ollama embed, 768 dim, gratis"
        )
        # Change ChromaDB path (different embed dim = different index)
        src = src.replace(
            "CHROMA_DB_PATH  = NOTEBOOK_DIR / 'pubmedqa_chroma'",
            "CHROMA_DB_PATH  = NOTEBOOK_DIR / 'pubmedqa_chroma_llama'  # separate index untuk nomic (768 dim)"
        )
        # Update print statements
        src = src.replace("(via OpenAI)", "(via Ollama local)")
        # Remove API key check
        src = src.replace(
            "if 'YOUR_API_KEY' in OPENAI_API_KEY:",
            "if False:  # No API key needed for Ollama"
        )
        # Remove print of OPENAI_API_KEY
        src = src.replace(
            "print('  OPENAI_API_KEY belum diisi! Set env var OPENAI_API_KEY atau isi di cell ini.')",
            "pass"
        )
        src = src.replace(
            "print(f'  OPENAI_API_KEY: {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}')",
            "print('  Mode: Full local (Ollama)')"
        )
        set_cell_source(nb['cells'][cfg_idx], src)

    # 3. Replace OpenAI client setup cell with Ollama setup
    client_idx = find_cell_idx(nb, lambda s: 'openai_client = OpenAI' in s and 'def openai_generate' in s)
    if client_idx >= 0:
        new_setup = '''# ============================================================
# Setup Ollama Client + Embeddings
# ============================================================

# Cek Ollama running
try:
    _models = ollama.list()
    _names = [m.get("model", m.get("name", "")) for m in _models.get("models", [])]
    print(f'Ollama siap. {len(_names)} model tersedia.')
except Exception as e:
    print(f'ERROR: Ollama tidak jalan: {e}')
    print('Pastikan Ollama service aktif: `ollama serve` di terminal lain')
    raise


def llama_generate(prompt: str, max_tokens: int = 300, temperature: float = TEMPERATURE) -> str:
    """Wrapper Ollama generate (untuk LLM)."""
    for attempt in range(5):
        try:
            response = ollama.generate(
                model=LLM_MODEL, prompt=prompt,
                options={
                    'temperature': temperature,
                    'seed': SEED,
                    'num_predict': max_tokens,
                }
            )
            return response['response'].strip()
        except Exception as e:
            err = str(e)
            print(f'  [Ollama Error {attempt+1}/5] {type(e).__name__}: {err[:100]}')
            if attempt == 4:
                raise
            time.sleep((attempt + 1) * 3)
    raise RuntimeError('Ollama gagal setelah 5 percobaan.')


def ollama_embed(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """Wrapper Ollama embeddings (untuk dense retrieval)."""
    embeddings = []
    for text in texts:
        for attempt in range(3):
            try:
                resp = ollama.embeddings(model=model, prompt=text)
                embeddings.append(resp['embedding'])
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2)
    return embeddings


# Smoke test
print('Testing Ollama LLM...')
_test = llama_generate('Reply with exactly: OK', max_tokens=10)
print(f'  Response: {_test!r}')

print('\\nTesting Ollama embeddings...')
_emb = ollama_embed(['aspirin reduces heart attack risk'])
print(f'  Embed dim: {len(_emb[0])} (expected 768 untuk nomic-embed-text)')

print('\\nSemua client siap:')
print(f'  Generator : {LLM_MODEL} (Ollama local)')
print(f'  Embedder  : {EMBED_MODEL} (Ollama local)')'''
        set_cell_source(nb['cells'][client_idx], new_setup)

    # 4. Replace all `openai_generate(` with `llama_generate(`
    replace_in_cells(nb, 'openai_generate(', 'llama_generate(')

    # 5. Replace `openai_embed(` with `ollama_embed(`
    replace_in_cells(nb, 'openai_embed(', 'ollama_embed(')

    # 6. Update labels/comments
    replace_in_cells(nb, 'via OpenAI', 'via Ollama')
    replace_in_cells(nb, 'GPT-4.1-mini', 'Llama 3.2')
    replace_in_cells(nb, 'OpenAI ({LLM_MODEL}', 'Ollama ({LLM_MODEL}')

    # 7. Update pip install
    replace_in_cells(nb, 'pip install openai rank-bm25 chromadb datasets',
                         'pip install ollama rank-bm25 chromadb datasets')

    return nb


def add_qr_to_notebook(nb: dict, uses_llama: bool) -> dict:
    """Add QR (query rewriting) step before retrieval."""
    nb = copy.deepcopy(nb)

    gen_func = 'llama_generate' if uses_llama else 'openai_generate'

    # Insert QR cell right before the retrieve_hybrid cell
    retrieve_idx = find_cell_idx(nb, lambda s: 'def retrieve_hybrid' in s and 'def retrieve_bm25_raw' in s)
    if retrieve_idx < 0:
        print('  WARN: could not find retrieve_hybrid cell')
        return nb

    # Build QR cell content
    qr_cell_content = f'''# ============================================================
# Query Rewriting
# ============================================================

{QR_PROMPT}


def rewrite_query(query: str) -> str:
    """Reformulasi query untuk meningkatkan kualitas retrieval."""
    prompt = QUERY_REWRITE_PROMPT.format(query=query)
    try:
        rewritten = {gen_func}(prompt, max_tokens=150, temperature=0.3)
        rewritten = rewritten.strip().replace('\\n', ' ')
        return rewritten if len(rewritten) >= 10 else query
    except Exception as e:
        print(f'  [QR Error] {{e}} -- pakai query asli')
        return query


# Test QR
print('Contoh Query Rewriting:')
for q in ['Does aspirin reduce the risk of MI?',
          'Can exercise prevent T2DM?']:
    rw = rewrite_query(q)
    print(f'\\nAsli    : {{q}}')
    print(f'Rewrite : {{rw}}')'''

    qr_cell = {
        'cell_type': 'code',
        'id': 'qr-cell',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
    }
    set_cell_source(qr_cell, qr_cell_content)

    # Insert BEFORE retrieve cell
    nb['cells'].insert(retrieve_idx, qr_cell)

    # Modify retrieve_hybrid to use rewritten query
    # Find retrieve_hybrid again (index shifted by 1 because we inserted)
    retrieve_idx = find_cell_idx(nb, lambda s: 'def retrieve_hybrid' in s and 'def retrieve_bm25_raw' in s)
    if retrieve_idx >= 0:
        src = get_cell_source(nb['cells'][retrieve_idx])
        # Add rewritten step at the start of retrieve_hybrid function
        # Pattern: find the bm25_results line and inject rewrite before it
        if 'def retrieve_hybrid(' in src:
            # Modify function signature to return rewritten too
            src = src.replace(
                'def retrieve_hybrid(\n    query: str,',
                'def retrieve_hybrid(\n    query: str,'
            )
            # Add rewrite step and use rewritten for both BM25 and dense
            src = src.replace(
                '    bm25_results  = retrieve_bm25_raw(query, k=k_bm25)\n    dense_results = retrieve_dense(query, k=k_dense)',
                '    # Query rewriting\n'
                '    rewritten = rewrite_query(query)\n\n'
                '    bm25_results  = retrieve_bm25_raw(rewritten, k=k_bm25)\n'
                '    dense_results = retrieve_dense(rewritten, k=k_dense)'
            )
            # Return rewritten alongside results
            src = src.replace(
                '    return results',
                '    return results, rewritten'
            )
            # Test call
            src = src.replace(
                'test_r = retrieve_hybrid(test_q)',
                'test_r, test_rw = retrieve_hybrid(test_q)'
            )
            # Test printing
            src = src.replace(
                "print(f'Query: {test_q}')",
                "print(f'Query asli : {test_q}')\nprint(f'Rewrite    : {test_rw}')"
            )
            set_cell_source(nb['cells'][retrieve_idx], src)

    # Update Phase 1 cell to capture rewritten query
    phase1_idx = find_cell_idx(
        nb, lambda s: 'phase1_results.append' in s and 'retrieve_hybrid' in s
    )
    if phase1_idx >= 0:
        src = get_cell_source(nb['cells'][phase1_idx])
        src = src.replace(
            'retrieved = retrieve_hybrid(q)',
            'retrieved, rewritten = retrieve_hybrid(q)'
        )
        # Add rewritten_query to the saved record
        src = src.replace(
            "'question'        : q,",
            "'question'        : q,\n            'rewritten_query' : rewritten,"
        )
        set_cell_source(nb['cells'][phase1_idx], src)

    return nb


def add_cr_to_notebook(nb: dict) -> dict:
    """Add CrossEncoder reranking to a Hybrid notebook.

    This takes the Baseline hybrid pipeline and adds:
    - CrossEncoder import + setup
    - Modify retrieve to take top-20 from RRF, rerank to top-5
    """
    nb = copy.deepcopy(nb)

    # 1. Add sentence_transformers import
    imp_idx = find_cell_idx(nb, lambda s: 'import chromadb' in s and 'from rank_bm25' in s)
    if imp_idx >= 0:
        src = get_cell_source(nb['cells'][imp_idx])
        if 'CrossEncoder' not in src:
            src = src.replace(
                'import chromadb',
                'import chromadb\nfrom sentence_transformers import CrossEncoder'
            )
            set_cell_source(nb['cells'][imp_idx], src)

    # 2. Update config: add TOP_K_CANDIDATES and RERANKER_MODEL
    cfg_idx = find_cell_idx(nb, lambda s: 'TOP_K_BM25' in s and 'CONFIG_NAME' in s)
    if cfg_idx >= 0:
        src = get_cell_source(nb['cells'][cfg_idx])
        if 'TOP_K_CANDIDATES' not in src:
            src = src.replace(
                'TOP_K_BM25      = 50   # BM25 ambil top-50',
                'TOP_K_BM25       = 50   # BM25 ambil top-50'
            )
            src = src.replace(
                'TOP_K_DENSE     = 50   # Dense ambil top-50',
                "TOP_K_DENSE      = 50   # Dense ambil top-50\n"
                "TOP_K_CANDIDATES = 20   # RRF ambil top-20 kandidat\n"
                "RERANKER_MODEL   = 'cross-encoder/ms-marco-MiniLM-L-6-v2'"
            )
            src = src.replace(
                'TOP_K_RETRIEVAL = 5    # Final top-5 setelah RRF',
                'TOP_K_RETRIEVAL  = 5    # Final top-5 setelah CrossEncoder'
            )
        set_cell_source(nb['cells'][cfg_idx], src)

    # 3. Modify chroma build cell is fine (still build index)
    # 4. Modify retrieve_hybrid cell: add CrossEncoder step
    #    Need to inject crossencoder load + modify retrieve function

    # Find the chroma-related setup cell or add CE setup after it
    chroma_idx = find_cell_idx(nb, lambda s: 'chromadb.PersistentClient' in s and 'build_or_load_chroma_index' in s)
    if chroma_idx >= 0:
        # Insert CrossEncoder setup cell right after chroma setup
        ce_cell = {
            'cell_type': 'code',
            'id': 'ce-setup',
            'metadata': {},
            'execution_count': None,
            'outputs': [],
        }
        ce_content = '''# ============================================================
# Load CrossEncoder Reranker
# ============================================================

print(f'Loading CrossEncoder: {RERANKER_MODEL}...')
t0 = time.time()
cross_encoder = CrossEncoder(RERANKER_MODEL)
print(f'CrossEncoder siap dalam {time.time()-t0:.1f} detik')

# Smoke test
_pairs = [
    ('Does aspirin prevent heart attacks?', 'Aspirin reduces platelet aggregation.'),
    ('Does aspirin prevent heart attacks?', 'Weather patterns affect agriculture.'),
]
_scores = cross_encoder.predict(_pairs)
print(f'Smoke test: relevan={_scores[0]:.3f}, tidak relevan={_scores[1]:.3f}')
assert _scores[0] > _scores[1], 'CrossEncoder gagal membedakan relevan vs tidak!'
print('Reranker berfungsi.')'''
        set_cell_source(ce_cell, ce_content)
        nb['cells'].insert(chroma_idx + 1, ce_cell)

    # 5. Modify retrieve_hybrid to add rerank step
    retrieve_idx = find_cell_idx(nb, lambda s: 'def retrieve_hybrid' in s and 'def retrieve_bm25_raw' in s)
    if retrieve_idx >= 0:
        src = get_cell_source(nb['cells'][retrieve_idx])
        # Change function to take top-20 candidates from RRF, then rerank to top-5
        # Original: fused[:k_final]
        # New: fused[:k_candidates], then CE rerank to k_final
        src = src.replace(
            'def retrieve_hybrid(\n    query: str,\n    k_bm25: int = TOP_K_BM25,\n    k_dense: int = TOP_K_DENSE,\n    k_final: int = TOP_K_RETRIEVAL\n)',
            'def retrieve_hybrid(\n    query: str,\n    k_bm25: int = TOP_K_BM25,\n    k_dense: int = TOP_K_DENSE,\n    k_candidates: int = TOP_K_CANDIDATES,\n    k_final: int = TOP_K_RETRIEVAL\n)'
        )
        # Change top-K selection: take candidates first
        src = src.replace(
            '    # Build RetrievalResult dengan semua skor\n'
            '    results = []\n'
            '    for doc_id, rrf_score in fused[:k_final]:',
            '    # Build candidates (top-K from RRF)\n'
            '    results = []\n'
            '    for doc_id, rrf_score in fused[:k_candidates]:'
        )
        # Add reranking step before returning
        src = src.replace(
            '    return results',
            '\n    # CrossEncoder reranking\n'
            '    pairs = [(query, r.document.text) for r in results]\n'
            '    ce_scores = cross_encoder.predict(pairs)\n'
            '    for r, s in zip(results, ce_scores):\n'
            '        r.reranker_score = float(s)\n'
            '    results = sorted(results, key=lambda r: r.reranker_score, reverse=True)[:k_final]\n\n'
            '    return results'
        )
        # Update test print to show CE score
        src = src.replace(
            "print(f'  [{i}] RRF={r.rrf_score:.4f} | BM25={r.bm25_score:.2f} | Dense={r.dense_score:.3f} '",
            "print(f'  [{i}] CE={r.reranker_score:.3f} | RRF={r.rrf_score:.4f} | BM25={r.bm25_score:.2f} | Dense={r.dense_score:.3f} '"
        )
        set_cell_source(nb['cells'][retrieve_idx], src)

    # 6. Update Phase 1 cell to save reranker_scores
    phase1_idx = find_cell_idx(
        nb, lambda s: 'phase1_results.append' in s and 'retrieve_hybrid' in s
    )
    if phase1_idx >= 0:
        src = get_cell_source(nb['cells'][phase1_idx])
        src = src.replace(
            "'rrf_scores'      : [r.rrf_score   for r in retrieved],",
            "'rrf_scores'      : [r.rrf_score   for r in retrieved],\n"
            "            'reranker_scores' : [r.reranker_score for r in retrieved],"
        )
        set_cell_source(nb['cells'][phase1_idx], src)

    return nb


def update_config_name(nb: dict, new_config_name: str) -> dict:
    """Update CONFIG_NAME in the config cell."""
    nb = copy.deepcopy(nb)
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = get_cell_source(cell)
        # Match any existing CONFIG_NAME string
        new_src = src
        for old_name in ['baseline_openai', 'baseline_claude', 'baseline_llama',
                         'cr_openai', 'cr_claude', 'cr_llama',
                         'qr_openai', 'qr_claude', 'qr_llama',
                         'qr_cr_openai', 'qr_cr_llama',
                         'hybrid_openai', 'hybrid_cr_openai', 'hybrid_claude']:
            new_src = new_src.replace(f"'{old_name}'", f"'{new_config_name}'")
        if new_src != src:
            set_cell_source(cell, new_src)
    return nb


def update_title(nb: dict, new_title: str, new_desc: str = None):
    """Update markdown title cell."""
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            src = get_cell_source(cell)
            # Replace heading line
            lines = src.split('\n')
            if lines and lines[0].startswith('# '):
                lines[0] = f'# {new_title}'
                new_src = '\n'.join(lines)
                cell['source'] = [l + '\n' for l in new_src.split('\n')[:-1]] + [new_src.split('\n')[-1]]
                break


# ================================================================
# BUILD NOTEBOOKS
# ================================================================

def main():
    # Load templates
    baseline_oai = load_notebook(BASELINE_OPENAI_PATH)
    cr_oai = load_notebook(CR_OPENAI_PATH)

    # Baseline Llama = baseline OpenAI but Llama/Ollama
    print('\n[1/6] Generating: 02 Baseline - Llama.ipynb')
    nb = transform_openai_to_llama(baseline_oai)
    nb = update_config_name(nb, 'baseline_llama')
    update_title(nb, '02 - RAG Baseline Hybrid dengan Llama 3.2')
    save_notebook(nb, NB_DIR / '02 Baseline - Llama.ipynb')

    # QR Llama = baseline Llama + QR
    print('[2/6] Generating: 03 QR - Llama.ipynb')
    nb = transform_openai_to_llama(baseline_oai)
    nb = add_qr_to_notebook(nb, uses_llama=True)
    nb = update_config_name(nb, 'qr_llama')
    update_title(nb, '03 - RAG Hybrid + Query Rewriting dengan Llama 3.2')
    save_notebook(nb, NB_DIR / '03 QR - Llama.ipynb')

    # QR OpenAI = baseline OpenAI + QR
    print('[3/6] Generating: 03.1 QR - OpenAI.ipynb')
    nb = copy.deepcopy(baseline_oai)
    nb = add_qr_to_notebook(nb, uses_llama=False)
    nb = update_config_name(nb, 'qr_openai')
    update_title(nb, '03.1 - RAG Hybrid + Query Rewriting dengan GPT-4.1-mini')
    save_notebook(nb, NB_DIR / '03.1 QR - OpenAI.ipynb')

    # CR Llama = baseline Llama + CR
    print('[4/6] Generating: 04 CR - Llama.ipynb')
    nb = transform_openai_to_llama(baseline_oai)
    nb = add_cr_to_notebook(nb)
    nb = update_config_name(nb, 'cr_llama')
    update_title(nb, '04 - RAG Hybrid + Context Reranking dengan Llama 3.2')
    save_notebook(nb, NB_DIR / '04 CR - Llama.ipynb')

    # QR+CR Llama = baseline Llama + QR + CR
    print('[5/6] Generating: 05 QR+CR - Llama.ipynb')
    nb = transform_openai_to_llama(baseline_oai)
    nb = add_cr_to_notebook(nb)
    nb = add_qr_to_notebook(nb, uses_llama=True)
    nb = update_config_name(nb, 'qr_cr_llama')
    update_title(nb, '05 - RAG Hybrid + QR + CR dengan Llama 3.2')
    save_notebook(nb, NB_DIR / '05 QR+CR - Llama.ipynb')

    # QR+CR OpenAI = CR OpenAI + QR
    print('[6/6] Generating: 05.1 QR+CR - OpenAI.ipynb')
    nb = copy.deepcopy(cr_oai)
    nb = add_qr_to_notebook(nb, uses_llama=False)
    nb = update_config_name(nb, 'qr_cr_openai')
    update_title(nb, '05.1 - RAG Hybrid + QR + CR dengan GPT-4.1-mini')
    save_notebook(nb, NB_DIR / '05.1 QR+CR - OpenAI.ipynb')

    # Verify syntax
    import ast
    print('\n=== Syntax verification ===')
    all_files = list(NB_DIR.glob('0*.ipynb'))
    for f in sorted(all_files):
        with open(f, 'r', encoding='utf-8') as fp:
            nb = json.load(fp)
        errors = []
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] != 'code':
                continue
            src = get_cell_source(cell)
            if 'pip install' in src and len(src) < 300:
                continue
            try:
                ast.parse(src)
            except SyntaxError as e:
                errors.append(f'cell[{i}]: {e.msg} @line {e.lineno}')
        status = 'OK' if not errors else f'FAIL ({len(errors)} errors)'
        print(f'  {f.name:<45} {status}')
        for err in errors[:2]:
            print(f'      {err}')


if __name__ == '__main__':
    main()
