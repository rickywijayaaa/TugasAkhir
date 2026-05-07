"""
Script untuk generate notebook 06.1-RAG-Hybrid-Claude.ipynb

Based on 06-RAG-Hybrid-OpenAI.ipynb dengan perubahan:
- LLM generation + evaluator: OpenAI GPT-4.1-mini -> Claude Haiku 4.5
- Dense embeddings: TETAP OpenAI text-embedding-3-small (Anthropic tidak punya embedding API)
- Retrieval pipeline: IDENTIK dengan 06 (BM25 + Dense via RRF)

Pipeline:
    query -> BM25 top-50 + Dense top-50 (OpenAI embed) -> RRF fusion -> top-5 -> Claude generate
"""

import json
import re
from pathlib import Path


def load_source_notebook():
    path = Path('C:/Users/Ricky Wijaya/Documents/STI/Semester 8/TA/Code TA/notebooks/06-RAG-Hybrid-OpenAI.ipynb')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def make_hybrid_claude_notebook(src_nb):
    """Transform 06 OpenAI notebook to 06.1 Claude version (embeddings stay OpenAI)."""
    nb = json.loads(json.dumps(src_nb))  # deep copy

    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            src = ''.join(cell['source'])
            new_src = src

            # Title
            new_src = new_src.replace(
                '# 06 — RAG Hybrid Retrieval (BM25 + Dense via RRF)',
                '# 06.1 — RAG Hybrid Retrieval dengan Claude Haiku 4.5'
            )
            # Pipeline diagram description
            new_src = new_src.replace(
                '-> OpenAI generate',
                '-> Claude Haiku 4.5 generate'
            )
            # Model section
            new_src = new_src.replace(
                '- Generator: `gpt-4.1-mini` via OpenAI API',
                '- Generator: `claude-haiku-4-5` via Anthropic API'
            )
            # Phase 1 time estimate (Claude should be similar speed)
            new_src = new_src.replace(
                'Estimasi waktu: ~15-20 menit dengan GPT-4.1-mini.',
                'Estimasi waktu: ~15-25 menit dengan Claude Haiku 4.5.'
            )

            if new_src != src:
                cell['source'] = [line + '\n' for line in new_src.split('\n')[:-1]] + [new_src.split('\n')[-1]]
            continue

        # Code cells
        src = ''.join(cell['source'])
        new_src = src

        # Detect cell type by keywords
        is_install_cell = src.strip().startswith('#') and 'pip install' in src
        is_imports_cell = 'from openai import OpenAI' in src and 'from rank_bm25' in src
        is_config_cell = 'OPENAI_API_KEY' in src and 'CONFIG_NAME' in src and 'TOP_K_BM25' in src
        is_openai_setup_cell = 'openai_client = OpenAI' in src and 'def openai_generate' in src

        # 1. Install cell: add anthropic
        if is_install_cell:
            new_src = new_src.replace(
                'pip install openai rank-bm25 chromadb datasets',
                'pip install anthropic openai rank-bm25 chromadb datasets'
            )

        # 2. Imports cell: ADD anthropic (keep openai for embeddings)
        elif is_imports_cell:
            # Add `import anthropic` right after `from openai import OpenAI`
            new_src = new_src.replace(
                'from openai import OpenAI\n',
                'from openai import OpenAI\nimport anthropic\n'
            )

        # 3. Config cell: add ANTHROPIC_API_KEY alongside OPENAI_API_KEY
        elif is_config_cell:
            # Security: force both API keys to use env vars
            new_src = re.sub(
                r"OPENAI_API_KEY = os\.environ\.get\('OPENAI_API_KEY',\s*'[^']*'\)",
                "OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YOUR_OPENAI_KEY_HERE')",
                new_src
            )
            # Add ANTHROPIC_API_KEY right after OPENAI_API_KEY line
            new_src = new_src.replace(
                "OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YOUR_OPENAI_KEY_HERE')",
                (
                    "OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YOUR_OPENAI_KEY_HERE')\n"
                    "ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', 'YOUR_ANTHROPIC_KEY_HERE')"
                )
            )
            # Change LLM_MODEL to Claude, but keep EMBED_MODEL as OpenAI
            new_src = new_src.replace(
                "LLM_MODEL   = 'gpt-4.1-mini'",
                "LLM_MODEL   = 'claude-haiku-4-5-20251001'  # Anthropic Claude Haiku 4.5"
            )
            # Add comment clarifying embedder stays OpenAI
            new_src = new_src.replace(
                "EMBED_MODEL = 'text-embedding-3-small'  # 1536 dim, $0.02 / 1M token",
                "EMBED_MODEL = 'text-embedding-3-small'  # OpenAI (Anthropic tidak punya embedding API), 1536 dim, $0.02 / 1M token"
            )
            # Change CONFIG_NAME
            new_src = new_src.replace(
                "CONFIG_NAME        = 'hybrid_openai'",
                "CONFIG_NAME        = 'hybrid_claude'"
            )
            # Update print labels
            new_src = new_src.replace(
                "print(f'  LLM          : {LLM_MODEL} (via OpenAI)')",
                "print(f'  LLM          : {LLM_MODEL} (via Anthropic)')"
            )
            # Update API key prompt
            new_src = new_src.replace(
                "if 'YOUR_API_KEY' in OPENAI_API_KEY:",
                "if 'YOUR_OPENAI_KEY' in OPENAI_API_KEY or 'YOUR_ANTHROPIC_KEY' in ANTHROPIC_API_KEY:"
            )
            new_src = new_src.replace(
                "print('  OPENAI_API_KEY belum diisi! Set env var OPENAI_API_KEY atau isi di cell ini.')",
                (
                    "print('  API key belum lengkap! Perlu DUA key:')\n"
                    "    print('    - OPENAI_API_KEY untuk embeddings (dense retrieval)')\n"
                    "    print('    - ANTHROPIC_API_KEY untuk generation + evaluator')"
                )
            )
            new_src = new_src.replace(
                "print(f'  OPENAI_API_KEY: {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}')",
                (
                    "print(f'  OPENAI_API_KEY    : {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}')\n"
                    "    print(f'  ANTHROPIC_API_KEY : {ANTHROPIC_API_KEY[:8]}...{ANTHROPIC_API_KEY[-4:]}')"
                )
            )

        # 4. OpenAI setup cell: ADD Claude client + wrapper, KEEP openai_client for embeddings
        elif is_openai_setup_cell:
            new_cell = '''# ============================================================
# Setup OpenAI Client (untuk embeddings) + Claude Client (untuk LLM)
# ============================================================

# OpenAI client: HANYA untuk dense embeddings (Anthropic tidak punya embedding API)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Claude client: untuk LLM generation + evaluator
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def claude_generate(prompt: str, max_tokens: int = 300, temperature: float = TEMPERATURE) -> str:
    """Wrapper Claude API dengan retry otomatis."""
    for attempt in range(5):
        try:
            response = claude_client.messages.create(
                model=LLM_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{'role': 'user', 'content': prompt}],
            )
            return response.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = (attempt + 1) * 10
            print(f'  [Claude Rate limit] Tunggu {wait}s... (attempt {attempt+1}/5)')
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code in (500, 502, 503):
                wait = (attempt + 1) * 5
                print(f'  [Claude Server error {e.status_code}] Tunggu {wait}s...')
                time.sleep(wait)
            else:
                print(f'  [Claude Error] {type(e).__name__}: {str(e)[:100]}')
                raise
        except Exception as e:
            print(f'  [Claude Unknown] {type(e).__name__}: {str(e)[:100]}')
            raise
    raise RuntimeError('Claude API gagal setelah 5 percobaan.')


def openai_embed(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """Wrapper OpenAI embeddings. HANYA untuk dense retrieval (Anthropic tidak ada embed API)."""
    for attempt in range(5):
        try:
            response = openai_client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in response.data]
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                wait = (attempt + 1) * 10
                print(f'  [Embed Rate limit] Tunggu {wait}s...')
                time.sleep(wait)
            elif '500' in err or '502' in err or '503' in err:
                wait = (attempt + 1) * 5
                print(f'  [Embed Server error] Tunggu {wait}s...')
                time.sleep(wait)
            else:
                print(f'  [Embed Error] {type(e).__name__}: {err[:100]}')
                raise
    raise RuntimeError('OpenAI embeddings gagal setelah 5 percobaan.')


# Smoke test
print('Testing Claude API (untuk generation)...')
_test = claude_generate('Reply with exactly: OK', max_tokens=5)
print(f'  Claude response: {_test!r}')

print('\\nTesting OpenAI Embed API (untuk dense retrieval)...')
_emb = openai_embed(['aspirin reduces heart attack risk'])
print(f'  Embed dim: {len(_emb[0])} (expected: 1536 for text-embedding-3-small)')

print('\\nSemua client siap:')
print(f'  Generator : {LLM_MODEL} (Anthropic)')
print(f'  Embedder  : {EMBED_MODEL} (OpenAI)')'''
            new_src = new_cell

        # 5. All other cells: replace openai_generate -> claude_generate for LLM calls
        else:
            # Only replace openai_generate calls (not openai_embed or openai_client for embeddings)
            new_src = new_src.replace('openai_generate(', 'claude_generate(')

            # Update comments/labels
            new_src = new_src.replace('via OpenAI', 'via Claude')
            new_src = new_src.replace('via OpenAI.', 'via Claude.')
            new_src = new_src.replace('Generate jawaban via Claude. PENTING', 'Generate jawaban via Claude. PENTING')

            # Output comparison labels
            new_src = new_src.replace(
                "Baseline OpenAI",
                "Baseline OpenAI"  # keep for comparison reference
            )
            # Evaluator cell header
            new_src = new_src.replace(
                '# Custom Zero-NaN Evaluator — 4 metrik via OpenAI',
                '# Custom Zero-NaN Evaluator — 4 metrik via Claude'
            )
            new_src = new_src.replace(
                "resp = openai_generate(prompt",
                "resp = claude_generate(prompt"
            )

            # Summary row label
            new_src = new_src.replace(
                "__LABEL__",
                "__LABEL__"  # keep as-is, it will be replaced below
            )
            new_src = new_src.replace(
                "Hybrid OpenAI",
                "Hybrid Claude"
            )

        # Apply changes
        cell['source'] = [line + '\n' for line in new_src.split('\n')[:-1]] + [new_src.split('\n')[-1]]
        cell['outputs'] = []
        cell['execution_count'] = None

    return nb


def main():
    src_nb = load_source_notebook()
    claude_nb = make_hybrid_claude_notebook(src_nb)

    # Cleanup: remove orphan cells that have no valid content
    cleaned_cells = []
    for cell in claude_nb['cells']:
        src = ''.join(cell.get('source', [])).strip()
        # Skip: empty cells
        if not src:
            continue
        # Skip: stray "pip install X" without # prefix (not in comment, not !pip)
        if src.startswith('pip install') and 'chromadb' in src and len(src) < 100:
            continue
        cleaned_cells.append(cell)
    claude_nb['cells'] = cleaned_cells

    out_path = Path('C:/Users/Ricky Wijaya/Documents/STI/Semester 8/TA/Code TA/notebooks/06.1-RAG-Hybrid-Claude.ipynb')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(claude_nb, f, indent=1, ensure_ascii=False)

    print(f'Saved: {out_path}')
    print(f'Cells: {len(claude_nb["cells"])}')

    # Syntax verify
    import ast
    all_ok = True
    for i, cell in enumerate(claude_nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        # Skip install cells (Jupyter magic commands)
        if 'pip install' in src and len(src) < 200:
            continue
        try:
            ast.parse(src)
        except SyntaxError as e:
            all_ok = False
            print(f'  SYNTAX ERROR cell [{i}]: {e.msg} at line {e.lineno}')
            print(f'    Preview: {src[:200]}')

    if all_ok:
        print('  All code cells: valid Python syntax')

    # Security check
    for cell in claude_nb['cells']:
        src = ''.join(cell.get('source', []))
        if 'sk-proj' in src or 'sk-ant-api' in src:
            print('  [SECURITY] Real API key leaked in cell!')
            return
    print('  [OK] No leaked API keys')


if __name__ == '__main__':
    main()
