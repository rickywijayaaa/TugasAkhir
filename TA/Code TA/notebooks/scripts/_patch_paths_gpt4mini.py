"""
Patch path di 12 notebook GPT-4.1-mini supaya pakai struktur baru:
- INDEXES_DIR -> notebooks/indexes/ (shared)
- RESULTS_DIR -> results/<kelompok>/ (per kelompok)

Logika resolve PROJECT_ROOT pakai walk-up parents agar notebook portable.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # notebooks/
print(f"notebooks root: {ROOT}")

# Mapping: notebook_path -> (kelompok_dir, bm25_index_name, chroma_dir_name)
TARGETS = {
    # Kelompok 1: BM25 tanpa ekspansi
    "10_bm25/gpt4mini/baseline.ipynb":  ("10_bm25", "pubmedqa_bm25.pkl", None),
    "10_bm25/gpt4mini/qr.ipynb":        ("10_bm25", "pubmedqa_bm25.pkl", None),
    "10_bm25/gpt4mini/cr.ipynb":        ("10_bm25", "pubmedqa_bm25.pkl", None),
    "10_bm25/gpt4mini/qr_cr.ipynb":     ("10_bm25", "pubmedqa_bm25.pkl", None),
    # Kelompok 2: Hybrid tanpa ekspansi
    "20_hybrid_tanpa_ekspansi/gpt4mini/baseline.ipynb": ("20_hybrid_tanpa_ekspansi", "pubmedqa_bm25.pkl", "pubmedqa_chroma"),
    "20_hybrid_tanpa_ekspansi/gpt4mini/qr.ipynb":       ("20_hybrid_tanpa_ekspansi", "pubmedqa_bm25.pkl", "pubmedqa_chroma"),
    "20_hybrid_tanpa_ekspansi/gpt4mini/cr.ipynb":       ("20_hybrid_tanpa_ekspansi", "pubmedqa_bm25.pkl", "pubmedqa_chroma"),
    "20_hybrid_tanpa_ekspansi/gpt4mini/qr_cr.ipynb":    ("20_hybrid_tanpa_ekspansi", "pubmedqa_bm25.pkl", "pubmedqa_chroma"),
    # Kelompok 3: Hybrid dengan ekspansi
    "30_hybrid_dengan_ekspansi/gpt4mini/baseline.ipynb": ("30_hybrid_dengan_ekspansi", "pubmedqa_bm25_ekspansi.pkl", "pubmedqa_chroma_ekspansi"),
    "30_hybrid_dengan_ekspansi/gpt4mini/qr.ipynb":       ("30_hybrid_dengan_ekspansi", "pubmedqa_bm25_ekspansi.pkl", "pubmedqa_chroma_ekspansi"),
    "30_hybrid_dengan_ekspansi/gpt4mini/cr.ipynb":       ("30_hybrid_dengan_ekspansi", "pubmedqa_bm25_ekspansi.pkl", "pubmedqa_chroma_ekspansi"),
    "30_hybrid_dengan_ekspansi/gpt4mini/qr_cr.ipynb":    ("30_hybrid_dengan_ekspansi", "pubmedqa_bm25_ekspansi.pkl", "pubmedqa_chroma_ekspansi"),
}


def make_path_setup_cell(kelompok_dir, bm25_name, chroma_name):
    """Generate code untuk path-setup cell yang dipasang di awal notebook."""
    chroma_block = ""
    if chroma_name:
        chroma_block = f"CHROMA_DB_PATH  = INDEXES_DIR / '{chroma_name}'\n"

    code = f'''# ============================================================
# PATH SETUP — auto-resolve PROJECT_ROOT
# ============================================================
from pathlib import Path

_HERE = Path('.').resolve()
PROJECT_ROOT = next((p for p in [_HERE] + list(_HERE.parents) if p.name == 'Code TA'), _HERE.parent.parent.parent)
NOTEBOOKS_V2 = PROJECT_ROOT / 'notebooks'
INDEXES_DIR  = NOTEBOOKS_V2 / 'indexes'
RESULTS_DIR  = PROJECT_ROOT / 'results' / '{kelompok_dir}'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BM25_INDEX_PATH = INDEXES_DIR / '{bm25_name}'
{chroma_block}NOTEBOOK_DIR    = INDEXES_DIR  # kompat lama: BM25_INDEX_PATH dan CHROMA_DB_PATH

print(f'PROJECT_ROOT  : {{PROJECT_ROOT}}')
print(f'INDEXES_DIR   : {{INDEXES_DIR}}')
print(f'RESULTS_DIR   : {{RESULTS_DIR}}')
print(f'BM25 index    : {{BM25_INDEX_PATH.name}}')
'''
    if chroma_name:
        code += f"print(f'Chroma DB     : {{CHROMA_DB_PATH.name}}')\n"

    return code


def _src_to_str(src):
    return src if isinstance(src, str) else ''.join(src)


def patch_notebook(nb_path: Path, kelompok_dir: str, bm25_name: str, chroma_name: str | None):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Locate the config cell (yang berisi NOTEBOOK_DIR atau BM25_INDEX_PATH)
    config_idx = None
    for i, c in enumerate(nb['cells'][:10]):
        if c['cell_type'] != 'code':
            continue
        src = _src_to_str(c['source'])
        if 'BM25_INDEX_PATH' in src or 'NOTEBOOK_DIR' in src or 'RESULTS_DIR' in src:
            config_idx = i
            break

    setup_code = make_path_setup_cell(kelompok_dir, bm25_name, chroma_name)
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": setup_code.splitlines(keepends=True),
    }

    if config_idx is not None:
        # Replace the lines that define paths within that cell, but easier: add new cell BEFORE it
        nb['cells'].insert(config_idx, new_cell)
        # Comment out the old path lines so they don't override
        old_cell = nb['cells'][config_idx + 1]
        old_src = _src_to_str(old_cell['source'])
        # mark old path lines
        out_lines = []
        for line in old_src.splitlines(keepends=True):
            stripped = line.strip()
            if stripped.startswith('NOTEBOOK_DIR') or stripped.startswith('BM25_INDEX_PATH') or stripped.startswith('CHROMA_DB_PATH') or stripped.startswith('RESULTS_DIR'):
                out_lines.append('# ' + line)  # comment out (was hardcoded path)
            else:
                out_lines.append(line)
        old_cell['source'] = out_lines
    else:
        # No existing config cell found; insert at beginning (after imports if any)
        nb['cells'].insert(1, new_cell)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    return config_idx


patched = 0
for rel_path, (kelompok_dir, bm25_name, chroma_name) in TARGETS.items():
    nb_path = ROOT / rel_path
    if not nb_path.exists():
        print(f"MISSING: {rel_path}")
        continue
    idx = patch_notebook(nb_path, kelompok_dir, bm25_name, chroma_name)
    print(f"Patched: {rel_path} (config cell at index {idx})")
    patched += 1

print(f"\nTotal notebook ter-patch: {patched}/{len(TARGETS)}")
