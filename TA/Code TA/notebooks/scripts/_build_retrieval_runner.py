"""Build notebook 40_evaluation/retrieval_metrics_runner.ipynb"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / '40_evaluation' / 'retrieval_metrics_runner.ipynb'


def md(text):
    lines = text.splitlines(keepends=True) or ['']
    return {'cell_type': 'markdown', 'metadata': {}, 'source': lines}


def code(text):
    lines = text.splitlines(keepends=True) or ['']
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': lines}


cells = []

cells.append(md("""# Runner B: Retrieval Metrics untuk 12 Konfigurasi GPT-4.1-mini

Notebook ini menghitung **Recall@5, MAP@5, nDCG@5, MRR, Hit@5, Precision@5** untuk 12 konfigurasi pada GPT-4.1-mini:

- Kelompok 1 (`10_bm25/`): baseline, qr, cr, qr_cr
- Kelompok 2 (`20_hybrid_tanpa_ekspansi/`): baseline, qr, cr, qr_cr
- Kelompok 3 (`30_hybrid_dengan_ekspansi/`): baseline, qr, cr, qr_cr

**Relevance criterion**: \\textit{chunk} relevan apabila `pubid == source paper pubid` (per PubMedQA Jin dkk. 2019).

**Output**:
- `results/<kelompok>/<config>_retrieval_metrics.json` (per-konfigurasi)
- `results/retrieval_metrics_summary.json` (ringkasan 12 konfigurasi)

Cepat — tidak memanggil LLM, hanya hitungan numerik berdasarkan phase1 yang sudah ada."""))

cells.append(md("## 1. Setup & Imports"))

cells.append(code("""import sys
import json
from pathlib import Path

_HERE = Path('.').resolve()
PROJECT_ROOT = next((p for p in [_HERE] + list(_HERE.parents) if p.name == 'Code TA'), _HERE.parent.parent.parent)
NOTEBOOKS_V2 = PROJECT_ROOT / 'notebooks'
SCRIPTS_DIR  = NOTEBOOKS_V2 / 'scripts'
INDEXES_DIR  = NOTEBOOKS_V2 / 'indexes'
RESULTS_DIR  = PROJECT_ROOT / 'results'

sys.path.insert(0, str(SCRIPTS_DIR))
from retrieval_metrics import (
    load_corpus_documents, build_text_to_meta_map, build_source_meta,
    evaluate_phase1, Document
)

print(f'PROJECT_ROOT : {PROJECT_ROOT}')
print(f'INDEXES_DIR  : {INDEXES_DIR}')
print(f'RESULTS_DIR  : {RESULTS_DIR}')"""))

cells.append(md("## 2. Load corpus (text_map untuk derive pubids dari context texts)"))

cells.append(code("""# Load dua versi korpus (asli dan ekspansi akronim) — text_map berbeda
print('Loading corpus asli (pubmedqa_bm25.pkl)...')
docs_asli = load_corpus_documents(INDEXES_DIR / 'pubmedqa_bm25.pkl')
text_map_asli = build_text_to_meta_map(docs_asli)
source_meta_asli = build_source_meta(docs_asli)
print(f'  {len(docs_asli)} chunks, {len(source_meta_asli)} unique pubids')

print('Loading corpus dengan ekspansi (pubmedqa_bm25_ekspansi.pkl)...')
docs_eks = load_corpus_documents(INDEXES_DIR / 'pubmedqa_bm25_ekspansi.pkl')
text_map_eks = build_text_to_meta_map(docs_eks)
source_meta_eks = build_source_meta(docs_eks)
print(f'  {len(docs_eks)} chunks, {len(source_meta_eks)} unique pubids')"""))

cells.append(md("## 3. Map config → text_map yang sesuai"))

cells.append(code("""# Mapping: (kelompok_dir, config) → (text_map, source_meta)
CONFIG_MAPS = {
    # Kelompok 1: BM25 tanpa ekspansi
    ('10_bm25', 'baseline'):  (text_map_asli, source_meta_asli),
    ('10_bm25', 'qr'):        (text_map_asli, source_meta_asli),
    ('10_bm25', 'cr'):        (text_map_asli, source_meta_asli),
    ('10_bm25', 'qr_cr'):     (text_map_asli, source_meta_asli),
    # Kelompok 2: Hybrid tanpa ekspansi
    ('20_hybrid_tanpa_ekspansi', 'baseline'):  (text_map_asli, source_meta_asli),
    ('20_hybrid_tanpa_ekspansi', 'qr'):        (text_map_asli, source_meta_asli),
    ('20_hybrid_tanpa_ekspansi', 'cr'):        (text_map_asli, source_meta_asli),
    ('20_hybrid_tanpa_ekspansi', 'qr_cr'):     (text_map_asli, source_meta_asli),
    # Kelompok 3: Hybrid dengan ekspansi
    ('30_hybrid_dengan_ekspansi', 'baseline'): (text_map_eks, source_meta_eks),
    ('30_hybrid_dengan_ekspansi', 'qr'):       (text_map_eks, source_meta_eks),
    ('30_hybrid_dengan_ekspansi', 'cr'):       (text_map_eks, source_meta_eks),
    ('30_hybrid_dengan_ekspansi', 'qr_cr'):    (text_map_eks, source_meta_eks),
}
print(f'Total konfigurasi: {len(CONFIG_MAPS)}')"""))

cells.append(md("## 4. Run evaluation untuk semua konfigurasi"))

cells.append(code("""summary = {}

for (kelompok, cfg), (text_map, source_meta) in CONFIG_MAPS.items():
    phase1_path = RESULTS_DIR / kelompok / f'{cfg}_phase1_answers.json'
    out_path    = RESULTS_DIR / kelompok / f'{cfg}_retrieval_metrics.json'

    if not phase1_path.exists():
        print(f'[SKIP] {kelompok}/{cfg} - no phase1 file')
        continue

    with open(phase1_path, 'r', encoding='utf-8') as f:
        phase1 = json.load(f)
    samples = phase1.get('results', phase1.get('answers', []))

    print(f'\\n=== {kelompok}/{cfg} ===')
    print(f'  Samples: {len(samples)}')

    result = evaluate_phase1(samples, text_map, source_meta)
    agg = result['aggregate']

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'kelompok': kelompok,
            'config': cfg,
            'n_samples': result['n_samples'],
            'aggregate': agg,
            'per_query': result['per_query'],
        }, f, indent=2)

    summary[f'{kelompok}/{cfg}'] = agg
    print(f'  Recall@5={agg[\"recall@5\"]:.4f}  MAP@5={agg[\"map@5\"]:.4f}  nDCG@5={agg[\"ndcg@5\"]:.4f}')
    print(f'  Saved: {out_path.name}')

# Save summary
summary_path = RESULTS_DIR / 'retrieval_metrics_summary.json'
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
print(f'\\nSummary saved: {summary_path}')"""))

cells.append(md("## 5. Ringkasan tabel"))

cells.append(code("""print(f'{\"Kelompok / Config\":<48} {\"Recall@5\":>10} {\"MAP@5\":>10} {\"nDCG@5\":>10} {\"MRR\":>10} {\"Hit@5\":>10}')
print('-' * 100)
for label, agg in summary.items():
    print(f'{label:<48} {agg[\"recall@5\"]:>10.4f} {agg[\"map@5\"]:>10.4f} {agg[\"ndcg@5\"]:>10.4f} {agg[\"mrr\"]:>10.4f} {agg[\"hit@5\"]:>10.4f}')"""))

notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'Created: {OUT}')
print(f'Total cells: {len(cells)}')
