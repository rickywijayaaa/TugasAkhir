"""
Build notebook 40_evaluation/ragas_phase2_runner.ipynb
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # notebooks/
OUT = ROOT / '40_evaluation' / 'ragas_phase2_runner.ipynb'


def md(text):
    lines = text.splitlines(keepends=True) or ['']
    return {'cell_type': 'markdown', 'metadata': {}, 'source': lines}


def code(text):
    lines = text.splitlines(keepends=True) or ['']
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': lines}


cells = []

cells.append(md("""# Runner A: Phase 2 RAGAS untuk Phase 1 yang Belum/Perlu Re-evaluasi (GPT-4.1-mini)

Notebook ini menjalankan evaluasi RAGAS (4 metrik) untuk konfigurasi yang phase 2 RAGAS-nya belum dijalankan ATAU perlu di-rerun karena ada perubahan Phase 1.

**Konfigurasi target (6 total):**

Kelompok 2 (Hybrid tanpa Ekspansi Akronim) — semua 4 belum ada phase 2:
- `baseline` — Hybrid retrieval saja
- `qr` — Hybrid + Query Rewriting (Versi A)
- `cr` — Hybrid + Context Reranking
- `qr_cr` — Hybrid + QR + CR

Kelompok 3 (Hybrid dengan Ekspansi Akronim) — 2 perlu re-run karena QR di-patch ke Versi A:
- `qr` — Hybrid + Ekspansi + QR (Versi A)
- `qr_cr` — Hybrid + Ekspansi + QR + CR (Versi A)

**Output:** `results/<kelompok>/{config}_phase2_custom.json`

Resume-aware: kalau sampel sudah dievaluasi, akan di-skip."""))

cells.append(md("## 1. Setup & Imports"))

cells.append(code("""import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from openai import OpenAI

# Progress bar (tqdm auto-detect notebook vs CLI)
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

# Resolve PROJECT_ROOT
_HERE = Path('.').resolve()
PROJECT_ROOT = next((p for p in [_HERE] + list(_HERE.parents) if p.name == 'Code TA'), _HERE.parent.parent.parent)
NOTEBOOKS_V2 = PROJECT_ROOT / 'notebooks'
SCRIPTS_DIR    = NOTEBOOKS_V2 / 'scripts'
RESULTS_BASE   = PROJECT_ROOT / 'results'

sys.path.insert(0, str(SCRIPTS_DIR))
from ragas_evaluator import evaluate_sample

print(f'PROJECT_ROOT : {PROJECT_ROOT}')
print(f'RESULTS_BASE : {RESULTS_BASE}')
print()
for k in ['20_hybrid_tanpa_ekspansi', '30_hybrid_dengan_ekspansi']:
    d = RESULTS_BASE / k
    print(f'[{k}]  files:')
    for p in sorted(d.glob('*.json')):
        print(f'  - {p.name}')"""))

cells.append(md("## 2. OpenAI client + generator wrapper"))

cells.append(code("""OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YOUR_OPENAI_KEY_HERE')
EVAL_MODEL = 'gpt-4.1-mini'  # model untuk RAGAS judge

client = OpenAI(api_key=OPENAI_API_KEY)


# Counter untuk track API calls
_api_call_count = [0]


def openai_generate(prompt: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
    \"\"\"Wrapper standar untuk pemanggilan OpenAI Chat Completion.\"\"\"
    _api_call_count[0] += 1
    resp = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def reset_api_counter():
    _api_call_count[0] = 0


def get_api_count():
    return _api_call_count[0]


# Smoke test
reset_api_counter()
print(openai_generate('Reply with: ok', max_tokens=5))
print(f'API calls so far: {get_api_count()}')"""))

cells.append(md("## 3. Konfigurasi target + helper fungsi dengan progress tracking"))

cells.append(code("""# Tuple (kelompok, config) — 6 target
CONFIGS = [
    ('20_hybrid_tanpa_ekspansi', 'baseline'),
    ('20_hybrid_tanpa_ekspansi', 'qr'),
    ('20_hybrid_tanpa_ekspansi', 'cr'),
    ('20_hybrid_tanpa_ekspansi', 'qr_cr'),
    ('30_hybrid_dengan_ekspansi', 'qr'),
    ('30_hybrid_dengan_ekspansi', 'qr_cr'),
]
REQUIRED_METRICS = ['faithfulness', 'context_recall', 'answer_relevancy', 'context_precision']


def format_eta(seconds):
    \"\"\"Format detik jadi human-readable.\"\"\"
    if seconds < 60:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        return f'{seconds/60:.1f}m'
    return f'{seconds/3600:.1f}h'


def run_phase2_for_config(kelompok: str, config_name: str, max_samples: int = 500, config_idx: int = 0, total_configs: int = 6):
    results_dir = RESULTS_BASE / kelompok
    phase1_path = results_dir / f'{config_name}_phase1_answers.json'
    phase2_path = results_dir / f'{config_name}_phase2_custom.json'
    label = f'{kelompok}/{config_name}'

    if not phase1_path.exists():
        print(f'[SKIP] {label}: phase1 file tidak ada')
        return None

    with open(phase1_path, 'r', encoding='utf-8') as f:
        phase1_data = json.load(f)

    samples = phase1_data.get('results', phase1_data.get('answers', []))[:max_samples]

    print()
    print('=' * 80)
    print(f'CONFIG {config_idx + 1}/{total_configs} : {label}  ({datetime.now().strftime(\"%H:%M:%S\")})')
    print('=' * 80)
    print(f'  Phase1 samples loaded: {len(samples)}')

    # Load existing phase2 untuk resume
    if phase2_path.exists():
        with open(phase2_path, 'r', encoding='utf-8') as f:
            p2 = json.load(f)
        done = {r['idx'] for r in p2.get('results', []) if all(m in r for m in REQUIRED_METRICS)}
        results = p2.get('results', [])
        print(f'  Resume mode  : {len(done)}/{len(samples)} sampel sudah lengkap')
    else:
        done, results = set(), []
        print(f'  Fresh start  : {len(samples)} sampel akan diproses')

    todo = [s for s in samples if s.get('idx') not in done]
    n_todo = len(todo)
    print(f'  Perlu eval   : {n_todo} sampel')

    if n_todo == 0:
        print(f'  [DONE] Semua sampel sudah ter-evaluasi.')
        return phase2_path

    # Reset counter API
    api_start = get_api_count()
    cfg_start = time.time()
    errors = 0

    # Progress bar dengan tqdm
    pbar = tqdm(
        todo,
        desc=f'{label[-30:]:<30}',
        unit='sample',
        ncols=120,
        bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )

    for k, sample in enumerate(pbar, 1):
        try:
            metrics = evaluate_sample(openai_generate, sample)
        except Exception as e:
            errors += 1
            pbar.set_postfix(errors=errors, api=get_api_count() - api_start)
            print(f'\\n  [ERROR] idx={sample.get(\"idx\")}: {type(e).__name__}: {str(e)[:80]}')
            continue

        row = {
            'idx': sample.get('idx'),
            'ground_truth': sample.get('ground_truth') or sample.get('final_decision'),
            'predicted_label': sample.get('predicted_label'),
            'is_correct': sample.get('is_correct'),
            **metrics,
        }
        results.append(row)

        # Update progress bar postfix dengan info berguna
        api_calls = get_api_count() - api_start
        elapsed = time.time() - cfg_start
        avg_per_sample = elapsed / k
        pbar.set_postfix(
            api=api_calls,
            avg=f'{avg_per_sample:.1f}s',
            err=errors,
        )

        # Save inkremental tiap 25 sampel
        if k % 25 == 0 or k == n_todo:
            with open(phase2_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'kelompok': kelompok,
                    'config': config_name,
                    'eval_model': EVAL_MODEL,
                    'metrics': REQUIRED_METRICS,
                    'results': results,
                }, f, indent=2)

    pbar.close()

    # Summary per-config
    cfg_elapsed = time.time() - cfg_start
    total_api = get_api_count() - api_start
    print()
    print(f'  Selesai dalam: {format_eta(cfg_elapsed)} ({cfg_elapsed:.0f}s)')
    print(f'  API calls    : {total_api} (avg {total_api/max(n_todo, 1):.1f}/sample)')
    print(f'  Errors       : {errors}')
    print(f'  Saved        : {phase2_path.name}')
    return phase2_path"""))

cells.append(md("""## 4. Run untuk semua 6 konfigurasi

Setiap konfigurasi ~500 sampel × 4 metrik × ~3 LLM calls per metrik = **~6000 LLM calls per konfigurasi**.

Estimasi total: **3-5 jam** untuk 6 konfigurasi. Progress bar akan menampilkan ETA real-time."""))

cells.append(code("""# Reset counter global
reset_api_counter()

global_start = time.time()
print(f'\\nMulai: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
print(f'Total konfigurasi: {len(CONFIGS)} | Sampel per konfigurasi: 500')
print(f'Estimasi waktu: 3-5 jam total\\n')

completed = []
for i, (kelompok, cfg) in enumerate(CONFIGS):
    result_path = run_phase2_for_config(kelompok, cfg, max_samples=500, config_idx=i, total_configs=len(CONFIGS))
    completed.append(((kelompok, cfg), result_path))

    # Update progress global setelah tiap config
    elapsed_total = time.time() - global_start
    remaining_configs = len(CONFIGS) - (i + 1)
    if i + 1 > 0:
        avg_per_config = elapsed_total / (i + 1)
        eta_total = avg_per_config * remaining_configs
        pct_done = ((i + 1) / len(CONFIGS)) * 100
        print()
        print(f'>>> Progress GLOBAL: {i+1}/{len(CONFIGS)} configs ({pct_done:.0f}%) | Elapsed: {format_eta(elapsed_total)} | ETA: {format_eta(eta_total)} <<<')

total_elapsed = time.time() - global_start
total_api = get_api_count()
print()
print('=' * 70)
print(f'[DONE] Semua {len(CONFIGS)} konfigurasi selesai dalam {format_eta(total_elapsed)} ({total_elapsed:.0f}s)')
print(f'Total API calls: {total_api}')
print(f'Selesai: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
print('=' * 70)"""))

cells.append(md("## 5. Verifikasi hasil + ringkasan"))

cells.append(code("""from statistics import mean

print(f'{\"Kelompok/Config\":<45} {\"N\":>4} {\"Acc\":>7} {\"Faith\":>7} {\"CRec\":>7} {\"ARel\":>7} {\"CPrec\":>7}')
print('-' * 90)

for kelompok, cfg in CONFIGS:
    phase2 = RESULTS_BASE / kelompok / f'{cfg}_phase2_custom.json'
    label = f'{kelompok}/{cfg}'
    if not phase2.exists():
        print(f'{label:<45} (no phase2 file)')
        continue
    with open(phase2, 'r', encoding='utf-8') as f:
        r = json.load(f).get('results', [])
    if not r:
        continue
    n = len(r)
    acc = sum(1 for x in r if x.get('is_correct')) / n
    def safe(k):
        vals = [x.get(k) for x in r if isinstance(x.get(k), (int, float))]
        return mean(vals) if vals else 0.0
    print(f'{label:<45} {n:>4} {acc:>7.4f} {safe(\"faithfulness\"):>7.4f} {safe(\"context_recall\"):>7.4f} {safe(\"answer_relevancy\"):>7.4f} {safe(\"context_precision\"):>7.4f}')"""))

# Build notebook JSON
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
