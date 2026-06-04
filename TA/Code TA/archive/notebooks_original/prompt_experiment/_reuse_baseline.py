"""Adapter: reuse existing baseline_openai results as P0_baseline results.

The baseline prompt in the main notebook IS identical to P0 in prompts.py,
so re-running P0 is wasted API calls. This script copies existing results
into the prompt_experiment/ folder with the new filename convention.

Usage:
    python _reuse_baseline.py [N_SAMPLES]

Default N_SAMPLES=100 (pilot). Set to 500 for full run.
"""

import json
import sys
from pathlib import Path

N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 100

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent if HERE.parent.name == 'notebooks' else HERE.parent
RESULTS_MAIN = PROJECT_ROOT / 'results'
RESULTS_PE   = PROJECT_ROOT / 'results' / 'prompt_experiment'
RESULTS_PE.mkdir(parents=True, exist_ok=True)

# (source_file_in_results, target_file_in_prompt_experiment)
COPIES = [
    # P0 baseline (no QR, no CR) <- baseline_openai
    ('baseline_openai_phase1_answers.json', 'P0_baseline__baseline_openai_phase1.json', 'baseline'),
    ('baseline_openai_phase2_custom.json',  'P0_baseline__baseline_openai_phase2.json', 'baseline'),
    # P0 + QR  <- qr_openai
    ('qr_openai_phase1_answers.json',       'P0_baseline__qr_openai_phase1.json', 'qr'),
    ('qr_openai_phase2_custom.json',        'P0_baseline__qr_openai_phase2.json', 'qr'),
    # P0 + CR  <- cr_openai
    ('cr_openai_phase1_answers.json',       'P0_baseline__cr_openai_phase1.json', 'cr'),
    ('cr_openai_phase2_custom.json',        'P0_baseline__cr_openai_phase2.json', 'cr'),
    # P0 + QR+CR  <- qr_cr_openai
    ('qr_cr_openai_phase1_answers.json',    'P0_baseline__qr_cr_openai_phase1.json', 'qr_cr'),
    ('qr_cr_openai_phase2_custom.json',     'P0_baseline__qr_cr_openai_phase2.json', 'qr_cr'),
]


def adapt(src: Path, dst: Path, method: str, n: int):
    if not src.exists():
        print(f'  SKIP (missing source): {src.name}')
        return False
    with open(src, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data['results'][:n]
    # Inject prompt_id + method metadata for the analysis notebook
    out = {
        'prompt_id'  : 'P0_baseline',
        'method'     : method,
        'llm_model'  : data.get('llm_model', 'gpt-4.1-mini'),
        'timestamp'  : data.get('timestamp'),
        'max_samples': n,
        'source_file': src.name,
        'metrics'    : data.get('metrics'),
        'evaluator'  : data.get('evaluator'),
        'results'    : results,
    }
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'  OK: {src.name} -> {dst.name} ({len(results)} samples)')
    return True


print(f'Reusing existing baseline results as P0 (n={N_SAMPLES} samples)\n')
print(f'  Source: {RESULTS_MAIN}')
print(f'  Target: {RESULTS_PE}\n')

count = 0
for src_name, dst_name, method in COPIES:
    if adapt(RESULTS_MAIN / src_name, RESULTS_PE / dst_name, method, N_SAMPLES):
        count += 1

print(f'\nDone. {count}/{len(COPIES)} files adapted.')
print('You can now SKIP running 01_P0_baseline.ipynb.')
