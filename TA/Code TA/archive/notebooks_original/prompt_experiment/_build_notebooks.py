"""Generates the 6 experiment notebooks. Run once: `python _build_notebooks.py`.

Each prompt notebook is a thin driver:
  1. Import shared + prompts.
  2. Set PROMPT_ID, MAX_SAMPLES, METHODS (toggles between pilot and full run).
  3. Run Phase 1 (generation) for each method.
  4. Run Phase 2 (evaluation) for each method.
  5. Print summary table.

The analysis notebook loads ALL phase2 results and produces a comparison table.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(*lines):
    return {'cell_type': 'markdown', 'metadata': {},
            'source': [l + '\n' for l in lines]}


def code(*lines):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': [l + '\n' for l in lines]}


def nb(cells):
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


# ---------------- prompt notebooks (5 of them) ----------------

PROMPT_NB_CONFIGS = [
    ('01_P0_baseline.ipynb',
     'P0_baseline', 'P0 - Baseline (current prompt)',
     'Baseline / control prompt. Same as the one used in main experiments. '
     'Serves as the reference for comparing all other prompt variants.'),

    ('02_P1_strict.ipynb',
     'P1_strict_grounding', 'P1 - Strict Grounding + Balanced Maybe',
     'Adds explicit grounding rules and rewrites the "maybe" rubric to remove '
     'the anti-maybe bias. Hypothesis: improves maybe accuracy without hurting '
     'yes/no accuracy.'),

    ('03_P2_cot.ipynb',
     'P2_zeroshot_cot', 'P2 - Zero-shot Chain-of-Thought',
     'Forces explicit step-by-step reasoning (Evidence -> Direction -> Strength '
     '-> Decision) before the final yes/no/maybe. Hypothesis: improves '
     'faithfulness and reduces confident wrong answers.'),

    ('04_P3_fewshot.ipynb',
     'P3_fewshot_cot', 'P3 - Few-shot Chain-of-Thought',
     'Same reasoning approach as P2, but anchored with 3 demonstrations '
     '(1 yes, 1 no, 1 maybe with subgroup-dependent finding). Hypothesis: '
     'format consistency and "maybe" handling improve via demonstrations.'),

    ('05_P4_medcot.ipynb',
     'P4_clinical_persona_cot', 'P4 - Clinical Persona + Structured Causal CoT',
     'Adapted from MedCoT-RAG (Wang et al., 2024). Adds a clinical evidence '
     'reviewer persona plus a 4-stage workflow: Identify Findings -> Assess '
     'Quality -> Weigh Evidence -> Decide. Hypothesis: gives the best '
     'faithfulness, esp. when combined with Context Reranking.'),
]


def build_prompt_notebook(filename, prompt_id, short_title, description):
    cells = [
        md(f'# {short_title}',
           '',
           f'**Prompt ID:** `{prompt_id}`  ',
           f'**LLM:** `gpt-4.1-mini` (OpenAI)  ',
           f'**Embedder:** `text-embedding-3-small`',
           '',
           description,
           '',
           '## Pilot vs Full mode',
           '',
           'By default this notebook runs in **pilot mode** to keep API cost low:',
           '- `MAX_SAMPLES = 100` (vs 500 in main experiments)',
           '- `METHODS = ["baseline"]` (vs all 4 in main experiments)',
           '',
           'After reviewing pilot results across all 5 prompts in the analysis '
           'notebook (`06_analysis.ipynb`), increase to full run by changing the '
           'two variables in the next cell.'),

        code('# ============================================================',
             '# EXPERIMENT CONFIG - adjust here',
             '# ============================================================',
             f"PROMPT_ID    = '{prompt_id}'",
             "",
             "# Pilot mode (cheap, fast)",
             "MAX_SAMPLES  = 100",
             "METHODS      = ['baseline']",
             "",
             "# Full run (uncomment when ready):",
             "# MAX_SAMPLES = 500",
             "# METHODS     = ['baseline', 'qr', 'cr', 'qr_cr']"),

        code('# ============================================================',
             '# Setup - import shared utilities + the chosen prompt',
             '# ============================================================',
             'import sys',
             'from pathlib import Path',
             '',
             '# Make sure the prompt_experiment folder is on sys.path',
             "sys.path.insert(0, str(Path('.').resolve()))",
             '',
             'import shared',
             'from prompts import PROMPTS, PROMPT_MAX_TOKENS',
             '',
             'PROMPT_TEMPLATE = PROMPTS[PROMPT_ID]',
             'MAX_TOKENS      = PROMPT_MAX_TOKENS[PROMPT_ID]',
             '',
             f"print(f'Prompt    : {{PROMPT_ID}}')",
             "print(f'Length    : {len(PROMPT_TEMPLATE)} chars')",
             "print(f'Max tokens: {MAX_TOKENS}')",
             "print(f'Samples   : {MAX_SAMPLES}')",
             "print(f'Methods   : {METHODS}')"),

        code('# Load PubMedQA, BM25 index, Chroma collection, CrossEncoder',
             '# (uses indexes already built by the main notebooks)',
             'shared.load_everything(max_samples=MAX_SAMPLES)'),

        md('## Smoke test'),

        code("# Smoke test on 1 sample to make sure the pipeline works end-to-end",
             "_data, *_ = shared.load_everything(max_samples=MAX_SAMPLES)",
             "_s   = _data[0]",
             "_q   = _s['question']",
             "_gt  = _s['final_decision']",
             "_ret, _used_q = shared.run_baseline(_q)",
             "_ans = shared.generate_answer(_q, _ret, PROMPT_TEMPLATE, max_tokens=MAX_TOKENS)",
             "_pred = shared.extract_label(_ans)",
             "print(f'Q: {_q}')",
             "print(f'GT: {_gt} | Pred: {_pred} | Match: {_gt == _pred}')",
             "print(f'Answer preview:')",
             "print(_ans[:500])"),

        md('## Phase 1 - generate answers for each method'),

        code("phase1_paths = {}",
             "for method in METHODS:",
             "    print(f'\\n=== Phase 1: prompt={PROMPT_ID}, method={method} ===')",
             "    p1 = shared.run_phase1(",
             "        prompt_id=PROMPT_ID,",
             "        prompt_template=PROMPT_TEMPLATE,",
             "        max_tokens=MAX_TOKENS,",
             "        method=method,",
             "        max_samples=MAX_SAMPLES,",
             "    )",
             "    phase1_paths[method] = p1"),

        md('## Phase 2 - evaluate (4 metrics)'),

        code("phase2_paths = {}",
             "for method, p1_path in phase1_paths.items():",
             "    print(f'\\n=== Phase 2: prompt={PROMPT_ID}, method={method} ===')",
             "    p2 = shared.run_phase2(p1_path, max_samples=MAX_SAMPLES)",
             "    phase2_paths[method] = p2"),

        md('## Summary'),

        code("print('=' * 80)",
             f"print(f'PROMPT: {{PROMPT_ID}}    (n={{MAX_SAMPLES}} per method)')",
             "print('=' * 80)",
             "for method, p2_path in phase2_paths.items():",
             "    shared.print_summary(p2_path)"),
    ]
    out = HERE / filename
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(nb(cells), f, indent=2, ensure_ascii=False)
    print(f'  wrote {out}')


# ---------------- analysis notebook ----------------

def build_analysis_notebook():
    cells = [
        md('# 06 - Prompt Experiment Analysis',
           '',
           'Load every `*_phase2.json` produced by notebooks 01-05 and produce '
           'a comparison table + simple visualisations across `prompt x method`.',
           '',
           '**Run notebooks 01-05 first**, in pilot or full mode.'),

        code("import json",
             "from pathlib import Path",
             "import pandas as pd",
             "",
             "RESULTS_DIR = Path('../results/prompt_experiment')",
             "files = sorted(RESULTS_DIR.glob('*_phase2.json'))",
             "print(f'Found {len(files)} phase2 files')",
             "for f in files: print('  ', f.name)"),

        code("def load_one(path):",
             "    with open(path, 'r', encoding='utf-8') as f:",
             "        data = json.load(f)",
             "    r = data['results']",
             "    if not r: return None",
             "    n = len(r)",
             "    return {",
             "        'file'             : path.stem,",
             "        'prompt_id'        : data.get('prompt_id'),",
             "        'method'           : data.get('method'),",
             "        'n'                : n,",
             "        'accuracy'         : sum(x['is_correct']        for x in r) / n,",
             "        'faithfulness'     : sum(x['faithfulness']      for x in r) / n,",
             "        'context_recall'   : sum(x['context_recall']    for x in r) / n,",
             "        'answer_relevancy' : sum(x['answer_relevancy']  for x in r) / n,",
             "        'context_precision': sum(x['context_precision'] for x in r) / n,",
             "    }",
             "",
             "rows = [load_one(p) for p in files]",
             "rows = [r for r in rows if r]",
             "df = pd.DataFrame(rows)",
             "df = df.sort_values(['prompt_id', 'method']).reset_index(drop=True)",
             "df"),

        md('## Pivot - prompt vs method, per metric'),

        code("for metric in ['accuracy', 'faithfulness', 'context_recall',",
             "               'answer_relevancy', 'context_precision']:",
             "    print(f'\\n=== {metric} ===')",
             "    pivot = df.pivot(index='prompt_id', columns='method', values=metric)",
             "    print(pivot.to_string(float_format=lambda x: f'{x:.3f}'))"),

        md('## Per-label accuracy (yes / no / maybe)',
           '',
           'Per-label accuracy is critical because the "maybe" class is heavily '
           'underrepresented in the baseline prompt (only ~1% predictions).'),

        code("per_label_rows = []",
             "for path in files:",
             "    with open(path, 'r', encoding='utf-8') as f:",
             "        data = json.load(f)",
             "    r = data['results']",
             "    if not r: continue",
             "    row = {'prompt_id': data.get('prompt_id'), 'method': data.get('method')}",
             "    for lbl in ['yes', 'no', 'maybe']:",
             "        sub = [x for x in r if x['ground_truth'] == lbl]",
             "        row[f'acc_{lbl}'] = sum(x['is_correct'] for x in sub) / len(sub) if sub else float('nan')",
             "        row[f'pred_{lbl}_count'] = sum(1 for x in r if x['predicted_label'] == lbl)",
             "    per_label_rows.append(row)",
             "",
             "per_label = pd.DataFrame(per_label_rows).sort_values(['prompt_id', 'method'])",
             "per_label.reset_index(drop=True)"),

        md('## Quick visual - bar chart of label accuracy per prompt (baseline method only)'),

        code("import matplotlib.pyplot as plt",
             "",
             "sub = per_label[per_label['method'] == 'baseline'].copy()",
             "if not sub.empty:",
             "    fig, ax = plt.subplots(figsize=(10, 5))",
             "    x = range(len(sub))",
             "    width = 0.25",
             "    ax.bar([i - width for i in x], sub['acc_yes'],   width, label='yes')",
             "    ax.bar(x,                       sub['acc_no'],    width, label='no')",
             "    ax.bar([i + width for i in x], sub['acc_maybe'], width, label='maybe')",
             "    ax.set_xticks(list(x))",
             "    ax.set_xticklabels(sub['prompt_id'].tolist(), rotation=20, ha='right')",
             "    ax.set_ylabel('Per-label accuracy')",
             "    ax.set_title('Per-label accuracy by prompt (baseline method)')",
             "    ax.legend()",
             "    ax.set_ylim(0, 1)",
             "    plt.tight_layout()",
             "    plt.show()",
             "else:",
             "    print('No baseline-method results found. Run at least one prompt notebook first.')"),

        md('## Recommendation generator',
           '',
           'Picks the best prompt per metric, and the best prompt for "maybe" '
           'class specifically (where current baseline is weakest).'),

        code("if not df.empty:",
             "    print('Best prompt per metric (across all methods):')",
             "    for metric in ['accuracy', 'faithfulness', 'context_recall',",
             "                   'answer_relevancy', 'context_precision']:",
             "        idx = df[metric].idxmax()",
             "        row = df.iloc[idx]",
             "        print(f'  {metric:<20} : {row[\"prompt_id\"]} on {row[\"method\"]} = {row[metric]:.3f}')",
             "",
             "    if not per_label.empty:",
             "        print('\\nBest prompt for \"maybe\" class (any method):')",
             "        idx = per_label['acc_maybe'].idxmax()",
             "        row = per_label.iloc[idx]",
             "        print(f'  {row[\"prompt_id\"]} on {row[\"method\"]} : acc_maybe = {row[\"acc_maybe\"]:.3f}')"),
    ]
    out = HERE / '06_analysis.ipynb'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(nb(cells), f, indent=2, ensure_ascii=False)
    print(f'  wrote {out}')


if __name__ == '__main__':
    print('Building prompt experiment notebooks...')
    for fn, pid, title, desc in PROMPT_NB_CONFIGS:
        build_prompt_notebook(fn, pid, title, desc)
    build_analysis_notebook()
    print('\nDone. Files in:', HERE)
