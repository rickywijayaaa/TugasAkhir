"""
Extract 5 interesting sample from PubMedQA results untuk UI visualization.

Strategy pemilihan sampel:
- Pilih sampel di mana beberapa config memberikan jawaban berbeda (menarik untuk perbandingan)
- Pastikan ada variasi ground truth (yes, no, maybe)
- Pastikan ada variasi kesulitan (ada yang mudah, ada yang hard)

Output: data JSON untuk realData.js
"""

import json
from pathlib import Path

RESULTS = Path('C:/Users/Ricky Wijaya/Documents/STI/Semester 8/TA/Code TA/results')

CONFIGS = [
    ('baseline_llama',  'baseline_phase1_answers.json',           'baseline_phase2_custom.json'),
    ('qr_llama',        'qr_phase1_answers.json',                 'qr_phase2_custom.json'),
    ('cr_llama',        'cr_phase1_answers.json',                 'cr_phase2_custom.json'),
    ('baseline_openai', 'baseline_openai_phase1_answers.json',    'baseline_openai_phase2_custom.json'),
    ('qr_openai',       'qr_openai_phase1_answers.json',          'qr_openai_phase2_custom.json'),
    ('cr_openai',       'cr_openai_phase1_answers.json',          'cr_openai_phase2_custom.json'),
    ('qr_cr_openai',    'combined_openai_phase1_answers.json',    'combined_openai_phase2_custom.json'),
    ('hybrid_openai',   'hybrid_openai_phase1_answers.json',      'hybrid_openai_phase2_custom.json'),
    ('hybrid_cr_openai','hybrid_cr_openai_phase1_answers.json',   'hybrid_cr_openai_phase2_custom.json'),
]

# Load all phase 1 + phase 2 data
p1_data = {}  # config -> {idx: sample}
p2_data = {}  # config -> {idx: metrics}

for cfg_name, p1_file, p2_file in CONFIGS:
    with open(RESULTS / p1_file, encoding='utf-8') as f:
        p1_data[cfg_name] = {r['idx']: r for r in json.load(f)['results']}
    try:
        with open(RESULTS / p2_file, encoding='utf-8') as f:
            p2_data[cfg_name] = {r['idx']: r for r in json.load(f)['results']}
    except FileNotFoundError:
        p2_data[cfg_name] = {}


# Find interesting sample indices
# Strategy: look at baseline_llama vs hybrid_cr_openai - find indices where they differ in correctness
# Also ensure diversity of ground truth labels

def pick_interesting_samples():
    # Get all 500 indices
    all_idx = sorted(p1_data['baseline_openai'].keys())

    candidates = []
    for idx in all_idx:
        bl_llama = p1_data['baseline_llama'].get(idx)
        hybrid_cr = p1_data['hybrid_cr_openai'].get(idx)
        if not bl_llama or not hybrid_cr:
            continue

        # Check if correctness differs
        bl_correct = bl_llama['is_correct']
        hy_correct = hybrid_cr['is_correct']

        # Count across configs how many get it right
        correct_count = sum(1 for cfg in ['baseline_openai','qr_openai','cr_openai','qr_cr_openai','hybrid_openai','hybrid_cr_openai']
                            if p1_data[cfg].get(idx, {}).get('is_correct', False))

        # Track different ground truths
        gt = bl_llama['ground_truth']

        candidates.append({
            'idx': idx,
            'gt': gt,
            'bl_llama_correct': bl_correct,
            'hybrid_cr_correct': hy_correct,
            'openai_correct_count': correct_count,
            'question_len': len(bl_llama['question']),
        })

    # Diverse selection strategy:
    # 1. One where baseline llama wrong, hybrid_cr right (shows improvement)
    # 2. One 'yes' label where most configs agree correctly
    # 3. One 'no' label (hard for llama)
    # 4. One 'maybe' label (hard for all)
    # 5. One where configs disagree (interesting variance)

    selected = []

    # 1. Pick a 'yes' sample that shows improvement (BL llama wrong -> Hybrid+CR right)
    for c in candidates:
        if c['gt'] == 'yes' and not c['bl_llama_correct'] and c['hybrid_cr_correct'] and c['openai_correct_count'] >= 5:
            selected.append(c)
            break

    # 2. Pick a 'no' sample (hard for llama, shows OpenAI's negative reasoning)
    for c in candidates:
        if c['gt'] == 'no' and not c['bl_llama_correct'] and c['hybrid_cr_correct'] and c['idx'] not in [s['idx'] for s in selected]:
            selected.append(c)
            break

    # 3. Pick a 'maybe' sample
    for c in candidates:
        if c['gt'] == 'maybe' and c['idx'] not in [s['idx'] for s in selected]:
            if c['hybrid_cr_correct'] or c['qr_cr_openai' if False else 'openai_correct_count'] >= 2:
                selected.append(c)
                break

    # 4. Pick a 'yes' where all configs correct (easy case)
    for c in candidates:
        if c['gt'] == 'yes' and c['bl_llama_correct'] and c['openai_correct_count'] == 6 and c['idx'] not in [s['idx'] for s in selected]:
            selected.append(c)
            break

    # 5. Pick a disagreement case (variance)
    for c in candidates:
        if c['openai_correct_count'] in [2, 3] and c['idx'] not in [s['idx'] for s in selected]:
            selected.append(c)
            break

    return [s['idx'] for s in selected[:5]]


selected_idx = pick_interesting_samples()
print(f'Selected sample indices: {selected_idx}\n')

# Build UI data structure
ui_samples = []
for sample_idx in selected_idx:
    sample_data = {
        'idx': sample_idx,
    }

    # Get question info from any config (they're all same)
    base = p1_data['baseline_openai'][sample_idx]
    sample_data['question'] = base['question']
    sample_data['ground_truth'] = base['ground_truth']
    sample_data['pubid'] = base['pubid']
    sample_data['reference'] = base['reference']

    # Get rewritten query (from QR config)
    qr_sample = p1_data['qr_openai'].get(sample_idx, {})
    sample_data['rewritten_query'] = qr_sample.get('rewritten_query') or qr_sample.get('question', '')

    # Collect per-config data
    sample_data['configs'] = {}
    for cfg_name, _, _ in CONFIGS:
        p1 = p1_data[cfg_name].get(sample_idx, {})
        p2 = p2_data[cfg_name].get(sample_idx, {})

        cfg_data = {
            'label': p1.get('predicted_label'),
            'is_correct': p1.get('is_correct'),
            'answer': p1.get('answer'),
            'faithfulness': p2.get('faithfulness'),
            'context_recall': p2.get('context_recall'),
            'answer_relevancy': p2.get('answer_relevancy'),
            'context_precision': p2.get('context_precision'),
        }
        sample_data['configs'][cfg_name] = cfg_data

    # Retrieved documents (use hybrid_cr as representative for all retrieval scores)
    base_config = 'hybrid_cr_openai' if sample_idx in p1_data['hybrid_cr_openai'] else 'baseline_openai'
    base_retrieval = p1_data[base_config].get(sample_idx, {})

    contexts = base_retrieval.get('contexts', [])
    bm25_scores = base_retrieval.get('retrieval_scores', [])
    dense_scores = base_retrieval.get('dense_scores', [])
    rrf_scores = base_retrieval.get('rrf_scores', [])
    reranker_scores = base_retrieval.get('reranker_scores', [])

    docs = []
    for i, ctx in enumerate(contexts[:5]):
        docs.append({
            'id': f'd{sample_idx}_{i+1}',
            'content': ctx[:500] + ('...' if len(ctx) > 500 else ''),
            'full_content': ctx,
            'bm25_score': bm25_scores[i] if i < len(bm25_scores) else None,
            'dense_score': dense_scores[i] if i < len(dense_scores) else None,
            'rrf_score': rrf_scores[i] if i < len(rrf_scores) else None,
            'reranker_score': reranker_scores[i] if i < len(reranker_scores) else None,
        })
    sample_data['retrieved_docs'] = docs

    ui_samples.append(sample_data)

# Save to JSON
output_path = Path('C:/Users/Ricky Wijaya/Documents/STI/Semester 8/TA/Code TA/ui/src/data/_samples.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(ui_samples, f, indent=2, ensure_ascii=False)

print(f'Saved {len(ui_samples)} samples to {output_path}')
print('\n=== Summary ===')
for i, s in enumerate(ui_samples, 1):
    print(f'\n#{i} idx={s["idx"]} GT={s["ground_truth"]}')
    print(f'   Q: {s["question"][:80]}...')
    print(f'   Predictions by config:')
    for cfg in ['baseline_llama','baseline_openai','cr_openai','qr_cr_openai','hybrid_openai','hybrid_cr_openai']:
        c = s['configs'][cfg]
        correct = '[OK]' if c['is_correct'] else '[X]'
        print(f'     {cfg:<20}: {c["label"]} {correct}')
