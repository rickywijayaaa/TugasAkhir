"""
Script untuk update 3 notebook Llama (02, 03, 04) agar:
1. Custom evaluator punya 4 metrik (+ answer_relevancy, + context_precision)
2. Phase 2 loop pakai smart resume (tidak re-compute metrik yang sudah ada)
3. Backward compatible dengan phase 2 JSON yang sudah ada (2 metrik)
"""

import json
import re
from pathlib import Path

NOTEBOOKS_DIR = Path("C:/Users/Ricky Wijaya/Documents/STI/Semester 8/TA/Code TA/notebooks")

# ============================================================
# NEW CELL CONTENTS
# ============================================================

NEW_EVAL_CELL = '''def _split_sentences(text: str) -> List[str]:
    """Pecah teks menjadi kalimat. Filter kalimat terlalu pendek (<15 char)."""
    parts = re.split(r'(?<=[.!?])\\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) >= 15]


def _llm_yes_no(prompt: str) -> bool:
    """Tanya LLM ya/tidak. Return True=yes, False=no. Fallback False jika gagal."""
    try:
        resp = ollama.generate(
            model=LLM_MODEL, prompt=prompt,
            options={'temperature': 0, 'seed': SEED, 'num_predict': 10}
        )
        return 'yes' in resp['response'].strip().lower()[:15]
    except Exception:
        return False  # konservatif: kalau gagal anggap "no"


# --- Metrik 1: Faithfulness ---
def compute_faithfulness(answer: str, contexts: List[str]) -> float:
    """Faithfulness: fraksi kalimat jawaban yang didukung konteks. 0.0-1.0, no NaN."""
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0
    ctx_text = '\\n'.join(f'[{i+1}] {c[:400]}' for i, c in enumerate(contexts))
    prompt_tmpl = (
        'Context:\\n{ctx}\\n\\n'
        'Statement: {sent}\\n\\n'
        'Is this statement directly supported by the context above? '
        'Answer with only "yes" or "no".'
    )
    supported = sum(
        1 for s in sentences
        if _llm_yes_no(prompt_tmpl.format(ctx=ctx_text, sent=s))
    )
    return supported / len(sentences)


# --- Metrik 2: Context Recall ---
def compute_context_recall(reference: str, contexts: List[str]) -> float:
    """Context Recall: fraksi fakta reference yang tercakup konteks. 0.0-1.0, no NaN."""
    sentences = _split_sentences(reference)
    if not sentences:
        return 0.0
    ctx_text = '\\n'.join(f'[{i+1}] {c[:400]}' for i, c in enumerate(contexts))
    prompt_tmpl = (
        'Context:\\n{ctx}\\n\\n'
        'Statement: {sent}\\n\\n'
        'Is this statement supported by the context above? '
        'Answer with only "yes" or "no".'
    )
    covered = sum(
        1 for s in sentences
        if _llm_yes_no(prompt_tmpl.format(ctx=ctx_text, sent=s))
    )
    return covered / len(sentences)


# --- Metrik 3: Answer Relevancy (BARU) ---
def compute_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy: fraksi kalimat jawaban yang relevan dengan pertanyaan.
    Mengukur apakah jawaban benar-benar menjawab pertanyaan (bukan off-topic).
    0.0-1.0, no NaN.
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0
    prompt_tmpl = (
        'Question: {question}\\n\\n'
        'Statement: {sent}\\n\\n'
        'Is this statement relevant to answering the question above? '
        'Answer with only "yes" or "no".'
    )
    relevant = sum(
        1 for s in sentences
        if _llm_yes_no(prompt_tmpl.format(question=question, sent=s))
    )
    return relevant / len(sentences)


# --- Metrik 4: Context Precision (BARU) ---
def compute_context_precision(question: str, contexts: List[str], reference: str) -> float:
    """
    Context Precision (Average Precision): apakah konteks relevan di rank atas?
    Formula: AP = sum(Precision@k * rel_k) / total_relevant
    0.0-1.0, no NaN.
    """
    if not contexts:
        return 0.0
    prompt_tmpl = (
        'Question: {question}\\n\\n'
        'Ground truth answer: {reference}\\n\\n'
        'Retrieved context: {ctx}\\n\\n'
        'Does this context contain information useful for correctly answering '
        'the question based on the ground truth? Answer with only "yes" or "no".'
    )
    relevance = []
    for ctx in contexts:
        is_rel = _llm_yes_no(prompt_tmpl.format(
            question=question, reference=reference[:300], ctx=ctx[:400]
        ))
        relevance.append(1 if is_rel else 0)

    total_relevant = sum(relevance)
    if total_relevant == 0:
        return 0.0

    precision_sum = 0.0
    relevant_count = 0
    for k, rel in enumerate(relevance):
        if rel:
            relevant_count += 1
            precision_at_k = relevant_count / (k + 1)
            precision_sum += precision_at_k
    return precision_sum / total_relevant


# --- Wrapper ---
def evaluate_custom(question: str, answer: str,
                    contexts: List[str], reference: str) -> Dict:
    """Wrapper evaluasi 1 sampel dengan 4 metrik. Selalu return dict tanpa NaN."""
    return {
        'faithfulness'      : compute_faithfulness(answer, contexts),
        'context_recall'    : compute_context_recall(reference, contexts),
        'answer_relevancy'  : compute_answer_relevancy(question, answer),
        'context_precision' : compute_context_precision(question, contexts, reference),
    }


# Smoke test
_test_ctx  = ['Aspirin reduces blood clotting and is used for heart attack prevention.']
_test_ans  = 'Aspirin helps prevent heart attacks. It works by reducing clotting.'
_test_ref  = 'Aspirin is used for heart attack prevention by reducing blood clotting.'
_r = evaluate_custom('Does aspirin prevent heart attacks?', _test_ans, _test_ctx, _test_ref)
print('Smoke test evaluate_custom (4 metrik):')
print(f'  faithfulness      = {_r["faithfulness"]:.3f}')
print(f'  context_recall    = {_r["context_recall"]:.3f}')
print(f'  answer_relevancy  = {_r["answer_relevancy"]:.3f}')
print(f'  context_precision = {_r["context_precision"]:.3f}')
print('Zero-NaN evaluator siap (4 metrik).')
'''


NEW_PHASE2_CELL = '''MAX_CUSTOM_SAMPLES  = 500
PHASE2_CUSTOM_PATH  = RESULTS_DIR / f'{CONFIG_NAME}_phase2_custom.json'

# Metrik yang harus ada di setiap sampel (untuk smart resume)
REQUIRED_METRICS = ['faithfulness', 'context_recall', 'answer_relevancy', 'context_precision']

with open(PHASE1_PATH, 'r', encoding='utf-8') as f:
    p1_custom = json.load(f)['results'][:MAX_CUSTOM_SAMPLES]

if PHASE2_CUSTOM_PATH.exists():
    with open(PHASE2_CUSTOM_PATH, 'r', encoding='utf-8') as f:
        p2_custom = json.load(f)['results']
    done_custom = {r['idx'] for r in p2_custom if all(m in r for m in REQUIRED_METRICS)}
    needs_upgrade = [r for r in p2_custom if not all(m in r for m in REQUIRED_METRICS)]
    print(f'Resume: {len(done_custom)}/{MAX_CUSTOM_SAMPLES} selesai dengan 4 metrik.')
    if needs_upgrade:
        print(f'Perlu upgrade: {len(needs_upgrade)} sampel (punya 2 metrik, butuh 2 metrik baru).')
else:
    p2_custom, done_custom, needs_upgrade = [], set(), []
    print(f'Mulai: {MAX_CUSTOM_SAMPLES} sampel (custom zero-NaN, 4 metrik).')

# --- TAHAP 1: Upgrade sampel yang sudah punya 2 metrik (faithfulness + context_recall) ---
if needs_upgrade:
    print(f'\\nTahap 1: Upgrade {len(needs_upgrade)} sampel ke 4 metrik (hanya hitung 2 metrik baru)...')
    t_up = time.time()
    p1_lookup = {r['idx']: r for r in p1_custom}

    for i, r in enumerate(needs_upgrade):
        src = p1_lookup[r['idx']]
        if 'answer_relevancy' not in r:
            r['answer_relevancy'] = compute_answer_relevancy(src['question'], src['answer'])
        if 'context_precision' not in r:
            r['context_precision'] = compute_context_precision(
                src['question'], src['contexts'], src['reference']
            )
        done_custom.add(r['idx'])

        if (i + 1) % 5 == 0 or i == len(needs_upgrade) - 1:
            with open(PHASE2_CUSTOM_PATH, 'w', encoding='utf-8') as f:
                json.dump({
                    'config': CONFIG_NAME, 'timestamp': datetime.now().isoformat(),
                    'max_samples': MAX_CUSTOM_SAMPLES,
                    'metrics': REQUIRED_METRICS,
                    'evaluator': 'custom_zero_nan_4metrics',
                    'results': p2_custom
                }, f, indent=2, ensure_ascii=False)
            done  = i + 1
            eta   = (time.time()-t_up)/done*(len(needs_upgrade)-done)/60 if done < len(needs_upgrade) else 0
            print(f'  upgrade [{done:3d}/{len(needs_upgrade)}] idx={r["idx"]} | '
                  f'ar={r["answer_relevancy"]:.2f} cp={r["context_precision"]:.2f} | ETA {eta:.1f} mnt')
    print(f'Tahap 1 selesai.\\n')

# --- TAHAP 2: Evaluasi sampel baru (4 metrik sekaligus) ---
remaining = [r for r in p1_custom if r['idx'] not in done_custom]
print(f'Tahap 2: Evaluasi {len(remaining)} sampel baru (4 metrik) | Estimasi ~{len(remaining)*1.0/60:.1f} jam\\n')

t0 = time.time()
for i, r in enumerate(remaining):
    scores = evaluate_custom(r['question'], r['answer'], r['contexts'], r['reference'])
    p2_custom.append({
        'idx': r['idx'], 'ground_truth': r['ground_truth'],
        'predicted_label': r['predicted_label'], 'is_correct': r['is_correct'],
        **scores
    })
    if (i + 1) % 5 == 0 or i == len(remaining) - 1:
        with open(PHASE2_CUSTOM_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                'config': CONFIG_NAME, 'timestamp': datetime.now().isoformat(),
                'max_samples': MAX_CUSTOM_SAMPLES,
                'metrics': REQUIRED_METRICS,
                'evaluator': 'custom_zero_nan_4metrics',
                'results': p2_custom
            }, f, indent=2, ensure_ascii=False)
        done  = i + 1
        total = len(remaining)
        eta   = (time.time()-t0)/done*(total-done)/60 if done < total else 0
        avg_f  = sum(x['faithfulness']      for x in p2_custom) / len(p2_custom)
        avg_cr = sum(x['context_recall']    for x in p2_custom) / len(p2_custom)
        avg_ar = sum(x['answer_relevancy']  for x in p2_custom) / len(p2_custom)
        avg_cp = sum(x['context_precision'] for x in p2_custom) / len(p2_custom)
        print(f'  [{done:3d}/{total}] idx={r["idx"]} | '
              f'f={scores["faithfulness"]:.2f} cr={scores["context_recall"]:.2f} '
              f'ar={scores["answer_relevancy"]:.2f} cp={scores["context_precision"]:.2f} | '
              f'avg: f={avg_f:.3f} cr={avg_cr:.3f} ar={avg_ar:.3f} cp={avg_cp:.3f} | ETA {eta:.1f} mnt')

print(f'\\nSelesai! -> {PHASE2_CUSTOM_PATH}')

# --- Ringkasan akhir ---
n   = len(p2_custom)
acc = sum(r['is_correct'] for r in p2_custom) / n
avg_f  = sum(r['faithfulness']      for r in p2_custom) / n
avg_cr = sum(r['context_recall']    for r in p2_custom) / n
avg_ar = sum(r['answer_relevancy']  for r in p2_custom) / n
avg_cp = sum(r['context_precision'] for r in p2_custom) / n
print(f'\\nRingkasan ({n} sampel, zero-NaN evaluator, 4 metrik):')
print(f'  Label Accuracy    : {acc:.1%}')
print(f'  Faithfulness      : {avg_f:.4f}')
print(f'  Context Recall    : {avg_cr:.4f}')
print(f'  Answer Relevancy  : {avg_ar:.4f}')
print(f'  Context Precision : {avg_cp:.4f}')
print(f'\\nBaris tabel skripsi:')
print(f'  | {CONFIG_NAME.upper()} | {acc:.3f} | {avg_f:.3f} | {avg_cr:.3f} | {avg_ar:.3f} | {avg_cp:.3f} |')
'''


# ============================================================
# UPDATE FUNCTION
# ============================================================

def find_cell_index(nb, predicate):
    """Return index of first cell matching predicate."""
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell.get('source', []))
            if predicate(src):
                return i
    return None


def source_to_list(content):
    """Convert string to ipynb source format (list of lines with \\n endings)."""
    lines = content.split('\n')
    # Each line ends with \n except the last (if no trailing newline)
    result = [line + '\n' for line in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    return result


def update_notebook(nb_path):
    print(f'\n{"="*70}')
    print(f'Updating: {nb_path.name}')
    print(f'{"="*70}')

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find evaluator cell (has def compute_faithfulness and def evaluate_custom)
    eval_idx = find_cell_index(
        nb,
        lambda s: 'def compute_faithfulness' in s and 'def evaluate_custom' in s
    )
    if eval_idx is None:
        print('  WARN: Evaluator cell not found!')
        return

    # Find phase 2 loop cell (uses MAX_CUSTOM_SAMPLES and has p2_custom.append)
    phase2_idx = find_cell_index(
        nb,
        lambda s: 'MAX_CUSTOM_SAMPLES' in s and 'p2_custom.append' in s
    )
    if phase2_idx is None:
        print('  WARN: Phase 2 loop cell not found!')
        return

    print(f'  Found: eval=cell[{eval_idx}], phase2=cell[{phase2_idx}]')

    # Replace evaluator cell
    nb['cells'][eval_idx]['source'] = source_to_list(NEW_EVAL_CELL)
    nb['cells'][eval_idx]['outputs'] = []
    nb['cells'][eval_idx]['execution_count'] = None
    print(f'  Updated evaluator cell ({eval_idx})')

    # Replace phase 2 cell
    nb['cells'][phase2_idx]['source'] = source_to_list(NEW_PHASE2_CELL)
    nb['cells'][phase2_idx]['outputs'] = []
    nb['cells'][phase2_idx]['execution_count'] = None
    print(f'  Updated phase 2 cell ({phase2_idx})')

    # Save
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'  Saved to {nb_path}')


def main():
    notebooks = [
        NOTEBOOKS_DIR / '02-RAG-Baseline-Evaluation.ipynb',
        NOTEBOOKS_DIR / '03-RAG-QueryRewriting.ipynb',
        NOTEBOOKS_DIR / '04-RAG-ContextReranking.ipynb',
    ]
    for nb in notebooks:
        update_notebook(nb)
    print('\nALL DONE!')


if __name__ == '__main__':
    main()
