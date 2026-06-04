"""Test acronym expansion impact on retrieval for 3 critical samples.

Compares original BM25 index vs acronym-expanded BM25 index for:
  - Sample 1: idx 16 (PMID 11729377, GT yes) — case study
  - Sample 2 + 3: random "all-wrong" samples with acronym in source paper

Output:
  Per sample, show BM25 ranks of source paper sections (BEFORE vs AFTER expansion).
  If METHODS/RESULTS source paper jumps from outside top-50 to inside top-10,
  hypothesis confirmed.
"""
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class Document:
    text: str
    pubid: str
    question: str
    section_label: str
    answer: str
    decision: str


import __main__
__main__.Document = Document


def tokenize(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower()).split()


def load_index(path):
    with open(path, 'rb') as f:
        saved = pickle.load(f)
    return saved['bm25'], saved['documents']


def find_source_chunk_ids(docs, pubid):
    return [(i, d.section_label) for i, d in enumerate(docs) if d.pubid == pubid]


def rank_in_bm25(bm25, query, doc_ids):
    """Return dict {doc_id: (rank, score)} for given doc_ids."""
    scores = bm25.get_scores(tokenize(query))
    sorted_ids = list(np.argsort(scores)[::-1])
    out = {}
    for did in doc_ids:
        rank = sorted_ids.index(did) + 1
        out[did] = (rank, float(scores[did]))
    return out


# ============================================================
# Test cases
# ============================================================
TEST_CASES = [
    {
        'idx': 16,
        'pubid': '11729377',
        'question': 'Is there still a need for living-related liver transplantation in children?',
        'gt': 'yes',
        'reason': 'Case study utama. Original LRT/SLT acronym tidak ter-expand, METHODS rank 179, RESULTS rank 62.',
    },
    {
        'idx': 286,
        'pubid': None,  # akan diambil otomatis
        'question': None,
        'gt': None,
        'reason': '"Do symptoms predict COPD in smokers?" — COPD adalah common acronym yang harusnya ter-expand.',
    },
    {
        'idx': 269,
        'pubid': None,
        'question': None,
        'gt': None,
        'reason': '"Does obstructive sleep apnea affect aerobic fitness?" — OSA acronym kemungkinan dipakai di METHODS/RESULTS.',
    },
]


def fill_test_metadata(test_cases, docs_orig):
    """Fill question, pubid, gt for test cases that have them as None."""
    # Need to load PubMedQA dataset
    from datasets import load_dataset
    ds = load_dataset('qiaojin/PubMedQA', 'pqa_labeled')['train']
    for tc in test_cases:
        if tc['question'] is None:
            item = ds[tc['idx']]
            tc['pubid'] = str(item['pubid'])
            tc['question'] = item['question']
            tc['gt'] = item['final_decision']
    return test_cases


def main():
    here = Path(__file__).parent
    notebooks = here.parent
    print('Loading indexes...')
    bm25_orig, docs_orig = load_index(notebooks / 'pubmedqa_bm25.pkl')
    bm25_exp, docs_exp = load_index(notebooks / 'pubmedqa_bm25_expanded.pkl')
    print(f'  Original index: {len(docs_orig)} docs')
    print(f'  Expanded index: {len(docs_exp)} docs')

    print('\nFilling test case metadata...')
    test_cases = fill_test_metadata(TEST_CASES, docs_orig)

    for tc in test_cases:
        print('\n' + '=' * 90)
        print(f'TEST CASE: idx={tc["idx"]}  PMID={tc["pubid"]}  GT={tc["gt"]}')
        print(f'Q: {tc["question"]}')
        print(f'Reason: {tc["reason"]}')
        print('=' * 90)

        # Find source paper chunks in both indexes
        src_orig = find_source_chunk_ids(docs_orig, tc['pubid'])
        src_exp  = find_source_chunk_ids(docs_exp, tc['pubid'])

        if not src_orig or not src_exp:
            print(f'  Source paper not found in corpus. Skipping.')
            continue

        # Show acronym expansion for this paper
        from acronym_expander import extract_acronyms_from_text
        all_text = ' '.join(d.text for _, d in [(i, docs_orig[i]) for i, _ in src_orig])
        in_paper_acr = extract_acronyms_from_text(all_text)
        if in_paper_acr:
            print(f'\n  In-paper acronyms detected: {dict(in_paper_acr)}')
        else:
            print(f'\n  No in-paper acronyms detected (only Layer 2 curated may apply)')

        # BM25 ranks ORIGINAL
        ranks_orig = rank_in_bm25(bm25_orig, tc['question'], [i for i, _ in src_orig])
        # BM25 ranks EXPANDED
        ranks_exp  = rank_in_bm25(bm25_exp,  tc['question'], [i for i, _ in src_exp])

        print(f'\n  Source paper sections — BM25 RANK comparison:')
        print(f'  {"Section":<28} {"BM25 ORIG":>12} {"BM25 EXP":>12} {"Δ rank":>10}')
        print(f'  {"-"*28:<28} {"-"*12:>12} {"-"*12:>12} {"-"*10:>10}')

        for (i_orig, label), (i_exp, _) in zip(src_orig, src_exp):
            r_orig, s_orig = ranks_orig[i_orig]
            r_exp, s_exp = ranks_exp[i_exp]
            delta = r_orig - r_exp
            arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '·')
            print(f'  {label:<28} {r_orig:>5d} ({s_orig:>5.2f}) {r_exp:>5d} ({s_exp:>5.2f})  {arrow} {delta:+4d}')

    # Summary
    print('\n' + '=' * 90)
    print('SUMMARY')
    print('=' * 90)
    print('Hipotesis: METHODS/RESULTS source paper akan naik signifikan di BM25 expanded')
    print('Indikator sukses: rank turun dari >50 ke <20 untuk setidaknya 1 sampel')


if __name__ == '__main__':
    main()
