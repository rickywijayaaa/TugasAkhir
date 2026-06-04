"""Build BM25 index with Schwartz-Hearst acronym expansion.

Output: notebooks/pubmedqa_bm25_sh.pkl
Input: PubMedQA dataset (first 500 papers, matches main notebook setup)

Run once: python build_sh_index.py
"""
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from acronym_expander_sh import expand_paper, extract_pairs_schwartz_hearst

from datasets import load_dataset
from rank_bm25 import BM25Okapi


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


def tokenize_bm25(text: str) -> list:
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower()).split()


CORPUS_SIZE = 500


def build():
    print('Loading PubMedQA dataset...')
    ds_full = load_dataset('qiaojin/PubMedQA', 'pqa_labeled')['train']
    ds = ds_full.select(range(CORPUS_SIZE))
    print(f'  Corpus size: {len(ds)} papers')

    print('\nApplying Schwartz-Hearst acronym expansion...')
    documents_expanded = []
    n_papers_with_sh_match = 0
    n_sh_acronyms_detected = 0

    for i, item in enumerate(ds):
        sections = item['context']['contexts']
        labels = item['context']['labels']

        # Layer 1 SH count
        sh_dict = extract_pairs_schwartz_hearst(' '.join(sections))
        if sh_dict:
            n_papers_with_sh_match += 1
            n_sh_acronyms_detected += len(sh_dict)

        expanded_sections, _ = expand_paper(list(sections))

        for ctx, label in zip(expanded_sections, labels):
            documents_expanded.append(Document(
                text=ctx.strip(),
                pubid=str(item['pubid']),
                question=item['question'],
                section_label=label,
                answer=item['long_answer'],
                decision=item['final_decision'],
            ))

        if (i + 1) % 100 == 0:
            print(f'  Processed {i+1}/{len(ds)} papers...')

    print(f'\nSchwartz-Hearst extraction stats:')
    print(f'  Papers with in-paper SH detected : {n_papers_with_sh_match} / {len(ds)} '
          f'({100*n_papers_with_sh_match/len(ds):.1f}%)')
    print(f'  Total SH acronyms detected        : {n_sh_acronyms_detected} '
          f'(avg {n_sh_acronyms_detected/max(1,n_papers_with_sh_match):.1f} per paper)')
    print(f'  Total chunks                       : {len(documents_expanded)}')

    print('\nBuilding BM25 index...')
    tokenized = [tokenize_bm25(d.text) for d in documents_expanded]
    bm25 = BM25Okapi(tokenized)

    out_path = Path(__file__).parent.parent / 'pubmedqa_bm25_sh.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump({'bm25': bm25, 'documents': documents_expanded}, f)
    print(f'\n[Saved] {out_path}')
    print(f'  Size: {out_path.stat().st_size / 1024:.1f} KB')

    return out_path


if __name__ == '__main__':
    build()
