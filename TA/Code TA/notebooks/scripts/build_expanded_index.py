"""Build new BM25 index with acronym expansion applied.

Output: notebooks/pubmedqa_bm25_expanded.pkl
Original index untouched at notebooks/pubmedqa_bm25.pkl

Usage:  python build_expanded_index.py
"""
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Make sure same directory in path
sys.path.insert(0, str(Path(__file__).parent))
from acronym_expander import expand_paper

from datasets import load_dataset
from rank_bm25 import BM25Okapi


# Same Document dataclass used by main notebooks (pickle compat)
@dataclass
class Document:
    text: str
    pubid: str
    question: str
    section_label: str
    answer: str
    decision: str


# Inject Document to __main__ for pickle compatibility
import __main__
__main__.Document = Document


def tokenize_bm25(text: str) -> list:
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower()).split()


CORPUS_SIZE = 500   # match original main notebook setup


def build():
    from acronym_expander import extract_acronyms_from_text

    print('Loading PubMedQA dataset...')
    ds_full = load_dataset('qiaojin/PubMedQA', 'pqa_labeled')['train']
    ds = ds_full.select(range(CORPUS_SIZE))
    print(f'  Corpus size: {len(ds)} papers (first {CORPUS_SIZE} to match main notebook)')

    print('\nApplying acronym expansion to each paper...')
    documents_expanded = []
    n_papers_with_in_paper = 0
    n_in_paper_acronyms = 0
    n_curated_used = 0  # number of papers where curated dict was applied

    for i, item in enumerate(ds):
        sections = item['context']['contexts']
        labels = item['context']['labels']

        # Layer 1 only count
        in_paper_dict = extract_acronyms_from_text(' '.join(sections))
        if in_paper_dict:
            n_papers_with_in_paper += 1
            n_in_paper_acronyms += len(in_paper_dict)

        expanded_sections, full_dict = expand_paper(list(sections))
        if len(full_dict) > len(in_paper_dict):
            n_curated_used += 1

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

    print(f'\nExpansion stats:')
    print(f'  Papers with in-paper acronym detected : {n_papers_with_in_paper} / {len(ds)} '
          f'({100*n_papers_with_in_paper/len(ds):.1f}%)')
    print(f'  Total in-paper acronyms detected       : {n_in_paper_acronyms} '
          f'(avg {n_in_paper_acronyms/max(1,n_papers_with_in_paper):.1f} per paper)')
    print(f'  Papers using curated dict (any layer)  : {n_curated_used} / {len(ds)}')
    print(f'  Total chunks                            : {len(documents_expanded)}')

    print('\nBuilding BM25 index...')
    tokenized = [tokenize_bm25(d.text) for d in documents_expanded]
    bm25 = BM25Okapi(tokenized)

    out_path = Path(__file__).parent.parent / 'pubmedqa_bm25_expanded.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump({'bm25': bm25, 'documents': documents_expanded}, f)
    print(f'\n[Saved] {out_path}')
    print(f'  Size: {out_path.stat().st_size / 1024:.1f} KB')

    return out_path


if __name__ == '__main__':
    build()
