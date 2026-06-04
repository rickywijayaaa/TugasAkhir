"""
Retrieval metrics module — Recall@K, MAP@K, nDCG@K, MRR, Hit@K, Precision@K.

Konsisten dengan evaluate_retrieval.py lama. Relevance criterion:
chunk relevan jika pubid == source paper pubid (per PubMedQA Jin et al. 2019).
"""
import json
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict


@dataclass
class Document:
    text: str
    pubid: str
    question: str
    section_label: str
    answer: str
    decision: str


# Hint untuk pickle: kelas Document tersedia di __main__
import __main__
__main__.Document = Document


def load_corpus_documents(pkl_path: Path) -> List[Document]:
    """Load list[Document] dari BM25 pickle index."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    # data biasanya tuple (bm25_index, documents) atau dict
    if isinstance(data, tuple) and len(data) >= 2:
        return data[1]
    if isinstance(data, dict):
        return data.get('documents') or data.get('docs') or []
    return data


def build_text_to_meta_map(documents: List[Document], prefix_len: int = 80) -> Dict[str, tuple]:
    """text[:prefix_len] -> (pubid, section_label)"""
    return {d.text[:prefix_len]: (d.pubid, d.section_label) for d in documents}


def build_source_meta(documents: List[Document]) -> Dict[str, list]:
    """pubid -> list of section labels yang tersedia di korpus untuk paper itu."""
    out = defaultdict(list)
    for d in documents:
        out[d.pubid].append(d.section_label)
    return dict(out)


def derive_pubids_sections(contexts: List[str], text_map: Dict[str, tuple], prefix_len: int = 80):
    pubids, sections = [], []
    for ctx in contexts:
        meta = text_map.get(ctx[:prefix_len])
        if meta:
            pubids.append(meta[0])
            sections.append(meta[1])
        else:
            pubids.append('?')
            sections.append('?')
    return pubids, sections


def compute_metrics(retrieved_pubids: List[str], retrieved_sections: List[str],
                    source_pubid: str, n_source_sections: int) -> dict:
    k = len(retrieved_pubids)
    rel = [1 if p == source_pubid else 0 for p in retrieved_pubids]
    n_rel = sum(rel)

    hit = 1 if n_rel > 0 else 0
    precision = n_rel / k if k else 0.0
    recall = min(n_rel / max(n_source_sections, 1), 1.0)

    mrr = 0.0
    for i, r in enumerate(rel):
        if r == 1:
            mrr = 1.0 / (i + 1)
            break

    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    n_ideal = min(n_source_sections, k)
    idcg = sum(1 / math.log2(i + 2) for i in range(n_ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    if n_rel == 0:
        map_score = 0.0
    else:
        cum_rel = 0
        prec_at_rel = []
        for i, r in enumerate(rel):
            if r == 1:
                cum_rel += 1
                prec_at_rel.append(cum_rel / (i + 1))
        map_score = sum(prec_at_rel) / n_source_sections

    return {
        'hit@5': hit,
        'precision@5': precision,
        'recall@5': recall,
        'mrr': mrr,
        'ndcg@5': ndcg,
        'map@5': map_score,
        'n_relevant_in_topK': n_rel,
    }


def aggregate(per_query: List[dict]) -> dict:
    n = len(per_query)
    if n == 0:
        return {}
    return {
        'hit@5': sum(m['hit@5'] for m in per_query) / n,
        'precision@5': sum(m['precision@5'] for m in per_query) / n,
        'recall@5': sum(m['recall@5'] for m in per_query) / n,
        'mrr': sum(m['mrr'] for m in per_query) / n,
        'ndcg@5': sum(m['ndcg@5'] for m in per_query) / n,
        'map@5': sum(m['map@5'] for m in per_query) / n,
    }


def evaluate_phase1(phase1_results: List[dict], text_map: Dict[str, tuple],
                    source_meta: Dict[str, list]) -> dict:
    """Evaluate satu file phase1 → return aggregate + per_query."""
    per_query = []
    for r in phase1_results:
        target_pubid = str(r['pubid'])
        pubids, sections = derive_pubids_sections(r.get('contexts', []), text_map)
        n_source_sections = len(source_meta.get(target_pubid, []))
        m = compute_metrics(pubids, sections, target_pubid, n_source_sections)
        m['idx'] = r.get('idx')
        m['source_pubid'] = target_pubid
        per_query.append(m)
    return {
        'aggregate': aggregate(per_query),
        'per_query': per_query,
        'n_samples': len(per_query),
    }
