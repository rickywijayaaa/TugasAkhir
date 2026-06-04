"""
RAGAS-style evaluator: custom zero-NaN implementation, 4 metrik.
Reusable module untuk runner notebook A (Phase 2 RAGAS evaluation).

Metrik:
- faithfulness     : proporsi statement jawaban yang didukung konteks
- context_recall   : proporsi statement ground-truth yang ter-cover konteks
- answer_relevancy : proporsi statement jawaban yang relevan ke pertanyaan
- context_precision: proporsi chunk konteks yang relevan ke pertanyaan
"""
import re
import time
from typing import List

# Default truncation - dinaikkan dari versi RAGAS bawaan agar chunk biomedis utuh
CONTEXT_TRUNCATE = 1500
REF_TRUNCATE = 800


def split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) >= 15]


def llm_yes_no(openai_generate_fn, prompt: str, max_retries: int = 2) -> bool:
    """Call LLM with retry, return True/False (yes/no)."""
    for attempt in range(max_retries + 1):
        try:
            resp = openai_generate_fn(prompt, max_tokens=10, temperature=0.0)
            return 'yes' in resp.lower()[:15]
        except Exception as e:
            if attempt == max_retries:
                return False
            time.sleep(2 ** attempt)
    return False


def compute_faithfulness(openai_generate_fn, answer: str, contexts: List[str]) -> float:
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0
    ctx_text = '\n'.join(f'[{i+1}] {c[:CONTEXT_TRUNCATE]}' for i, c in enumerate(contexts))
    prompt_tmpl = (
        'Context:\n{ctx}\n\n'
        'Statement: {sent}\n\n'
        'Is this statement directly supported by the context above? '
        'Answer with only "yes" or "no".'
    )
    supported = sum(1 for s in sentences
                    if llm_yes_no(openai_generate_fn, prompt_tmpl.format(ctx=ctx_text, sent=s)))
    return supported / len(sentences)


def compute_context_recall(openai_generate_fn, reference: str, contexts: List[str]) -> float:
    sentences = split_sentences(reference[:REF_TRUNCATE])
    if not sentences:
        return 0.0
    ctx_text = '\n'.join(f'[{i+1}] {c[:CONTEXT_TRUNCATE]}' for i, c in enumerate(contexts))
    prompt_tmpl = (
        'Context:\n{ctx}\n\n'
        'Statement: {sent}\n\n'
        'Is this statement supported by the context above? '
        'Answer with only "yes" or "no".'
    )
    covered = sum(1 for s in sentences
                  if llm_yes_no(openai_generate_fn, prompt_tmpl.format(ctx=ctx_text, sent=s)))
    return covered / len(sentences)


def compute_answer_relevancy(openai_generate_fn, question: str, answer: str) -> float:
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0
    prompt_tmpl = (
        'Question: {question}\n\n'
        'Statement: {sent}\n\n'
        'Is this statement relevant to answering the question above? '
        'Answer with only "yes" or "no".'
    )
    relevant = sum(1 for s in sentences
                   if llm_yes_no(openai_generate_fn, prompt_tmpl.format(question=question, sent=s)))
    return relevant / len(sentences)


def compute_context_precision(openai_generate_fn, question: str, contexts: List[str]) -> float:
    if not contexts:
        return 0.0
    prompt_tmpl = (
        'Question: {question}\n\n'
        'Context chunk: {chunk}\n\n'
        'Is this context chunk useful for answering the question above? '
        'Answer with only "yes" or "no".'
    )
    relevant = sum(1 for c in contexts
                   if llm_yes_no(openai_generate_fn, prompt_tmpl.format(question=question, chunk=c[:CONTEXT_TRUNCATE])))
    return relevant / len(contexts)


def evaluate_sample(openai_generate_fn, sample: dict) -> dict:
    """Evaluate satu sampel phase1 → return 4 RAGAS metrik."""
    question = sample.get('question', '')
    answer = sample.get('answer', '') or sample.get('llm_answer', '')
    contexts = sample.get('contexts', []) or sample.get('retrieved_contexts', [])
    reference = (
        sample.get('reference', '')
        or sample.get('long_answer', '')
        or sample.get('ground_truth', '')
    )

    if isinstance(contexts, list) and contexts and isinstance(contexts[0], dict):
        contexts = [c.get('text', '') for c in contexts]

    return {
        'faithfulness': compute_faithfulness(openai_generate_fn, answer, contexts),
        'context_recall': compute_context_recall(openai_generate_fn, reference, contexts),
        'answer_relevancy': compute_answer_relevancy(openai_generate_fn, question, answer),
        'context_precision': compute_context_precision(openai_generate_fn, question, contexts),
    }
