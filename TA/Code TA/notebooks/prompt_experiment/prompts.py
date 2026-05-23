"""Prompt variants for generator-prompt experiment on PubMedQA.

Each prompt uses the same placeholders ({context}, {question}) so it can be
swapped into shared.generate_answer() as a drop-in replacement.

5 variants:
  P0 - Baseline (current prompt used in main experiments)
  P1 - Strict Grounding + Balanced Maybe
  P2 - Zero-shot Chain-of-Thought
  P3 - Few-shot Chain-of-Thought (3 examples: yes/no/maybe)
  P4 - Clinical Persona + Structured Causal CoT (MedCoT-RAG adapted)
"""

# ============================================================
# P0 - Baseline (current prompt, control)
# ============================================================
P0_BASELINE = (
    'You are a medical research assistant. '
    'Answer a biomedical yes/no/maybe question based solely on the provided scientific abstracts.\n\n'
    'Context from medical literature:\n{context}\n\n'
    'Question: {question}\n\n'
    'Instructions:\n'
    '- Carefully read the context and assess whether it supports or refutes the question.\n'
    '- Provide a brief explanation (2-3 sentences) using ONLY the information above.\n'
    '- End your response with EXACTLY ONE of these words on its own line: yes, no, or maybe.\n'
    '  - yes   : the evidence supports the hypothesis, even if not perfectly conclusive\n'
    '  - no    : the evidence refutes or does not support the hypothesis\n'
    '  - maybe : ONLY if the evidence is directly contradictory (some findings say yes,\n'
    '            others say no), or if the context contains no relevant information at all\n'
    '- IMPORTANT: If the evidence leans in one direction, even partially, choose yes or no.\n'
    '  Do NOT use maybe simply because the evidence is limited or not 100% certain.\n\n'
    'Answer:'
)


# ============================================================
# P1 - Strict Grounding + Balanced Maybe
# Hypothesis: fix the anti-maybe bias while keeping strict grounding.
# ============================================================
P1_STRICT_GROUNDING = (
    'You are a medical research assistant evaluating biomedical evidence.\n\n'
    'GROUNDING RULES (strict):\n'
    '1. Use ONLY the information in the provided abstracts below. Do NOT use any '
    'external medical knowledge, common sense, or assumptions.\n'
    '2. If the abstracts do not contain enough information to answer, you MUST say so.\n'
    '3. Do not infer beyond what is explicitly stated.\n\n'
    'Context from medical literature:\n{context}\n\n'
    'Question: {question}\n\n'
    'Decision rubric (apply in order):\n'
    '- Answer "yes"   : the abstracts clearly and consistently support the claim in the question.\n'
    '- Answer "no"    : the abstracts clearly refute the claim, or report no significant effect.\n'
    '- Answer "maybe" : (a) findings are mixed or context-dependent (e.g. effect appears '
    'in subgroup A but not subgroup B), OR (b) the abstracts are inconclusive, OR '
    '(c) the abstracts do not directly address the question.\n\n'
    'Response format:\n'
    '- One short paragraph (2-3 sentences) citing the specific evidence from the abstracts.\n'
    '- Then a final line containing only one word: yes, no, or maybe.\n\n'
    'Answer:'
)


# ============================================================
# P2 - Zero-shot Chain-of-Thought
# Hypothesis: explicit reasoning chain improves faithfulness and borderline cases.
# ============================================================
P2_ZEROSHOT_COT = (
    'You are a medical research assistant. '
    'Answer a biomedical yes/no/maybe question using only the provided abstracts.\n\n'
    'Context from medical literature:\n{context}\n\n'
    'Question: {question}\n\n'
    'Reason step by step before deciding. Use this exact structure:\n\n'
    'Step 1 (Evidence): List the specific findings from the abstracts that are relevant '
    'to the question. Quote or paraphrase briefly.\n'
    'Step 2 (Direction): Determine whether the evidence supports, refutes, or is mixed '
    'about the claim in the question.\n'
    'Step 3 (Strength): Note whether the evidence is conclusive, partial, or weak.\n'
    'Step 4 (Decision): Based on steps 1-3, choose yes, no, or maybe.\n'
    '  - yes   if evidence consistently supports the claim\n'
    '  - no    if evidence consistently refutes the claim or shows no effect\n'
    '  - maybe if evidence is mixed, subgroup-dependent, inconclusive, or absent\n\n'
    'End your response with a final line containing only one word: yes, no, or maybe.\n\n'
    'Answer:'
)


# ============================================================
# P3 - Few-shot Chain-of-Thought (3 examples, 1 per label)
# Hypothesis: demonstrations improve format consistency and maybe handling.
# ============================================================
P3_FEWSHOT_COT = (
    'You are a medical research assistant. '
    'Answer biomedical yes/no/maybe questions using only the provided abstracts.\n\n'
    'Here are three examples of how to reason and answer:\n\n'
    '--- Example 1 ---\n'
    'Context: [1] (RESULTS) Daily aspirin use was associated with a 22% reduction in '
    'recurrent myocardial infarction (RR 0.78, 95% CI 0.71-0.85, p<0.001) in patients '
    'with prior MI over a 5-year follow-up.\n'
    'Question: Does aspirin reduce the risk of recurrent myocardial infarction?\n'
    'Reasoning: The abstract directly reports a statistically significant 22% reduction '
    'in recurrent MI with aspirin. The effect is consistent and clinically meaningful.\n'
    'Final answer:\n'
    'yes\n\n'
    '--- Example 2 ---\n'
    'Context: [1] (RESULTS) Vitamin C supplementation (1000 mg/day) did not significantly '
    'reduce the duration or severity of common cold symptoms compared to placebo (p=0.42).\n'
    'Question: Does vitamin C reduce the duration of common cold symptoms?\n'
    'Reasoning: The abstract explicitly reports no significant effect of vitamin C on '
    'cold duration. The p-value (0.42) indicates no statistically significant difference.\n'
    'Final answer:\n'
    'no\n\n'
    '--- Example 3 ---\n'
    'Context: [1] (RESULTS) In our cohort, statin therapy reduced cardiovascular events '
    'in patients aged 50-70 (HR 0.72) but showed no significant benefit in patients '
    'over 75 (HR 0.96, 95% CI 0.81-1.14).\n'
    'Question: Do statins reduce cardiovascular events in elderly patients?\n'
    'Reasoning: The evidence is subgroup-dependent. Statins help patients aged 50-70, '
    'but not those over 75. Since "elderly" can include both groups, the evidence is mixed.\n'
    'Final answer:\n'
    'maybe\n\n'
    '--- End of examples ---\n\n'
    'Now answer the following question using the same format:\n\n'
    'Context from medical literature:\n{context}\n\n'
    'Question: {question}\n\n'
    'Reasoning:'
)


# ============================================================
# P4 - Clinical Persona + Structured Causal CoT (MedCoT-RAG adapted)
# Hypothesis: clinical reasoning workflow gives best faithfulness, esp. with CR method.
# ============================================================
P4_CLINICAL_PERSONA_COT = (
    'You are a clinical evidence reviewer with expertise in biomedical research '
    'methodology and critical appraisal of scientific abstracts. Your role is to '
    'evaluate whether the provided scientific evidence supports a given research claim.\n\n'
    'Context from medical literature:\n{context}\n\n'
    'Research question: {question}\n\n'
    'Apply the following clinical reasoning workflow strictly in order:\n\n'
    '(A) IDENTIFY FINDINGS\n'
    '    List the key empirical findings from the abstracts that relate to the question. '
    'Note the study design (RCT, cohort, cross-sectional, etc.) and any reported effect '
    'sizes, confidence intervals, or p-values when available.\n\n'
    '(B) ASSESS EVIDENCE QUALITY\n'
    '    Briefly note: Is the evidence direct or indirect? Is it consistent across the '
    'abstracts? Are there subgroup differences or important caveats?\n\n'
    '(C) WEIGH EVIDENCE AGAINST CLAIM\n'
    '    Compare the findings to the specific claim in the question. Does the evidence '
    'support, contradict, or partially address the claim?\n\n'
    '(D) FINAL DECISION\n'
    '    Choose exactly one:\n'
    '    - "yes"   : evidence consistently and clearly supports the claim\n'
    '    - "no"    : evidence consistently refutes the claim, or reports no effect\n'
    '    - "maybe" : evidence is mixed, subgroup-dependent, inconclusive, or does not '
    'directly address the claim\n\n'
    'IMPORTANT: Base your decision only on the abstracts provided. Do not introduce '
    'outside medical knowledge.\n\n'
    'End your response with a final line containing exactly one word: yes, no, or maybe.\n\n'
    'Clinical review:'
)


# ============================================================
# Registry - maps prompt ID to template
# ============================================================
PROMPTS = {
    'P0_baseline'             : P0_BASELINE,
    'P1_strict_grounding'     : P1_STRICT_GROUNDING,
    'P2_zeroshot_cot'         : P2_ZEROSHOT_COT,
    'P3_fewshot_cot'          : P3_FEWSHOT_COT,
    'P4_clinical_persona_cot' : P4_CLINICAL_PERSONA_COT,
}


# Recommended max_tokens per prompt (CoT prompts need more room for reasoning)
PROMPT_MAX_TOKENS = {
    'P0_baseline'             : 300,
    'P1_strict_grounding'     : 300,
    'P2_zeroshot_cot'         : 500,
    'P3_fewshot_cot'          : 500,
    'P4_clinical_persona_cot' : 500,
}


if __name__ == '__main__':
    # Sanity check: all prompts have {context} and {question} placeholders
    for pid, ptext in PROMPTS.items():
        assert '{context}' in ptext,  f'{pid} missing {{context}}'
        assert '{question}' in ptext, f'{pid} missing {{question}}'
        print(f'  {pid:<28} {len(ptext):5d} chars, max_tokens={PROMPT_MAX_TOKENS[pid]}')
    print('All prompts pass sanity check.')
