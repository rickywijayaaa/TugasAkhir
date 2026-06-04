"""Acronym extractor + expansion for biomedical text preprocessing.

Two-layer approach:
  Layer 1 (primary): in-paper acronym detection via regex pattern
                     "<full form> (<ACRONYM>)"
  Layer 2 (fallback): curated medical dictionary (~150 entries)

Strategy: bidirectional expansion - replace "ACRONYM" jadi "ACRONYM (full form)"
sehingga query original maupun expanded sama-sama match di BM25.
"""

import re
from typing import Dict, List, Tuple

from acronym_dict import COMMON_MEDICAL_ACRONYMS, AMBIGUOUS_ACRONYMS


# Pattern hanya untuk capture '(ACRONYM)' — kita scan backwards untuk full form
ACRONYM_IN_PARENS = re.compile(r'\(([A-Z][A-Za-z0-9]{1,7}s?)\)')

SKIP_WORDS = {
    'a', 'an', 'the', 'of', 'and', 'or', 'in', 'on', 'to', 'for', 'with', 'by',
    'at', 'from', 'as', 'is', 'are', 'be',
}


def get_initials(phrase: str) -> str:
    """Get lowercase initials from a phrase, splitting on whitespace AND hyphens."""
    # Split on whitespace and hyphens, treat each piece as a word
    pieces = re.split(r'[\s\-]+', phrase.lower())
    initials = []
    for w in pieces:
        if not w or w in SKIP_WORDS:
            continue
        if w[0].isalpha():
            initials.append(w[0])
    return ''.join(initials)


def validate_acronym(full_form: str, acronym: str) -> bool:
    """Check if acronym letters match full form initials (split on whitespace+hyphens)."""
    initials = get_initials(full_form)
    if not initials:
        return False

    ac_clean = acronym.rstrip('s').lower()
    init_clean = initials.rstrip('s')

    # Exact match
    if ac_clean == init_clean:
        return True

    # Allow acronym to be a subsequence of initials (skip some words like articles)
    # E.g. "split-liver transplantation" -> initials "slt", acronym "slt" -> match
    # E.g. "in vitro fertilization" -> initials "ivf", acronym "ivf" -> match
    if len(ac_clean) <= len(init_clean):
        i = 0
        for c in ac_clean:
            i = init_clean.find(c, i)
            if i == -1:
                return False
            i += 1
        return True
    return False


def _exact_initials_match(phrase: str, acronym: str) -> bool:
    """Strict: initials of phrase (split on whitespace+hyphens, skip stopwords)
    EXACTLY equal acronym (case-insensitive, plural-tolerant)."""
    initials = get_initials(phrase)
    ac_clean = acronym.rstrip('s').lower()
    init_clean = initials.rstrip('s')
    return ac_clean == init_clean


def extract_acronyms_from_text(text: str) -> Dict[str, str]:
    """Find '<full form> (ABC)' patterns by scanning back from each parenthesized acronym.

    Strategy:
      1. For each '(ACRONYM)', look back up to 8 words.
      2. STOP scan if hit boundary: '.', ',', ';', '(', ')'.
      3. Try word counts in increasing order, ACCEPT FIRST exact initials match.
      4. Fallback: subsequence match (allow skip articles) only if no exact found.
    """
    found = {}
    for match in ACRONYM_IN_PARENS.finditer(text):
        acronym = match.group(1)
        if len(acronym) < 2:
            continue
        upper_count = sum(1 for c in acronym if c.isupper())
        if upper_count < 2:
            continue

        end_pos = match.start()
        before = text[max(0, end_pos - 200):end_pos].rstrip()

        # Stop at boundary characters when scanning backwards
        # Find rightmost boundary, take only text after it
        boundary_match = re.search(r'[.;:!?\(\)\[\]"]', before[::-1])
        if boundary_match:
            cutoff = len(before) - boundary_match.start()
            before = before[cutoff:].lstrip(' ,')

        words = re.findall(r'\S+', before)
        if not words:
            continue

        ac_clean = acronym.rstrip('s')
        target_len = len(ac_clean)

        # Try EXACT match first, increasing word count
        # (target_len, target_len+1 to handle hyphen giving extra initial)
        accepted = None
        for n in range(target_len, min(target_len + 3, len(words)) + 1):
            if n > len(words):
                break
            candidate = ' '.join(words[-n:])
            if _exact_initials_match(candidate, acronym):
                accepted = candidate
                break

        # Fallback: subsequence match (less strict, may include articles)
        if not accepted:
            for n in range(target_len + 1, min(target_len + 4, len(words)) + 1):
                if n > len(words):
                    break
                candidate = ' '.join(words[-n:])
                if validate_acronym(candidate, acronym):
                    accepted = candidate
                    break

        if accepted:
            # Strip leading stopwords for cleaner output
            tokens = accepted.split()
            while tokens and tokens[0].lower() in SKIP_WORDS:
                tokens.pop(0)
            if tokens:
                full_normalized = re.sub(r'\s+', ' ', ' '.join(tokens).lower())
                found[acronym] = full_normalized
    return found


def get_acronym_dict_for_paper(paper_sections: List[str]) -> Dict[str, str]:
    """Build per-paper acronym dictionary.

    Combine in-paper detection (Layer 1, priority) + curated dict (Layer 2, fallback).
    """
    full_text = ' '.join(paper_sections)

    # Layer 1: in-paper detection (highest priority)
    in_paper = extract_acronyms_from_text(full_text)

    # Layer 2: curated dict (only for acronyms NOT defined in this paper)
    final_dict = dict(in_paper)
    for ac, full in COMMON_MEDICAL_ACRONYMS.items():
        if ac in AMBIGUOUS_ACRONYMS:
            # Skip ambiguous ones if not defined in this paper
            if ac not in in_paper:
                continue
        if ac not in final_dict:
            final_dict[ac] = full

    return final_dict


def expand_text(text: str, acronym_dict: Dict[str, str]) -> str:
    """Replace acronym occurrences with 'ACRONYM (full form)'.

    Bidirectional strategy: keep original acronym + add full form in parens.
    Preserves searchability for both original and expanded queries.

    Skip the FIRST definition (since it's already 'X Y Z (ABC)' format).
    """
    if not acronym_dict:
        return text

    # Sort by length descending to avoid partial-match issues
    sorted_acronyms = sorted(acronym_dict.keys(), key=lambda x: -len(x))

    expanded = text
    for ac in sorted_acronyms:
        full = acronym_dict[ac]
        # Pattern: word boundary + acronym + word boundary
        # Skip if already followed by '(' (already defined)
        # Use negative lookahead to avoid double-expansion
        pattern = rf'\b({re.escape(ac)})\b(?!\s*\()'
        replacement = rf'\1 ({full})'
        expanded = re.sub(pattern, replacement, expanded)

    return expanded


def expand_paper(paper_sections: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Apply acronym expansion to all sections of a paper.

    Returns (expanded_sections, acronym_dict_used).
    """
    acronym_dict = get_acronym_dict_for_paper(paper_sections)
    expanded = [expand_text(section, acronym_dict) for section in paper_sections]
    return expanded, acronym_dict


# ============================================================
# Smoke test
# ============================================================
if __name__ == '__main__':
    test_paper = [
        # OBJECTIVE
        "To assess and compare the value of split-liver transplantation (SLT) "
        "and living-related liver transplantation (LRT).",
        # SUMMARY BACKGROUND
        "The concept of SLT results from the development of reduced-size "
        "transplantation. A further development of SLT, the in situ split "
        "technique, is derived from LRT. The combination of SLT and LRT has "
        "abolished deaths on the waiting list.",
        # METHODS
        "Outcomes and postoperative liver function of 43 primary LRT patients "
        "were compared with those of 49 primary SLT patients (14 ex situ, 35 "
        "in situ) with known graft weight performed between 1996 and 2000.",
        # RESULTS
        "After a median follow-up of 35 months, actual patient survival rates "
        "were 82% in the SLT group and 88% in the LRT group. Primary nonfunction "
        "was 12% in SLT vs 2.3% in LRT.",
    ]

    expanded, acronyms = expand_paper(test_paper)
    print('Detected acronyms (in-paper + curated):')
    for ac, full in sorted(acronyms.items()):
        marker = ' (in-paper)' if ac in extract_acronyms_from_text(' '.join(test_paper)) else ' (curated)'
        print(f'  {ac:8s} -> {full}{marker}')

    print('\nBefore vs After expansion (METHODS section):')
    print(f'  BEFORE: {test_paper[2]}')
    print(f'  AFTER : {expanded[2]}')

    print('\nBefore vs After expansion (RESULTS section):')
    print(f'  BEFORE: {test_paper[3]}')
    print(f'  AFTER : {expanded[3]}')
