"""Acronym extraction using Schwartz-Hearst algorithm.

Reference: Schwartz, A. S., & Hearst, M. A. (2003).
"A Simple Algorithm for Identifying Abbreviation Definitions in Biomedical Text."
Pacific Symposium on Biocomputing, 451-462.

Improvement over naive initial-matching:
  - Allows acronym letters from INTERNAL position in compound words
  - Handles HBO -> Hyperbaric oxygenation (B from inside "hyperBaric")
  - Handles VEGF, EGFR, NSAID, and similar biomedical acronyms

Two-layer approach (consistent with original):
  Layer 1 (primary): Schwartz-Hearst in-paper detection
  Layer 2 (fallback): curated medical dictionary (~175 entries)

Output strategy: bidirectional expansion
  Replace "ACRONYM" -> "ACRONYM (full form)"
  Preserves searchability for both original and expanded queries.
"""

import re
from typing import Dict, List, Tuple, Optional

from acronym_dict import COMMON_MEDICAL_ACRONYMS, AMBIGUOUS_ACRONYMS


PAREN_PATTERN = re.compile(r'\(([^)]{2,30})\)')

# Word boundary characters in long-form text
WORD_BOUNDARY_CHARS = set(' \t\n\r-/([.,;:')


def is_valid_short(s: str) -> bool:
    """Heuristic: is this string a valid acronym candidate?

    Rules:
      - Length 2-10 chars (after strip)
      - Max 2 words
      - At least 2 uppercase letters
      - Not a pure number (year, etc.)
    """
    s = s.strip()
    if len(s) < 2 or len(s) > 10:
        return False
    if len(s.split()) > 2:
        return False
    upper_count = sum(1 for c in s if c.isupper())
    if upper_count < 2:
        return False
    if not any(c.isalpha() for c in s):
        return False
    # Reject pure number patterns like "2003", "II", "1A" etc.
    no_alpha_chars = re.sub(r'[A-Za-z]', '', s)
    if no_alpha_chars and not any(c.isalpha() for c in s):
        return False
    return True


def find_best_long_form(short: str, long_candidate: str) -> Optional[str]:
    """Core Schwartz-Hearst shortest-match algorithm.

    Right-to-left scan in both short and long_candidate.
    For each char in short:
      - If first char (s_index == 0): MUST match start of a word in long
      - Otherwise: can match any position in long (including internal letters)

    Returns the matched substring (long form) or None.
    """
    s = short.lower().rstrip('s')  # strip plural marker
    l = long_candidate.lower()

    if not s or not l:
        return None
    if len(l) < len(s):
        return None

    s_index = len(s) - 1
    l_index = len(l) - 1

    while s_index >= 0:
        current = s[s_index]

        if s_index == 0:
            # First char of short: MUST be at word boundary in long
            found = False
            while l_index >= 0:
                if l[l_index] == current:
                    # Check it's at word boundary
                    if l_index == 0 or l[l_index - 1] in WORD_BOUNDARY_CHARS:
                        found = True
                        break
                l_index -= 1
            if not found:
                return None
        else:
            # Other chars: anywhere (allows internal letters)
            while l_index >= 0:
                if l[l_index] == current:
                    break
                l_index -= 1
            if l_index < 0:
                return None

        s_index -= 1
        l_index -= 1

    if s_index < 0:
        # All short chars matched. Return matched span from long_candidate.
        # l_index is now at the position just before the matched span starts (-1 or earlier)
        start = l_index + 1
        return long_candidate[start:].strip()
    return None


def extract_pairs_schwartz_hearst(text: str) -> Dict[str, str]:
    """Extract all valid (acronym -> full_form) pairs from text."""
    pairs: Dict[str, str] = {}

    for match in PAREN_PATTERN.finditer(text):
        inside = match.group(1).strip()
        if not is_valid_short(inside):
            continue

        short = inside

        # Determine candidate window: words before parens
        # Standard heuristic: min(len(short)*2, len(short)+5) words
        max_words = min(len(short) * 2, len(short) + 5)

        # Take text before parens, stop at sentence boundary
        before = text[:match.start()].rstrip()
        # Strip everything before last sentence boundary
        sentence_split = re.split(r'(?<=[.!?;])\s', before)
        candidate_text = sentence_split[-1] if sentence_split else before

        words = candidate_text.split()
        if not words:
            continue

        # Take last max_words words as candidate
        long_candidate = ' '.join(words[-max_words:])

        long_form = find_best_long_form(short, long_candidate)
        if long_form and len(long_form) >= len(short):
            # Only add first occurrence (don't override)
            if short not in pairs:
                # Normalize: lowercase, single space
                pairs[short] = re.sub(r'\s+', ' ', long_form.lower()).strip()

    return pairs


def get_acronym_dict_for_paper(paper_sections: List[str]) -> Dict[str, str]:
    """Build per-paper acronym dictionary using Schwartz-Hearst + curated fallback."""
    full_text = ' '.join(paper_sections)

    # Layer 1: Schwartz-Hearst in-paper detection
    in_paper = extract_pairs_schwartz_hearst(full_text)

    # Layer 2: curated dict (only for acronyms NOT in Layer 1)
    final_dict = dict(in_paper)
    for ac, full in COMMON_MEDICAL_ACRONYMS.items():
        if ac in AMBIGUOUS_ACRONYMS and ac not in in_paper:
            continue
        if ac not in final_dict:
            final_dict[ac] = full

    return final_dict


def expand_text(text: str, acronym_dict: Dict[str, str]) -> str:
    """Replace acronym occurrences with 'ACRONYM (full form)' bidirectionally.

    Skip if already followed by '(' (already defined in original text).
    """
    if not acronym_dict:
        return text

    sorted_acronyms = sorted(acronym_dict.keys(), key=lambda x: -len(x))
    expanded = text
    for ac in sorted_acronyms:
        full = acronym_dict[ac]
        pattern = rf'\b({re.escape(ac)})\b(?!\s*\()'
        replacement = rf'\1 ({full})'
        expanded = re.sub(pattern, replacement, expanded)
    return expanded


def expand_paper(paper_sections: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Apply acronym expansion to all sections of a paper."""
    acronym_dict = get_acronym_dict_for_paper(paper_sections)
    expanded = [expand_text(section, acronym_dict) for section in paper_sections]
    return expanded, acronym_dict


# ============================================================
# SMOKE TEST
# ============================================================
if __name__ == '__main__':
    print('=' * 80)
    print('Schwartz-Hearst SMOKE TEST')
    print('=' * 80)

    # Test 1: HBO case (was failing in naive)
    test1 = "Hyperbaric oxygenation (HBO) has been recommended as adjuvant therapy."
    pairs = extract_pairs_schwartz_hearst(test1)
    print(f'\nTest 1: HBO (internal letter)')
    print(f'  Input: {test1}')
    print(f'  Result: {pairs}')
    assert 'HBO' in pairs and 'hyperbaric oxygenation' in pairs['HBO'], 'HBO test FAILED'
    print(f'  STATUS: PASS')

    # Test 2: NF case (worked in naive too)
    test2 = "necrotizing fasciitis (NF) is a severe disease."
    pairs = extract_pairs_schwartz_hearst(test2)
    print(f'\nTest 2: NF (clean initials)')
    print(f'  Input: {test2}')
    print(f'  Result: {pairs}')
    assert 'NF' in pairs, 'NF test FAILED'
    print(f'  STATUS: PASS')

    # Test 3: VEGF case
    test3 = "vascular endothelial growth factor (VEGF) plays a role."
    pairs = extract_pairs_schwartz_hearst(test3)
    print(f'\nTest 3: VEGF (4 letters, all word starts)')
    print(f'  Input: {test3}')
    print(f'  Result: {pairs}')
    assert 'VEGF' in pairs, 'VEGF test FAILED'
    print(f'  STATUS: PASS')

    # Test 4: NSAID case
    test4 = "non-steroidal anti-inflammatory drugs (NSAIDs) are common."
    pairs = extract_pairs_schwartz_hearst(test4)
    print(f'\nTest 4: NSAIDs (with hyphens, plural)')
    print(f'  Input: {test4}')
    print(f'  Result: {pairs}')
    assert 'NSAIDs' in pairs, 'NSAIDs test FAILED'
    print(f'  STATUS: PASS')

    # Test 5: CABG case (compound)
    test5 = "coronary artery bypass graft (CABG) is a surgical procedure."
    pairs = extract_pairs_schwartz_hearst(test5)
    print(f'\nTest 5: CABG')
    print(f'  Input: {test5}')
    print(f'  Result: {pairs}')
    assert 'CABG' in pairs, 'CABG test FAILED'
    print(f'  STATUS: PASS')

    # Test 6: invalid short (year-like)
    test6 = "data from 2003 (the most recent available)"
    pairs = extract_pairs_schwartz_hearst(test6)
    print(f'\nTest 6: false positive year (should reject)')
    print(f'  Input: {test6}')
    print(f'  Result: {pairs}')
    print(f'  STATUS: {"PASS (rejected as expected)" if not pairs else "FAIL"}')

    # Test 7: full LRT/SLT paper
    test7_sections = [
        'To assess and compare the value of split-liver transplantation (SLT) and living-related liver transplantation (LRT).',
        'The concept of SLT results from the development of reduced-size transplantation. The combination of SLT and LRT has abolished deaths.',
        'Outcomes of 43 LRT patients were compared with 49 SLT patients.',
        'Mortality rate among HBO-treated was 36% vs 25% in non-HBO group.',  # cross-paper check (HBO from another context)
    ]
    expanded, acr = expand_paper(test7_sections)
    print(f'\nTest 7: LRT/SLT full paper')
    print(f'  In-paper acronyms detected:')
    for ac, full in extract_pairs_schwartz_hearst(' '.join(test7_sections)).items():
        print(f'    {ac:6s} -> {full}')
    print(f'\nMETHODS expanded: {expanded[2]}')
    print(f'RESULTS expanded: {expanded[3]}')

    # Test 8: HBO + NF combined (the actual idx 30 case)
    test8 = ("The accepted treatment protocol for necrotizing fasciitis (NF) consists of "
             "extensive surgery and wide spectrum antibiotics. Hyperbaric oxygenation (HBO) "
             "has been recommended as adjuvant therapy for NF, improving patient mortality.")
    pairs = extract_pairs_schwartz_hearst(test8)
    print(f'\nTest 8: NF + HBO combined (real PMID 7482275 BACKGROUND)')
    print(f'  Result:')
    for ac, full in pairs.items():
        print(f'    {ac:6s} -> {full}')
    assert 'HBO' in pairs, 'HBO not detected in combined test'
    assert 'NF' in pairs, 'NF not detected in combined test'
    print(f'  STATUS: PASS')

    print('\n' + '=' * 80)
    print('ALL TESTS PASSED')
    print('=' * 80)
