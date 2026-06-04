"""Build Excel showing Claude Haiku 4.5 answers for verification of hedging behavior.

Output: notebooks/claude_answers_inspection.xlsx with 4 sheets:
  Sheet 1 - Overview (summary stats per config)
  Sheet 2 - All Answers (500 answers, baseline config)
  Sheet 3 - Hedge Highlights (only answers with hedge patterns)
  Sheet 4 - Cross-LLM Compare (Claude vs GPT vs Llama for same questions)
"""
import json
import re
from pathlib import Path
from collections import Counter

import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

HERE = Path(__file__).parent
RESULTS = HERE.parent / 'results'
OUT = HERE / 'claude_answers_inspection.xlsx'

# === Hedge pattern markers ===
HEDGE_PATTERNS = [
    r'\bhowever\b',
    r'\bbased on (?:the |provided )?(?:context|abstracts?)\b',
    r'\bcannot (?:determine|conclude|verify|assess)\b',
    r'\binsufficient\b',
    r'\bdoes not (?:provide|include|specify|address|directly)\b',
    r'\bdo not (?:provide|include|specify|address|directly)\b',
    r'\bnot enough\b',
    r'\bunclear\b',
    r'\bmay (?:suggest|indicate|imply)\b',
    r'\bmight\b',
    r'\bappears? to\b',
    r'\bsuggests? (?:that|but)\b',
    r'\blimited\b',
    r'\bwithout (?:additional|more|further)\b',
    r'\bonly (?:describes?|details?|mentions?|provides?)\b',
    r'\bnot (?:mentioned|stated|reported|discussed)\b',
    r'\bno (?:information|data|details?|results?|conclusion)\b',
    r'\bnot (?:fully|directly|explicitly|clearly)\b',
]
HEDGE_RE = re.compile('|'.join(HEDGE_PATTERNS), re.IGNORECASE)


def count_hedges(text):
    return len(HEDGE_RE.findall(text or ''))


def load_phase1(path):
    if not path.exists():
        return None
    with open(path, encoding='utf-8', errors='replace') as f:
        return json.load(f)['results']


def load_phase2(path):
    if not path.exists():
        return None
    with open(path, encoding='utf-8', errors='replace') as f:
        return json.load(f)['results']


# ============================================================
# Style helpers
# ============================================================
NAVY    = '1F4E79'
GOLD    = 'D4A957'
GREEN   = '4ADE80'
RED     = 'F87171'
ORANGE  = 'FB923C'
LIGHT_RED  = 'FEE2E2'
LIGHT_GRN  = 'D1FAE5'
LIGHT_YEL  = 'FEF3C7'
LIGHT_BLUE = 'DBEAFE'
LIGHT_GRY  = 'F3F4F6'

HEADER_FONT = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
HEADER_FILL = PatternFill(start_color=NAVY, end_color=NAVY, fill_type='solid')
HEADER_ALIGN = Alignment(horizontal='center', vertical='center', wrap_text=True)

BODY_FONT = Font(name='Calibri', size=10)
WRAP = Alignment(vertical='top', wrap_text=True)
CENTER = Alignment(horizontal='center', vertical='center')

THIN = Side(border_style='thin', color='CCCCCC')
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def style_header_row(ws, row=1, last_col=None):
    if last_col is None:
        last_col = ws.max_column
    for col in range(1, last_col + 1):
        cell = ws.cell(row=row, column=col)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = HEADER_ALIGN
        cell.border = BORDER
    ws.row_dimensions[row].height = 32


def fill_row(ws, row, fill_color):
    for col in range(1, ws.max_column + 1):
        ws.cell(row=row, column=col).fill = PatternFill(
            start_color=fill_color, end_color=fill_color, fill_type='solid'
        )


def label_color(label):
    return {'yes': LIGHT_GRN, 'no': LIGHT_RED, 'maybe': LIGHT_YEL}.get(label, '')


# ============================================================
# SHEET 1 - Overview
# ============================================================
def build_overview(wb):
    ws = wb.create_sheet('1_Overview')
    ws['A1'] = 'CLAUDE HAIKU 4.5 — Inspeksi Jawaban (n=500 PubMedQA)'
    ws['A1'].font = Font(size=16, bold=True, color=NAVY)
    ws.merge_cells('A1:G1')

    ws['A3'] = ('Tujuan: validasi klaim bahwa Claude memberikan "hedge" (epistemik humility) '
                'yang menurunkan faithfulness tapi menaikkan maybe-recall.')
    ws['A3'].font = Font(size=11, italic=True, color='6B7280')
    ws['A3'].alignment = WRAP
    ws.merge_cells('A3:G3')
    ws.row_dimensions[3].height = 32

    # Summary table
    headers = ['Konfigurasi', 'n', 'Acc Total', 'Acc yes', 'Acc no', 'Acc maybe', 'Faithfulness']
    for i, h in enumerate(headers, 1):
        ws.cell(row=5, column=i, value=h)
    style_header_row(ws, row=5, last_col=len(headers))

    row = 6
    for cfg in ['baseline', 'qr', 'cr', 'qr_cr']:
        p1 = load_phase1(RESULTS / f'{cfg}_claude_phase1_answers.json')
        p2 = load_phase2(RESULTS / f'{cfg}_claude_phase2_custom.json')
        if not p1 or not p2:
            continue
        n = len(p1)
        acc = sum(x['is_correct'] for x in p1) / n
        accs = {}
        for lbl in ['yes','no','maybe']:
            sub = [x for x in p1 if x['ground_truth']==lbl]
            accs[lbl] = sum(x['is_correct'] for x in sub)/len(sub) if sub else 0
        faith = sum(x['faithfulness'] for x in p2) / len(p2)
        cells = [cfg, n, f'{acc:.3f}', f'{accs["yes"]:.3f}', f'{accs["no"]:.3f}',
                 f'{accs["maybe"]:.3f}', f'{faith:.3f}']
        for i, v in enumerate(cells, 1):
            c = ws.cell(row=row, column=i, value=v)
            c.font = BODY_FONT
            c.alignment = CENTER if i > 1 else Alignment(horizontal='left', vertical='center')
            c.border = BORDER
        # Highlight maybe column for visual
        if accs['maybe'] > 0.15:
            ws.cell(row=row, column=6).fill = PatternFill(
                start_color=LIGHT_GRN, end_color=LIGHT_GRN, fill_type='solid')
        row += 1

    # Insight box
    row += 2
    ws.cell(row=row, column=1, value='INSIGHT KUNCI').font = Font(size=12, bold=True, color=NAVY)
    row += 1
    insights = [
        '• Maybe accuracy Claude 17-24% (jauh di atas Llama/OpenAI yang 0-12%)',
        '• Faithfulness Claude 0.55-0.60 (di bawah GPT 0.88-0.90)',
        '• Trade-off ini diduga akibat Constitutional AI training (Anthropic)',
        '• Hedging muncul saat konteks tidak konklusif → Claude memilih maybe + tambah disclaimer',
    ]
    for ins in insights:
        c = ws.cell(row=row, column=1, value=ins)
        c.font = Font(size=11)
        c.alignment = Alignment(vertical='top', wrap_text=True)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
        row += 1

    # Sheet navigation
    row += 2
    ws.cell(row=row, column=1, value='NAVIGASI SHEET').font = Font(size=12, bold=True, color=NAVY)
    row += 1
    nav = [
        '→ Sheet "2_All_Answers": Semua 500 jawaban Claude baseline (lengkap)',
        '→ Sheet "3_Hedge_Highlights": Filter jawaban dengan hedge pattern terbanyak',
        '→ Sheet "4_Cross_LLM_Compare": Banding jawaban Claude vs GPT vs Llama untuk sampel sama',
        '→ Sheet "5_All_LLMs_Wrong": 85 sampel di mana semua 4 LLM gagal (information gap)',
    ]
    for n in nav:
        c = ws.cell(row=row, column=1, value=n)
        c.font = Font(size=11)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
        row += 1

    ws.column_dimensions['A'].width = 22
    for col in 'BCDEFG':
        ws.column_dimensions[col].width = 14


# ============================================================
# SHEET 2 - All Answers
# ============================================================
def build_all_answers(wb):
    ws = wb.create_sheet('2_All_Answers')
    p1 = load_phase1(RESULTS / 'baseline_claude_phase1_answers.json')
    p2 = load_phase2(RESULTS / 'baseline_claude_phase2_custom.json')
    p2_lookup = {x['idx']: x for x in p2} if p2 else {}

    headers = ['Idx', 'PMID', 'Question', 'GT', 'Pred', 'Match',
               'Faith', 'Hedge#', 'Answer (Claude)']
    for i, h in enumerate(headers, 1):
        ws.cell(row=1, column=i, value=h)
    style_header_row(ws)

    for row_i, r in enumerate(p1, 2):
        idx = r['idx']
        p2r = p2_lookup.get(idx, {})
        hedge_n = count_hedges(r['answer'])
        pmid_url = f'https://pubmed.ncbi.nlm.nih.gov/{r["pubid"]}/'
        cells = [
            r['idx'], r['pubid'], r['question'],
            r['ground_truth'], r['predicted_label'],
            'YES' if r['is_correct'] else 'NO',
            f"{p2r.get('faithfulness', 0):.3f}",
            hedge_n,
            r['answer'],
        ]
        for col_i, v in enumerate(cells, 1):
            c = ws.cell(row=row_i, column=col_i, value=v)
            c.font = BODY_FONT
            c.alignment = WRAP
            c.border = BORDER
        # Make PMID a hyperlink
        ws.cell(row=row_i, column=2).hyperlink = pmid_url
        ws.cell(row=row_i, column=2).font = Font(color=NAVY, underline='single', size=10)
        # Color GT and Pred cells
        ws.cell(row=row_i, column=4).fill = PatternFill(
            start_color=label_color(r['ground_truth']), end_color=label_color(r['ground_truth']),
            fill_type='solid')
        ws.cell(row=row_i, column=5).fill = PatternFill(
            start_color=label_color(r['predicted_label']), end_color=label_color(r['predicted_label']),
            fill_type='solid')
        # Match: green/red
        ws.cell(row=row_i, column=6).fill = PatternFill(
            start_color=LIGHT_GRN if r['is_correct'] else LIGHT_RED,
            end_color=LIGHT_GRN if r['is_correct'] else LIGHT_RED,
            fill_type='solid')
        ws.cell(row=row_i, column=6).alignment = CENTER
        # Hedge intensity color
        if hedge_n >= 4:
            ws.cell(row=row_i, column=8).fill = PatternFill(
                start_color=LIGHT_RED, end_color=LIGHT_RED, fill_type='solid')
        elif hedge_n >= 2:
            ws.cell(row=row_i, column=8).fill = PatternFill(
                start_color=LIGHT_YEL, end_color=LIGHT_YEL, fill_type='solid')
        ws.cell(row=row_i, column=8).alignment = CENTER

    # Column widths
    widths = [6, 12, 50, 8, 8, 8, 8, 8, 80]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    # Freeze header
    ws.freeze_panes = 'A2'
    # Auto-filter
    ws.auto_filter.ref = ws.dimensions


# ============================================================
# SHEET 3 - Hedge Highlights (only top hedges)
# ============================================================
def build_hedge_highlights(wb):
    ws = wb.create_sheet('3_Hedge_Highlights')
    p1 = load_phase1(RESULTS / 'baseline_claude_phase1_answers.json')

    # Compute hedge count, sort
    enriched = [(count_hedges(r['answer']), r) for r in p1]
    enriched.sort(key=lambda x: -x[0])
    top_hedges = [r for n, r in enriched if n >= 3][:80]  # top 80 with hedge >= 3

    ws['A1'] = f'Top {len(top_hedges)} jawaban Claude dengan hedge pattern terbanyak'
    ws['A1'].font = Font(size=14, bold=True, color=NAVY)
    ws.merge_cells('A1:G1')

    ws['A2'] = ('Hedge patterns: "however", "based on context", "cannot determine", "may suggest", '
                '"appears to", "limited", "does not provide", "unclear", dll. '
                'Hedge tinggi → indikasi epistemik humility (Constitutional AI signature).')
    ws['A2'].font = Font(size=10, italic=True, color='6B7280')
    ws['A2'].alignment = WRAP
    ws.merge_cells('A2:G2')
    ws.row_dimensions[2].height = 36

    headers = ['Hedge#', 'GT', 'Pred', 'Match', 'Question', 'Answer (with hedge highlights)', 'PMID']
    for i, h in enumerate(headers, 1):
        ws.cell(row=4, column=i, value=h)
    style_header_row(ws, row=4, last_col=len(headers))

    for row_i, (r, hedge_n) in enumerate(zip(top_hedges, [count_hedges(r['answer']) for r in top_hedges]), 5):
        cells = [
            hedge_n,
            r['ground_truth'], r['predicted_label'],
            'YES' if r['is_correct'] else 'NO',
            r['question'],
            r['answer'],
            r['pubid'],
        ]
        for col_i, v in enumerate(cells, 1):
            c = ws.cell(row=row_i, column=col_i, value=v)
            c.font = BODY_FONT
            c.alignment = WRAP
            c.border = BORDER
        # PMID link
        ws.cell(row=row_i, column=7).hyperlink = f'https://pubmed.ncbi.nlm.nih.gov/{r["pubid"]}/'
        ws.cell(row=row_i, column=7).font = Font(color=NAVY, underline='single', size=10)
        # Hedge severity color
        if hedge_n >= 6:
            ws.cell(row=row_i, column=1).fill = PatternFill(
                start_color=RED, end_color=RED, fill_type='solid')
            ws.cell(row=row_i, column=1).font = Font(color='FFFFFF', bold=True)
        elif hedge_n >= 4:
            ws.cell(row=row_i, column=1).fill = PatternFill(
                start_color=ORANGE, end_color=ORANGE, fill_type='solid')
            ws.cell(row=row_i, column=1).font = Font(color='FFFFFF', bold=True)
        else:
            ws.cell(row=row_i, column=1).fill = PatternFill(
                start_color=LIGHT_YEL, end_color=LIGHT_YEL, fill_type='solid')
        ws.cell(row=row_i, column=1).alignment = CENTER
        # GT/Pred color
        ws.cell(row=row_i, column=2).fill = PatternFill(
            start_color=label_color(r['ground_truth']), end_color=label_color(r['ground_truth']),
            fill_type='solid')
        ws.cell(row=row_i, column=3).fill = PatternFill(
            start_color=label_color(r['predicted_label']), end_color=label_color(r['predicted_label']),
            fill_type='solid')
        ws.cell(row=row_i, column=4).fill = PatternFill(
            start_color=LIGHT_GRN if r['is_correct'] else LIGHT_RED,
            end_color=LIGHT_GRN if r['is_correct'] else LIGHT_RED, fill_type='solid')
        ws.cell(row=row_i, column=4).alignment = CENTER

    widths = [10, 8, 8, 8, 45, 80, 12]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = 'A5'
    ws.auto_filter.ref = f'A4:G{4 + len(top_hedges)}'


# ============================================================
# SHEET 4 - Cross-LLM Compare
# ============================================================
def build_cross_llm(wb):
    ws = wb.create_sheet('4_Cross_LLM_Compare')

    # Load baseline answers from 4 LLMs
    llms = {
        'Claude':  RESULTS / 'baseline_claude_phase1_answers.json',
        'OpenAI':  RESULTS / 'baseline_openai_phase1_answers.json',
        'Llama33': RESULTS / 'baseline_llama_phase1_answers.json',
        'Llama32': RESULTS / 'baseline_llama32_phase1_answers.json',
    }
    data = {k: load_phase1(p) or [] for k, p in llms.items()}
    if not data['Claude']:
        ws['A1'] = 'No Claude data available'
        return

    ws['A1'] = 'Perbandingan jawaban Claude vs LLM lain (sampel di mana Claude jawab maybe)'
    ws['A1'].font = Font(size=14, bold=True, color=NAVY)
    ws.merge_cells('A1:H1')

    ws['A2'] = ('Filter: hanya sampel di mana Claude prediksi "maybe" (untuk lihat behavior khas Claude). '
                'Bandingkan apakah LLM lain juga maybe atau commit ke yes/no.')
    ws['A2'].font = Font(size=10, italic=True, color='6B7280')
    ws['A2'].alignment = WRAP
    ws.merge_cells('A2:H2')
    ws.row_dimensions[2].height = 36

    headers = ['Idx', 'PMID', 'Question', 'GT',
               'Claude', 'OpenAI', 'Llama33', 'Llama32',
               'Claude Answer (preview)']
    for i, h in enumerate(headers, 1):
        ws.cell(row=4, column=i, value=h)
    style_header_row(ws, row=4, last_col=len(headers))

    # Build idx lookup
    lookups = {k: {x['idx']: x for x in data[k]} for k in data}

    # Filter: Claude said maybe
    claude_maybe_idx = [x['idx'] for x in data['Claude'] if x['predicted_label'] == 'maybe']

    row = 5
    for idx in claude_maybe_idx:
        c = lookups['Claude'].get(idx)
        if not c: continue
        oa = lookups['OpenAI'].get(idx, {})
        l33 = lookups['Llama33'].get(idx, {})
        l32 = lookups['Llama32'].get(idx, {})

        cells = [
            idx, c['pubid'], c['question'], c['ground_truth'],
            c['predicted_label'],
            oa.get('predicted_label', '-'),
            l33.get('predicted_label', '-'),
            l32.get('predicted_label', '-'),
            c['answer'][:400] + ('...' if len(c['answer']) > 400 else ''),
        ]
        for col_i, v in enumerate(cells, 1):
            cell = ws.cell(row=row, column=col_i, value=v)
            cell.font = BODY_FONT
            cell.alignment = WRAP
            cell.border = BORDER
        # PMID link
        ws.cell(row=row, column=2).hyperlink = f'https://pubmed.ncbi.nlm.nih.gov/{c["pubid"]}/'
        ws.cell(row=row, column=2).font = Font(color=NAVY, underline='single', size=10)
        # Color GT
        ws.cell(row=row, column=4).fill = PatternFill(
            start_color=label_color(c['ground_truth']), end_color=label_color(c['ground_truth']),
            fill_type='solid')
        # Color predictions per LLM (compare to GT)
        for col_i, llm_key in enumerate(['Claude','OpenAI','Llama33','Llama32'], 5):
            pred = cells[col_i - 1]
            if pred == c['ground_truth']:
                ws.cell(row=row, column=col_i).fill = PatternFill(
                    start_color=LIGHT_GRN, end_color=LIGHT_GRN, fill_type='solid')
            elif pred == '-':
                ws.cell(row=row, column=col_i).fill = PatternFill(
                    start_color=LIGHT_GRY, end_color=LIGHT_GRY, fill_type='solid')
            else:
                ws.cell(row=row, column=col_i).fill = PatternFill(
                    start_color=LIGHT_RED, end_color=LIGHT_RED, fill_type='solid')
            ws.cell(row=row, column=col_i).alignment = CENTER
        row += 1

    widths = [6, 12, 45, 8, 9, 9, 9, 9, 70]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = 'A5'
    if row > 5:
        ws.auto_filter.ref = f'A4:I{row - 1}'


# ============================================================
# SHEET 5 - All-LLM-Wrong (85 sampel di mana 4 LLM gagal)
# ============================================================
def build_all_wrong(wb):
    """Sampel di mana ke-4 LLM (baseline) memberikan jawaban salah.
    Ini menunjukkan information gap antara konteks dan jawaban benar."""
    ws = wb.create_sheet('5_All_LLMs_Wrong')

    # Load 4 LLM phase1
    llms = {
        'Claude':  RESULTS / 'baseline_claude_phase1_answers.json',
        'OpenAI':  RESULTS / 'baseline_openai_phase1_answers.json',
        'Llama33': RESULTS / 'baseline_llama_phase1_answers.json',
        'Llama32': RESULTS / 'baseline_llama32_phase1_answers.json',
    }
    data = {k: load_phase1(p) or [] for k, p in llms.items()}
    if not all(data.values()):
        ws['A1'] = 'Tidak semua data LLM tersedia'
        return

    # Build idx-keyed lookup
    lookups = {k: {x['idx']: x for x in data[k]} for k in data}

    # Find samples where ALL 4 are wrong
    common_idx = set(lookups['Claude'].keys())
    for k in ['OpenAI', 'Llama33', 'Llama32']:
        common_idx &= set(lookups[k].keys())

    all_wrong = []
    for idx in sorted(common_idx):
        rs = [lookups[k][idx] for k in ['Claude', 'OpenAI', 'Llama33', 'Llama32']]
        if all(not r['is_correct'] for r in rs):
            all_wrong.append((idx, rs))

    # ===== Header section =====
    ws['A1'] = (f'85 SAMPEL DI MANA SEMUA 4 LLM GAGAL  '
                f'(n={len(all_wrong)} dari 500 = {100*len(all_wrong)/500:.1f}%)')
    ws['A1'].font = Font(size=14, bold=True, color=NAVY)
    ws.merge_cells('A1:K1')

    ws['A2'] = ('Hipotesis: kegagalan ini sebagian disebabkan information gap struktural — '
                'jawaban benar memerlukan informasi yang ada di CONCLUSIONS (long_answer) '
                'tapi conclusions sengaja DISISIHKAN dari corpus untuk mencegah answer leakage. '
                'Lihat case study idx 16 (PMID 11729377) sebagai contoh klasik.')
    ws['A2'].font = Font(size=10, italic=True, color='6B7280')
    ws['A2'].alignment = WRAP
    ws.merge_cells('A2:K2')
    ws.row_dimensions[2].height = 50

    # Distribution summary
    gt_counts = Counter(rs[0]['ground_truth'] for _, rs in all_wrong)
    ws['A4'] = 'Distribusi GT untuk sampel "all-wrong":'
    ws['A4'].font = Font(size=11, bold=True, color=NAVY)
    ws['B4'] = (f'yes: {gt_counts.get("yes", 0)} | '
                f'no: {gt_counts.get("no", 0)} | '
                f'maybe: {gt_counts.get("maybe", 0)}  '
                f'(maybe paling sulit, sesuai prediksi class imbalance)')
    ws['B4'].font = Font(size=11)
    ws.merge_cells('B4:K4')

    # ===== Table =====
    headers = ['Idx', 'PMID', 'GT', 'Claude', 'OpenAI', 'Llama33', 'Llama32',
               'Question', 'Long Answer (gold conclusion)', 'Section Labels', 'Reference']
    for i, h in enumerate(headers, 1):
        ws.cell(row=6, column=i, value=h)
    style_header_row(ws, row=6, last_col=len(headers))

    # Pre-load PubMedQA dataset for context.labels and long_answer
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(HERE.parent / 'data' / 'pubmedqa_pqa_labeled'))
        ds_lookup = {idx: ds[idx] for idx in [a[0] for a in all_wrong]}
    except Exception as e:
        print(f'  Warning: cannot load dataset for long_answer: {e}')
        ds_lookup = {}

    for row_i, (idx, rs) in enumerate(all_wrong, 7):
        ds_entry = ds_lookup.get(idx, {})
        labels = ', '.join(ds_entry.get('context', {}).get('labels', [])) if ds_entry else '-'
        long_ans = ds_entry.get('long_answer', '-') if ds_entry else (rs[0].get('reference', '-'))
        cells = [
            idx, rs[0]['pubid'], rs[0]['ground_truth'],
            rs[0]['predicted_label'],  # Claude
            rs[1]['predicted_label'],  # OpenAI
            rs[2]['predicted_label'],  # Llama33
            rs[3]['predicted_label'],  # Llama32
            rs[0]['question'],
            long_ans,
            labels,
            rs[0].get('reference', '-'),
        ]
        for col_i, v in enumerate(cells, 1):
            c = ws.cell(row=row_i, column=col_i, value=v)
            c.font = BODY_FONT
            c.alignment = WRAP
            c.border = BORDER

        # PMID hyperlink
        pmid_cell = ws.cell(row=row_i, column=2)
        pmid_cell.hyperlink = f'https://pubmed.ncbi.nlm.nih.gov/{rs[0]["pubid"]}/'
        pmid_cell.font = Font(color=NAVY, underline='single', size=10)

        # Color GT
        ws.cell(row=row_i, column=3).fill = PatternFill(
            start_color=label_color(rs[0]['ground_truth']),
            end_color=label_color(rs[0]['ground_truth']),
            fill_type='solid')
        ws.cell(row=row_i, column=3).alignment = CENTER

        # Color all 4 LLM predictions (red because all wrong)
        for col_i in [4, 5, 6, 7]:
            ws.cell(row=row_i, column=col_i).fill = PatternFill(
                start_color=LIGHT_RED, end_color=LIGHT_RED, fill_type='solid')
            ws.cell(row=row_i, column=col_i).alignment = CENTER

        # Highlight long_answer column with light yellow (information that LLM cannot see)
        ws.cell(row=row_i, column=9).fill = PatternFill(
            start_color=LIGHT_YEL, end_color=LIGHT_YEL, fill_type='solid')

    # Column widths
    widths = [6, 12, 8, 9, 9, 9, 9, 45, 60, 25, 45]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = 'D7'  # freeze idx, pmid, gt, predictions
    ws.auto_filter.ref = f'A6:K{6 + len(all_wrong)}'

    print(f'  Sheet 5: {len(all_wrong)} all-wrong samples added')


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    wb = Workbook()
    # Delete default sheet
    wb.remove(wb.active)

    print('Building Sheet 1: Overview...')
    build_overview(wb)
    print('Building Sheet 2: All Answers (500)...')
    build_all_answers(wb)
    print('Building Sheet 3: Hedge Highlights...')
    build_hedge_highlights(wb)
    print('Building Sheet 4: Cross-LLM Compare...')
    build_cross_llm(wb)
    print('Building Sheet 5: All-LLM-Wrong (information gap analysis)...')
    build_all_wrong(wb)

    wb.save(OUT)
    print(f'\n[Saved] {OUT}')
    print(f'  Size: {OUT.stat().st_size / 1024:.1f} KB')
