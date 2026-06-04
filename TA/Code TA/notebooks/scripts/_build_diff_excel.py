"""Build Excel showing differences between Original vs BM25-Expanded baseline OpenAI.

Sheets:
  1_Overview          - summary stats + comparison
  2_All_Wrong         - 136 sampel salah di expanded run (drill down)
  3_NEW_Correct       - 30 sampel gain berkat expansion (was wrong, now correct)
  4_NEW_Wrong         - 12 sampel loss dari expansion (was correct, now wrong)
  5_Both_Wrong        - 124 sampel masih salah (persistent failure)
  6_Per_Class_Stats   - detail per kelas yes/no/maybe
"""
import json
from pathlib import Path
from collections import Counter

import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

HERE = Path(__file__).parent
RESULTS = HERE.parent.parent / 'results'
OUT = HERE / 'comparison_original_vs_expanded.xlsx'

# ============================================================
# Style
# ============================================================
NAVY = '1F4E79'; GOLD = 'D4A957'; GREEN = '4ADE80'; RED = 'F87171'
LRED = 'FEE2E2'; LGRN = 'D1FAE5'; LYEL = 'FEF3C7'; LBLU = 'DBEAFE'
LGRY = 'F3F4F6'; LPUR = 'E9D5FF'

HEADER_FONT = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
HEADER_FILL = PatternFill('solid', fgColor=NAVY)
HEADER_ALIGN = Alignment(horizontal='center', vertical='center', wrap_text=True)
BODY_FONT = Font(name='Calibri', size=10)
WRAP = Alignment(vertical='top', wrap_text=True)
CENTER = Alignment(horizontal='center', vertical='center', wrap_text=True)
THIN = Side(border_style='thin', color='CCCCCC')
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def style_header(ws, row=1, last_col=None):
    if last_col is None: last_col = ws.max_column
    for col in range(1, last_col + 1):
        c = ws.cell(row=row, column=col)
        c.font = HEADER_FONT; c.fill = HEADER_FILL
        c.alignment = HEADER_ALIGN; c.border = BORDER
    ws.row_dimensions[row].height = 32


def label_color(label):
    return {'yes': LGRN, 'no': LRED, 'maybe': LYEL}.get(label, '')


# ============================================================
# Load data
# ============================================================
print('Loading results...')
with open(RESULTS / 'baseline_openai_phase1_answers.json', encoding='utf-8') as f:
    old = json.load(f)['results']
with open(RESULTS / 'BM25_Expansion' / 'bm25expanded_baseline_openai_phase1_answers.json', encoding='utf-8') as f:
    new = json.load(f)['results']

old_idx = {r['idx']: r for r in old}
new_idx = {r['idx']: r for r in new}

# Categorize
new_correct, new_wrong, both_wrong, both_correct = [], [], [], []
for idx in old_idx:
    o = old_idx[idx]; n = new_idx.get(idx)
    if not n: continue
    if o['is_correct'] and n['is_correct']: both_correct.append(idx)
    elif not o['is_correct'] and not n['is_correct']: both_wrong.append(idx)
    elif not o['is_correct'] and n['is_correct']: new_correct.append(idx)
    elif o['is_correct'] and not n['is_correct']: new_wrong.append(idx)

print(f'  Both correct: {len(both_correct)}, Both wrong: {len(both_wrong)}')
print(f'  NEW correct (gain): {len(new_correct)}, NEW wrong (loss): {len(new_wrong)}')


# ============================================================
# Sheet builders
# ============================================================
def build_overview(wb):
    ws = wb.create_sheet('1_Overview')
    ws['A1'] = 'BM25 EXPANSION - Comparison vs Original (Baseline OpenAI, n=500 PubMedQA)'
    ws['A1'].font = Font(size=15, bold=True, color=NAVY)
    ws.merge_cells('A1:G1')

    ws['A3'] = ('Hipotesis: acronym expansion preprocessing menyelesaikan vocabulary '
                'mismatch BM25 untuk biomedical text dengan banyak akronim, sehingga '
                'METHODS dan RESULTS dari paper sumber lebih sering masuk top-5.')
    ws['A3'].font = Font(size=11, italic=True, color='6B7280')
    ws['A3'].alignment = WRAP
    ws.merge_cells('A3:G3')
    ws.row_dimensions[3].height = 36

    # Summary table
    headers = ['Metric', 'Original BM25', 'Expanded BM25', 'Δ (pp)', 'Catatan']
    for i, h in enumerate(headers, 1):
        ws.cell(row=5, column=i, value=h)
    style_header(ws, row=5, last_col=len(headers))

    # Compute stats
    def compute(data):
        n = len(data)
        acc = sum(r['is_correct'] for r in data) / n
        accs = {}
        for lbl in ['yes', 'no', 'maybe']:
            sub = [r for r in data if r['ground_truth'] == lbl]
            accs[lbl] = sum(r['is_correct'] for r in sub) / len(sub) if sub else 0
        bal = sum(accs.values()) / 3
        return {'n': n, 'acc': acc, 'bal': bal, **{f'acc_{k}': v for k, v in accs.items()}}

    s_old = compute(old); s_new = compute(new)

    rows = [
        ('Overall accuracy',   s_old['acc'],       s_new['acc'],        ''),
        ('Balanced accuracy',  s_old['bal'],       s_new['bal'],        'Rata-rata akurasi 3 kelas'),
        ('Acc yes class',      s_old['acc_yes'],   s_new['acc_yes'],    'Kelas mayoritas (n=275)'),
        ('Acc no class',       s_old['acc_no'],    s_new['acc_no'],     'Kelas minoritas (n=159)'),
        ('Acc maybe class',    s_old['acc_maybe'], s_new['acc_maybe'],  'Kelas tersulit (n=66)'),
    ]
    row = 6
    for name, vo, vn, note in rows:
        delta = (vn - vo) * 100
        ws.cell(row=row, column=1, value=name).font = Font(size=11, bold=True, color=NAVY)
        ws.cell(row=row, column=2, value=f'{vo:.1%}')
        ws.cell(row=row, column=3, value=f'{vn:.1%}')
        ws.cell(row=row, column=4, value=f'{delta:+.2f} pp')
        ws.cell(row=row, column=5, value=note).alignment = Alignment(vertical='center')
        # Color delta
        if delta > 0.5:
            ws.cell(row=row, column=4).fill = PatternFill('solid', fgColor=LGRN)
            ws.cell(row=row, column=4).font = Font(bold=True, color='065F46')
        elif delta < -0.5:
            ws.cell(row=row, column=4).fill = PatternFill('solid', fgColor=LRED)
            ws.cell(row=row, column=4).font = Font(bold=True, color='991B1B')
        for c in range(1, 6):
            ws.cell(row=row, column=c).border = BORDER
            ws.cell(row=row, column=c).alignment = Alignment(vertical='center', horizontal='left' if c == 1 or c == 5 else 'center')
        row += 1

    # Sample-level changes box
    row += 2
    ws.cell(row=row, column=1, value='SAMPLE-LEVEL CHANGES').font = Font(size=12, bold=True, color=NAVY)
    row += 1
    rows = [
        ('Both correct (no change)', len(both_correct), 'Stabil — sampel correct di kedua versi'),
        ('Both wrong (no change)',   len(both_wrong),   'Persistent failure — bukan masalah retrieval'),
        ('NEW correct (gain)',       len(new_correct),  'Berkat expansion — drilldown di Sheet 3'),
        ('NEW wrong (loss)',         len(new_wrong),    'Regresi dari expansion — drilldown di Sheet 4'),
    ]
    for name, count, note in rows:
        ws.cell(row=row, column=1, value=name)
        ws.cell(row=row, column=2, value=count)
        ws.cell(row=row, column=3, value=note)
        if 'gain' in name.lower():
            ws.cell(row=row, column=2).fill = PatternFill('solid', fgColor=LGRN)
        elif 'loss' in name.lower():
            ws.cell(row=row, column=2).fill = PatternFill('solid', fgColor=LRED)
        ws.cell(row=row, column=2).alignment = CENTER
        ws.cell(row=row, column=2).font = Font(bold=True, size=12)
        ws.cell(row=row, column=3).alignment = Alignment(vertical='center', wrap_text=True)
        ws.merge_cells(start_row=row, start_column=3, end_row=row, end_column=5)
        row += 1

    # Net gain
    row += 1
    net = len(new_correct) - len(new_wrong)
    ws.cell(row=row, column=1, value='Net gain').font = Font(size=12, bold=True, color=NAVY)
    ws.cell(row=row, column=2, value=f'+{net}' if net >= 0 else str(net))
    ws.cell(row=row, column=2).font = Font(size=14, bold=True, color='065F46' if net > 0 else '991B1B')
    ws.cell(row=row, column=2).alignment = CENTER
    ws.cell(row=row, column=3, value=f'= {(net / len(new) * 100):+.2f} pp peningkatan')

    # Navigation
    row += 3
    ws.cell(row=row, column=1, value='NAVIGASI SHEET').font = Font(size=12, bold=True, color=NAVY)
    row += 1
    nav = [
        '→ Sheet "2_All_Wrong"     : 136 sampel masih salah di expanded (browse + filter)',
        '→ Sheet "3_NEW_Correct"   : 30 sampel jadi correct berkat expansion (drilldown gain)',
        '→ Sheet "4_NEW_Wrong"     : 12 sampel jadi wrong (regresi - investigate)',
        '→ Sheet "5_Both_Wrong"    : 124 persistent failure (bukan masalah retrieval)',
        '→ Sheet "6_Per_Class"     : detail per kelas yes/no/maybe',
    ]
    for n in nav:
        ws.cell(row=row, column=1, value=n).font = Font(size=11)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=5)
        row += 1

    # Column widths
    widths = {'A': 28, 'B': 18, 'C': 18, 'D': 14, 'E': 36, 'F': 14, 'G': 14}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w


def build_diff_sheet(wb, sheet_name, idx_list, title, intro):
    """Sheet that shows old vs new comparison for given idx list."""
    ws = wb.create_sheet(sheet_name)
    ws['A1'] = title
    ws['A1'].font = Font(size=14, bold=True, color=NAVY)
    ws.merge_cells('A1:K1')

    ws['A2'] = intro
    ws['A2'].font = Font(size=10, italic=True, color='6B7280')
    ws['A2'].alignment = WRAP
    ws.merge_cells('A2:K2')
    ws.row_dimensions[2].height = 36

    headers = ['Idx', 'PMID', 'GT', 'Old Pred', 'New Pred',
               'Question', 'OLD: Source paper sections', 'NEW: Source paper sections',
               'OLD answer (preview)', 'NEW answer (preview)', 'Reference']
    for i, h in enumerate(headers, 1):
        ws.cell(row=4, column=i, value=h)
    style_header(ws, row=4, last_col=len(headers))

    for row_i, idx in enumerate(idx_list, 5):
        o = old_idx.get(idx); n = new_idx.get(idx)
        if not o or not n: continue

        # Source paper sections - find which sections of the question's paper were retrieved
        target_pubid = str(o['pubid'])

        old_sections = []
        if 'context_pubids' in o:
            for pid, sec in zip(o['context_pubids'], o.get('context_sections', ['?']*5)):
                if pid == target_pubid:
                    old_sections.append(sec)
        else:
            # Original baseline doesn't store context_pubids — try to detect via reference
            old_sections = ['(metadata not available)']

        new_sections = []
        for pid, sec in zip(n['context_pubids'], n['context_sections']):
            if pid == target_pubid:
                new_sections.append(sec)

        cells = [
            idx, target_pubid, o['ground_truth'],
            o['predicted_label'], n['predicted_label'],
            o['question'],
            ', '.join(old_sections) if old_sections else '(none from source)',
            ', '.join(new_sections) if new_sections else '(none from source)',
            o['answer'][:300] + ('...' if len(o['answer']) > 300 else ''),
            n['answer'][:300] + ('...' if len(n['answer']) > 300 else ''),
            o.get('reference', '-'),
        ]
        for col_i, v in enumerate(cells, 1):
            c = ws.cell(row=row_i, column=col_i, value=v)
            c.font = BODY_FONT; c.alignment = WRAP; c.border = BORDER

        # PMID hyperlink
        ws.cell(row=row_i, column=2).hyperlink = f'https://pubmed.ncbi.nlm.nih.gov/{target_pubid}/'
        ws.cell(row=row_i, column=2).font = Font(color=NAVY, underline='single', size=10)

        # GT color
        ws.cell(row=row_i, column=3).fill = PatternFill('solid', fgColor=label_color(o['ground_truth']))
        # Old pred: green if correct vs GT, red if wrong
        ws.cell(row=row_i, column=4).fill = PatternFill('solid',
            fgColor=LGRN if o['predicted_label'] == o['ground_truth'] else LRED)
        # New pred
        ws.cell(row=row_i, column=5).fill = PatternFill('solid',
            fgColor=LGRN if n['predicted_label'] == n['ground_truth'] else LRED)
        for col_i in [3, 4, 5]:
            ws.cell(row=row_i, column=col_i).alignment = CENTER

        # Highlight if source sections are different (showed expansion impact)
        if set(old_sections) != set(new_sections) and new_sections:
            ws.cell(row=row_i, column=8).fill = PatternFill('solid', fgColor=LGRN)
            ws.cell(row=row_i, column=8).font = Font(bold=True, size=10, color='065F46')

    widths = [6, 12, 6, 9, 9, 38, 30, 30, 50, 50, 35]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = 'F5'
    if idx_list:
        ws.auto_filter.ref = f'A4:K{4 + len(idx_list)}'


def build_per_class(wb):
    ws = wb.create_sheet('6_Per_Class_Stats')
    ws['A1'] = 'Per-Class Accuracy Detail'
    ws['A1'].font = Font(size=14, bold=True, color=NAVY)
    ws.merge_cells('A1:H1')

    ws['A3'] = ('Tabel per-class menunjukkan di kelas mana acronym expansion paling membantu. '
                'Hipotesis: kelas "no" paling banyak diuntungkan karena BM25 expansion mengangkat '
                'METHODS/RESULTS yang berisi negative findings.')
    ws['A3'].font = Font(size=10, italic=True, color='6B7280')
    ws['A3'].alignment = WRAP
    ws.merge_cells('A3:H3')
    ws.row_dimensions[3].height = 32

    headers = ['Kelas GT', 'n_total', 'Old correct', 'New correct',
               'Old acc', 'New acc', 'Δ (pp)', 'Net gain']
    for i, h in enumerate(headers, 1):
        ws.cell(row=5, column=i, value=h)
    style_header(ws, row=5, last_col=len(headers))

    row = 6
    for lbl in ['yes', 'no', 'maybe']:
        sub_old = [r for r in old if r['ground_truth'] == lbl]
        sub_new = [r for r in new if r['ground_truth'] == lbl]
        n = len(sub_old)
        c_old = sum(r['is_correct'] for r in sub_old)
        c_new = sum(r['is_correct'] for r in sub_new)
        delta = (c_new - c_old) / n * 100 if n else 0
        net = c_new - c_old

        ws.cell(row=row, column=1, value=lbl).fill = PatternFill('solid', fgColor=label_color(lbl))
        ws.cell(row=row, column=2, value=n)
        ws.cell(row=row, column=3, value=c_old)
        ws.cell(row=row, column=4, value=c_new)
        ws.cell(row=row, column=5, value=f'{c_old/n:.1%}')
        ws.cell(row=row, column=6, value=f'{c_new/n:.1%}')
        ws.cell(row=row, column=7, value=f'{delta:+.2f} pp')
        ws.cell(row=row, column=8, value=f'{net:+d}')
        if delta > 1:
            ws.cell(row=row, column=7).fill = PatternFill('solid', fgColor=LGRN)
            ws.cell(row=row, column=8).fill = PatternFill('solid', fgColor=LGRN)
        elif delta < -1:
            ws.cell(row=row, column=7).fill = PatternFill('solid', fgColor=LRED)
            ws.cell(row=row, column=8).fill = PatternFill('solid', fgColor=LRED)
        for c in range(1, 9):
            ws.cell(row=row, column=c).border = BORDER
            ws.cell(row=row, column=c).alignment = CENTER
            if c >= 5:
                ws.cell(row=row, column=c).font = Font(bold=True)
        row += 1

    # Insight box
    row += 2
    ws.cell(row=row, column=1, value='INSIGHT KUNCI').font = Font(size=12, bold=True, color=NAVY)
    row += 1
    insights = [
        '• Yes class: hampir tidak berubah (~88.5%) — sudah baik di baseline, tidak banyak ruang improve.',
        '• No class: gain TERBESAR (~9pp) — banyak temuan negative ada di RESULTS section yang sebelumnya tidak ter-retrieve.',
        '• Maybe class: gain modest (~3pp) — masih bottleneck di prompt anti-maybe bias, bukan retrieval.',
        '• Kesimpulan: BM25 expansion paling efektif untuk pertanyaan yang jawabannya negatif/null finding.',
    ]
    for ins in insights:
        ws.cell(row=row, column=1, value=ins).font = Font(size=11)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        ws.cell(row=row, column=1).alignment = WRAP
        row += 1

    widths = [12, 10, 12, 12, 10, 10, 12, 12]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


# ============================================================
# Build workbook
# ============================================================
if __name__ == '__main__':
    wb = Workbook()
    wb.remove(wb.active)

    print('Building Sheet 1: Overview...')
    build_overview(wb)

    print(f'Building Sheet 2: All Wrong ({sum(1 for r in new if not r["is_correct"])} samples)...')
    all_wrong_idx = [idx for idx in new_idx if not new_idx[idx]['is_correct']]
    build_diff_sheet(wb, '2_All_Wrong', all_wrong_idx,
                     f'Semua sampel SALAH di expanded run ({len(all_wrong_idx)} dari 500)',
                     'Drilldown semua kasus yang masih salah setelah BM25 expansion. '
                     'Filter via auto-filter: bandingkan old vs new prediction, lihat sections '
                     'mana dari paper sumber yang ter-retrieve.')

    print(f'Building Sheet 3: NEW Correct ({len(new_correct)} samples)...')
    build_diff_sheet(wb, '3_NEW_Correct', new_correct,
                     f'NEW CORRECT: {len(new_correct)} sampel jadi correct berkat expansion',
                     'Sampel di mana original SALAH tapi expanded BENAR. '
                     'Highlight hijau di kolom "NEW: Source paper sections" menandakan '
                     'expansion berhasil retrieve sections baru yang membantu LLM jawab benar.')

    print(f'Building Sheet 4: NEW Wrong ({len(new_wrong)} samples)...')
    build_diff_sheet(wb, '4_NEW_Wrong', new_wrong,
                     f'NEW WRONG (regresi): {len(new_wrong)} sampel jadi salah dari expansion',
                     'Sampel di mana original BENAR tapi expanded SALAH. '
                     'Investigate: apakah expansion menambahkan noise yang membingungkan LLM? '
                     'Atau ada false positive expansion?')

    print(f'Building Sheet 5: Both Wrong ({len(both_wrong)} samples)...')
    build_diff_sheet(wb, '5_Both_Wrong', both_wrong,
                     f'PERSISTENT FAILURE: {len(both_wrong)} sampel masih salah di kedua versi',
                     'Bukan masalah BM25 expansion. Kemungkinan: information gap struktural '
                     '(jawaban di CONCLUSIONS yang tidak di-retrieve), reasoning failure LLM, '
                     'atau ground truth ambiguous.')

    print('Building Sheet 6: Per-Class Stats...')
    build_per_class(wb)

    wb.save(OUT)
    print(f'\n[Saved] {OUT}')
    print(f'  Size: {OUT.stat().st_size / 1024:.1f} KB')
