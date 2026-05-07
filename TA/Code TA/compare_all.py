import json, sys
sys.stdout.reconfigure(encoding='utf-8')

RESULTS = 'C:/Users/Ricky Wijaya/Documents/STI/Semester 8/TA/Code TA/results'

configs = [
    ('BL Llama',     'baseline_phase1_answers.json', 'baseline_phase2_custom.json'),
    ('QR Llama',     'qr_phase1_answers.json', 'qr_phase2_custom.json'),
    ('CR Llama',     'cr_phase1_answers.json', 'cr_phase2_custom.json'),
    ('BL OpenAI',    'baseline_openai_phase1_answers.json', 'baseline_openai_phase2_custom.json'),
    ('QR OpenAI',    'qr_openai_phase1_answers.json', 'qr_openai_phase2_custom.json'),
    ('CR OpenAI',    'cr_openai_phase1_answers.json', 'cr_openai_phase2_custom.json'),
    ('QR+CR OpenAI', 'combined_openai_phase1_answers.json', 'combined_openai_phase2_custom.json'),
    ('Hybrid OpenAI','hybrid_openai_phase1_answers.json', 'hybrid_openai_phase2_custom.json'),
]

data = {}
for name, p1f, p2f in configs:
    with open(f'{RESULTS}/{p1f}', encoding='utf-8') as f:
        results = json.load(f)['results']
    n = len(results)
    correct = sum(r['is_correct'] for r in results)
    pl = {}
    for lbl in ['yes','no','maybe']:
        sub = [r for r in results if r['ground_truth'] == lbl]
        pl[lbl] = (sum(r['is_correct'] for r in sub), len(sub))
    pd = {}
    for r in results:
        pd[r['predicted_label']] = pd.get(r['predicted_label'], 0) + 1
    met = {}
    try:
        with open(f'{RESULTS}/{p2f}', encoding='utf-8') as f:
            p2r = json.load(f)['results']
        for m in ['faithfulness','context_recall','answer_relevancy','context_precision']:
            vals = [r[m] for r in p2r if m in r]
            if len(vals) >= 100:
                met[m] = sum(vals)/len(vals)
    except:
        pass
    data[name] = {'acc': correct/n, 'correct': correct, 'n': n, 'pl': pl, 'pd': pd, 'met': met}


# ============================================================
# TABEL 1: LLAMA (delta vs BL Llama)
# ============================================================
bl = data['BL Llama']
lcfg = ['BL Llama','QR Llama','CR Llama']

print('TABEL 1: LLAMA 3.2 (delta vs Baseline Llama)')
print('='*95)
print(f"{'Metrik':<18} | {'BL Llama':<18} {'':>7} | {'QR Llama':<18} {'delta':>7} | {'CR Llama':<18} {'delta':>7}")
print('-'*95)

def delta_pct(v, ref):
    return f"{(v-ref)*100:+.1f}%"
def delta_abs(v, ref):
    return f"{v-ref:+.4f}"

def pla(d, lbl):
    c, t = d['pl'][lbl]
    return f"{c}/{t} ({c/t:.1%})"

# Label Accuracy
for c in lcfg:
    d = data[c]
print(f"{'Label Accuracy':<18} | {bl['acc']:.1%} ({bl['correct']}/500) {'base':>10} | {data['QR Llama']['acc']:.1%} ({data['QR Llama']['correct']}/500) {delta_pct(data['QR Llama']['acc'], bl['acc']):>7} | {data['CR Llama']['acc']:.1%} ({data['CR Llama']['correct']}/500) {delta_pct(data['CR Llama']['acc'], bl['acc']):>7}")
print(f"{'yes':<18} | {pla(bl,'yes'):<18} {'base':>7} | {pla(data['QR Llama'],'yes'):<18} {delta_pct(227/275, 249/275):>7} | {pla(data['CR Llama'],'yes'):<18} {delta_pct(245/275, 249/275):>7}")
print(f"{'no':<18} | {pla(bl,'no'):<18} {'base':>7} | {pla(data['QR Llama'],'no'):<18} {delta_pct(21/159, 28/159):>7} | {pla(data['CR Llama'],'no'):<18} {delta_pct(23/159, 28/159):>7}")
print(f"{'maybe':<18} | {pla(bl,'maybe'):<18} {'base':>7} | {pla(data['QR Llama'],'maybe'):<18} {delta_pct(2/66, 1/66):>7} | {pla(data['CR Llama'],'maybe'):<18} {delta_pct(2/66, 1/66):>7}")
print(f"{'Faithfulness':<18} | {bl['met'].get('faithfulness',0):.4f}{'':>14} {'base':>7} | {data['QR Llama']['met'].get('faithfulness',0):.4f}{'':>14} {delta_abs(data['QR Llama']['met'].get('faithfulness',0), bl['met']['faithfulness']):>7} | {data['CR Llama']['met'].get('faithfulness',0):.4f}{'':>14} {delta_abs(data['CR Llama']['met'].get('faithfulness',0), bl['met']['faithfulness']):>7}")
print(f"{'Context Recall':<18} | {bl['met'].get('context_recall',0):.4f}{'':>14} {'base':>7} | {data['QR Llama']['met'].get('context_recall',0):.4f}{'':>14} {delta_abs(data['QR Llama']['met'].get('context_recall',0), bl['met']['context_recall']):>7} | {data['CR Llama']['met'].get('context_recall',0):.4f}{'':>14} {delta_abs(data['CR Llama']['met'].get('context_recall',0), bl['met']['context_recall']):>7}")
print(f"{'Pred (y/n/m)':<18} | {bl['pd'].get('yes',0)}/{bl['pd'].get('no',0)}/{bl['pd'].get('maybe',0)}{'':>11} {'base':>7} | {data['QR Llama']['pd'].get('yes',0)}/{data['QR Llama']['pd'].get('no',0)}/{data['QR Llama']['pd'].get('maybe',0)}{'':>10} {'':>7} | {data['CR Llama']['pd'].get('yes',0)}/{data['CR Llama']['pd'].get('no',0)}/{data['CR Llama']['pd'].get('maybe',0)}{'':>10} {'':>7}")


# ============================================================
# TABEL 2: OpenAI (delta vs BL OpenAI)
# ============================================================
bl2 = data['BL OpenAI']
print('\n\nTABEL 2: GPT-4.1-MINI (delta vs Baseline OpenAI)')
print('='*140)
ocfg = ['BL OpenAI','QR OpenAI','CR OpenAI','QR+CR OpenAI','Hybrid OpenAI']
print(f"{'Metrik':<18}", end='')
for c in ocfg:
    short = c.replace(' OpenAI','')
    if c == 'BL OpenAI':
        print(f" | {short+' (base)':<22}", end='')
    else:
        print(f" | {short:<14} {'delta':>7}", end='')
print()
print('-'*140)

def row_openai(label, vals_and_deltas):
    """vals_and_deltas: list of (val_str, delta_str) per config"""
    line = f"{label:<18}"
    for i, (v, d) in enumerate(vals_and_deltas):
        if i == 0:
            line += f" | {v:<22}"
        else:
            line += f" | {v:<14} {d:>7}"
    print(line)

# Build rows
acc_vd = []
for c in ocfg:
    d = data[c]
    v = f"{d['acc']:.1%} ({d['correct']}/500)"
    delta = f"{(d['acc']-bl2['acc'])*100:+.1f}%" if c != 'BL OpenAI' else 'base'
    acc_vd.append((v, delta))
row_openai('Label Accuracy', acc_vd)

for lbl, total in [('yes', 275), ('no', 159), ('maybe', 66)]:
    vd = []
    for c in ocfg:
        d = data[c]
        corr, tot = d['pl'][lbl]
        v = f"{corr}/{tot} ({corr/tot:.1%})"
        bl_corr = bl2['pl'][lbl][0]
        delta = f"{(corr/tot - bl_corr/tot)*100:+.1f}%" if c != 'BL OpenAI' else 'base'
        vd.append((v, delta))
    row_openai(lbl, vd)

for m_name, m_key in [('Faithfulness','faithfulness'), ('Context Recall','context_recall'),
                       ('Ans Relevancy','answer_relevancy'), ('Ctx Precision','context_precision')]:
    vd = []
    for c in ocfg:
        d = data[c]
        val = d['met'].get(m_key, 0)
        v = f"{val:.4f}"
        bl_val = bl2['met'].get(m_key, 0)
        delta = f"{val-bl_val:+.4f}" if c != 'BL OpenAI' else 'base'
        vd.append((v, delta))
    row_openai(m_name, vd)

# Pred dist row
vd = []
for c in ocfg:
    d = data[c]
    v = f"{d['pd'].get('yes',0)}/{d['pd'].get('no',0)}/{d['pd'].get('maybe',0)}"
    vd.append((v, ''))
row_openai('Pred (y/n/m)', vd)


# ============================================================
# TABEL 3: RANKING
# ============================================================
print('\n\nTABEL 3: RANKING KESELURUHAN')
print('='*100)
all_sorted = sorted(data.items(), key=lambda x: -x[1]['acc'])
bl_acc = data['BL Llama']['acc']
for rank, (name, d) in enumerate(all_sorted, 1):
    f_val = d['met'].get('faithfulness', 0)
    no_c, no_t = d['pl']['no']
    delta_bll = (d['acc'] - bl_acc) * 100
    print(f"  #{rank} {name:<18} | acc={d['acc']:.1%} ({d['correct']}/500) "
          f"| faith={f_val:.4f} "
          f"| no={no_c}/{no_t} ({no_c/no_t:.1%}) "
          f"| vs BL Llama: {delta_bll:+.1f}%")
