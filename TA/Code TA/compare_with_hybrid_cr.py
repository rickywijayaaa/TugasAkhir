import json, sys
sys.stdout.reconfigure(encoding='utf-8')

RESULTS = 'C:/Users/Ricky Wijaya/Documents/STI/Semester 8/TA/Code TA/results'

configs = [
    ('BL Llama',        'baseline_phase1_answers.json', 'baseline_phase2_custom.json'),
    ('QR Llama',        'qr_phase1_answers.json', 'qr_phase2_custom.json'),
    ('CR Llama',        'cr_phase1_answers.json', 'cr_phase2_custom.json'),
    ('BL OpenAI',       'baseline_openai_phase1_answers.json', 'baseline_openai_phase2_custom.json'),
    ('QR OpenAI',       'qr_openai_phase1_answers.json', 'qr_openai_phase2_custom.json'),
    ('CR OpenAI',       'cr_openai_phase1_answers.json', 'cr_openai_phase2_custom.json'),
    ('QR+CR OpenAI',    'combined_openai_phase1_answers.json', 'combined_openai_phase2_custom.json'),
    ('Hybrid OpenAI',   'hybrid_openai_phase1_answers.json', 'hybrid_openai_phase2_custom.json'),
    ('Hybrid+CR OpenAI','hybrid_cr_openai_phase1_answers.json', 'hybrid_cr_openai_phase2_custom.json'),
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


print('\n' + '='*60)
print('RANKING KESELURUHAN (9 Konfigurasi)')
print('='*60)
bl_acc = data['BL Llama']['acc']
bl2_acc = data['BL OpenAI']['acc']
all_sorted = sorted(data.items(), key=lambda x: -x[1]['acc'])
for rank, (name, d) in enumerate(all_sorted, 1):
    f_val = d['met'].get('faithfulness', 0)
    no_c, no_t = d['pl']['no']
    delta_bll = (d['acc'] - bl_acc) * 100
    delta_blo = (d['acc'] - bl2_acc) * 100 if 'OpenAI' in name else None
    blo_str = f' | vs BL OpenAI: {delta_blo:+.1f}%' if delta_blo is not None else ''
    print(f"  #{rank} {name:<18} | acc={d['acc']:.1%} ({d['correct']}/500) "
          f"| faith={f_val:.4f}{blo_str}")


print('\n\n' + '='*140)
print('TABEL: GPT-4.1-MINI KONFIGURASI (delta vs Baseline OpenAI)')
print('='*140)
bl2 = data['BL OpenAI']
ocfg = ['BL OpenAI','QR OpenAI','CR OpenAI','QR+CR OpenAI','Hybrid OpenAI','Hybrid+CR OpenAI']

print(f"{'Metrik':<16}", end='')
for c in ocfg:
    short = c.replace(' OpenAI','').replace('Baseline','BL')
    if c == 'BL OpenAI':
        print(f" | {short+' (base)':<20}", end='')
    else:
        print(f" | {short:<12} {'delta':>7}", end='')
print()
print('-'*140)

def row_oa(label, vals, deltas):
    line = f"{label:<16}"
    for i, c in enumerate(ocfg):
        if i == 0:
            line += f" | {vals[i]:<20}"
        else:
            line += f" | {vals[i]:<12} {deltas[i]:>7}"
    print(line)

# Accuracy
vals = [f"{data[c]['acc']:.1%} ({data[c]['correct']}/500)" for c in ocfg]
deltas = ['base'] + [f"{(data[c]['acc']-bl2['acc'])*100:+.1f}%" for c in ocfg[1:]]
row_oa('Label Accuracy', vals, deltas)

for lbl, total in [('yes', 275), ('no', 159), ('maybe', 66)]:
    vals = [f"{data[c]['pl'][lbl][0]}/{total} ({data[c]['pl'][lbl][0]/total:.1%})" for c in ocfg]
    deltas = ['base'] + [f"{(data[c]['pl'][lbl][0]/total - bl2['pl'][lbl][0]/total)*100:+.1f}%" for c in ocfg[1:]]
    row_oa(lbl, vals, deltas)

for m_name, m_key in [('Faithfulness','faithfulness'), ('Context Recall','context_recall'),
                       ('Ans Relevancy','answer_relevancy'), ('Ctx Precision','context_precision')]:
    vals = [f"{data[c]['met'].get(m_key, 0):.4f}" for c in ocfg]
    deltas = ['base'] + [f"{data[c]['met'].get(m_key, 0) - bl2['met'].get(m_key, 0):+.4f}" for c in ocfg[1:]]
    row_oa(m_name, vals, deltas)

vals = [f"{data[c]['pd'].get('yes',0)}/{data[c]['pd'].get('no',0)}/{data[c]['pd'].get('maybe',0)}" for c in ocfg]
deltas = [''] * len(ocfg)
row_oa('Pred (y/n/m)', vals, deltas)


# ============================================================
# HYBRID+CR vs HYBRID (delta)
# ============================================================
print('\n\n' + '='*90)
print('HYBRID+CR vs HYBRID (kontribusi tambahan CrossEncoder di atas Hybrid)')
print('='*90)
hyb = data['Hybrid OpenAI']
hcr = data['Hybrid+CR OpenAI']
print(f"\n{'Metrik':<20} | {'Hybrid':<20} | {'Hybrid+CR':<20} | {'Delta':>12}")
print('-'*90)
print(f"{'Label Accuracy':<20} | {hyb['acc']:.1%} ({hyb['correct']}/500){'':>5} | {hcr['acc']:.1%} ({hcr['correct']}/500){'':>5} | {(hcr['acc']-hyb['acc'])*100:+.1f}%")
for lbl, total in [('yes', 275), ('no', 159), ('maybe', 66)]:
    h = hyb['pl'][lbl][0]
    c = hcr['pl'][lbl][0]
    print(f"{lbl:<20} | {h}/{total} ({h/total:.1%}){'':>9} | {c}/{total} ({c/total:.1%}){'':>9} | {(c/total-h/total)*100:+.1f}%")
for m_name, m_key in [('Faithfulness','faithfulness'), ('Context Recall','context_recall'),
                       ('Ans Relevancy','answer_relevancy'), ('Ctx Precision','context_precision')]:
    h = hyb['met'].get(m_key, 0)
    c = hcr['met'].get(m_key, 0)
    print(f"{m_name:<20} | {h:.4f}{'':>14} | {c:.4f}{'':>14} | {c-h:+.4f}")


# ============================================================
# HYBRID+CR vs CR OpenAI (retrieval trio vs CR only)
# ============================================================
print('\n\n' + '='*90)
print('HYBRID+CR vs CR OpenAI (kontribusi tambahan Dense retrieval di atas CR)')
print('='*90)
cr = data['CR OpenAI']
print(f"\n{'Metrik':<20} | {'CR OpenAI':<20} | {'Hybrid+CR':<20} | {'Delta':>12}")
print('-'*90)
print(f"{'Label Accuracy':<20} | {cr['acc']:.1%} ({cr['correct']}/500){'':>5} | {hcr['acc']:.1%} ({hcr['correct']}/500){'':>5} | {(hcr['acc']-cr['acc'])*100:+.1f}%")
for lbl, total in [('yes', 275), ('no', 159), ('maybe', 66)]:
    h = cr['pl'][lbl][0]
    c = hcr['pl'][lbl][0]
    print(f"{lbl:<20} | {h}/{total} ({h/total:.1%}){'':>9} | {c}/{total} ({c/total:.1%}){'':>9} | {(c/total-h/total)*100:+.1f}%")
for m_name, m_key in [('Faithfulness','faithfulness'), ('Context Recall','context_recall'),
                       ('Ans Relevancy','answer_relevancy'), ('Ctx Precision','context_precision')]:
    h = cr['met'].get(m_key, 0)
    c = hcr['met'].get(m_key, 0)
    print(f"{m_name:<20} | {h:.4f}{'':>14} | {c:.4f}{'':>14} | {c-h:+.4f}")


# ============================================================
# DISTRIBUSI
# ============================================================
print('\n\n' + '='*70)
print('DISTRIBUSI PREDIKSI vs GROUND TRUTH')
print('='*70)
print(f'Ground Truth: yes=275 (55%), no=159 (32%), maybe=66 (13%)\n')
for name in ['BL OpenAI','QR OpenAI','CR OpenAI','QR+CR OpenAI','Hybrid OpenAI','Hybrid+CR OpenAI']:
    d = data[name]
    y = d['pd'].get('yes',0)
    n = d['pd'].get('no',0)
    m = d['pd'].get('maybe',0)
    dev = abs(y-275) + abs(n-159) + abs(m-66)
    print(f"  {name:<20}: y={y:>3} n={n:>3} m={m:>3} | total deviasi={dev}")
