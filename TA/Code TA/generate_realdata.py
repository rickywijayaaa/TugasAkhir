import json

with open('C:/Users/Ricky Wijaya/Documents/STI/Semester 8/TA/Code TA/ui/src/data/_samples.json') as f:
    samples = json.load(f)

# Shorten answers and contexts
for s in samples:
    for cfg_key in s['configs']:
        ans = s['configs'][cfg_key].get('answer')
        if ans and len(ans) > 800:
            s['configs'][cfg_key]['answer'] = ans[:800] + '...'
    for doc in s.get('retrieved_docs', []):
        if doc.get('full_content') and len(doc['full_content']) > 400:
            doc['content'] = doc['full_content'][:400] + '...'
        elif doc.get('content') and len(doc['content']) > 400:
            doc['content'] = doc['content'][:400] + '...'
        doc.pop('full_content', None)

short_labels = {
    29: 'Visceral Adipose Tissue',
    13: 'Primary Care Cancer Risk',
    57: 'HPV & Pterygium',
    0: 'Mitochondria Lace Plant',
    1: 'Landolt C vs Snellen E',
}
for s in samples:
    s['short_label'] = short_labels.get(s['idx'], f"Sample {s['idx']}")

# RAGAS violation annotations (handcrafted for illustration)
ragas_annotations = {
    0: {
        'answer_sentences': [
            {
                'text': 'The abstracts indicate that mitochondrial dynamics, including changes in distribution, motility, and membrane potential, are closely associated with programmed cell death (PCD) in lace plant leaves.',
                'violations': []
            },
            {
                'text': 'Additionally, inhibition of mitochondrial permeability transition pore formation via cyclosporine A treatment reduced leaf perforations and altered mitochondrial behavior, suggesting mitochondria actively contribute to leaf remodeling during PCD.',
                'violations': ['faithfulness'],
                'reason_faithfulness': 'Klaim "inhibition via cyclosporine A reduced leaf perforations" tidak secara eksplisit didukung oleh konteks yang disediakan. Konteks menyebutkan CsA tetapi hasil spesifik ini tidak ditemukan.',
            },
        ],
        'reference_sentences': [
            {
                'text': 'Results depicted mitochondrial dynamics in vivo as PCD progresses within the lace plant.',
                'covered_by_context': True,
            },
            {
                'text': 'This is the first report of mitochondria and chloroplasts moving on transvacuolar strands to form a ring structure surrounding the nucleus during developmental PCD.',
                'covered_by_context': False,
                'reason': 'Detail "ring structure surrounding nucleus" tidak ditemukan dalam 5 konteks teratas. Retrieval melewatkan fakta spesifik ini.',
            },
            {
                'text': 'For the first time, we have shown the feasibility for the use of CsA in a whole plant system.',
                'covered_by_context': True,
            },
            {
                'text': 'Our findings implicate the mitochondria as playing a critical and early role in developmentally regulated PCD.',
                'covered_by_context': True,
            },
        ],
        'contexts_relevance': [
            {'doc_idx': 0, 'relevant': True,  'reason': 'Membahas peran mitokondria dalam PCD lace plant - fit langsung dengan pertanyaan.'},
            {'doc_idx': 1, 'relevant': True,  'reason': 'Menjelaskan dinamika mitokondria in vivo selama PCD - highly relevant.'},
            {'doc_idx': 2, 'relevant': False, 'reason': 'Dokumen tentang topik berbeda, bukan lace plant PCD spesifik.'},
            {'doc_idx': 3, 'relevant': False, 'reason': 'Tidak mengandung informasi tentang mitochondria atau leaf remodeling.'},
            {'doc_idx': 4, 'relevant': True,  'reason': 'Membahas metodologi PCD observation yang mendukung jawaban.'},
        ],
    },
    29: {
        'answer_sentences': [
            {
                'text': 'The provided studies indicate that measuring visceral adipose tissue (VAT) area at a specific CT slice location (3 cm above the lower margin of L3) shows a strong correlation with VAT volume (r = 0.853) and body weight changes (r = 0.902).',
                'violations': [],
            },
            {
                'text': 'This suggests that VAT area measurement at a single level can effectively represent VAT volume.',
                'violations': [],
            },
        ],
        'reference_sentences': [
            {
                'text': 'VAT area measurement at a single level 3 cm above the lower margin of the L3 vertebra is feasible and can reflect changes in VAT volume and body weight.',
                'covered_by_context': True,
            },
        ],
        'contexts_relevance': [
            {'doc_idx': 0, 'relevant': True, 'reason': 'Menjelaskan tujuan studi untuk mencari CT slice optimal representasi VAT.'},
            {'doc_idx': 1, 'relevant': True, 'reason': 'Membahas metodologi pengukuran VAT dan korelasi dengan volume.'},
            {'doc_idx': 2, 'relevant': True, 'reason': 'Mengandung hasil korelasi kuantitatif yang dikutip jawaban.'},
            {'doc_idx': 3, 'relevant': True, 'reason': 'Supporting document tentang subject characteristics.'},
            {'doc_idx': 4, 'relevant': True, 'reason': 'Konteks metodologis tentang CT slice location.'},
        ],
    },
}

for s in samples:
    ann = ragas_annotations.get(s['idx'])
    if ann:
        s['ragas_annotations'] = ann

js_content = """// Auto-generated from results JSON + handcrafted RAGAS annotations

// AGGREGATE STATS (9 konfigurasi, 500 sampel)
export const AGGREGATE_STATS = {
  baseline_llama:     { accuracy: 55.6, hallucination: 44.4, faithfulness: 0.5596, contextRecall: 0.8222, answerRelevancy: null, contextPrecision: null, correctCount: 278, model: 'Llama 3.2' },
  qr_llama:           { accuracy: 50.0, hallucination: 50.0, faithfulness: 0.5381, contextRecall: 0.7851, answerRelevancy: null, contextPrecision: null, correctCount: 250, model: 'Llama 3.2' },
  cr_llama:           { accuracy: 54.0, hallucination: 46.0, faithfulness: 0.5986, contextRecall: 0.8422, answerRelevancy: null, contextPrecision: null, correctCount: 270, model: 'Llama 3.2' },
  baseline_openai:    { accuracy: 65.4, hallucination: 34.6, faithfulness: 0.8933, contextRecall: 0.7914, answerRelevancy: 0.9709, contextPrecision: 0.6983, correctCount: 327, model: 'GPT-4.1-mini' },
  qr_openai:          { accuracy: 65.4, hallucination: 34.6, faithfulness: 0.8958, contextRecall: 0.7819, answerRelevancy: 0.9714, contextPrecision: 0.6696, correctCount: 327, model: 'GPT-4.1-mini' },
  cr_openai:          { accuracy: 68.4, hallucination: 31.6, faithfulness: 0.8980, contextRecall: 0.8203, answerRelevancy: 0.9753, contextPrecision: 0.7306, correctCount: 342, model: 'GPT-4.1-mini' },
  qr_cr_openai:       { accuracy: 68.2, hallucination: 31.8, faithfulness: 0.8102, contextRecall: 0.7373, answerRelevancy: 0.8963, contextPrecision: 0.6758, correctCount: 341, model: 'GPT-4.1-mini' },
  hybrid_openai:      { accuracy: 69.2, hallucination: 30.8, faithfulness: 0.8930, contextRecall: 0.7998, answerRelevancy: 0.9835, contextPrecision: 0.7365, correctCount: 346, model: 'GPT-4.1-mini' },
  hybrid_cr_openai:   { accuracy: 69.8, hallucination: 30.2, faithfulness: 0.8837, contextRecall: 0.8259, answerRelevancy: 0.9800, contextPrecision: 0.7434, correctCount: 349, model: 'GPT-4.1-mini' },
}

// CONFIG METADATA
export const CONFIGS = [
  { key: 'baseline_llama',   label: 'Baseline (BM25)',          model: 'Llama 3.2',    techniques: [],                                       color: '#6b7280', useCrossEncoder: false, useHybrid: false, useQR: false },
  { key: 'qr_llama',         label: '+ Query Rewriting',        model: 'Llama 3.2',    techniques: ['QR'],                                    color: '#94a3b8', useCrossEncoder: false, useHybrid: false, useQR: true },
  { key: 'cr_llama',         label: '+ Context Reranking',      model: 'Llama 3.2',    techniques: ['CR'],                                    color: '#64748b', useCrossEncoder: true,  useHybrid: false, useQR: false },
  { key: 'baseline_openai',  label: 'Baseline (BM25)',          model: 'GPT-4.1-mini', techniques: [],                                        color: '#a78bfa', useCrossEncoder: false, useHybrid: false, useQR: false },
  { key: 'qr_openai',        label: '+ Query Rewriting',        model: 'GPT-4.1-mini', techniques: ['QR'],                                    color: '#60a5fa', useCrossEncoder: false, useHybrid: false, useQR: true },
  { key: 'cr_openai',        label: '+ Context Reranking',      model: 'GPT-4.1-mini', techniques: ['CR'],                                    color: '#2dd4bf', useCrossEncoder: true,  useHybrid: false, useQR: false },
  { key: 'qr_cr_openai',     label: '+ QR + CR',                model: 'GPT-4.1-mini', techniques: ['QR','CR'],                               color: '#c084fc', useCrossEncoder: true,  useHybrid: false, useQR: true },
  { key: 'hybrid_openai',    label: '+ Hybrid Retrieval',       model: 'GPT-4.1-mini', techniques: ['Hybrid'],                                color: '#fbbf24', useCrossEncoder: false, useHybrid: true,  useQR: false },
  { key: 'hybrid_cr_openai', label: '+ Hybrid + CR',            model: 'GPT-4.1-mini', techniques: ['Hybrid','CR'],                           color: '#f97316', useCrossEncoder: true,  useHybrid: true,  useQR: false },
]

// GROUND TRUTH DISTRIBUTION
export const GROUND_TRUTH_DIST = { yes: 275, no: 159, maybe: 66, total: 500 }

// PER-LABEL ACCURACY
export const PER_LABEL_ACCURACY = {
  baseline_llama:     { yes: { correct: 249, total: 275 }, no: { correct: 28, total: 159 }, maybe: { correct: 1, total: 66 } },
  qr_llama:           { yes: { correct: 227, total: 275 }, no: { correct: 21, total: 159 }, maybe: { correct: 2, total: 66 } },
  cr_llama:           { yes: { correct: 245, total: 275 }, no: { correct: 23, total: 159 }, maybe: { correct: 2, total: 66 } },
  baseline_openai:    { yes: { correct: 229, total: 275 }, no: { correct: 93, total: 159 }, maybe: { correct: 5, total: 66 } },
  qr_openai:          { yes: { correct: 222, total: 275 }, no: { correct: 96, total: 159 }, maybe: { correct: 9, total: 66 } },
  cr_openai:          { yes: { correct: 238, total: 275 }, no: { correct: 101, total: 159 }, maybe: { correct: 3, total: 66 } },
  qr_cr_openai:       { yes: { correct: 228, total: 275 }, no: { correct: 106, total: 159 }, maybe: { correct: 7, total: 66 } },
  hybrid_openai:      { yes: { correct: 243, total: 275 }, no: { correct: 100, total: 159 }, maybe: { correct: 3, total: 66 } },
  hybrid_cr_openai:   { yes: { correct: 244, total: 275 }, no: { correct: 101, total: 159 }, maybe: { correct: 4, total: 66 } },
}

// RAGAS METRIC DEFINITIONS (Indonesian)
export const RAGAS_METRICS_INFO = {
  faithfulness: {
    name: 'Faithfulness',
    icon: 'shield',
    color: 'emerald',
    description: 'Mengukur konsistensi faktual antara jawaban dan konteks. Setiap kalimat jawaban dicek apakah didukung oleh konteks.',
    formula: 'F = (kalimat jawaban yang didukung konteks) / (total kalimat jawaban)',
    violation_means: 'Halusinasi - LLM menulis fakta yang tidak ada di konteks.',
  },
  context_recall: {
    name: 'Context Recall',
    icon: 'search',
    color: 'sky',
    description: 'Mengukur kelengkapan konteks. Setiap klaim di reference dicek apakah tercakup dalam konteks yang di-retrieve.',
    formula: 'CR = (klaim reference yang tercakup) / (total klaim reference)',
    violation_means: 'Retrieval incomplete - konteks tidak memuat info yang ada di reference.',
  },
  answer_relevancy: {
    name: 'Answer Relevancy',
    icon: 'target',
    color: 'violet',
    description: 'Mengukur relevansi jawaban terhadap pertanyaan. Setiap kalimat jawaban dicek apakah relevan dengan pertanyaan.',
    formula: 'AR = (kalimat jawaban yang relevan) / (total kalimat jawaban)',
    violation_means: 'Jawaban off-topic - kalimat tidak menjawab pertanyaan.',
  },
  context_precision: {
    name: 'Context Precision',
    icon: 'filter',
    color: 'amber',
    description: 'Mengukur kualitas ranking konteks. Konteks relevan idealnya di peringkat atas (Average Precision).',
    formula: 'CP = sum(Precision@k * rel_k) / total_relevant',
    violation_means: 'Retrieval bising - dokumen tidak relevan muncul di rank atas.',
  },
}

// SAMPLES (5 sample real dari PubMedQA)
"""

js_content += 'export const SAMPLES = '
js_content += json.dumps(samples, indent=2, ensure_ascii=False)
js_content += '\n'

output = 'C:/Users/Ricky Wijaya/Documents/STI/Semester 8/TA/Code TA/ui/src/data/realData.js'
with open(output, 'w', encoding='utf-8') as f:
    f.write(js_content)
print(f'Saved {output}')
print(f'File size: {len(js_content)} chars, {len(js_content.splitlines())} lines')
