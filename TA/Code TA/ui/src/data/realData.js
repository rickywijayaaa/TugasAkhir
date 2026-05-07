// Auto-generated from results JSON + handcrafted RAGAS annotations

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
export const SAMPLES = [
  {
    "idx": 29,
    "question": "Visceral adipose tissue area measurement at a single level: can it represent visceral adipose tissue volume?",
    "ground_truth": "yes",
    "pubid": "28707539",
    "reference": "VAT area measurement at a single level 3 cm above the lower margin of the L3 vertebra is feasible and can reflect changes in VAT volume and body weight. Advances in knowledge: As VAT area at a CT slice 3cm above the lower margin of L3 can best reflect interval changes in VAT volume and body weight, VAT area measurement should be selected at this location.",
    "rewritten_query": "Can measurement of visceral adipose tissue area at a single anatomical level accurately represent total visceral adipose tissue volume?",
    "configs": {
      "baseline_llama": {
        "label": "maybe",
        "is_correct": false,
        "answer": "The question of whether visceral adipose tissue area measurement at a single level can represent visceral adipose tissue volume is supported by the context of study [1], which aims to determine the optimal CT slice location for measuring VAT area. The study's objective suggests that accurate and sensitive measurement of VAT area is necessary, implying that VAT area can be used as an indicator of changes in VAT volume. However, it does not explicitly state that VAT area directly represents VAT volume.\n\nmaybe",
        "faithfulness": 0.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      },
      "qr_llama": {
        "label": "yes",
        "is_correct": true,
        "answer": "The question of whether visceral adipose tissue area measurement at a single level can represent visceral adipose tissue volume is supported by the context. The study in abstract [1] aimed to determine the optimal CT slice location for measuring VAT area, suggesting that VAT area can be used as an indicator of VAT volume. Additionally, the fact that VAT area needs to be accurate and sensitive to change for risk monitoring implies that VAT area measurement at a single level may be sufficient to represent visceral adipose tissue volume.\n\nyes",
        "faithfulness": 0.0,
        "context_recall": 1.0,
        "answer_relevancy": null,
        "context_precision": null
      },
      "cr_llama": {
        "label": "yes",
        "is_correct": true,
        "answer": "Based on the provided scientific abstracts, the strongest correlations of VAT area with VAT volume and body weight changes were shown in a slice 3 cm above the lower margin of L3. This suggests that measuring VAT area at this specific location can provide accurate information about changes in VAT volume and body weight.\n\nThe evidence supports the hypothesis that VAT area measurement at a single level, specifically at the slice 3 cm above the lower margin of L3, can represent visceral adipose tissue volume.\n\nyes",
        "faithfulness": 0.6666666666666666,
        "context_recall": 1.0,
        "answer_relevancy": null,
        "context_precision": null
      },
      "baseline_openai": {
        "label": "yes",
        "is_correct": true,
        "answer": "The first abstract specifically addresses the measurement of visceral adipose tissue (VAT) and aims to determine the CT slice location where VAT area best reflects changes in VAT volume and body weight. This implies that measuring VAT area at a single level can represent VAT volume to some extent, as the study focuses on identifying the most representative slice. Therefore, the evidence supports that VAT area measurement at a single level can represent VAT volume.\n\nyes",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      },
      "qr_openai": {
        "label": "yes",
        "is_correct": true,
        "answer": "The first abstract specifically addresses the measurement of visceral adipose tissue (VAT) and aims to determine the CT slice location where VAT area best reflects changes in VAT volume and body weight. This implies that measuring VAT area at a single level can represent VAT volume to some extent, as the study is focused on identifying the optimal slice for this purpose.\n\nyes",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      },
      "cr_openai": {
        "label": "yes",
        "is_correct": true,
        "answer": "The study described in abstracts [1] and [2] indicates that measuring visceral adipose tissue (VAT) area at a specific CT slice location (3 cm above the lower margin of L3) shows a strong correlation with VAT volume (r = 0.853). This suggests that VAT area measurement at this single level can effectively represent VAT volume and changes in body weight.\n\nyes",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      },
      "qr_cr_openai": {
        "label": "yes",
        "is_correct": true,
        "answer": "The study described in abstracts [1] and [2] indicates that measuring visceral adipose tissue (VAT) area at a specific CT slice location (3 cm above the lower margin of L3) shows a strong correlation with VAT volume (r = 0.853). This suggests that VAT area measurement at this single level can effectively represent VAT volume.\n\nyes",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      },
      "hybrid_openai": {
        "label": "yes",
        "is_correct": true,
        "answer": "The provided abstracts indicate that measuring visceral adipose tissue (VAT) area at a specific CT slice (3 cm above the lower margin of L3) shows strong correlations with VAT volume (r = 0.853) and body weight changes (r = 0.902). This suggests that VAT area measurement at this single level can reliably represent VAT volume.\n\nyes",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 0.8333333333333333
      },
      "hybrid_cr_openai": {
        "label": "yes",
        "is_correct": true,
        "answer": "The provided studies indicate that measuring visceral adipose tissue (VAT) area at a specific CT slice location (3 cm above the lower margin of L3) shows a strong correlation with VAT volume (r = 0.853) and body weight changes (r = 0.902). This suggests that VAT area measurement at a single level can effectively represent VAT volume.\n\nyes",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      }
    },
    "retrieved_docs": [
      {
        "id": "d29_1",
        "content": "Measurement of visceral adipose tissue (VAT) needs to be accurate and sensitive to change for risk monitoring. The purpose of this study is to determine the CT slice location where VAT area can best reflect changes in VAT volume and body weight.",
        "bm25_score": 59.680777801282666,
        "dense_score": 0.6775335669517517,
        "rrf_score": 0.03278688524590164,
        "reranker_score": 3.6674087047576904
      },
      {
        "id": "d29_2",
        "content": "The strongest correlations of VAT area with VAT volume and body weight changes were shown in a slice 3 cm above the lower margin of L3 with r values of 0.853 and 0.902, respectively.",
        "bm25_score": 11.733642437414682,
        "dense_score": 0.5328667759895325,
        "rrf_score": 0.02964254577157803,
        "reranker_score": -8.294675827026367
      },
      {
        "id": "d29_3",
        "content": "The levels of bone formation and resorption can be assessed at the tissue level by bone histomorphometry on transiliac bone biopsies. Systemic biochemical markers of bone turnover reflect the overall bone formation and resorption at the level of the entire skeleton but cannot discriminate the different skeletal compartments.",
        "bm25_score": 20.247037101906624,
        "dense_score": 0.3745112419128418,
        "rrf_score": 0.030621785881252923,
        "reranker_score": -9.165178298950195
      },
      {
        "id": "d29_4",
        "content": "Lymphedema may be identified by simpler circumference changes as compared with changes in limb volume.",
        "bm25_score": 0.0,
        "dense_score": 0.4133901596069336,
        "rrf_score": 0.015625,
        "reranker_score": -9.80626106262207
      },
      {
        "id": "d29_5",
        "content": "To estimate changes in uterine myoma volume during pregnancy.",
        "bm25_score": 0.0,
        "dense_score": 0.4010261297225952,
        "rrf_score": 0.015151515151515152,
        "reranker_score": -10.08538818359375
      }
    ],
    "short_label": "Visceral Adipose Tissue",
    "ragas_annotations": {
      "answer_sentences": [
        {
          "text": "The provided studies indicate that measuring visceral adipose tissue (VAT) area at a specific CT slice location (3 cm above the lower margin of L3) shows a strong correlation with VAT volume (r = 0.853) and body weight changes (r = 0.902).",
          "violations": []
        },
        {
          "text": "This suggests that VAT area measurement at a single level can effectively represent VAT volume.",
          "violations": []
        }
      ],
      "reference_sentences": [
        {
          "text": "VAT area measurement at a single level 3 cm above the lower margin of the L3 vertebra is feasible and can reflect changes in VAT volume and body weight.",
          "covered_by_context": true
        }
      ],
      "contexts_relevance": [
        {
          "doc_idx": 0,
          "relevant": true,
          "reason": "Menjelaskan tujuan studi untuk mencari CT slice optimal representasi VAT."
        },
        {
          "doc_idx": 1,
          "relevant": true,
          "reason": "Membahas metodologi pengukuran VAT dan korelasi dengan volume."
        },
        {
          "doc_idx": 2,
          "relevant": true,
          "reason": "Mengandung hasil korelasi kuantitatif yang dikutip jawaban."
        },
        {
          "doc_idx": 3,
          "relevant": true,
          "reason": "Supporting document tentang subject characteristics."
        },
        {
          "doc_idx": 4,
          "relevant": true,
          "reason": "Konteks metodologis tentang CT slice location."
        }
      ]
    }
  },
  {
    "idx": 13,
    "question": "Prompting Primary Care Providers about Increased Patient Risk As a Result of Family History: Does It Work?",
    "ground_truth": "no",
    "pubid": "25957366",
    "reference": "No change occurred upon instituting simple, at-the-visit family history prompts geared to improve PCPs' ability to identify patients at high risk for 6 common conditions. The results are both surprising and disappointing. Further studies should examine physicians' perception of the utility of prompts for family history risk.",
    "rewritten_query": "Does prompting primary care providers about increased patient risk due to family history of hereditary diseases improve clinical outcomes or risk assessment accuracy?",
    "configs": {
      "baseline_llama": {
        "label": "yes",
        "is_correct": false,
        "answer": "The provided abstracts suggest that prompting primary care providers about family history may have a minimal impact on their behavior. While there was an increase in physicians discussing family history (n = 159, 22.9%), reviewing family history (n = 372, 53.5%), and ordering tests/referrals (n = 5, 0.7%) during visits with patients at moderate or strong familial risk for certain diseases, the overall change in problem summary list documentation of risk status or screening interventions was not significant. This indicates that prompting primary care providers about family history may have a limited effect on changing their behavior.\n\nyes",
        "faithfulness": 0.6666666666666666,
        "context_recall": 0.6666666666666666,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      },
      "qr_llama": {
        "label": "yes",
        "is_correct": false,
        "answer": "Based on abstracts [3] and [1], prompting primary care providers about increased patient risk as a result of family history appears to be effective. The study implemented automated prompts highlighting familial risk for various diseases, which led to PCP response to the prompts and adding family history risk to problem summary lists. Additionally, the study found that patients with moderate or strong familial risk benefited from these interventions.\n\nyes",
        "faithfulness": 0.6666666666666666,
        "context_recall": 1.0,
        "answer_relevancy": null,
        "context_precision": null
      },
      "cr_llama": {
        "label": "yes",
        "is_correct": false,
        "answer": "The study found that automated, tailored prompts about family history risk did not change primary care physician behavior, as there was no significant increase in problem summary list documentation of risk status or screening interventions for any of the 6 diseases. The physicians either reviewed family history (53.5%), discussed family history (22.9%), or did not address it at all (22.3%). This suggests that prompting primary care providers about increased patient risk as a result of family history may not be effective in changing their behavior.\n\nyes",
        "faithfulness": 0.6666666666666666,
        "context_recall": 1.0,
        "answer_relevancy": null,
        "context_precision": null
      },
      "baseline_openai": {
        "label": "no",
        "is_correct": true,
        "answer": "The study described in abstracts [1]-[3] implemented automated, tailored prompts about family history risk during primary care visits. Despite some physician engagement with the prompts (reviewing or discussing family history), there was no significant change in documentation of risk status or in patient screening interventions for the diseases studied. This indicates that prompting PCPs about increased patient risk due to family history did not effectively change their clinical behavior.\n\nno",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 0.0
      },
      "qr_openai": {
        "label": "no",
        "is_correct": true,
        "answer": "The study described in abstracts [1], [2], and [3] implemented automated, tailored prompts for primary care providers about familial risk for several diseases. Despite these prompts, there was no significant change in documentation of risk status or screening interventions by the providers. This suggests that prompting PCPs about increased patient risk due to family history did not effectively change their behavior.\n\nno",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 0.0
      },
      "cr_openai": {
        "label": "no",
        "is_correct": true,
        "answer": "The study described in the abstracts implemented automated, tailored prompts about familial risk during primary care visits and found that while physicians often reviewed or discussed family history, there was no significant change in documentation of risk status or screening interventions. This suggests that prompting primary care providers about increased patient risk due to family history did not effectively change provider behavior or patient care outcomes.\n\nno",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 0.0
      },
      "qr_cr_openai": {
        "label": "no",
        "is_correct": true,
        "answer": "The provided abstracts indicate that automated, tailored prompts about family history risk were implemented and physician responses to these prompts were documented. However, there was no significant change in problem summary list documentation of risk status or screening interventions for any of the six diseases, suggesting that the prompts did not effectively change primary care provider behavior in a meaningful way.\n\nno",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 0.0
      },
      "hybrid_openai": {
        "label": "no",
        "is_correct": true,
        "answer": "The study described in the abstracts implemented automated, tailored prompts for primary care physicians about familial risk for several diseases. Despite these prompts, there was no significant change in documentation of risk status or screening interventions by the physicians. This suggests that prompting PCPs about increased patient risk due to family history did not effectively change their behavior.\n\nno",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      },
      "hybrid_cr_openai": {
        "label": "no",
        "is_correct": true,
        "answer": "The study described in the abstracts implemented automated, tailored prompts about familial risk for several diseases during primary care visits. Despite these prompts, there was no significant change in documentation of risk status or screening interventions by primary care providers. Therefore, prompting PCPs about increased patient risk due to family history did not effectively change their behavior or improve care outcomes in this context.\n\nno",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 0.0
      }
    },
    "retrieved_docs": [
      {
        "id": "d13_1",
        "content": "Electronic health records have the potential to facilitate family history use by primary care physicians (PCPs) to provide personalized care. The objective of this study was to determine whether automated, at-the-visit tailored prompts about family history risk change PCP behavior.",
        "bm25_score": 28.577024172280318,
        "dense_score": 0.7020673155784607,
        "rrf_score": 0.03278688524590164,
        "reranker_score": 4.446298599243164
      },
      {
        "id": "d13_2",
        "content": "Automated, tailored prompts highlighting familial risk for heart disease, stroke, diabetes, and breast, colorectal, or ovarian cancer were implemented during 2011 to 2012. Medical records of a cohort of community-based primary care patients, aged 35 to 65 years, who previously participated in our Family Healthware study and had a moderate or strong familial risk for any of the 6 diseases were subs...",
        "bm25_score": 22.945027714698305,
        "dense_score": 0.6984501481056213,
        "rrf_score": 0.03225806451612903,
        "reranker_score": 3.4828195571899414
      },
      {
        "id": "d13_3",
        "content": "The 492 eligible patients had 847 visits during the study period; 152 visits had no documentation of response to a family history prompt. Of the remaining 695 visits, physician responses were reviewed family history (n = 372, 53.5%), discussed family history (n = 159, 22.9%), not addressed (n = 155, 22.3%), and reviewed family history and ordered tests/referrals (n = 5, 0.7%). There was no signifi...",
        "bm25_score": 19.191623741346103,
        "dense_score": 0.671113133430481,
        "rrf_score": 0.031746031746031744,
        "reranker_score": -1.855872392654419
      },
      {
        "id": "d13_4",
        "content": "There is increasing pressure on mental health providers to reduce the duration of treatments, while retaining level of quality and effectiveness. The risk is that the population is underserved and therefore needs new treatment episodes. The primary aim of this study was to investigate whether duration of treatment and return into mental health care were related.",
        "bm25_score": 16.508760674098397,
        "dense_score": 0.0,
        "rrf_score": 0.015151515151515152,
        "reranker_score": -5.4636030197143555
      },
      {
        "id": "d13_5",
        "content": "Health care providers, which include house staff, nurses, and respiratory care practitioners who provide care to patients with asthma in the primary general medicine clinic or the pulmonary medicine clinic of a university-county hospital in which patients were referred, were surveyed and assigned a performance score regarding the knowledge base of the appropriate use of metered-dose inhalers. Pati...",
        "bm25_score": 18.33965859044743,
        "dense_score": 0.0,
        "rrf_score": 0.015625,
        "reranker_score": -8.935224533081055
      }
    ],
    "short_label": "Primary Care Cancer Risk"
  },
  {
    "idx": 57,
    "question": "Human papillomavirus and pterygium. Is the virus a risk factor?",
    "ground_truth": "maybe",
    "pubid": "17179167",
    "reference": "The low presence of HPV DNA in pterygia does not support the hypothesis that HPV is involved in the development of pterygia in Denmark.",
    "rewritten_query": "Is human papillomavirus infection a risk factor for the development of ocular pterygium?",
    "configs": {
      "baseline_llama": {
        "label": "yes",
        "is_correct": false,
        "answer": "The study aims to investigate a large sample of pterygia for the presence of HPV in order to clarify the putative association between pterygia and HPV. Although the results are inconclusive, previous studies have investigated human papillomavirus as a risk factor for the development of pterygia. However, the current study's findings may provide more conclusive evidence regarding this association.\n\nyes",
        "faithfulness": 0.3333333333333333,
        "context_recall": 1.0,
        "answer_relevancy": null,
        "context_precision": null
      },
      "qr_llama": {
        "label": "yes",
        "is_correct": false,
        "answer": "Based on the provided abstracts, there is no direct evidence to support a clear association between human papillomavirus (HPV) and pterygium. While HPV has been investigated as a potential risk factor for pterygia in some studies, the results are inconclusive. The only relevant study mentioned in the context is aimed at investigating HPV presence in pterygia, but its objectives do not imply that it found a conclusive association.\n\nyes",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": null,
        "context_precision": null
      },
      "cr_llama": {
        "label": "yes",
        "is_correct": false,
        "answer": "The study aimed to investigate a large sample of pterygia for the presence of HPV in order to clarify the putative association between pterygia and HPV. Although the results are inconclusive, previous studies have investigated human papillomavirus (HPV) as a risk factor for the development of pterygia. However, the current study's findings do not provide conclusive evidence either way.\n\nyes",
        "faithfulness": 0.3333333333333333,
        "context_recall": 1.0,
        "answer_relevancy": null,
        "context_precision": null
      },
      "baseline_openai": {
        "label": "maybe",
        "is_correct": true,
        "answer": "The provided abstract states that several studies have investigated HPV as a risk factor for pterygium but the results are inconclusive. The aim of the study mentioned is to clarify the association, implying that current evidence does not definitively support HPV as a risk factor for pterygium.\n\nmaybe",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      },
      "qr_openai": {
        "label": "maybe",
        "is_correct": true,
        "answer": "The provided abstract states that several studies have investigated HPV as a risk factor for pterygium but the results are inconclusive. The aim of the study mentioned was to clarify this association, implying that current evidence does not definitively support HPV as a risk factor for pterygium.\n\nmaybe",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      },
      "cr_openai": {
        "label": "no",
        "is_correct": false,
        "answer": "The provided abstract states that several studies have investigated HPV as a risk factor for pterygium but the results are inconclusive. This suggests that there is no clear evidence supporting HPV as a risk factor for pterygium.\n\nno",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      },
      "qr_cr_openai": {
        "label": "maybe",
        "is_correct": true,
        "answer": "The provided abstract states that several studies have investigated HPV as a risk factor for pterygia but the results are inconclusive. The aim of the study mentioned was to clarify this association, implying that current evidence does not definitively support HPV as a risk factor for pterygium.\n\nmaybe",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      },
      "hybrid_openai": {
        "label": "no",
        "is_correct": false,
        "answer": "The provided abstracts indicate that several studies have investigated HPV as a risk factor for pterygium but found inconclusive results. A large sample of pterygium specimens was tested for HPV presence using PCR and other methods, but no definitive association is reported in the given context. Therefore, the evidence does not support HPV as a risk factor for pterygium.\n\nno",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 0.8333333333333333
      },
      "hybrid_cr_openai": {
        "label": "maybe",
        "is_correct": true,
        "answer": "The abstracts indicate that several studies have investigated HPV as a risk factor for pterygium, but the results are inconclusive. One study aimed to clarify this association by testing a large sample of pterygia for HPV presence, suggesting uncertainty remains about HPV's role as a risk factor.\n\nmaybe",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 0.8333333333333333
      }
    },
    "retrieved_docs": [
      {
        "id": "d57_1",
        "content": "Pterygium is a disease of unknown origin and pathogenesis that might be vision threatening. It is characterised by a wing-like conjunctival overgrowth of the cornea. Several studies have investigated human papillomavirus (HPV) as a risk factor for the development of pterygia, but the results are inconclusive.AIM: To investigate a large sample of pterygia for the presence of HPV in order to clarify...",
        "bm25_score": 32.769692257967414,
        "dense_score": 0.7748761773109436,
        "rrf_score": 0.03278688524590164,
        "reranker_score": 7.658145427703857
      },
      {
        "id": "d57_2",
        "content": "Vaginal intraepithelial neoplasia is a little known disease which could be related to risk factors different from simple HPV infections.",
        "bm25_score": 0.0,
        "dense_score": 0.4598041772842407,
        "rrf_score": 0.015625,
        "reranker_score": -4.6866021156311035
      },
      {
        "id": "d57_3",
        "content": "100 specimens of pterygium from Danish patients and 20 normal conjunctival biopsy specimens were investigated for the presence of HPV with PCR technique using beta-globin primers to access the quality of the extracted DNA and the HPV primers MY09/11 and GP5+/6+. HPV-positive specimens underwent subsequent HPV typing with type-specific HPV primers and further investigation with DNA in situ hybridis...",
        "bm25_score": 11.645746998153331,
        "dense_score": 0.5789098143577576,
        "rrf_score": 0.02736726874657909,
        "reranker_score": -5.9708991050720215
      },
      {
        "id": "d57_4",
        "content": "Hypertension is one of the important contributing factors linked with both causation and development of kidney disease. It is a multifactorial, polygenic, and complex disorder due to interaction of several risk genes with environmental factors. The present study was aimed to explore genetic polymorphism in ACE-1 gene as a risk factor for CKD among hypertensive patients.",
        "bm25_score": 17.28250840881114,
        "dense_score": 0.0,
        "rrf_score": 0.016129032258064516,
        "reranker_score": -7.068215370178223
      },
      {
        "id": "d57_5",
        "content": "To critically assess the evidence that appendiceal perforation is a risk factor for subsequent tubal infertility or ectopic pregnancy.",
        "bm25_score": 16.5445459269182,
        "dense_score": 0.0,
        "rrf_score": 0.014925373134328358,
        "reranker_score": -7.347508430480957
      }
    ],
    "short_label": "HPV & Pterygium"
  },
  {
    "idx": 0,
    "question": "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
    "ground_truth": "yes",
    "pubid": "21645374",
    "reference": "Results depicted mitochondrial dynamics in vivo as PCD progresses within the lace plant, and highlight the correlation of this organelle with other organelles during developmental PCD. To the best of our knowledge, this is the first report of mitochondria and chloroplasts moving on transvacuolar strands to form a ring structure surrounding the nucleus during developmental PCD. Also, for the first time, we have shown the feasibility for the use of CsA in a whole plant system. Overall, our findings implicate the mitochondria as playing a critical and early role in developmentally regulated PCD in the lace plant.",
    "rewritten_query": "Do mitochondria contribute to the regulation of programmed cell death and tissue remodeling in lace plant (Aponogeton madagascariensis) leaves?",
    "configs": {
      "baseline_llama": {
        "label": "yes",
        "is_correct": true,
        "answer": "Based on the provided scientific abstracts, it appears that mitochondria play a role in programmed cell death (PCD) in lace plant leaves. The study found that mitochondrial dynamics were delineated into four categories based on characteristics including distribution, motility, and membrane potential, and that these dynamics changed as PCD progressed. Additionally, the treatment with cyclosporine A resulted in leaves with mitochondrial dynamics similar to non-PCD cells, suggesting a possible role of mitochondrial permeability transition pore (PTP) formation during PCD.\n\nyes",
        "faithfulness": 0.0,
        "context_recall": 0.5,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      },
      "qr_llama": {
        "label": "yes",
        "is_correct": true,
        "answer": "Based on the provided scientific abstracts, it appears that mitochondria play a role in programmed cell death (PCD) in lace plant leaves. The study found that mitochondrial dynamics changed as PCD progressed, and that cyclosporine A treatment, which affects mitochondrial permeability transition pore formation, resulted in reduced PCD. Additionally, the fact that mitochondrial dynamics were similar to those of non-PCD cells after CsA treatment suggests a possible role for mitochondria in regulating PCD.\n\nyes",
        "faithfulness": 0.0,
        "context_recall": 0.5,
        "answer_relevancy": null,
        "context_precision": null
      },
      "cr_llama": {
        "label": "yes",
        "is_correct": true,
        "answer": "Based on the provided scientific abstracts, it appears that mitochondria play a role in programmed cell death (PCD) in lace plant leaves. The study found that mitochondrial dynamics changed as PCD progressed, and cyclosporine A treatment, which affects mitochondrial permeability transition pore formation, resulted in reduced PCD. Additionally, the fact that mitochondrial dynamics were similar to those of non-PCD cells after CsA treatment suggests a possible role for mitochondria in regulating PCD.\n\nyes",
        "faithfulness": 0.0,
        "context_recall": 0.5,
        "answer_relevancy": null,
        "context_precision": null
      },
      "baseline_openai": {
        "label": "yes",
        "is_correct": true,
        "answer": "The abstracts indicate that mitochondrial dynamics, including changes in distribution, motility, and membrane potential, are closely associated with programmed cell death (PCD) in lace plant leaves. Additionally, inhibition of mitochondrial permeability transition pore formation via cyclosporine A treatment reduced leaf perforations and altered mitochondrial behavior, suggesting mitochondria actively contribute to leaf remodeling during PCD.\n\nyes",
        "faithfulness": 0.5,
        "context_recall": 0.75,
        "answer_relevancy": 1.0,
        "context_precision": 0.5
      },
      "qr_openai": {
        "label": "yes",
        "is_correct": true,
        "answer": "The abstracts indicate that mitochondrial dynamics change during programmed cell death (PCD) in lace plant leaves, with different mitochondrial states correlating with stages of PCD. Additionally, treatment with cyclosporine A, which affects mitochondrial permeability transition pore formation, resulted in fewer leaf perforations and mitochondrial behavior resembling non-PCD cells, suggesting mitochondria are involved in the remodeling process during PCD.\n\nyes",
        "faithfulness": 0.5,
        "context_recall": 0.75,
        "answer_relevancy": 1.0,
        "context_precision": 0.3333333333333333
      },
      "cr_openai": {
        "label": "yes",
        "is_correct": true,
        "answer": "The abstracts indicate that mitochondrial dynamics, including distribution, motility, and membrane potential, change during programmed cell death (PCD) in lace plant leaves. Treatment with cyclosporine A, which affects mitochondrial permeability transition pore formation, reduced leaf perforations and altered mitochondrial behavior to resemble non-PCD cells, suggesting mitochondria play a role in leaf remodeling during PCD.\n\nyes",
        "faithfulness": 0.5,
        "context_recall": 0.75,
        "answer_relevancy": 1.0,
        "context_precision": 0.5
      },
      "qr_cr_openai": {
        "label": "yes",
        "is_correct": true,
        "answer": "The abstracts describe that programmed cell death (PCD) in lace plant leaves involves mitochondrial dynamics, including changes in distribution, motility, and membrane potential. Treatment with cyclosporine A, which affects mitochondrial permeability transition pore formation, reduced leaf perforations and altered mitochondrial behavior to resemble non-PCD cells, indicating mitochondria play a role in leaf remodeling during PCD.\n\nyes",
        "faithfulness": 1.0,
        "context_recall": 0.75,
        "answer_relevancy": 1.0,
        "context_precision": 0.5
      },
      "hybrid_openai": {
        "label": "yes",
        "is_correct": true,
        "answer": "The abstracts describe that programmed cell death (PCD) in lace plant leaves involves mitochondrial dynamics, including changes in distribution, motility, and membrane potential. Treatment with cyclosporine A, which affects mitochondrial permeability transition pore formation, resulted in fewer leaf perforations and mitochondrial behavior similar to non-PCD cells, indicating mitochondria play a role in the leaf remodeling process during PCD.\n\nyes",
        "faithfulness": 1.0,
        "context_recall": 0.75,
        "answer_relevancy": 1.0,
        "context_precision": 0.5
      },
      "hybrid_cr_openai": {
        "label": "yes",
        "is_correct": true,
        "answer": "The abstracts indicate that mitochondrial dynamics, including distribution, motility, and membrane potential, change during programmed cell death (PCD) in lace plant leaves. Treatment with cyclosporine A, which affects mitochondrial permeability transition pore formation, reduced leaf perforations and altered mitochondrial behavior to resemble non-PCD cells, suggesting mitochondria play a role in leaf remodeling during PCD.\n\nyes",
        "faithfulness": 0.5,
        "context_recall": 0.75,
        "answer_relevancy": 1.0,
        "context_precision": 0.5
      }
    },
    "retrieved_docs": [
      {
        "id": "d0_1",
        "content": "Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponogeton madagascariensis) produces perforations in its leaves through PCD. The leaves of the plant consist of a latticework of longitudinal and transverse veins enclosing areoles. PCD occurs in the cells at the center of these areoles and progresses outwards, stopping approximately five cells from th...",
        "bm25_score": 51.176322588950015,
        "dense_score": 0.6082525253295898,
        "rrf_score": 0.03278688524590164,
        "reranker_score": 6.258399963378906
      },
      {
        "id": "d0_2",
        "content": "The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis. A single areole within a window stage leaf (PCD is occurring) was divided into three areas based on the progression of PCD; cells that will not undergo PCD (NPCD), cells in early stages of PCD (EPCD), and cells in late stages of PCD (LPCD). Window stage leaves were...",
        "bm25_score": 23.687623117736297,
        "dense_score": 0.5995238423347473,
        "rrf_score": 0.03225806451612903,
        "reranker_score": -5.053024768829346
      },
      {
        "id": "d0_3",
        "content": "Plant acclimation in the cold (2 degrees C) brought about retardation of leaf expansion, concomitant with development of freezing resistance. These effects were associated with the increases in leaf tensile stiffness, cell wall and pectin contents, pectin methylesterase (EC 3.1.1.11) activity and the low-methylated pectin content, independently of the genotype studied. However, the cold-induced mo...",
        "bm25_score": 15.864032223212822,
        "dense_score": 0.3681046962738037,
        "rrf_score": 0.03125763125763126,
        "reranker_score": -9.369415283203125
      },
      {
        "id": "d0_4",
        "content": "Cyclical pamidronate therapy in a 2-year-old child with skeletal fragility resulted in remodelling of vertebral fractures and improvement in bone mineral density (BMD) at distal radial and spinal sites. The BMD at both sites decreased precipitously within 24 months of stopping treatment, raising the question as to whether bisphosphonates can be stopped in a growing child with skeletal fragility.",
        "bm25_score": 12.140434534342253,
        "dense_score": 0.0,
        "rrf_score": 0.014285714285714285,
        "reranker_score": -9.751298904418945
      },
      {
        "id": "d0_5",
        "content": "The hypothesis was tested that pectin content and methylation degree participate in regulation of cell wall mechanical properties and in this way may affect tissue growth and freezing resistance over the course of plant cold acclimation and de-acclimation.",
        "bm25_score": 13.255019207010264,
        "dense_score": 0.3587340712547302,
        "rrf_score": 0.03055037313432836,
        "reranker_score": -10.405147552490234
      }
    ],
    "short_label": "Mitochondria Lace Plant",
    "ragas_annotations": {
      "answer_sentences": [
        {
          "text": "The abstracts indicate that mitochondrial dynamics, including changes in distribution, motility, and membrane potential, are closely associated with programmed cell death (PCD) in lace plant leaves.",
          "violations": []
        },
        {
          "text": "Additionally, inhibition of mitochondrial permeability transition pore formation via cyclosporine A treatment reduced leaf perforations and altered mitochondrial behavior, suggesting mitochondria actively contribute to leaf remodeling during PCD.",
          "violations": [
            "faithfulness"
          ],
          "reason_faithfulness": "Klaim \"inhibition via cyclosporine A reduced leaf perforations\" tidak secara eksplisit didukung oleh konteks yang disediakan. Konteks menyebutkan CsA tetapi hasil spesifik ini tidak ditemukan."
        }
      ],
      "reference_sentences": [
        {
          "text": "Results depicted mitochondrial dynamics in vivo as PCD progresses within the lace plant.",
          "covered_by_context": true
        },
        {
          "text": "This is the first report of mitochondria and chloroplasts moving on transvacuolar strands to form a ring structure surrounding the nucleus during developmental PCD.",
          "covered_by_context": false,
          "reason": "Detail \"ring structure surrounding nucleus\" tidak ditemukan dalam 5 konteks teratas. Retrieval melewatkan fakta spesifik ini."
        },
        {
          "text": "For the first time, we have shown the feasibility for the use of CsA in a whole plant system.",
          "covered_by_context": true
        },
        {
          "text": "Our findings implicate the mitochondria as playing a critical and early role in developmentally regulated PCD.",
          "covered_by_context": true
        }
      ],
      "contexts_relevance": [
        {
          "doc_idx": 0,
          "relevant": true,
          "reason": "Membahas peran mitokondria dalam PCD lace plant - fit langsung dengan pertanyaan."
        },
        {
          "doc_idx": 1,
          "relevant": true,
          "reason": "Menjelaskan dinamika mitokondria in vivo selama PCD - highly relevant."
        },
        {
          "doc_idx": 2,
          "relevant": false,
          "reason": "Dokumen tentang topik berbeda, bukan lace plant PCD spesifik."
        },
        {
          "doc_idx": 3,
          "relevant": false,
          "reason": "Tidak mengandung informasi tentang mitochondria atau leaf remodeling."
        },
        {
          "doc_idx": 4,
          "relevant": true,
          "reason": "Membahas metodologi PCD observation yang mendukung jawaban."
        }
      ]
    }
  },
  {
    "idx": 1,
    "question": "Landolt C and snellen e acuity: differences in strabismus amblyopia?",
    "ground_truth": "no",
    "pubid": "16418930",
    "reference": "Using the charts described, there was only a slight overestimation of visual acuity by the Snellen E compared to the Landolt C, even in strabismus amblyopia. Small differences in the lower visual acuity range have to be considered.",
    "rewritten_query": "Are there differences in visual acuity measurements between Landolt C and Snellen E charts in patients with strabismic amblyopia?",
    "configs": {
      "baseline_llama": {
        "label": "yes",
        "is_correct": false,
        "answer": "The study abstracts suggest that there are small differences between Landolt C acuity (LR) and Snellen E acuity (SE), with a mean difference of 0.55 lines in both groups, including those with strabismus amblyopia. The results also indicate that the eyes with strabismus amblyopia showed similar improvements in visual acuity when using LR and SE charts. However, the study did not specifically investigate differences between LR and SE in strabismus amblyopia.\n\nyes",
        "faithfulness": 0.3333333333333333,
        "context_recall": 1.0,
        "answer_relevancy": 0.6666666666666666,
        "context_precision": 1.0
      },
      "qr_llama": {
        "label": "yes",
        "is_correct": false,
        "answer": "The study found that the mean difference between Landolt C acuity (LR) and Snellen E acuity (SE) was 0.55 lines in the entire group, with similar values for the eyes with strabismus amblyopia. The results also showed that the eye showing improvement of acuity had a history of amblyopia, but this information does not directly address whether there are differences between Landolt C and Snellen E acuity specifically in strabismus amblyopia.\n\nyes",
        "faithfulness": 0.5,
        "context_recall": 1.0,
        "answer_relevancy": null,
        "context_precision": null
      },
      "cr_llama": {
        "label": "yes",
        "is_correct": false,
        "answer": "The study abstracts suggest that there are small differences between Landolt C acuity (LR) and Snellen E acuity (SE), with a mean difference of 0.55 lines in both groups, including those with strabismus amblyopia. However, the improvements in visual acuity for eyes with strabismus amblyopia were significant, with a mean improvement of 3.3 and 1.9 lines logMAR in distance and near acuity, respectively. This suggests that while there may be some initial differences between LR and SE, the eyes with strabismus amblyopia can still show substantial improvements in visual acuity.\n\nyes",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": null,
        "context_precision": null
      },
      "baseline_openai": {
        "label": "no",
        "is_correct": true,
        "answer": "The provided abstracts indicate that differences between Landolt C acuity and Snellen E acuity in eyes with strabismus amblyopia are small, with mean decimal values and line differences being minimal (0.14 vs. 0.16 and 0.55 lines difference, respectively). This suggests that while there are slight differences, they are not substantial.\n\nno",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 0.8333333333333333
      },
      "qr_openai": {
        "label": "yes",
        "is_correct": false,
        "answer": "The provided abstracts indicate that in patients with strabismus amblyopia, the differences between Landolt C acuity and Snellen E acuity are small. Specifically, the mean decimal values and the mean difference in lines between the two tests were minimal, with Snellen E showing slightly higher values but only by about 0.55 lines. This suggests that while there is a measurable difference, it is minor and consistent.\n\nyes",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      },
      "cr_openai": {
        "label": "yes",
        "is_correct": false,
        "answer": "The provided abstracts indicate that differences between Landolt C acuity and Snellen E acuity in eyes with strabismus amblyopia are small. Specifically, the mean decimal values and mean difference in lines were very close, with Snellen E acuity showing slightly higher values but only by about 0.55 lines. This suggests minimal clinically relevant differences between the two measures in strabismus amblyopia.\n\nyes",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      },
      "qr_cr_openai": {
        "label": "no",
        "is_correct": true,
        "answer": "The provided abstracts indicate that differences between Landolt C acuity and Snellen E acuity in patients with strabismus amblyopia are small. Specifically, the mean decimal values and mean difference in lines between the two tests were minimal, with Snellen E acuity slightly higher but only by about 0.55 lines. This suggests that the two methods yield comparable results in strabismus amblyopia.\n\nno",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 0.8333333333333333
      },
      "hybrid_openai": {
        "label": "no",
        "is_correct": true,
        "answer": "The provided abstracts indicate that differences between Landolt C acuity and Snellen E acuity in patients with strabismus amblyopia are small. Specifically, the mean decimal values and mean difference in lines between the two tests were minimal, with Snellen E showing slightly higher values but only by about 0.55 lines. This suggests that the two methods yield comparable results in strabismus amblyopia.\n\nno",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      },
      "hybrid_cr_openai": {
        "label": "yes",
        "is_correct": false,
        "answer": "The provided abstracts indicate that differences between Landolt C acuity and Snellen E acuity in eyes with strabismus amblyopia are small. Specifically, the mean decimal values and mean difference in lines were very close, with Snellen E acuity showing slightly higher values but only by about 0.55 lines. This suggests minimal clinically relevant differences between the two measures in strabismus amblyopia.\n\nyes",
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0
      }
    },
    "retrieved_docs": [
      {
        "id": "d1_1",
        "content": "Differences between Landolt C acuity (LR) and Snellen E acuity (SE) were small. The mean decimal values for LR and SE were 0.25 and 0.29 in the entire group and 0.14 and 0.16 for the eyes with strabismus amblyopia. The mean difference between LR and SE was 0.55 lines in the entire group and 0.55 lines for the eyes with strabismus amblyopia, with higher values of SE in both groups. The results of t...",
        "bm25_score": 48.3118980323831,
        "dense_score": 0.7574276924133301,
        "rrf_score": 0.03278688524590164,
        "reranker_score": 8.946990966796875
      },
      {
        "id": "d1_2",
        "content": "100 patients (age 8 - 90 years, median 60.5 years) with various eye disorders, among them 39 with amblyopia due to strabismus, and 13 healthy volunteers were tested. Charts with the Snellen E and the Landolt C (Precision Vision) which mimic the ETDRS charts were used to assess visual acuity. Three out of 5 optotypes per line had to be correctly identified, while wrong answers were monitored. In th...",
        "bm25_score": 38.359580393471965,
        "dense_score": 0.7089611887931824,
        "rrf_score": 0.03225806451612903,
        "reranker_score": 5.589302062988281
      },
      {
        "id": "d1_3",
        "content": "Assessment of visual acuity depends on the optotypes used for measurement. The ability to recognize different optotypes differs even if their critical details appear under the same visual angle. Since optotypes are evaluated on individuals with good visual acuity and without eye disorders, differences in the lower visual acuity range cannot be excluded. In this study, visual acuity measured with t...",
        "bm25_score": 36.25669042444246,
        "dense_score": 0.6874179840087891,
        "rrf_score": 0.031746031746031744,
        "reranker_score": 2.0349202156066895
      },
      {
        "id": "d1_4",
        "content": "12 such cases were detected. In nine of these the eye showing improvement of acuity had a history of amblyopia. The mean improvement in distance and near acuity in amblyopic eyes by 12 months was 3.3 and 1.9 lines logMAR respectively. The improvement in acuity generally occurred between 1 and 12 months from baseline and remained stable over the period of follow up.",
        "bm25_score": 21.064257887950983,
        "dense_score": 0.6331771612167358,
        "rrf_score": 0.03125,
        "reranker_score": -3.3910257816314697
      },
      {
        "id": "d1_5",
        "content": "The mean difference in SE between non-cycloplegic and cycloplegic measurements was 0.68â€‰Â±â€‰0.83Â D (95Â % CI, 0.64-0.72). Significantly greater differences were observed in hypermetropes than myopes (1.30â€‰Â±â€‰0.90Â D versus 0.46â€‰Â±â€‰0.68Â D, pâ€‰<â€‰0.001). Moderate hypermetropes (2 to 5Â D) demonstrated significantly greater refractive error than mild (0.5 to 2Â D) or severe (>5Â D) hyp...",
        "bm25_score": 0.0,
        "dense_score": 0.46591997146606445,
        "rrf_score": 0.014492753623188406,
        "reranker_score": -10.641268730163574
      }
    ],
    "short_label": "Landolt C vs Snellen E"
  }
]
