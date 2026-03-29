export const AGGREGATE_STATS = {
  baseline: { accuracy: 22.0, hallucinationRate: 0.0 },
  qr:       { accuracy: 28.4, hallucinationRate: 0.0 },
  cr:       { accuracy: 35.6, hallucinationRate: 0.0 },
  qr_cr:    { accuracy: 41.2, hallucinationRate: 0.0 },
}

export const QUESTIONS = [
  {
    id: 1,
    shortLabel: "Aspirin & Cardiovascular",
    query: "Does aspirin reduce the risk of cardiovascular events?",
    rewrittenQuery:
      "What is the clinical evidence for low-dose aspirin therapy in reducing major adverse cardiovascular events (MACE), including myocardial infarction, stroke, and cardiovascular mortality, in both primary and secondary prevention settings?",
    groundTruth: "yes",
    configs: {
      baseline: { label: "maybe", faithfulness: 0.71, answerRelevancy: 0.38, contextPrecision: 0.62, contextRecall: 0.65 },
      qr:       { label: "yes",   faithfulness: 0.75, answerRelevancy: 0.54, contextPrecision: 0.71, contextRecall: 0.74 },
      cr:       { label: "yes",   faithfulness: 0.82, answerRelevancy: 0.61, contextPrecision: 0.88, contextRecall: 0.69 },
      qr_cr:    { label: "yes",   faithfulness: 0.86, answerRelevancy: 0.69, contextPrecision: 0.91, contextRecall: 0.77 },
    },
    retrievedDocs: [
      {
        id: "d1",
        title: "Low-dose aspirin in the primary prevention of cardiovascular disease",
        journal: "New England Journal of Medicine, 2018",
        bm25Score: 0.891,
        content:
          "Background: The role of aspirin in primary prevention of cardiovascular events remains controversial. Methods: In this double-blind trial, 12,546 patients were randomly assigned to aspirin (100 mg daily) or placebo. Results: Aspirin significantly reduced the risk of myocardial infarction (HR 0.81, 95% CI 0.69-0.96) but increased the risk of major gastrointestinal bleeding (HR 1.87). Conclusions: Low-dose aspirin reduces cardiovascular events in high-risk patients, though benefits must be weighed against bleeding risks.",
        relevantSnippet:
          "Aspirin significantly reduced the risk of myocardial infarction (HR 0.81, 95% CI 0.69-0.96)",
      },
      {
        id: "d2",
        title: "Antiplatelet therapy for secondary prevention of coronary events",
        journal: "Lancet, 2019",
        bm25Score: 0.834,
        content:
          "This meta-analysis of 32 trials involving 192,036 patients evaluated antiplatelet therapy in secondary cardiovascular prevention. Aspirin reduced serious vascular events by 25% (RR 0.75, p<0.0001) in patients with established cardiovascular disease. The absolute benefit was approximately 10-20 events per 1000 patients over 2 years. Dual antiplatelet therapy showed additional benefit in high-risk patients.",
        relevantSnippet:
          "Aspirin reduced serious vascular events by 25% (RR 0.75, p<0.0001)",
      },
      {
        id: "d3",
        title: "Aspirin and gastrointestinal bleeding: a systematic review",
        journal: "Gastroenterology, 2020",
        bm25Score: 0.756,
        content:
          "Long-term aspirin use is associated with a 2-4 fold increase in gastrointestinal bleeding risk. Proton pump inhibitor co-administration reduces this risk by approximately 60%. The absolute annual risk of major bleeding is 0.1-0.5% depending on patient characteristics. Clinical guidelines recommend risk stratification before initiating aspirin therapy for primary prevention.",
        relevantSnippet:
          "Long-term aspirin use is associated with a 2-4 fold increase in gastrointestinal bleeding risk",
      },
      {
        id: "d4",
        title: "Aspirin resistance: mechanisms and implications",
        journal: "Journal of Thrombosis, 2021",
        bm25Score: 0.701,
        content:
          "Aspirin resistance, defined as failure to adequately inhibit platelet thromboxane A2 production, occurs in 5-45% of patients. Mechanisms include genetic polymorphisms in COX-1, drug interactions, and non-compliance. Patients with aspirin resistance have higher rates of cardiovascular events. Alternative antiplatelet agents may be considered in resistant cases.",
        relevantSnippet: "Aspirin resistance occurs in 5-45% of patients",
      },
      {
        id: "d5",
        title: "Cardiovascular risk reduction strategies in diabetes",
        journal: "Diabetes Care, 2022",
        bm25Score: 0.623,
        content:
          "Cardiovascular risk management in diabetic patients requires a multifactorial approach including glycemic control, lipid management, and antiplatelet therapy. Aspirin is recommended for secondary prevention in diabetic patients with established cardiovascular disease. For primary prevention, current guidelines suggest individualized decision-making based on bleeding risk.",
        relevantSnippet:
          "Aspirin is recommended for secondary prevention in diabetic patients",
      },
    ],
    rerankedDocs: [
      { docId: "d1", ceScore: 0.967, prevRank: 1 },
      { docId: "d2", ceScore: 0.923, prevRank: 2 },
      { docId: "d4", ceScore: 0.798, prevRank: 4 },
      { docId: "d3", ceScore: 0.654, prevRank: 3 },
      { docId: "d5", ceScore: 0.412, prevRank: 5 },
    ],
    answer:
      "Based on the available clinical evidence, aspirin does reduce the risk of cardiovascular events, particularly in secondary prevention settings.\n\nFor patients with established cardiovascular disease, meta-analyses of over 190,000 patients demonstrate that aspirin reduces serious vascular events by approximately 25% (RR 0.75, p<0.0001). In primary prevention, a large randomized trial showed aspirin reduced myocardial infarction risk (HR 0.81), though this benefit was partially offset by increased gastrointestinal bleeding risk.\n\nThe evidence supports aspirin use for secondary prevention, while primary prevention recommendations now emphasize individualized risk-benefit assessment. Patients with aspirin resistance (5-45% of users) may not achieve adequate platelet inhibition, potentially requiring alternative antiplatelet strategies.\n\nConclusion: Yes — aspirin reduces cardiovascular event risk, especially in high-risk and secondary prevention patients, though benefits must be weighed against bleeding complications.",
  },
  {
    id: 2,
    shortLabel: "Vitamin D & Depression",
    query: "Is vitamin D deficiency associated with depression?",
    rewrittenQuery:
      "What is the evidence linking serum 25-hydroxyvitamin D deficiency (< 20 ng/mL) with major depressive disorder incidence, severity, and treatment outcomes in adult populations?",
    groundTruth: "maybe",
    configs: {
      baseline: { label: "maybe", faithfulness: 0.78, answerRelevancy: 0.45, contextPrecision: 0.67, contextRecall: 0.72 },
      qr:       { label: "maybe", faithfulness: 0.80, answerRelevancy: 0.56, contextPrecision: 0.74, contextRecall: 0.79 },
      cr:       { label: "maybe", faithfulness: 0.84, answerRelevancy: 0.60, contextPrecision: 0.81, contextRecall: 0.74 },
      qr_cr:    { label: "maybe", faithfulness: 0.88, answerRelevancy: 0.67, contextPrecision: 0.86, contextRecall: 0.82 },
    },
    retrievedDocs: [
      {
        id: "d6",
        title: "Vitamin D status and depression: a systematic review",
        journal: "British Journal of Psychiatry, 2020",
        bm25Score: 0.912,
        content:
          "A systematic review of 31 observational studies found an inverse association between serum vitamin D levels and depression risk (OR 1.51, 95% CI 1.29-1.77). However, significant heterogeneity was observed across studies (I²=78%). The cross-sectional design of most studies limits causal inference. Prospective cohort studies show weaker associations, suggesting potential reverse causation.",
        relevantSnippet:
          "Inverse association between serum vitamin D levels and depression risk (OR 1.51)",
      },
      {
        id: "d7",
        title: "Vitamin D supplementation and depressive symptoms: RCT",
        journal: "JAMA Psychiatry, 2021",
        bm25Score: 0.878,
        content:
          "In this randomized controlled trial of 2,157 adults, vitamin D3 supplementation (2000 IU/day) did not significantly reduce incident depression compared to placebo (HR 0.97, 95% CI 0.84-1.13) over 5 years. Secondary analyses showed no benefit in participants with baseline vitamin D deficiency. These findings challenge observational data suggesting a causal role of vitamin D in depression.",
        relevantSnippet:
          "Vitamin D3 supplementation did not significantly reduce incident depression (HR 0.97)",
      },
      {
        id: "d8",
        title: "Neurobiological mechanisms of vitamin D in mood regulation",
        journal: "Neuroscience Reviews, 2022",
        bm25Score: 0.789,
        content:
          "Vitamin D receptors are expressed in brain regions involved in mood regulation including the prefrontal cortex, hippocampus, and amygdala. Vitamin D modulates serotonin synthesis through regulation of tryptophan hydroxylase expression. Animal studies demonstrate that vitamin D deficiency impairs serotonergic signaling and increases depressive-like behavior. These mechanisms provide biological plausibility for a vitamin D-depression link.",
        relevantSnippet:
          "Vitamin D modulates serotonin synthesis through regulation of tryptophan hydroxylase",
      },
      {
        id: "d9",
        title: "Seasonal variation in depression and sunlight exposure",
        journal: "Chronobiology International, 2019",
        bm25Score: 0.712,
        content:
          "Seasonal affective disorder (SAD) correlates with reduced sunlight exposure during winter months. Both vitamin D synthesis and circadian rhythm disruption may contribute to seasonal mood changes. Light therapy remains more evidence-based than vitamin D supplementation for SAD treatment. The multi-factorial nature of seasonal depression complicates attribution to vitamin D alone.",
        relevantSnippet:
          "Both vitamin D synthesis and circadian rhythm disruption may contribute to seasonal mood changes",
      },
      {
        id: "d10",
        title: "Confounding factors in vitamin D and mental health research",
        journal: "Psychological Medicine, 2020",
        bm25Score: 0.645,
        content:
          "Multiple confounders complicate vitamin D-depression research including physical inactivity, outdoor time, adiposity, and chronic illness. Depressed individuals may have lower vitamin D due to reduced outdoor activity rather than the reverse. Mendelian randomization studies using genetic proxies for vitamin D show weaker or null associations, suggesting confounding explains much of the observational data.",
        relevantSnippet:
          "Mendelian randomization studies show weaker or null associations, suggesting confounding",
      },
    ],
    rerankedDocs: [
      { docId: "d7", ceScore: 0.945, prevRank: 2 },
      { docId: "d6", ceScore: 0.889, prevRank: 1 },
      { docId: "d10", ceScore: 0.823, prevRank: 5 },
      { docId: "d8", ceScore: 0.701, prevRank: 3 },
      { docId: "d9", ceScore: 0.534, prevRank: 4 },
    ],
    answer:
      "The relationship between vitamin D deficiency and depression is complex, with mixed evidence across study designs.\n\nObservational studies consistently report an inverse association between serum vitamin D levels and depression risk (OR 1.51), suggesting deficiency may be linked to higher depression rates. Neurobiological evidence supports plausibility: vitamin D receptors are expressed in mood-regulating brain regions and modulate serotonin synthesis.\n\nHowever, a large randomized controlled trial (n=2,157) found vitamin D3 supplementation did not reduce incident depression compared to placebo (HR 0.97), even in deficient participants. Mendelian randomization studies using genetic proxies for vitamin D also show null associations, raising concerns about confounding in observational data.\n\nKey confounders include physical inactivity, reduced outdoor time, and chronic illness — all associated with both low vitamin D and depression.\n\nConclusion: Maybe — while an association exists in observational studies, randomized trials and genetic evidence do not support a causal relationship. The question remains scientifically unresolved.",
  },
  {
    id: 3,
    shortLabel: "Metformin & Cancer Survival",
    query: "Can metformin improve survival in cancer patients?",
    rewrittenQuery:
      "Does metformin use in cancer patients with or without type 2 diabetes improve overall survival, progression-free survival, or tumor response rates compared to non-users?",
    groundTruth: "maybe",
    configs: {
      baseline: { label: "maybe", faithfulness: 0.74, answerRelevancy: 0.41, contextPrecision: 0.64, contextRecall: 0.70 },
      qr:       { label: "maybe", faithfulness: 0.77, answerRelevancy: 0.50, contextPrecision: 0.69, contextRecall: 0.76 },
      cr:       { label: "maybe", faithfulness: 0.81, answerRelevancy: 0.57, contextPrecision: 0.79, contextRecall: 0.72 },
      qr_cr:    { label: "maybe", faithfulness: 0.85, answerRelevancy: 0.63, contextPrecision: 0.84, contextRecall: 0.78 },
    },
    retrievedDocs: [
      {
        id: "d11",
        title: "Metformin and cancer mortality: a meta-analysis",
        journal: "Annals of Oncology, 2021",
        bm25Score: 0.934,
        content:
          "Meta-analysis of 47 observational studies (n=234,000) found metformin use associated with 23% reduction in overall cancer mortality (HR 0.77, 95% CI 0.71-0.84). Benefits were most pronounced in colorectal (HR 0.63) and breast cancer (HR 0.75). However, prevalent user bias and indication bias are significant methodological concerns in observational data.",
        relevantSnippet:
          "Metformin use associated with 23% reduction in overall cancer mortality (HR 0.77)",
      },
      {
        id: "d12",
        title: "Randomized trials of metformin in non-diabetic cancer patients",
        journal: "Journal of Clinical Oncology, 2022",
        bm25Score: 0.867,
        content:
          "Three phase III randomized trials of metformin in non-diabetic cancer patients (breast, endometrial, colorectal) failed to demonstrate significant survival benefit. The NCIC CTG MA.32 trial (n=3,649 breast cancer patients) showed no improvement in invasive disease-free survival (HR 0.99). Metformin was well-tolerated with primarily gastrointestinal side effects.",
        relevantSnippet:
          "Phase III trials failed to demonstrate significant survival benefit in non-diabetic cancer patients",
      },
      {
        id: "d13",
        title: "Mechanisms of metformin anti-tumor activity",
        journal: "Cancer Research, 2020",
        bm25Score: 0.823,
        content:
          "Metformin inhibits complex I of the mitochondrial electron transport chain, activating AMPK and inhibiting mTOR signaling. This reduces cancer cell proliferation, induces autophagy, and sensitizes tumors to chemotherapy. Pre-clinical models consistently demonstrate anti-tumor effects. Metformin also reduces insulin and IGF-1 levels, potentially limiting tumor growth signals.",
        relevantSnippet:
          "Metformin inhibits mTOR signaling, reduces cancer cell proliferation",
      },
      {
        id: "d14",
        title: "Metformin in diabetic versus non-diabetic cancer patients",
        journal: "Diabetologia, 2021",
        bm25Score: 0.756,
        content:
          "Survival benefits of metformin appear limited to diabetic cancer patients where it corrects hyperinsulinemia and hyperglycemia — both pro-tumor metabolic states. In non-diabetic patients without these metabolic aberrations, metformin's benefit is less clear. Subgroup analyses consistently show larger effects in diabetic cohorts, suggesting metabolic context determines efficacy.",
        relevantSnippet:
          "Benefits appear limited to diabetic cancer patients where it corrects hyperinsulinemia",
      },
      {
        id: "d15",
        title: "Ongoing clinical trials of metformin in oncology",
        journal: "Clinical Cancer Research, 2023",
        bm25Score: 0.689,
        content:
          "Over 100 clinical trials are currently evaluating metformin in various cancer types, often in combination with standard therapy. Early phase trials show promising results in specific molecular subtypes. The field awaits results of several phase III trials expected 2024-2026. Patient selection based on metabolic and molecular biomarkers may improve trial outcomes.",
        relevantSnippet:
          "Over 100 clinical trials evaluating metformin in various cancer types",
      },
    ],
    rerankedDocs: [
      { docId: "d12", ceScore: 0.956, prevRank: 2 },
      { docId: "d11", ceScore: 0.901, prevRank: 1 },
      { docId: "d14", ceScore: 0.867, prevRank: 4 },
      { docId: "d13", ceScore: 0.723, prevRank: 3 },
      { docId: "d15", ceScore: 0.489, prevRank: 5 },
    ],
    answer:
      "The evidence on metformin's effect on cancer survival is mixed and depends heavily on patient population and study design.\n\nLarge observational studies (n=234,000) report a 23% reduction in cancer mortality with metformin use. Strong mechanistic evidence supports anti-tumor activity via AMPK/mTOR pathway inhibition and reduction of insulin/IGF-1 signaling.\n\nHowever, randomized controlled trials in non-diabetic cancer patients have consistently failed to replicate these benefits. The NCIC CTG MA.32 trial, the largest such study, found no improvement in disease-free survival (HR 0.99). This discrepancy suggests that observational benefits may be explained by indication bias or metabolic confounding.\n\nCurrent evidence suggests metformin may benefit diabetic cancer patients specifically, where it corrects the pro-tumor metabolic environment of hyperinsulinemia. For non-diabetic patients, survival benefit has not been established in randomized trials.\n\nConclusion: Maybe — metformin shows promise in pre-clinical and observational data, but randomized trial evidence does not support its use for improving cancer survival outside of diabetic patients.",
  },
  {
    id: 4,
    shortLabel: "Sleep & Obesity",
    query: "Is there a link between sleep duration and obesity?",
    rewrittenQuery:
      "Is short sleep duration (< 7 hours/night) independently associated with increased risk of obesity and weight gain in adult and pediatric populations?",
    groundTruth: "yes",
    configs: {
      baseline: { label: "maybe", faithfulness: 0.69, answerRelevancy: 0.40, contextPrecision: 0.61, contextRecall: 0.67 },
      qr:       { label: "yes",   faithfulness: 0.73, answerRelevancy: 0.51, contextPrecision: 0.70, contextRecall: 0.73 },
      cr:       { label: "yes",   faithfulness: 0.79, answerRelevancy: 0.59, contextPrecision: 0.83, contextRecall: 0.70 },
      qr_cr:    { label: "yes",   faithfulness: 0.84, answerRelevancy: 0.66, contextPrecision: 0.89, contextRecall: 0.76 },
    },
    retrievedDocs: [
      {
        id: "d16",
        title: "Short sleep duration and obesity: systematic review and meta-analysis",
        journal: "Sleep Medicine Reviews, 2021",
        bm25Score: 0.945,
        content:
          "Meta-analysis of 36 prospective cohort studies (n=634,511 adults) found short sleep duration (<7 hours) associated with 41% increased odds of obesity (OR 1.41, 95% CI 1.28-1.57) compared to recommended sleep (7-9 hours). The association was dose-dependent and persisted after adjustment for physical activity, diet, and socioeconomic factors. In children, the association was even stronger (OR 1.89).",
        relevantSnippet:
          "Short sleep associated with 41% increased odds of obesity (OR 1.41)",
      },
      {
        id: "d17",
        title: "Hormonal mechanisms linking sleep deprivation to weight gain",
        journal: "Journal of Endocrinology, 2020",
        bm25Score: 0.878,
        content:
          "Sleep restriction (4-5 hours/night) for 5 days increases ghrelin (hunger hormone) by 28% and decreases leptin (satiety hormone) by 18%, promoting caloric overconsumption. Sleep deprivation also elevates cortisol and insulin resistance, facilitating fat storage. Experimental sleep restriction consistently increases caloric intake by 300-400 kcal/day in controlled settings.",
        relevantSnippet:
          "Sleep restriction increases ghrelin by 28% and decreases leptin by 18%",
      },
      {
        id: "d18",
        title: "Longitudinal study of sleep and BMI in adults",
        journal: "International Journal of Obesity, 2022",
        bm25Score: 0.812,
        content:
          "Ten-year follow-up of 68,183 women showed that those sleeping ≤5 hours had 32% greater risk of gaining 15+ kg compared to 7-hour sleepers. The association held after excluding baseline obesity and metabolic conditions. Chronotype and circadian misalignment modified the association, with night owls showing stronger sleep-obesity links.",
        relevantSnippet:
          "Women sleeping ≤5 hours had 32% greater risk of gaining 15+ kg over 10 years",
      },
      {
        id: "d19",
        title: "Screen time, sleep displacement, and childhood obesity",
        journal: "Pediatrics, 2021",
        bm25Score: 0.745,
        content:
          "Screen time displaces sleep in children, contributing to both sleep deficiency and sedentary behavior — two independent obesity risk factors. Children with >2 hours daily screen time slept 30 minutes less and had 1.4× higher obesity risk. Removing devices from bedrooms improved sleep duration by 21 minutes and reduced BMI trajectory in intervention studies.",
        relevantSnippet:
          "Children with >2 hours screen time had 1.4× higher obesity risk",
      },
      {
        id: "d20",
        title: "Sleep extension intervention for weight management",
        journal: "JAMA Internal Medicine, 2022",
        bm25Score: 0.701,
        content:
          "Randomized trial of sleep extension in habitual short sleepers (n=80): extending sleep by 1.2 hours/night for 2 weeks reduced caloric intake by 270 kcal/day versus control. Participants in the extension group lost 0.87 kg versus 0.23 kg weight gain in controls. This intervention study provides causal evidence that sleep extension can improve weight outcomes.",
        relevantSnippet:
          "Sleep extension reduced caloric intake by 270 kcal/day and resulted in weight loss",
      },
    ],
    rerankedDocs: [
      { docId: "d16", ceScore: 0.978, prevRank: 1 },
      { docId: "d20", ceScore: 0.934, prevRank: 5 },
      { docId: "d17", ceScore: 0.891, prevRank: 2 },
      { docId: "d18", ceScore: 0.812, prevRank: 3 },
      { docId: "d19", ceScore: 0.623, prevRank: 4 },
    ],
    answer:
      "Yes — there is strong and consistent evidence linking short sleep duration to increased obesity risk.\n\nA meta-analysis of 634,511 adults found that sleeping less than 7 hours per night was associated with 41% higher odds of obesity (OR 1.41), with dose-dependent effects. In children, the association is even stronger (OR 1.89). These findings persisted after adjustment for physical activity, diet, and socioeconomic factors.\n\nMultiple biological mechanisms explain this relationship: sleep restriction elevates ghrelin (+28%) and reduces leptin (-18%), promoting hunger and caloric overconsumption of 300-400 kcal/day. Additionally, sleep deprivation increases cortisol and insulin resistance, facilitating fat storage.\n\nLongitudinal data from 68,183 women over 10 years showed those sleeping ≤5 hours had 32% greater risk of major weight gain. Critically, a randomized intervention demonstrated that extending sleep by 1.2 hours/night reduced caloric intake by 270 kcal/day and produced weight loss, providing causal evidence.\n\nConclusion: Yes — short sleep duration is independently and causally linked to obesity risk through well-established hormonal and behavioral mechanisms.",
  },
  {
    id: 5,
    shortLabel: "Probiotics & IBS",
    query: "Do probiotics help in treating irritable bowel syndrome?",
    rewrittenQuery:
      "What is the efficacy of probiotic supplementation (specific strains or multi-strain formulations) in reducing IBS symptom severity scores, abdominal pain, bloating, and stool consistency in randomized controlled trials?",
    groundTruth: "yes",
    configs: {
      baseline: { label: "maybe", faithfulness: 0.76, answerRelevancy: 0.44, contextPrecision: 0.66, contextRecall: 0.69 },
      qr:       { label: "maybe", faithfulness: 0.78, answerRelevancy: 0.50, contextPrecision: 0.70, contextRecall: 0.73 },
      cr:       { label: "yes",   faithfulness: 0.83, answerRelevancy: 0.62, contextPrecision: 0.85, contextRecall: 0.71 },
      qr_cr:    { label: "yes",   faithfulness: 0.87, answerRelevancy: 0.68, contextPrecision: 0.90, contextRecall: 0.79 },
    },
    retrievedDocs: [
      {
        id: "d21",
        title: "Probiotics for irritable bowel syndrome: systematic review",
        journal: "American Journal of Gastroenterology, 2021",
        bm25Score: 0.956,
        content:
          "Systematic review and meta-analysis of 53 RCTs (n=5,545 patients) found probiotics superior to placebo for global IBS symptom improvement (RR 1.82, 95% CI 1.54-2.15) and abdominal pain reduction (SMD -0.34). Multi-strain formulations showed greater efficacy than single-strain. Benefits were observed across IBS subtypes (IBS-D, IBS-C, IBS-M). Treatment duration of ≥8 weeks showed better outcomes.",
        relevantSnippet:
          "Probiotics superior to placebo for global IBS symptom improvement (RR 1.82)",
      },
      {
        id: "d22",
        title: "Lactobacillus and Bifidobacterium in IBS management",
        journal: "Gut, 2020",
        bm25Score: 0.901,
        content:
          "Double-blind RCT of Lactobacillus plantarum 299v in 214 IBS patients showed significant improvement in IBS Symptom Severity Score (IBS-SSS) versus placebo (-89 vs -42 points, p=0.003). Bloating and flatulence showed greatest improvement. The probiotic was well-tolerated with no serious adverse events. 4-week post-treatment follow-up showed sustained benefits.",
        relevantSnippet:
          "Lactobacillus plantarum 299v showed significant IBS-SSS improvement (-89 vs -42 points)",
      },
      {
        id: "d23",
        title: "Gut microbiome dysbiosis in IBS pathogenesis",
        journal: "Nature Reviews Gastroenterology, 2022",
        bm25Score: 0.834,
        content:
          "IBS patients exhibit reduced microbial diversity, decreased Firmicutes/Bacteroidetes ratio, and enrichment of pro-inflammatory species. Post-infectious IBS shows most pronounced dysbiosis. Probiotics may restore eubiosis, reduce intestinal permeability, and modulate the gut-brain axis. Fecal microbiota transplantation trials show 30-40% symptom response rates.",
        relevantSnippet:
          "Probiotics may restore eubiosis, reduce intestinal permeability, and modulate the gut-brain axis",
      },
      {
        id: "d24",
        title: "Strain-specific effects of probiotics in functional GI disorders",
        journal: "Alimentary Pharmacology & Therapeutics, 2021",
        bm25Score: 0.767,
        content:
          "Probiotic efficacy in IBS is highly strain-specific. VSL#3 (multi-strain) showed significant improvement in bloating and quality of life. Bifidobacterium infantis 35624 demonstrated particular efficacy for IBS-D. Head-to-head comparisons reveal significant inter-strain variability. Current evidence does not support universal probiotic recommendations; strain selection should be evidence-based.",
        relevantSnippet:
          "Probiotic efficacy in IBS is highly strain-specific; strain selection should be evidence-based",
      },
      {
        id: "d25",
        title: "Dietary fiber and synbiotics in IBS treatment",
        journal: "Clinical Nutrition, 2022",
        bm25Score: 0.712,
        content:
          "Combining probiotics with prebiotics (synbiotics) may enhance efficacy through improved probiotic survival and colonization. Synbiotic formulations showed 15% greater symptom improvement than probiotics alone in a meta-analysis of 12 trials. Soluble fiber (psyllium) independently improves IBS symptoms. The NICE guideline recommends fiber adjustment as first-line intervention before probiotics.",
        relevantSnippet:
          "Synbiotics showed 15% greater symptom improvement than probiotics alone",
      },
    ],
    rerankedDocs: [
      { docId: "d21", ceScore: 0.983, prevRank: 1 },
      { docId: "d22", ceScore: 0.945, prevRank: 2 },
      { docId: "d24", ceScore: 0.867, prevRank: 4 },
      { docId: "d23", ceScore: 0.756, prevRank: 3 },
      { docId: "d25", ceScore: 0.634, prevRank: 5 },
    ],
    answer:
      "Yes — probiotics are effective in treating irritable bowel syndrome, though efficacy is strain-specific.\n\nA meta-analysis of 53 randomized controlled trials (n=5,545) found probiotics significantly superior to placebo for global IBS symptom improvement (RR 1.82) and abdominal pain reduction. Multi-strain formulations outperformed single-strain products. A double-blind RCT of Lactobacillus plantarum 299v demonstrated a nearly doubling of symptom score improvement versus placebo (-89 vs -42 IBS-SSS points).\n\nThe biological rationale is well-established: IBS involves gut microbiome dysbiosis, and probiotics restore microbial balance, reduce intestinal permeability, and modulate the gut-brain axis. Treatment duration of ≥8 weeks and multi-strain formulations yield better outcomes.\n\nImportantly, probiotic efficacy is highly strain-specific — not all probiotic products are equivalent. Combining probiotics with prebiotics (synbiotics) may provide an additional 15% benefit.\n\nConclusion: Yes — probiotics reduce IBS symptoms with good evidence from multiple RCTs, though product selection should be evidence-based using clinically validated strains.",
  },
]
