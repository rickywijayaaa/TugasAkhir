/**
 * 6 sample questions from PubMedQA, chosen because they are:
 * - Easy for non-medical readers to follow
 * - Relevant to everyday health concerns
 * - Spread across topics (nutrition, sleep, treatment, mental health, heart, diet)
 * - **All in our 500-PMID corpus** so retrieval consistently lands on the
 *   expected paper and the confidence guard rail returns HIGH.
 *
 * Each entry holds:
 *  - pubid    : original PubMed ID
 *  - topic    : label for the sample card
 *  - icon     : emoji used as a visual cue
 *  - question : PubMedQA question (English) — phrased close to the original
 *               paper question so cross-encoder rerank scores stay high
 *  - answer   : warm, natural-language mock answer (markdown) hand-crafted
 *               from the abstract; used only when the backend is offline
 */

export const SAMPLE_QUESTIONS = [
  {
    pubid: '20684175',
    topic: 'Nutrition',
    icon: '🌿',
    question: 'Does vitamin D supplementation enhance regulatory T cells in healthy adults?',
    answer: `**Yes, there is evidence to support this.**

A clinical study found that giving vitamin D supplements to apparently healthy adults **significantly increased the number of regulatory T cells (Tregs)**. Tregs are part of the immune system that help keep immune responses balanced and prevent the body from overreacting.

This finding is interesting because vitamin D deficiency has often been linked to autoimmune diseases (where the immune system attacks the body itself). In other words, vitamin D seems to act like a "traffic controller" that helps the immune system stay in balance.

Keep in mind that this effect was studied in healthy adults, so the impact on people with specific medical conditions may need a direct conversation with a doctor.`,
  },
  {
    pubid: '17076091',
    topic: 'Sleep',
    icon: '😴',
    question: 'Does sleep apnea affect aerobic fitness?',
    answer: `**The answer depends on how severe the condition is.**

A study of US Air Force aviators found that, **in general, obstructive sleep apnea (OSA) did not predict a decline in aerobic fitness** as measured by cycle ergometry.

However, **patients with severe apnea** (an apnea-hypopnea index above 20) showed a **statistically significant drop in aerobic fitness** compared with the normal population. So mild-to-moderate apnea may leave fitness largely intact, but once it becomes severe, the impact starts to show.

One important caveat: the study was conducted on a generally fit population (military pilots), so results in the general population could differ. If you have symptoms of sleep apnea (loud snoring, daytime sleepiness), it's worth seeing a doctor — the impact reaches beyond fitness to heart and brain health too.`,
  },
  {
    pubid: '26370095',
    topic: 'Treatment',
    icon: '💊',
    question: 'Are financial incentives effective for helping pregnant women quit smoking?',
    answer: `**Yes, and they turn out to be very cost-effective.**

This study analyzed the cost-effectiveness of a financial incentive program (up to £400) added on top of standard smoking-cessation support for pregnant women. The results showed that **financial incentives for quitting smoking in pregnancy are highly cost-effective**, at an incremental cost of around **£482 per Quality-Adjusted Life Year (QALY)** — well below the usual healthcare decision threshold of £20,000–30,000 per QALY.

In other words, paying pregnant women to quit smoking not only helps them stop, but also saves **substantial long-term healthcare costs** linked to smoking-related complications (low birth weight, preterm birth, later cardiovascular disease, and more).

It's a striking example of how a simple, incentive-based intervention can have a big public-health payoff.`,
  },
  {
    pubid: '12920330',
    topic: 'Mental Health',
    icon: '🧠',
    question: 'Do somatic complaints predict subsequent symptoms of depression?',
    answer: `**The evidence is partial — yes for women, but the link isn't as strong as you might expect.**

Researchers followed an initially healthy group of community adults for **five years** as part of the RENO Diet-Heart Study. They measured **physical (somatic) complaints** at the start, then checked for depressive symptoms five years later.

The results showed that **higher somatic complaints did predict later depression — but only in women**, not in men. And even for women, the effect was real but **weaker than other factors like age and income**. The strongest predictor of future depression was simply the person's baseline mood score.

So while paying attention to recurring unexplained physical symptoms (fatigue, aches, stomach issues) can be useful, they're just one signal among many. If you're noticing patterns that concern you, talking to a healthcare professional gives a clearer picture than focusing on any single sign.`,
  },
  {
    pubid: '24019262',
    topic: 'Heart Health',
    icon: '❤️',
    question: 'Does high blood pressure reduce the risk of chronic low back pain?',
    answer: `**Surprisingly, yes — at least according to this large Norwegian study.**

Researchers analyzed data from the **HUNT health survey**, including **nearly 40,000 adults** who had never used blood pressure medication, plus a follow-up study tracking another **17,000 people** over about 10 years.

They found an **inverse relationship**: people with **higher blood pressure had a slightly lower risk of developing chronic low back pain**. For women specifically, every 10 mmHg increase in pulse pressure was linked to about a **7% lower risk** of chronic back pain over time. The pattern showed up clearly in women, but the results in men were less consistent.

Why might this happen? One leading theory is that higher blood pressure can subtly reduce pain sensitivity — a phenomenon already documented for migraines and headaches.

That said, **this doesn't mean high blood pressure is good for you**. Hypertension is still a major risk factor for heart disease and stroke, which far outweigh any pain-related benefit. Treat this as an intriguing scientific finding, not a recommendation.`,
  },
  {
    pubid: '17595200',
    topic: 'Diet & Weight',
    icon: '🥦',
    question: 'Is there an intrauterine influence on obesity?',
    answer: `**The evidence suggests no — at least not a major one.**

The idea behind this question is that if a mother's body weight during pregnancy heavily shapes her child's later obesity risk, the **mother's BMI should predict the child's BMI much more strongly than the father's BMI** does (since the father isn't involved in the pregnancy environment).

Researchers tested this using data from the **ALSPAC study** in the UK, looking at over **4,600 parent-child trios**. They compared how strongly each parent's BMI predicted the child's BMI at age 7.5.

The result: **both parents' BMI predicted the child's BMI about equally**. If pregnancy environment were the dominant factor, we'd expect a much stronger mother-child link. Instead, the similarity between father and mother suggests that **genetics and shared family lifestyle** (eating habits, activity levels, food environment at home) likely matter more than what happens during pregnancy itself.

So while a healthy pregnancy is important for many reasons, the long-term obesity story seems to be written more by everyday family habits than by the womb.`,
  },
]
