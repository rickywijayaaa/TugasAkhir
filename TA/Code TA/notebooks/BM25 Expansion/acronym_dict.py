"""Curated dictionary of ~150 most common biomedical acronyms.

Sumber: combinasi MeSH headings, common ICD entries, medical statistics,
study designs, dan procedures yang paling sering muncul di PubMed abstracts.

Format: ACRONYM (uppercase) -> full form (lowercase, untuk konsistensi BM25).
"""

COMMON_MEDICAL_ACRONYMS = {
    # ===== Conditions / Diseases =====
    "MI":     "myocardial infarction",
    "AMI":    "acute myocardial infarction",
    "CHF":    "congestive heart failure",
    "CAD":    "coronary artery disease",
    "CHD":    "coronary heart disease",
    "CVD":    "cardiovascular disease",
    "HTN":    "hypertension",
    "DM":     "diabetes mellitus",
    "T1DM":   "type 1 diabetes mellitus",
    "T2DM":   "type 2 diabetes mellitus",
    "DKA":    "diabetic ketoacidosis",
    "COPD":   "chronic obstructive pulmonary disease",
    "OSA":    "obstructive sleep apnea",
    "OSAS":   "obstructive sleep apnea syndrome",
    "GERD":   "gastroesophageal reflux disease",
    "IBD":    "inflammatory bowel disease",
    "IBS":    "irritable bowel syndrome",
    "UC":     "ulcerative colitis",
    "CKD":    "chronic kidney disease",
    "AKI":    "acute kidney injury",
    "ESRD":   "end stage renal disease",
    "CRF":    "chronic renal failure",
    "AF":     "atrial fibrillation",
    "DVT":    "deep vein thrombosis",
    "PE":     "pulmonary embolism",
    "TB":     "tuberculosis",
    "HIV":    "human immunodeficiency virus",
    "AIDS":   "acquired immunodeficiency syndrome",
    "HCV":    "hepatitis C virus",
    "HBV":    "hepatitis B virus",
    "HCC":    "hepatocellular carcinoma",
    "RA":     "rheumatoid arthritis",
    "OA":     "osteoarthritis",
    "MS":     "multiple sclerosis",
    "AD":     "Alzheimer disease",
    "PD":     "Parkinson disease",
    "ASD":    "autism spectrum disorder",
    "ADHD":   "attention deficit hyperactivity disorder",
    "PTSD":   "post traumatic stress disorder",
    "MDD":    "major depressive disorder",
    "BPD":    "bipolar disorder",
    "TBI":    "traumatic brain injury",
    "SCI":    "spinal cord injury",
    "ALS":    "amyotrophic lateral sclerosis",

    # ===== Procedures / Interventions =====
    "PCI":    "percutaneous coronary intervention",
    "CABG":   "coronary artery bypass graft",
    "TAVR":   "transcatheter aortic valve replacement",
    "TAVI":   "transcatheter aortic valve implantation",
    "ICD":    "implantable cardioverter defibrillator",
    "CRT":    "cardiac resynchronization therapy",
    "ECMO":   "extracorporeal membrane oxygenation",
    "CPR":    "cardiopulmonary resuscitation",
    "CABP":   "coronary artery bypass procedure",
    "IVF":    "in vitro fertilization",
    "ART":    "assisted reproductive technology",
    "C-section": "cesarean section",
    "ERCP":   "endoscopic retrograde cholangiopancreatography",
    "EUS":    "endoscopic ultrasound",

    # ===== Imaging =====
    "MRI":    "magnetic resonance imaging",
    "CT":     "computed tomography",
    "PET":    "positron emission tomography",
    "SPECT":  "single photon emission computed tomography",
    "US":     "ultrasound",
    "ECG":    "electrocardiogram",
    "EKG":    "electrocardiogram",
    "EEG":    "electroencephalogram",
    "DXA":    "dual energy x-ray absorptiometry",
    "DEXA":   "dual energy x-ray absorptiometry",

    # ===== Measurements / Vitals =====
    "BP":     "blood pressure",
    "SBP":    "systolic blood pressure",
    "DBP":    "diastolic blood pressure",
    "HR":     "heart rate",
    "RR":     "respiratory rate",
    "BMI":    "body mass index",
    "WHR":    "waist to hip ratio",
    "GFR":    "glomerular filtration rate",
    "eGFR":   "estimated glomerular filtration rate",
    "EF":     "ejection fraction",
    "LVEF":   "left ventricular ejection fraction",
    "FEV1":   "forced expiratory volume in one second",
    "FVC":    "forced vital capacity",
    "HbA1c":  "hemoglobin A1c",
    "LDL":    "low density lipoprotein",
    "HDL":    "high density lipoprotein",
    "TG":     "triglycerides",
    "TC":     "total cholesterol",
    "FBG":    "fasting blood glucose",

    # ===== Drugs / Treatments =====
    "NSAID":  "non steroidal anti inflammatory drug",
    "NSAIDs": "non steroidal anti inflammatory drugs",
    "ACEi":   "angiotensin converting enzyme inhibitor",
    "ARB":    "angiotensin receptor blocker",
    "BB":     "beta blocker",
    "CCB":    "calcium channel blocker",
    "PPI":    "proton pump inhibitor",
    "SSRI":   "selective serotonin reuptake inhibitor",
    "SNRI":   "serotonin norepinephrine reuptake inhibitor",
    "TCA":    "tricyclic antidepressant",
    "DOAC":   "direct oral anticoagulant",
    "NOAC":   "novel oral anticoagulant",
    "LMWH":   "low molecular weight heparin",
    "OCP":    "oral contraceptive pill",
    "HRT":    "hormone replacement therapy",
    "PDE5":   "phosphodiesterase type 5",

    # ===== Cancer =====
    "NHL":    "non Hodgkin lymphoma",
    "AML":    "acute myeloid leukemia",
    "CML":    "chronic myeloid leukemia",
    "ALL":    "acute lymphoblastic leukemia",
    "CLL":    "chronic lymphocytic leukemia",
    "NSCLC":  "non small cell lung cancer",
    "SCLC":   "small cell lung cancer",
    "BRCA":   "breast cancer",
    "PSA":    "prostate specific antigen",
    "CEA":    "carcinoembryonic antigen",
    "CA":     "cancer antigen",

    # ===== Lab / Biomarkers =====
    "CRP":    "C reactive protein",
    "ESR":    "erythrocyte sedimentation rate",
    "CBC":    "complete blood count",
    "WBC":    "white blood cell",
    "RBC":    "red blood cell",
    "PLT":    "platelet",
    "INR":    "international normalized ratio",
    "PT":     "prothrombin time",
    "PTT":    "partial thromboplastin time",
    "AST":    "aspartate aminotransferase",
    "ALT":    "alanine aminotransferase",
    "ALP":    "alkaline phosphatase",
    "BUN":    "blood urea nitrogen",
    "Cr":     "creatinine",
    "TSH":    "thyroid stimulating hormone",
    "T3":     "triiodothyronine",
    "T4":     "thyroxine",
    "PTH":    "parathyroid hormone",
    "BNP":    "B type natriuretic peptide",
    "NT-proBNP": "N terminal pro B type natriuretic peptide",
    "TNF":    "tumor necrosis factor",
    "IL":     "interleukin",

    # ===== Study Design / Statistics =====
    "RCT":    "randomized controlled trial",
    "RCTs":   "randomized controlled trials",
    "OR":     "odds ratio",
    "RR":     "relative risk",
    "HR":     "hazard ratio",
    "CI":     "confidence interval",
    "ICC":    "intraclass correlation coefficient",
    "AUC":    "area under the curve",
    "ROC":    "receiver operating characteristic",
    "NNT":    "number needed to treat",
    "NNH":    "number needed to harm",
    "ITT":    "intention to treat",
    "PP":     "per protocol",
    "QoL":    "quality of life",
    "QOL":    "quality of life",
    "HRQOL":  "health related quality of life",

    # ===== Healthcare Systems =====
    "ICU":    "intensive care unit",
    "NICU":   "neonatal intensive care unit",
    "PICU":   "pediatric intensive care unit",
    "ED":     "emergency department",
    "ER":     "emergency room",
    "OR":     "operating room",
    "PCP":    "primary care physician",
    "GP":     "general practitioner",
    "EHR":    "electronic health record",
    "EMR":    "electronic medical record",
    "PRO":    "patient reported outcome",

    # ===== Anatomy / Physiology =====
    "CNS":    "central nervous system",
    "PNS":    "peripheral nervous system",
    "GI":     "gastrointestinal",
    "GU":     "genitourinary",
    "MSK":    "musculoskeletal",
    "LV":     "left ventricular",
    "RV":     "right ventricular",
    "LA":     "left atrium",
    "RA":     "right atrium",  # (also rheumatoid arthritis - context dependent)
    "DNA":    "deoxyribonucleic acid",
    "RNA":    "ribonucleic acid",
    "mRNA":   "messenger ribonucleic acid",

    # ===== Specialties / Misc =====
    "OB":     "obstetrics",
    "GYN":    "gynecology",
    "ENT":    "ear nose and throat",
    "ORL":    "otorhinolaryngology",
    "NICU":   "neonatal intensive care unit",
    "WHO":    "World Health Organization",
    "FDA":    "Food and Drug Administration",
    "NIH":    "National Institutes of Health",
    "CDC":    "Centers for Disease Control",
}


# ===== Acronym yang ambiguous (perlu context-aware handling) =====
# Skip dari static dict, biarkan in-paper detection yang handle
AMBIGUOUS_ACRONYMS = {
    "PR",    # public relations / pulse rate / progesterone receptor
    "HR",    # heart rate / hazard ratio (already in dict but ambiguous)
    "OR",    # odds ratio / operating room
    "RA",    # rheumatoid arthritis / right atrium
    "MS",    # multiple sclerosis / mass spectrometry / mitral stenosis
    "CT",    # computed tomography / clinical trial
    "PT",    # physical therapy / prothrombin time / patient
    "OT",    # occupational therapy
}


if __name__ == '__main__':
    print(f'Total curated acronyms: {len(COMMON_MEDICAL_ACRONYMS)}')
    print(f'Ambiguous (skip from auto-expansion): {len(AMBIGUOUS_ACRONYMS)}')
