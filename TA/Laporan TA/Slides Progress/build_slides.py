"""Build progress bimbingan TA slides — minimalis, biru tua accent."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "Code TA" / "notebooks" / "figures"
OUT = ROOT / "Slides Progress" / "Progress-Bimbingan.pptx"

# Palette (Charcoal Minimal + biru tua accent)
NAVY = RGBColor(0x1F, 0x4E, 0x79)        # accent
INK = RGBColor(0x21, 0x21, 0x21)         # heading
BODY = RGBColor(0x44, 0x44, 0x44)        # body
MUTED = RGBColor(0x88, 0x88, 0x88)       # captions
LINE = RGBColor(0xDD, 0xDD, 0xDD)
BG_SOFT = RGBColor(0xF7, 0xF9, 0xFC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

# 16:9
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height

BLANK = prs.slide_layouts[6]


def add_slide():
    s = prs.slides.add_slide(BLANK)
    # white background (default already white) — set explicit for safety
    return s


def add_text(slide, x, y, w, h, text, *, size=14, bold=False, color=BODY,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Calibri"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0)
    tf.margin_right = Inches(0)
    tf.margin_top = Inches(0)
    tf.margin_bottom = Inches(0)
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return tb


def add_bullets(slide, x, y, w, h, items, *, size=14, color=BODY, bullet_color=NAVY,
                line_spacing=1.25, font="Calibri"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0)
    tf.margin_right = Inches(0)
    tf.margin_top = Inches(0)
    tf.margin_bottom = Inches(0)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = line_spacing
        p.space_after = Pt(4)
        # bullet square
        r1 = p.add_run()
        r1.text = "■  "
        r1.font.name = font
        r1.font.size = Pt(size)
        r1.font.color.rgb = bullet_color
        r1.font.bold = True
        # body
        r2 = p.add_run()
        r2.text = item
        r2.font.name = font
        r2.font.size = Pt(size)
        r2.font.color.rgb = color
    return tb


def add_rect(slide, x, y, w, h, fill, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(0.5)
    sh.shadow.inherit = False
    return sh


def add_accent_bar(slide, x, y, h=0.5, w=0.08, color=NAVY):
    return add_rect(slide, x, y, w, h, color)


def add_page_header(slide, eyebrow, title):
    """Top accent bar + eyebrow label + slide title."""
    add_accent_bar(slide, 0.6, 0.55, h=0.42, w=0.06)
    add_text(slide, 0.78, 0.5, 8, 0.32, eyebrow.upper(),
             size=10, bold=True, color=NAVY)
    add_text(slide, 0.78, 0.78, 12, 0.7, title,
             size=26, bold=True, color=INK, font="Calibri")
    # thin separator
    add_rect(slide, 0.6, 1.55, 12.13, 0.02, LINE)


def add_footer(slide, page_num, total):
    add_text(slide, 0.6, 7.05, 6, 0.3,
             "Progress TA — Ricky Wijaya 18222043",
             size=9, color=MUTED)
    add_text(slide, 11.5, 7.05, 1.2, 0.3,
             f"{page_num} / {total}",
             size=9, color=MUTED, align=PP_ALIGN.RIGHT)


# ──────────────────────── SLIDE 1 — COVER ────────────────────────
TOTAL = 12
s = add_slide()
# left navy band
add_rect(s, 0, 0, 4.6, 7.5, NAVY)
# decorative thin lines
add_rect(s, 0.7, 6.4, 1.2, 0.04, WHITE)

add_text(s, 0.7, 0.7, 3.4, 0.35, "TUGAS AKHIR",
         size=11, bold=True, color=WHITE, font="Calibri")
add_text(s, 0.7, 1.05, 3.4, 0.3, "Program Studi Sistem dan Teknologi Informasi",
         size=10, color=RGBColor(0xCA, 0xDC, 0xFC), font="Calibri")
add_text(s, 0.7, 1.3, 3.4, 0.3, "Institut Teknologi Bandung",
         size=10, color=RGBColor(0xCA, 0xDC, 0xFC), font="Calibri")

add_text(s, 0.7, 5.6, 3.4, 0.3, "Bimbingan",
         size=11, bold=True, color=WHITE)
add_text(s, 0.7, 5.95, 3.4, 0.3, "29 April 2026",
         size=11, color=RGBColor(0xCA, 0xDC, 0xFC))

# right side
add_text(s, 5.0, 1.6, 8, 0.45, "PROGRESS",
         size=14, bold=True, color=NAVY, font="Calibri")
add_text(s, 5.0, 2.0, 8, 0.9, "Progress Tugas Akhir",
         size=42, bold=True, color=INK, font="Calibri")
add_rect(s, 5.0, 3.05, 1.2, 0.05, NAVY)

add_text(s, 5.0, 3.3, 8, 1.6,
         "Mitigasi Halusinasi pada RAG-LLM Menggunakan "
         "Query Rewriting, Context Reranking, dan "
         "Active Hallucination Detection",
         size=16, color=BODY, font="Calibri")

add_text(s, 5.0, 5.6, 4, 0.3, "Disusun oleh",
         size=10, bold=True, color=MUTED)
add_text(s, 5.0, 5.85, 6, 0.35, "Ricky Wijaya — 18222043",
         size=14, bold=True, color=INK)

add_text(s, 5.0, 6.35, 4, 0.3, "Pembimbing",
         size=10, bold=True, color=MUTED)
add_text(s, 5.0, 6.6, 8, 0.35, "Dr. Fetty Fitriyanti Lubis, S.T., M.T.",
         size=12, color=BODY)


# ──────────────── SLIDE 2 — RECAP MASALAH & TUJUAN ────────────────
s = add_slide()
add_page_header(s, "01 — Konteks", "Recap Masalah & Tujuan")

# two columns
# left: Masalah
add_rect(s, 0.6, 1.85, 5.95, 4.9, BG_SOFT)
add_accent_bar(s, 0.85, 2.1, h=0.3, w=0.06)
add_text(s, 1.05, 2.05, 5, 0.35, "MASALAH",
         size=11, bold=True, color=NAVY)
add_text(s, 0.85, 2.55, 5.6, 1.0,
         "LLM rentan hallucination — keluaran terdengar meyakinkan tapi tidak faktual.",
         size=15, bold=True, color=INK)
add_text(s, 0.85, 3.7, 5.6, 0.7,
         "RAG (Retrieval-Augmented Generation) mengurangi tapi tidak "
         "menghilangkan halusinasi karena:",
         size=12.5, color=BODY)
add_bullets(s, 0.85, 4.55, 5.6, 2.0, [
    "Retrieval bisa gagal (kueri ambigu, dokumen tidak relevan)",
    "Generation bisa tidak setia pada konteks yang benar",
    "Konteks bising mendorong LLM mengarang isi",
], size=12)

# right: Tujuan
add_rect(s, 6.85, 1.85, 5.9, 4.9, BG_SOFT)
add_accent_bar(s, 7.1, 2.1, h=0.3, w=0.06)
add_text(s, 7.3, 2.05, 5, 0.35, "TUJUAN TA",
         size=11, bold=True, color=NAVY)
add_text(s, 7.1, 2.55, 5.5, 1.0,
         "Bandingkan RAG standar vs RAG + 3 teknik mitigasi.",
         size=15, bold=True, color=INK)
add_bullets(s, 7.1, 3.7, 5.5, 2.7, [
    "Query Rewriting — perbaiki kueri sebelum retrieval",
    "Context Reranking — pilih dokumen paling relevan",
    "Active Hallucination Detection — verifikasi output LLM",
    "Dataset: PubMedQA pqa_labeled, n=500 (yes/no/maybe)",
], size=13)

add_footer(s, 2, TOTAL)


# ──────────────── SLIDE 3 — RAG PIPELINE ────────────────
s = add_slide()
add_page_header(s, "02 — Sistem", "RAG Pipeline yang Digunakan")

# Pipeline diagram horizontal
stages = [
    ("User Query", "Pertanyaan medis\ndari pengguna"),
    ("Query Rewriting", "Opsional — LLM\nmenulis ulang kueri"),
    ("Retriever", "BM25 atau\nHybrid (BM25+Dense)"),
    ("Context Reranking", "Opsional — CrossEncoder\nrerank top-20 → top-5"),
    ("LLM Generation", "Llama / OpenAI /\nClaude jawab"),
    ("Final Answer", "yes / no / maybe"),
]
n = len(stages)
margin_x = 0.6
total_w = 12.13
gap = 0.15
box_w = (total_w - gap * (n - 1)) / n
box_y = 2.1
box_h = 1.6

for i, (title, sub) in enumerate(stages):
    x = margin_x + i * (box_w + gap)
    add_rect(s, x, box_y, box_w, box_h, BG_SOFT, line=LINE)
    add_text(s, x + 0.05, box_y + 0.15, box_w - 0.1, 0.4,
             f"0{i+1}", size=11, bold=True, color=NAVY)
    add_text(s, x + 0.1, box_y + 0.5, box_w - 0.2, 0.5,
             title, size=12.5, bold=True, color=INK)
    add_text(s, x + 0.1, box_y + 1.0, box_w - 0.2, 0.55,
             sub, size=9.5, color=BODY)
    if i < n - 1:
        # arrow between
        ax = x + box_w + 0.005
        ay = box_y + box_h / 2
        ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_TRIANGLE,
                                Inches(ax), Inches(ay - 0.06),
                                Inches(0.14), Inches(0.12))
        ar.rotation = 30
        ar.fill.solid()
        ar.fill.fore_color.rgb = NAVY
        ar.line.fill.background()

# Below: 4 konfigurasi
add_text(s, 0.6, 4.1, 12, 0.35,
         "4 KONFIGURASI YANG DIBANDINGKAN",
         size=11, bold=True, color=NAVY)

configs = [
    ("Baseline", "Retriever → LLM"),
    ("+ QR", "QR → Retriever → LLM"),
    ("+ CR", "Retriever → CR → LLM"),
    ("+ QR + CR", "QR → Retriever → CR → LLM"),
]
cw = (12.13 - 0.3 * 3) / 4
for i, (name, flow) in enumerate(configs):
    x = 0.6 + i * (cw + 0.3)
    add_rect(s, x, 4.5, cw, 1.1, WHITE, line=NAVY)
    add_text(s, x + 0.15, 4.62, cw - 0.3, 0.4,
             name, size=14, bold=True, color=NAVY)
    add_text(s, x + 0.15, 5.05, cw - 0.3, 0.5,
             flow, size=10.5, color=BODY)

# 3 LLM tested
add_text(s, 0.6, 5.95, 12, 0.35,
         "3 LLM YANG DIUJI",
         size=11, bold=True, color=NAVY)
add_bullets(s, 0.85, 6.3, 12, 0.7, [
    "Llama 3.3 70B (via Groq) — open-source benchmark",
    "GPT-4.1-mini (OpenAI) — closed-source efficient",
    "Claude Haiku 4.5 (Anthropic) — closed-source efficient",
], size=12)

add_footer(s, 3, TOTAL)


# ──────────────── SLIDE 4 — CHUNKING STRATEGY ────────────────
s = add_slide()
add_page_header(s, "03 — Data", "Chunking Strategy")

# Left: Visual stats
add_rect(s, 0.6, 1.85, 5.95, 5, NAVY)
add_text(s, 0.85, 2.1, 5, 0.4, "PUBMEDQA CORPUS",
         size=11, bold=True, color=RGBColor(0xCA, 0xDC, 0xFC))

add_text(s, 0.85, 2.6, 5.5, 1.4, "1.706",
         size=72, bold=True, color=WHITE)
add_text(s, 0.85, 4.1, 5.5, 0.4, "chunks total",
         size=14, color=RGBColor(0xCA, 0xDC, 0xFC))

add_rect(s, 0.85, 4.7, 5, 0.02, RGBColor(0x6F, 0x9D, 0xC8))

add_text(s, 0.85, 4.9, 5.5, 0.4, "500 abstrak × ~3.4 chunk per abstrak",
         size=12, color=WHITE)
add_text(s, 0.85, 5.4, 5.5, 0.4, "200–400 token per chunk",
         size=12, color=WHITE)
add_text(s, 0.85, 5.9, 5.5, 0.4, "Sumber: PubMed scientific abstracts",
         size=12, color=WHITE)

# Right: Strategy explanation
add_text(s, 6.85, 2.0, 6, 0.4, "PER-SECTION CHUNKING",
         size=11, bold=True, color=NAVY)
add_text(s, 6.85, 2.4, 6, 0.7,
         "Setiap abstrak dipecah berdasarkan section ilmiahnya:",
         size=13, color=INK)

sections = [
    ("Background", "konteks penelitian"),
    ("Methods", "metode/protokol"),
    ("Results", "temuan numerik"),
    ("Conclusion", "ringkasan jawaban"),
]
for i, (lbl, desc) in enumerate(sections):
    y = 3.2 + i * 0.5
    add_rect(s, 6.85, y, 0.15, 0.35, NAVY)
    add_text(s, 7.15, y, 2.2, 0.35, lbl,
             size=12.5, bold=True, color=INK)
    add_text(s, 9.0, y, 3.7, 0.35, desc,
             size=11.5, color=BODY)

add_text(s, 6.85, 5.45, 6, 0.4, "ALASAN",
         size=11, bold=True, color=NAVY)
add_bullets(s, 6.85, 5.85, 6, 1.2, [
    "Menjaga koherensi semantik per topik",
    "Ukuran pas untuk context window LLM",
    "Hindari memotong klausa medis penting",
], size=11.5)

add_footer(s, 4, TOTAL)


# ──────────────── SLIDE 5 — EMBEDDING & RETRIEVER ────────────────
s = add_slide()
add_page_header(s, "04 — Sistem", "Embedding & Retriever — 2 Variasi")

# left card — BM25 only
def variant_card(x, y, w, h, label, title, color_bg, items, footer_note):
    add_rect(s, x, y, w, h, color_bg, line=LINE)
    add_text(s, x + 0.25, y + 0.2, w - 0.5, 0.3, label,
             size=10, bold=True, color=NAVY)
    add_text(s, x + 0.25, y + 0.5, w - 0.5, 0.5, title,
             size=17, bold=True, color=INK)
    add_rect(s, x + 0.25, y + 1.0, 0.8, 0.04, NAVY)
    add_bullets(s, x + 0.25, y + 1.15, w - 0.5, h - 1.7,
                items, size=12)
    add_text(s, x + 0.25, y + h - 0.45, w - 0.5, 0.35,
             footer_note, size=10, color=MUTED, font="Calibri")

variant_card(
    0.6, 1.85, 5.95, 5.0,
    "VARIASI 1",
    "BM25-only (Sparse)",
    WHITE,
    [
        "Library: rank_bm25 (BM25Okapi)",
        "Tokenisasi sederhana, exact keyword match",
        "Top-5 dokumen langsung ke LLM",
        "+ Cocok untuk istilah medis spesifik (nama obat, dosis, kode)",
        "− Lemah untuk parafrasa & sinonim biomedis",
    ],
    "Trade-off: ringan, deterministik, tapi miss semantic match",
)

variant_card(
    6.85, 1.85, 5.9, 5.0,
    "VARIASI 2",
    "Hybrid (BM25 + Dense)",
    BG_SOFT,
    [
        "BM25 top-50 ⊕ Dense top-50",
        "Dense: OpenAI text-embedding-3-small (1536-dim)",
        "Vector store: ChromaDB persistent",
        "Fusion: Reciprocal Rank Fusion (k=60)",
        "Final top-5 ke LLM",
    ],
    "Menangkap exact match (BM25) + semantik (Dense)",
)

add_footer(s, 5, TOTAL)


# ──────────────── SLIDE 6 — RERANKER ────────────────
s = add_slide()
add_page_header(s, "05 — Sistem", "Reranker (Context Reranking)")

# Left: model card
add_rect(s, 0.6, 1.85, 5.95, 5, BG_SOFT)
add_text(s, 0.85, 2.1, 5, 0.35, "MODEL",
         size=11, bold=True, color=NAVY)
add_text(s, 0.85, 2.5, 5.5, 0.6, "CrossEncoder",
         size=22, bold=True, color=INK)
add_text(s, 0.85, 3.15, 5.5, 0.5, "ms-marco-MiniLM-L-6-v2",
         size=14, color=NAVY, font="Consolas")

add_rect(s, 0.85, 3.85, 0.8, 0.04, NAVY)

specs = [
    ("Parameter", "22M (CPU-friendly)"),
    ("Arsitektur", "BERT cross-encoder"),
    ("Training data", "MS MARCO passage ranking"),
    ("Pair scoring", "(query, doc) → relevance score"),
]
for i, (k, v) in enumerate(specs):
    y = 4.05 + i * 0.55
    add_text(s, 0.85, y, 2, 0.35, k,
             size=11, bold=True, color=MUTED)
    add_text(s, 2.7, y, 3.8, 0.35, v,
             size=12, color=INK)

# Right: pipeline flow
add_text(s, 6.85, 2.05, 6, 0.35, "ALUR KERJA",
         size=11, bold=True, color=NAVY)

steps = [
    ("Retrieve top-20", "BM25 atau Hybrid kembalikan 20 kandidat"),
    ("Pair scoring", "CrossEncoder skor (query, doc) untuk tiap kandidat"),
    ("Re-rank", "Urutkan ulang berdasarkan skor relevansi"),
    ("Pilih top-5", "5 dokumen terbaik diteruskan ke LLM"),
]
for i, (title, desc) in enumerate(steps):
    y = 2.55 + i * 0.95
    # number circle
    n_shape = s.shapes.add_shape(
        MSO_SHAPE.OVAL,
        Inches(6.85), Inches(y), Inches(0.5), Inches(0.5))
    n_shape.fill.solid()
    n_shape.fill.fore_color.rgb = NAVY
    n_shape.line.fill.background()
    n_tf = n_shape.text_frame
    n_tf.margin_left = 0
    n_tf.margin_right = 0
    n_tf.margin_top = 0
    n_tf.margin_bottom = 0
    n_p = n_tf.paragraphs[0]
    n_p.alignment = PP_ALIGN.CENTER
    n_r = n_p.add_run()
    n_r.text = str(i + 1)
    n_r.font.bold = True
    n_r.font.size = Pt(13)
    n_r.font.color.rgb = WHITE

    add_text(s, 7.55, y - 0.05, 5.2, 0.4, title,
             size=13.5, bold=True, color=INK)
    add_text(s, 7.55, y + 0.35, 5.2, 0.5, desc,
             size=11, color=BODY)

# Tujuan note
add_rect(s, 6.85, 6.4, 6, 0.5, NAVY)
add_text(s, 7.0, 6.45, 5.7, 0.4,
         "Tujuan: kurangi context noise sebelum input ke LLM",
         size=11.5, bold=True, color=WHITE)

add_footer(s, 6, TOTAL)


# ──────────────── SLIDE 7 — SETUP EKSPERIMEN ────────────────
s = add_slide()
add_page_header(s, "06 — Eksperimen", "Setup Eksperimen")

# Left: stats grid 2x2
def stat_box(x, y, w, h, big, label, sub=None):
    add_rect(s, x, y, w, h, WHITE, line=LINE)
    add_text(s, x + 0.2, y + 0.25, w - 0.4, 0.7, big,
             size=32, bold=True, color=NAVY)
    add_text(s, x + 0.2, y + 1.0, w - 0.4, 0.4, label,
             size=11.5, bold=True, color=INK)
    if sub:
        add_text(s, x + 0.2, y + 1.4, w - 0.4, 0.4, sub,
                 size=10, color=MUTED)

bw, bh = 2.85, 1.85
gx, gy = 0.6, 1.95
gap_x = 0.15
stat_box(gx, gy, bw, bh, "500", "Sampel test", "PubMedQA pqa_labeled")
stat_box(gx + (bw + gap_x), gy, bw, bh, "12", "Konfigurasi", "3 LLM × 4 teknik")
stat_box(gx, gy + bh + 0.15, bw, bh, "1.706", "Chunks corpus", "per-section")
stat_box(gx + (bw + gap_x), gy + bh + 0.15, bw, bh, "0.0", "Temperature", "deterministik")

# Right: details
add_text(s, 6.85, 2.0, 6, 0.4, "DISTRIBUSI LABEL",
         size=11, bold=True, color=NAVY)
labels = [("yes", 55), ("no", 32), ("maybe", 13)]
for i, (lbl, pct) in enumerate(labels):
    y = 2.45 + i * 0.55
    add_text(s, 6.85, y, 1.2, 0.4, lbl,
             size=12, bold=True, color=INK)
    # bar
    bar_w_total = 4.5
    bar_w = bar_w_total * pct / 100
    add_rect(s, 7.95, y + 0.1, bar_w_total, 0.22, BG_SOFT)
    add_rect(s, 7.95, y + 0.1, bar_w, 0.22, NAVY)
    add_text(s, 12.55, y, 0.6, 0.4, f"{pct}%",
             size=11, color=MUTED)

add_text(s, 6.85, 4.4, 6, 0.4, "METRIK EVALUASI",
         size=11, bold=True, color=NAVY)
add_bullets(s, 6.85, 4.8, 6.0, 2.0, [
    "Label Accuracy (yes/no/maybe vs ground truth)",
    "Hallucination rate = 1 − accuracy",
    "RAGAS: Faithfulness (jawaban ter-grounding ke konteks)",
    "RAGAS: Context Recall (cakupan bukti relevan)",
], size=12)

add_footer(s, 7, TOTAL)


# ──────────────── SLIDE 8 — HASIL ACCURACY ────────────────
def result_slide(eyebrow, title, image_path, headline, insights, slide_num):
    s = add_slide()
    add_page_header(s, eyebrow, title)
    # Image left (60% width)
    img_x, img_y, img_w, img_h = 0.6, 1.85, 7.5, 4.85
    add_rect(s, img_x, img_y, img_w, img_h, BG_SOFT, line=LINE)
    if image_path.exists():
        # fit image inside
        from PIL import Image
        with Image.open(image_path) as im:
            iw, ih = im.size
        ratio = min(img_w / (iw / 96), img_h / (ih / 96))  # rough
        # Use python-pptx native sizing — just set width, let height scale
        pic = s.shapes.add_picture(str(image_path),
                                   Inches(img_x + 0.1),
                                   Inches(img_y + 0.1),
                                   width=Inches(img_w - 0.2))
        # If picture exceeds box height, rescale
        if pic.height > Inches(img_h - 0.2):
            new_h = Inches(img_h - 0.2)
            scale = new_h / pic.height
            pic.height = new_h
            pic.width = int(pic.width * scale)
            # center horizontally inside the box
            pic.left = Inches(img_x + (img_w - pic.width / 914400) / 2)
            pic.top = Inches(img_y + 0.1)
    else:
        add_text(s, img_x + 0.5, img_y + 2.0, img_w - 1, 0.5,
                 f"[Image not found: {image_path.name}]",
                 size=12, color=MUTED, align=PP_ALIGN.CENTER)

    # Right: headline + insights
    rx = 8.4
    rw = 4.4
    add_text(s, rx, 1.85, rw, 0.35, "HEADLINE",
             size=10.5, bold=True, color=NAVY)
    add_text(s, rx, 2.2, rw, 1.7, headline,
             size=15, bold=True, color=INK)
    add_rect(s, rx, 3.95, 0.8, 0.04, NAVY)
    add_text(s, rx, 4.1, rw, 0.35, "INSIGHT",
             size=10.5, bold=True, color=NAVY)
    add_bullets(s, rx, 4.45, rw, 2.5, insights, size=11.5)

    add_footer(s, slide_num, TOTAL)


result_slide(
    "07 — Hasil",
    "Akurasi Antar LLM × Teknik",
    FIG / "A_accuracy_comparison.png",
    "Context Reranking konsisten meningkatkan akurasi di semua model.",
    [
        "Pemilihan model LLM mempengaruhi accuracy secara signifikan.",
        "Penambahan teknik mitigasi tidak selalu konsisten antar model.",
        "CR konsisten naikkan akurasi vs RAG biasa di semua LLM.",
        "QR paradoks: BM25 score ↑ tapi accuracy ↓ (over-specification).",
    ],
    8,
)

# ──────────────── SLIDE 9 — PER-LABEL ────────────────
result_slide(
    "08 — Hasil",
    "Per-Label Accuracy",
    FIG / "D_per_label_accuracy.png",
    "Bias prediksi yes — performa drop drastis untuk label no & maybe.",
    [
        "Recall label \"yes\" sangat tinggi (>85% di semua konfigurasi).",
        "Recall label \"no\" rendah (~17% di Llama Baseline).",
        "Recall label \"maybe\" sangat rendah (1–3%) — distribusi imbalanced.",
        "CR sedikit memperbaiki tapi belum signifikan.",
        "Bias ini di sisi generation prompting, bukan retrieval.",
    ],
    9,
)

# ──────────────── SLIDE 10 — RAGAS ────────────────
result_slide(
    "09 — Hasil",
    "RAGAS: Faithfulness & Context Recall",
    FIG / "B_ragas_heatmap.png",
    "Context Reranking mencapai Faithfulness & Context Recall tertinggi.",
    [
        "Faithfulness: CR > Baseline > QR — jawaban CR lebih ter-grounding.",
        "Context Recall: CR menemukan lebih banyak bukti relevan.",
        "QR menurunkan kedua metrik karena keyword drift di BM25.",
        "Mendukung temuan label accuracy: CR teknik paling stabil.",
    ],
    10,
)

# ──────────────── SLIDE 11 — HYBRID vs BM25 ────────────────
result_slide(
    "10 — Hasil",
    "Hybrid Retrieval vs BM25-only",
    FIG / "F1_bm25_vs_hybrid_accuracy.png",
    "Hybrid retrieval mengungguli BM25-only di hampir semua konfigurasi.",
    [
        "Hybrid (BM25+Dense) konsisten naikkan accuracy ~2–4 pp vs BM25.",
        "Kombinasi terbaik: Hybrid + CR + GPT-4.1-mini.",
        "Dense menangkap parafrasa medis yang miss di keyword match.",
        "Trade-off: butuh API embedding (~$0.03 index + $0.001/query).",
    ],
    11,
)


# ──────────────── SLIDE 12 — NEXT STEP ────────────────
s = add_slide()
add_page_header(s, "11 — Penutup", "Next Step & Pertanyaan Diskusi")

# Two columns: Done + To Do
add_text(s, 0.6, 1.85, 6, 0.35, "✓  SUDAH SELESAI",
         size=11, bold=True, color=RGBColor(0x2C, 0x6E, 0x49))
add_bullets(s, 0.6, 2.2, 6, 2.3, [
    "Implementasi Baseline, QR, CR, dan QR+CR.",
    "Eksperimen multi-LLM (Llama 3.3 70B, GPT-4.1-mini, Claude Haiku 4.5).",
    "Eksperimen multi-retriever (BM25-only vs Hybrid).",
    "Evaluasi 12 konfigurasi × 500 sampel (Phase 1 + Phase 2 RAGAS).",
], size=11.5, bullet_color=RGBColor(0x2C, 0x6E, 0x49))

add_text(s, 6.85, 1.85, 6, 0.35, "→  AKAN DIKERJAKAN",
         size=11, bold=True, color=NAVY)
add_bullets(s, 6.85, 2.2, 6, 2.3, [
    "Implementasi Active Hallucination Detection (notebook 05).",
    "Konfigurasi gabungan E (QR + CR + AHD).",
    "Penulisan Bab IV (Perancangan), V (Implementasi), VI (Evaluasi).",
    "Revisi Abstrak (saat ini masih placeholder template).",
], size=11.5)

# bottom: Pertanyaan diskusi
add_rect(s, 0.6, 4.85, 12.13, 1.95, BG_SOFT)
add_accent_bar(s, 0.85, 5.05, h=0.3, w=0.06)
add_text(s, 1.05, 5.0, 11, 0.35, "PERTANYAAN UNTUK DISKUSI",
         size=11, bold=True, color=NAVY)
add_bullets(s, 0.85, 5.45, 11.5, 1.3, [
    "Apakah scope AHD cukup pakai self-consistency check, atau perlu confidence-based detection juga?",
    "Apakah Hybrid retrieval bisa masuk sebagai kontribusi tambahan thesis (ablation retriever)?",
], size=12.5)

add_footer(s, 12, TOTAL)


# ──────────────── SAVE ────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"Saved: {OUT}")
print(f"Slides: {len(prs.slides)}")
