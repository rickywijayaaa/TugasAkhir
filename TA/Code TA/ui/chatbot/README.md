# BioRAG Chatbot

Halaman chatbot biomedis untuk demo sistem tanya jawab berbasis RAG-LLM. Terpisah dari project visualisasi RAG utama (`ui/`).

## Tech Stack

Konsisten dengan project utama:
- React 18 + Vite 5
- TailwindCSS 3.4
- Font: Inter (Google Fonts)

Tambahan untuk chatbot:
- `react-markdown` + `remark-gfm` — render markdown pada jawaban assistant

## Struktur

```
ui/chatbot/
├── package.json
├── vite.config.js              (port 5174, beda dari project utama)
├── tailwind.config.js
├── postcss.config.js
├── index.html
└── src/
    ├── main.jsx                Entry React
    ├── App.jsx                 Root component (orchestrator)
    ├── index.css               Tailwind + custom utilities
    ├── components/
    │   ├── Header.jsx          Logo + tombol "Percakapan baru"
    │   ├── WelcomeState.jsx    State awal: heading + 6 sample cards
    │   ├── SampleCard.jsx      Kartu pertanyaan sampel (clickable)
    │   ├── ChatMessages.jsx    Container daftar pesan + auto-scroll
    │   ├── MessageBubble.jsx   Bubble user/assistant + markdown
    │   ├── TypingIndicator.jsx Three-dot animasi loading
    │   └── ChatInput.jsx       Textarea + tombol kirim
    ├── hooks/
    │   └── useChat.js          State management (messages, loading, errors)
    ├── services/
    │   ├── chatService.js      Mock chat service (fuzzy match)
    │   └── systemPrompt.js     System prompt natural (untuk backend nanti)
    └── data/
        └── sampleQuestions.js  6 sample Q&A (PubMedQA + jawaban natural)
```

## Cara Menjalankan

```bash
cd ui/chatbot
npm install
npm run dev
```

Buka [http://localhost:5174](http://localhost:5174).

## Mode Mock (Saat Ini)

Backend RAG belum tersedia. Saat ini chatbot beroperasi dalam **mock mode**:

1. **Sample questions** — 6 pertanyaan dengan jawaban natural yang sudah di-pre-craft (dari abstract PubMedQA, ditulis ulang dengan tone ramah). Klik kartu = pertanyaan otomatis dikirim ke chat.

2. **Free-typed questions** — fuzzy-match terhadap 6 sample question via token overlap. Kalau match (skor ≥ 30%), tampilkan jawaban sample tersebut. Kalau tidak match, tampilkan fallback ramah.

3. **Simulasi network delay** — 1.5-2.5 detik untuk meniru waktu inferensi LLM nyata.

## Integrasi Backend (Nanti)

Saat backend RAG aktif, file `src/services/chatService.js` diganti dengan call ke endpoint:

```javascript
export async function askChat(question) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  return res.json()  // { answer, contexts?, source? }
}
```

System prompt yang digunakan didokumentasikan di `src/services/systemPrompt.js`.
Prompt ini diadaptasi dari "OPSI A: Evidence-First Citation Prompt" yang ada di
`notebooks/BM25 Expansion/12_prompt_testing_sh.ipynb` (Cell 7) — dengan penyesuaian:
- Drop yes/no/maybe label
- Tone lebih hangat (ramah, bukan asisten kaku)
- Tambah instruksi simplifikasi istilah medis

## Sumber Data

6 sample questions dipilih dari **PubMedQA `pqa_labeled` subset** dengan kriteria:
- Mudah dipahami non-medis
- Berkaitan dengan kesehatan sehari-hari
- Bervariasi topik: Nutrisi, Olahraga, Tidur, Pengobatan, Kesehatan Mental, Diabetes

PubMed ID disertakan di setiap jawaban assistant (link ke pubmed.ncbi.nlm.nih.gov).

## Design Principles

- **Clean & minimalist** — putih dominan, slate accent, satu warna utama (slate-900)
- **Readability first** — line-height nyaman, kontras cukup, font Inter
- **Responsive** — 1 col mobile (sm), 2 col tablet/desktop untuk sample grid
- **Auto-scroll** — pesan baru otomatis scroll ke bawah
- **Accessibility** — aria-label di tombol, focus styling, keyboard navigation (Enter / Shift+Enter)
