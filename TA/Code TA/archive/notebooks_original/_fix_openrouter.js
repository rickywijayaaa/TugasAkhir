const fs = require('fs');
const nb = JSON.parse(fs.readFileSync('02.1-RAG-Baseline-Groq.ipynb', 'utf8'));

function getSrc(cell) { return cell.source.join(''); }
function setSrc(cell, src) {
  cell.source = src.split('\n').map((l, i, arr) => i < arr.length - 1 ? l + '\n' : l);
  if (cell.source[cell.source.length - 1] === '') cell.source.pop();
}

for (let i = 0; i < nb.cells.length; i++) {
  const cell = nb.cells[i];
  let s = getSrc(cell);
  const orig = s;

  // ---- TITLE (markdown cell 0) ----
  if (cell.cell_type === 'markdown' && s.includes('Cerebras API')) {
    s = s.replace(/Cerebras API/g, 'OpenRouter API');
    s = s.replace(/Cerebras/g, 'OpenRouter');
    s = s.replace('llama-3.3-70b', 'llama-3.3-70b-instruct');
  }

  // ---- INSTALL ----
  if (s.includes('cerebras-cloud-sdk')) {
    s = s.replace('cerebras-cloud-sdk', 'openai');
  }

  // ---- IMPORTS ----
  if (s.includes('from cerebras.cloud.sdk import Cerebras')) {
    s = s.replace(
      'from cerebras.cloud.sdk import Cerebras',
      'from openai import OpenAI'
    );
  }

  // ---- CONFIG ----
  if (s.includes('CEREBRAS_API_KEY') && s.includes('LLM_MODEL')) {
    // Replace env var name
    s = s.replace(/CEREBRAS_API_KEY/g, 'OPENROUTER_API_KEY');
    s = s.replace("os.environ.get('OPENROUTER_API_KEY', 'YOUR_API_KEY_HERE')",
                  "os.environ.get('OPENROUTER_API_KEY', 'YOUR_API_KEY_HERE')");
    // Model name
    s = s.replace("'llama-3.3-70b'", "'meta-llama/llama-3.3-70b-instruct'");
    s = s.replace('# Cerebras: sama dengan Groq llama-3.3-70b-versatile',
                  '# OpenRouter: sama model Llama 3.3 70B');
    // Print text
    s = s.replace(/via Cerebras API/g, 'via OpenRouter API');
    s = s.replace("'YOUR_API_KEY' in CEREBRAS_API_KEY",
                  "'YOUR_API_KEY' in OPENROUTER_API_KEY");
  }
  // Catch any remaining CEREBRAS refs in config
  if (s.includes('CEREBRAS_API_KEY')) {
    s = s.replace(/CEREBRAS_API_KEY/g, 'OPENROUTER_API_KEY');
  }

  // ---- CLIENT SETUP ----
  if (s.includes('cerebras_client = Cerebras(')) {
    // Rewrite the whole cell
    s = `# ============================================================
# Setup OpenRouter Client + Utility
# ============================================================

openrouter_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url='https://openrouter.ai/api/v1',
)

# Jeda antar request (20 RPM pada free tier = 1 req per 3 detik)
LLM_DELAY = 3.0  # detik antar request
_last_call_time = 0


def llm_generate(prompt: str, max_tokens: int = 300, temperature: float = TEMPERATURE) -> str:
    """
    Wrapper OpenRouter API dengan:
    1. Jeda otomatis antar request (LLM_DELAY) agar tidak kena rate limit
    2. Retry otomatis jika tetap kena 429
    """
    global _last_call_time

    # Jeda otomatis
    elapsed = time.time() - _last_call_time
    if elapsed < LLM_DELAY:
        time.sleep(LLM_DELAY - elapsed)

    for attempt in range(5):
        try:
            _last_call_time = time.time()
            response = openrouter_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                wait = (attempt + 1) * 15  # 15s, 30s, 45s, 60s, 75s
                print(f'  [Rate limit] Tunggu {wait}s... (attempt {attempt+1}/5)')
                time.sleep(wait)
            else:
                print(f'  [OpenRouter Error] {type(e).__name__}: {err[:80]}')
                raise
    raise RuntimeError('OpenRouter API gagal setelah 5 percobaan.')


# Smoke test
print('Testing OpenRouter API...')
_test = llm_generate('Reply with exactly: OK', max_tokens=5)
print(f'Response: {_test!r}')
print(f'Delay antar request: {LLM_DELAY}s')
print('OpenRouter client siap!')`;
  }

  // ---- All other cells: replace remaining cerebras refs ----
  s = s.replace(/cerebras_client/g, 'openrouter_client');
  s = s.replace(/Cerebras API/g, 'OpenRouter API');
  s = s.replace(/Cerebras client/g, 'OpenRouter client');
  s = s.replace(/Cerebras Error/g, 'OpenRouter Error');
  s = s.replace(/Cerebras API gagal/g, 'OpenRouter API gagal');
  s = s.replace(/Cerebras Baseline/g, 'OpenRouter Baseline');
  s = s.replace(/\(Cerebras\)/g, '(OpenRouter)');
  s = s.replace(/via Cerebras/g, 'via OpenRouter');
  // Markdown cells
  if (cell.cell_type === 'markdown') {
    s = s.replace(/Cerebras/g, 'OpenRouter');
  }

  if (s !== orig) {
    setSrc(cell, s);
    cell.outputs = [];
    console.log(`[${i}] Updated`);
  }
}

fs.writeFileSync('02.1-RAG-Baseline-Groq.ipynb', JSON.stringify(nb, null, 1));
console.log('\n=== Saved ===');
