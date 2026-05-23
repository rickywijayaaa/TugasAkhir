"""Mask all hard-coded API keys across .py and .ipynb files in this repo.

Replaces the actual key string with a placeholder and ensures the runtime path
reads from os.environ instead. The fix is conservative: it only replaces the
literal key string inside common assignment patterns. Existing code that already
uses os.environ.get(...) is left alone.

Run from repo root:
    "C:/Users/Ricky Wijaya/AppData/Local/Programs/Python/Python311/python.exe" _mask_api_keys.py
"""
import os, re, json, sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
EXCLUDE_DIRS = {".git", "venv", "__pycache__", "node_modules"}
EXCLUDE_DIR_PARTS = {"chroma"}  # skip any folder containing 'chroma'

# Pattern that matches actual API key VALUES
KEY_PATTERNS = [
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{30,}"),
    re.compile(r"sk-or-v[0-9]-[a-f0-9]{40,}"),
    re.compile(r"sk-ant-api[0-9]{2}-[A-Za-z0-9_\-]{30,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{30,}"),
]

# When a key is found, the literal "...key string..." is replaced with the
# placeholder below. Surrounding code already reads from os.environ in most
# notebooks; this just kills the hard-coded value.
PLACEHOLDER = "REDACTED_SET_VIA_ENV"

# Also handle the common `# os.environ["OPENAI_API_KEY"] = "<REDACTED — set via shell env or .env file>"` pattern —
# we rewrite the entire line to just a comment so the env var must come from
# elsewhere (a .env file, shell export, etc).
ENV_ASSIGN_PATTERNS = [
    # python source
    re.compile(r'os\.environ\[\s*"(OPENAI_API_KEY|ANTHROPIC_API_KEY|OPENROUTER_API_KEY)"\s*\]\s*=\s*"sk-[^"]+"'),
    re.compile(r"os\.environ\[\s*'(OPENAI_API_KEY|ANTHROPIC_API_KEY|OPENROUTER_API_KEY)'\s*\]\s*=\s*'sk-[^']+'"),
]

def make_env_replacement(match):
    var = match.group(1)
    return f'# os.environ["{var}"] = "<REDACTED — set via shell env or .env file>"'

def sanitize_text(text):
    changed = False
    # First: replace env-var assignments with safe comment
    for pat in ENV_ASSIGN_PATTERNS:
        new_text, n = pat.subn(make_env_replacement, text)
        if n > 0:
            changed = True
            text = new_text
    # Then: replace any remaining bare key strings
    for pat in KEY_PATTERNS:
        new_text, n = pat.subn(PLACEHOLDER, text)
        if n > 0:
            changed = True
            text = new_text
    return text, changed

def sanitize_py(path):
    src = path.read_text(encoding="utf-8", errors="replace")
    new, changed = sanitize_text(src)
    if changed:
        path.write_text(new, encoding="utf-8")
    return changed

def sanitize_ipynb(path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        nb = json.loads(raw)
    except Exception as e:
        print(f"  SKIP (not valid JSON): {path.relative_to(ROOT)} — {e}")
        return False
    any_changed = False
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        if isinstance(src, list):
            joined = "".join(src)
            new, changed = sanitize_text(joined)
            if changed:
                any_changed = True
                cell["source"] = new.splitlines(keepends=True)
        elif isinstance(src, str):
            new, changed = sanitize_text(src)
            if changed:
                any_changed = True
                cell["source"] = new
        # Also wipe any outputs that contain keys (paranoid: API keys can leak via cell outputs)
        for out in cell.get("outputs", []):
            for k, v in list(out.items()):
                if isinstance(v, str):
                    new, changed = sanitize_text(v)
                    if changed:
                        any_changed = True
                        out[k] = new
                elif isinstance(v, list):
                    new_list = []
                    for item in v:
                        if isinstance(item, str):
                            new_item, ch = sanitize_text(item)
                            if ch: any_changed = True
                            new_list.append(new_item)
                        else:
                            new_list.append(item)
                    out[k] = new_list
                elif isinstance(v, dict):
                    for kk, vv in list(v.items()):
                        if isinstance(vv, list):
                            new_list = []
                            for item in vv:
                                if isinstance(item, str):
                                    new_item, ch = sanitize_text(item)
                                    if ch: any_changed = True
                                    new_list.append(new_item)
                                else:
                                    new_list.append(item)
                            v[kk] = new_list
                        elif isinstance(vv, str):
                            new, ch = sanitize_text(vv)
                            if ch: any_changed = True
                            v[kk] = new
    if any_changed:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return any_changed


def walk_and_sanitize():
    changed_files = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if d not in EXCLUDE_DIRS
                   and not any(part in d.lower() for part in EXCLUDE_DIR_PARTS)]
        for f in files:
            path = Path(root) / f
            if path.suffix == ".py":
                if sanitize_py(path):
                    changed_files.append(path)
            elif path.suffix == ".ipynb":
                if sanitize_ipynb(path):
                    changed_files.append(path)
    return changed_files


if __name__ == "__main__":
    print(f"Sanitizing under: {ROOT}\n")
    changed = walk_and_sanitize()
    print(f"\nSanitized {len(changed)} files:")
    for p in changed:
        print(f"  - {p.relative_to(ROOT)}")
