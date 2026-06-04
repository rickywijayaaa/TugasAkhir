"""Scan all .tex files in current dir, find cited keys vs defined keys in .bib."""
import re, os, sys

cited = set()
cite_pat = re.compile(r"\\(?:cite|parencite|textcite|autocite|citeauthor|citeyear)\*?(?:\[[^\]]*\])?\{([^}]+)\}")

for f in sorted(os.listdir(".")):
    if not f.endswith(".tex"): continue
    txt = open(f, encoding="utf-8", errors="ignore").read()
    for m in cite_pat.findall(txt):
        for k in m.split(","):
            cited.add(k.strip())

defined = set()
with open("daftar-pustaka.bib", encoding="utf-8") as fh:
    for m in re.finditer(r"^@[a-zA-Z]+\{([^,]+),", fh.read(), re.M):
        defined.add(m.group(1).strip())

missing = sorted(cited - defined)
unused  = sorted(defined - cited)
both    = sorted(cited & defined)

print(f"Cited keys total : {len(cited)}")
print(f"Defined keys total: {len(defined)}")
print(f"\nMISSING from .bib ({len(missing)}):")
for k in missing: print(f"  - {k}")
print(f"\nDefined but unused ({len(unused)}):")
for k in unused: print(f"  - {k}")
print(f"\nDefined AND cited ({len(both)}):")
for k in both: print(f"  - {k}")
