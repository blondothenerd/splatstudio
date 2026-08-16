#!/usr/bin/env python3
from pathlib import Path
import re, sys
root = Path(__file__).resolve().parents[1]
ignore_parts = {".git", "runtime", "backend", "models", "projects", ".splat_studio"}
patterns = [
    re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    re.compile(r"/home/[A-Za-z0-9._-]+/"),
    re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\\]+\\\\"),
]
failures=[]
for path in root.rglob('*'):
    if not path.is_file() or any(part in ignore_parts for part in path.parts): continue
    if path.suffix.lower() not in {'.py','.md','.txt','.toml','.yml','.yaml','.command','.json'} and path.name not in {'.gitignore'}: continue
    try: text=path.read_text(encoding='utf-8')
    except Exception: continue
    for pat in patterns:
        for match in pat.finditer(text): failures.append((path.relative_to(root), match.group(0)))
if failures:
    print('Machine-specific absolute paths found:')
    for path, value in failures: print(f'  {path}: {value}')
    sys.exit(1)
print('Portability check passed: no user-home absolute paths found.')
