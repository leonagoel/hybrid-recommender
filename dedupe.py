import re

with open('backend/main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
seen_imports = set()
for line in lines:
    if line.startswith('from ') or line.startswith('import '):
        # normalize
        norm = re.sub(r'\s*#.*$', '', line).strip()
        if norm in seen_imports:
            continue
        seen_imports.add(norm)
    new_lines.append(line)

with open('backend/main.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
