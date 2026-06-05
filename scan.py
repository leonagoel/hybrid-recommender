import os
import re

PATTERNS = {
    'eval/exec': r'\b(eval|exec)\s*\(',
    'subprocess': r'\b(subprocess\.|os\.system|popen)',
    'sql_injection': r'(execute|query)\s*\([^,]*%[^,]*\)|(execute|query)\s*\([^,]*\+[^,]*\)|(execute|query)\s*\([^\)]*f".*\{.*\}',
    'pickle': r'\bpickle\.loads?\(',
    'yaml': r'\byaml\.load\(',
    'path_traversal': r'open\s*\(\s*f["\'].*\{',
    'open_redirect': r'(RedirectResponse|redirect)\s*\(\s*[a-zA-Z_]',
    'ssrf': r'requests\.(get|post|put|delete|request)\s*\(\s*[a-zA-Z_]',
    'csrf': r'csrf',
    'jwt_secret': r'jwt\.encode',
    'cors': r'CORSMiddleware',
    'xxe': r'xml\.etree'
}

def scan_dir(d):
    for root, _, files in os.walk(d):
        for f in files:
            if not f.endswith('.py'):
                continue
            path = os.path.join(root, f)
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.readlines()
                    for i, line in enumerate(content):
                        for name, pat in PATTERNS.items():
                            if re.search(pat, line, re.IGNORECASE):
                                print(f"{path}:{i+1}:{name} -> {line.strip()}")
            except Exception as e:
                pass

if __name__ == '__main__':
    scan_dir('C:\\Users\\user\\.gemini\\antigravity\\scratch\\hybrid-recommender')
