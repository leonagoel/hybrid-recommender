import ast

def deduplicate_ast(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        source = f.read()
    
    tree = ast.parse(source)
    seen_names = set()
    to_remove = []
    
    # We only care about top-level functions and classes
    # But wait, we want to KEEP the FIRST definition and remove the LATER ones?
    # Actually, Python keeps the LATER one. But usually the first one is the "original" and the later one is the duplicate.
    # Wait, in the flake8 error: `backend/main.py:1945:1: F811 redefinition of unused 'build_models' from line 1766`
    # Flake8 points to the LATER one as the redefinition. So we should remove the LATER ones.
    
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in seen_names:
                to_remove.append((node.lineno, node.end_lineno))
            else:
                seen_names.add(node.name)
    
    # We also have `defaultdict` redefinition which is an import:
    # backend/main.py:2871:5: F811 redefinition of unused 'defaultdict' from line 39
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Remove lines from bottom to top so indices don't shift
    for start, end in sorted(to_remove, reverse=True):
        del lines[start-1:end]
        
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(lines)

if __name__ == '__main__':
    deduplicate_ast('backend/main.py')
