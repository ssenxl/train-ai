from pathlib import Path
lines = Path('step3b_extract_codes.py').read_text().splitlines()
target_terms = ['BOX_LABELS', 'กล่อง / ถาด']
for idx, line in enumerate(lines, 1):
    if any(term in line for term in target_terms) or '4.2 กล่อง' in line or '4.1 ถาด' in line:
        print(idx, line.strip())
