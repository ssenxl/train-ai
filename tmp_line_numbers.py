from pathlib import Path
path = Path('stepD_postprocess_SI0_merge.py')
with path.open(encoding='utf-8') as f:
    for idx, line in enumerate(f,1):
        if 'SIDE_MODEL_PATH' in line or 'predict_side' in line or 'arrow_fronts' in line:
            print(idx)
