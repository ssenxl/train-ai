"""Orchestrate the relevant steps from STEP 1 to the final pipeline.

This script simply invokes the existing sequence of scripts that are actually required
for production. It is meant to be executed from the repository root.

Usage:
    python run_full_pipeline.py
"""

import subprocess
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "step1_extract_keys.py",
    "step2_find_excels.py",
    "step3_extract_images.py",
    "step3b_extract_codes.py",
    "step3c_merge_codes_all.py",
    "step4a_merge_codes.py",
    "stepD_final_pipeline.py",
    "stepD_postprocess_SI0_merge.py",
]

MISSING_LOG = ROOT / "missing_files.json"

def log_missing(script: str, reason: str) -> None:
    data = []
    if MISSING_LOG.exists():
        try:
            with MISSING_LOG.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    entry = {"script": script, "reason": reason}
    data.append(entry)
    with MISSING_LOG.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_step(script: str) -> None:
    path = ROOT / script
    if not path.exists():
        log_missing(script, f"Script not found: {path}")
        raise FileNotFoundError(f"Script not found: {path}")
    print("\n" + "=" * 60)
    print(f"RUNNING {script}")
    print("=" * 60 + "\n")
    subprocess.run([sys.executable, str(path)], check=True)


def main() -> None:
    for script in SCRIPTS:
        run_step(script)


if __name__ == "__main__":
    main()
