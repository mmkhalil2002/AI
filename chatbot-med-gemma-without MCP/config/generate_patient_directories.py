#!/usr/bin/env python3
"""Create synthetic patient folders containing ONLY medical files.

Input: data/xray-data-MMDDYY, data/ekg-MMDDYY, data/bloodtest-data-MMDDYY.
Output: patient_records/P0001/{xrays,ekgs,blood_tests}/...

Blood-test .json files are medical data and are preserved. No index.json or
record.json files are created. Assignments are random, NOT real patient links.
"""
import random
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'patient_records'
FOLDER_RULES = {
    'xrays': (re.compile(r'^xray-data-\d{6}$', re.I), {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}),
    'ekgs': (re.compile(r'^ekg-\d{6}$', re.I), {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}),
    'blood_tests': (re.compile(r'^bloodtest-data-\d{6}$', re.I), {'.json'}),
}


def discover_files():
    """Collect eligible source files from the date-stamped input folders."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f'Missing data directory: {DATA_DIR}')
    found = {kind: [] for kind in FOLDER_RULES}
    for folder in sorted(DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue
        for kind, (pattern, extensions) in FOLDER_RULES.items():
            if pattern.fullmatch(folder.name):
                found[kind].extend(
                    file for file in sorted(folder.iterdir())
                    if file.is_file() and file.suffix.lower() in extensions
                )
    return found


def generate_patient_folders(count, pools):
    """Randomly copy up to three files of each type into each synthetic patient."""
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f'{OUTPUT_DIR} is not empty. Rename or move it before running again; '
            'existing patient files will not be overwritten.'
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(1, count + 1):
        patient_dir = OUTPUT_DIR / f'P{i:04d}'
        counts = {}
        for kind, files in pools.items():
            destination_dir = patient_dir / kind
            destination_dir.mkdir(parents=True, exist_ok=True)
            chosen = random.sample(files, k=random.randint(1, min(3, len(files)))) if files else []
            for src in chosen:
                # Include the source folder name to prevent collisions across dates.
                shutil.copy2(src, destination_dir / f'{src.parent.name}__{src.name}')
            counts[kind] = len(chosen)
        print(f'{patient_dir.name}: ' + ', '.join(f'{k}={v}' for k, v in counts.items()))
    return OUTPUT_DIR


def main():
    print('Synthetic Patient Directory Generator (no metadata JSON)')
    pools = discover_files()
    print('Available:', ', '.join(f'{k}={len(v)}' for k, v in pools.items()))
    if not any(pools.values()):
        print('No supported medical files found.')
        return
    while True:
        try:
            count = int(input('Enter number of patients to generate: ').strip())
            if 1 <= count <= 10000:
                break
        except ValueError:
            pass
        print('Enter a whole number between 1 and 10000.')
    location = generate_patient_folders(count, pools)
    print(f'Created {count} synthetic patient directories in {location}')
    print('No index.json or record.json created; blood-test JSON files are retained.')
    print('Synthetic assignments are for testing only, not real patient identity.')


if __name__ == '__main__':
    main()
