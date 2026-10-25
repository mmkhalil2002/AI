#!/usr/bin/env python3
"""Create synthetic patient folders containing ONLY medical files.

Input: data/xray-data-MMDDYY, data/ekg-MMDDYY, data/bloodtest-data-MMDDYY.
Output: <BASE_OUTPUT_DIR>/patient_records/P0001/{xrays,ekgs,blood_tests}/...

Blood-test .json files are medical data and are preserved. No index.json or
record.json files are created. Assignments are random, NOT real patient links.
"""
import argparse
import os
import random
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / '.env'


def read_settings():
    """Read generator paths from med_data/.env; process environment wins."""
    settings = {}
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            settings[key] = value
    return settings


SETTINGS = read_settings()


def resolve_directory(name, default):
    value = os.environ.get(name, SETTINGS.get(name, default))
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


BASE_INPUT_DIR = resolve_directory('BASE_INPUT_DIR', '.')
BASE_OUTPUT_DIR = resolve_directory('BASE_OUTPUT_DIR', '.')
DATA_DIR = BASE_INPUT_DIR / 'data'
OUTPUT_DIR = BASE_OUTPUT_DIR / 'patient_records'
FOLDER_RULES = {
    'xrays': (re.compile(r'^xray-data-\d{6}$', re.I), {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}),
    'ekgs': (re.compile(r'^ekg-(?:data-)?\d{6}$', re.I), {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}),
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


def validate_output_path(output_dir, pools):
    """Reject output locations that overlap input data or contain existing files."""
    output = output_dir.expanduser().resolve()
    input_roots = [DATA_DIR.resolve()]
    input_roots.extend(path.resolve() for files in pools.values() for path in files)
    for source in input_roots:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f'Output path overlaps source data: {output} and {source}')
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f'Output path exists and is not a directory: {output}')
    if output.is_dir() and any(output.iterdir()):
        raise FileExistsError(
            f'{output} is not empty. Rename or move it before running again; '
            'existing files will not be overwritten.'
        )
    return output


def plan_assignments(count, pools, seed=None):
    """Return deterministic, random demo assignments without touching the filesystem."""
    rng = random.Random(seed)
    assignments = []
    for i in range(1, count + 1):
        patient = f'P{i:04d}'
        selected = {}
        for kind, files in pools.items():
            inventory = sorted(files, key=lambda path: str(path).casefold())
            amount = rng.randint(1, min(3, len(inventory))) if inventory else 0
            selected[kind] = rng.sample(inventory, amount) if amount else []
        assignments.append((patient, selected))
    return assignments


def destination_name(src, used):
    """Make a unique destination basename while retaining the source name."""
    base = f'{src.parent.name}__{src.name}'
    candidate = base
    stem, suffix = Path(base).stem, Path(base).suffix
    index = 2
    while candidate.casefold() in used:
        candidate = f'{stem}__{index}{suffix}'
        index += 1
    used.add(candidate.casefold())
    return candidate


def generate_patient_folders(count, pools, output_dir=None, seed=None, dry_run=False):
    """Plan or copy randomly grouped demo records, optionally with a reproducible seed."""
    output = validate_output_path(Path(output_dir) if output_dir is not None else OUTPUT_DIR, pools)
    assignments = plan_assignments(count, pools, seed)
    print('DEMO DATA: random file grouping does not represent real patient identity or history.')
    if seed is not None:
        print(f'Random seed: {seed}')
    for patient, selected in assignments:
        counts = {kind: len(files) for kind, files in selected.items()}
        print(f'{patient}: ' + ', '.join(f'{kind}={n}' for kind, n in counts.items()))
        for kind, files in selected.items():
            used = set()
            for src in files:
                name = destination_name(src, used)
                print(f'  {kind}/{name} <- {src}')
    if dry_run:
        print(f'Dry run: no files or directories were written. Destination: {output}')
        return output

    for patient, selected in assignments:
        patient_dir = output / patient
        for kind, files in selected.items():
            destination_dir = patient_dir / kind
            destination_dir.mkdir(parents=True, exist_ok=True)
            used = set()
            for src in files:
                shutil.copy2(src, destination_dir / destination_name(src, used))
    return output


def parse_args(argv=None):
    """Parse optional CLI controls; omitting --count preserves interactive use."""
    parser = argparse.ArgumentParser(
        description='Create demo patient folders using random medical-file grouping.'
    )
    parser.add_argument('--dry-run', action='store_true', help='show planned assignments without writing files')
    parser.add_argument('--count', type=int, help='number of demo folders (1-10000); prompt if omitted')
    parser.add_argument('--seed', type=int, help='seed for repeatable file assignments')
    parser.add_argument('--output', type=Path, help='explicit output directory (defaults to med_data/.env setting)')
    args = parser.parse_args(argv)
    if args.count is not None and not 1 <= args.count <= 10000:
        parser.error('--count must be between 1 and 10000')
    return args


def main(argv=None):
    args = parse_args(argv)
    print('Synthetic Patient Directory Generator (DEMO DATA; no metadata JSON)')
    pools = discover_files()
    print('Available:', ', '.join(f'{k}={len(v)}' for k, v in pools.items()))
    if not any(pools.values()):
        print('No supported medical files found.')
        return
    count = args.count
    while count is None:
        try:
            entered = int(input('Enter number of patients to generate: ').strip())
            if 1 <= entered <= 10000:
                count = entered
            else:
                print('Enter a whole number between 1 and 10000.')
        except ValueError:
            print('Enter a whole number between 1 and 10000.')
    location = generate_patient_folders(count, pools, args.output, args.seed, args.dry_run)
    if args.dry_run:
        print(f'Planned {count} synthetic demo directories in {location}')
    else:
        print(f'Created {count} synthetic demo patient directories in {location}')
    print('No index.json or record.json created; blood-test JSON files are retained.')
    print('Synthetic assignments are for testing only, not real patient identity.')


if __name__ == '__main__':
    main()
