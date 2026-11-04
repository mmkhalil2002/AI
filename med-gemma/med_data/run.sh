#!/usr/bin/env bash
# Starts the importer from this directory on Ubuntu and other Linux systems.
set -u
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$script_dir" || exit 1

if command -v python3 >/dev/null 2>&1; then
    python3 generate_patient_data.py
elif command -v python >/dev/null 2>&1; then
    python generate_patient_data.py
else
    echo "[ERROR] Python 3 was not found. Install Python 3 and try again." >&2
    exit 1
fi
