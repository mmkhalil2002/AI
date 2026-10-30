#!/usr/bin/env python3
"""Interactive MedGemma patient-date data importer (Windows / Ubuntu).

Uses BASE_INPUT_DIR and BASE_OUTPUT_DIR from .env beside this script.
Creates BASE_OUTPUT_DIR/patient_records/PATIENT_ID/YYYY-MM-DD/HH-MM_AM/{xrays,ekgs,blood_tests}.
Only standard-library modules are required. Educational images come from
Wikimedia Commons; blood test records are synthetic educational examples.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://commons.wikimedia.org/w/api.php"
AGENT = "MedGemma-Educational-Patient-Importer/1.0 (training and demonstration)"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
SCRIPT_DIR = Path(__file__).resolve().parent
# The script is normally placed in med_data/ and the GUI .env in python/.
ENV_CANDIDATES = [SCRIPT_DIR / ".env", SCRIPT_DIR.parent / "python" / ".env"]
MAX_RECORDS = 100
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024

BLOOD_EXAMPLES = [
    {"title": "Iron-Deficiency Anemia Pattern", "results": [
        {"test": "Hemoglobin", "value": 8.7, "unit": "g/dL", "reference_range": "12.0-16.0", "flag": "LOW"},
        {"test": "MCV", "value": 68, "unit": "fL", "reference_range": "80-100", "flag": "LOW"}]},
    {"title": "Bacterial Infection and Inflammation Pattern", "results": [
        {"test": "WBC", "value": 17.8, "unit": "x10^3/uL", "reference_range": "4.0-11.0", "flag": "HIGH"},
        {"test": "CRP", "value": 112, "unit": "mg/L", "reference_range": "<10", "flag": "HIGH"}]},
    {"title": "Poorly Controlled Diabetes Pattern", "results": [
        {"test": "Fasting glucose", "value": 238, "unit": "mg/dL", "reference_range": "70-99", "flag": "HIGH"},
        {"test": "HbA1c", "value": 9.4, "unit": "%", "reference_range": "<5.7", "flag": "HIGH"}]},
    {"title": "Kidney Impairment Pattern", "results": [
        {"test": "Creatinine", "value": 2.4, "unit": "mg/dL", "reference_range": "0.6-1.3", "flag": "HIGH"},
        {"test": "eGFR", "value": 29, "unit": "mL/min/1.73m2", "reference_range": ">=60", "flag": "LOW"}]},
    {"title": "Liver Injury Pattern", "results": [
        {"test": "ALT", "value": 286, "unit": "U/L", "reference_range": "7-56", "flag": "HIGH"},
        {"test": "AST", "value": 192, "unit": "U/L", "reference_range": "10-40", "flag": "HIGH"}]},
]


def read_env(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE lines without installing python-dotenv."""
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def configured_directory(key: str) -> Path:
    """Resolve a path from .env relative to the directory containing .env.

    On both Windows and Ubuntu, forward and backward slashes are accepted in
    relative paths, such as BASE_OUTPUT_DIR=../python.
    """
    env_file = next((p for p in ENV_CANDIDATES if p.is_file()), None)
    if env_file is None:
        raise ValueError("Missing .env. Put it beside this script in med_data/.")
    values = read_env(env_file)
    raw = os.environ.get(key) or values.get(key)
    if not raw or not raw.strip():
        raise ValueError(f"{key} is missing from {env_file}")
    # Normalize separators for portability. Avoid changing drive-qualified
    # Windows paths when actually running on Windows.
    raw = os.path.expandvars(raw.strip())
    if os.name != "nt":
        raw = raw.replace("\\", "/")
    directory = Path(raw).expanduser()
    if not directory.is_absolute():
        directory = env_file.parent / directory
    return directory.resolve()


def output_root() -> Path:
    """Read BASE_OUTPUT_DIR (the configured location for patient records)."""
    directory = configured_directory("BASE_OUTPUT_DIR")
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"BASE_OUTPUT_DIR is not a directory: {directory}")
    print(f"[INFO] BASE_OUTPUT_DIR: {directory}")
    return directory


def prompt_patient_id() -> tuple[str, str, str]:
    """Collect patient identity fields and build ``firstname-lastname-MMDDYYYY``."""
    while True:
        first_name = input("First name: ").strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z '-]{0,39}", first_name) and first_name[-1] not in {" ", "-", "'"}:
            break
        print("[ERROR] First name must be 1-40 letters and may include spaces, hyphens, or apostrophes.")

    while True:
        last_name = input("Last name: ").strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z '-]{0,39}", last_name) and last_name[-1] not in {" ", "-", "'"}:
            break
        print("[ERROR] Last name must be 1-40 letters and may include spaces, hyphens, or apostrophes.")

    while True:
        dob = input("Date of birth (MMDDYYYY): ").strip()
        try:
            # Parsing also rejects invalid values such as month 13 or day 32.
            if not re.fullmatch(r"\d{8}", dob):
                raise ValueError
            datetime.strptime(dob, "%m%d%Y")
            return f"{first_name}-{last_name}-{dob}", f"{first_name} {last_name}", dob
        except ValueError:
            print("[ERROR] Enter a valid eight-digit date of birth in MMDDYYYY format, for example 04251986.")


def prompt_date() -> str:
    """Normalize input to ISO YYYY-MM-DD; accept MM/DD/YYYY too."""
    while True:
        raw = input("Examination date (YYYY-MM-DD or MM/DD/YYYY): ").strip()
        for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass
        print("[ERROR] Invalid date. Example: 2026-10-25")



def prompt_exam_time() -> str:
    """Ask for 12-hour examination time; return a Windows-safe folder name.

    Example: 05:20 PM becomes 05-20_PM. Different times on the same
    examination date are stored in separate directories beneath that date.
    """
    while True:
        raw = input("Examination time (HH:MM AM/PM, e.g. 05:20 PM): ").strip()
        if not re.fullmatch(r"(?:0[1-9]|1[0-2]):[0-5][0-9] (?:AM|PM)", raw, re.IGNORECASE):
            print("[ERROR] Use HH:MM AM or PM, for example 05:20 PM.")
            continue
        parsed = datetime.strptime(raw.upper(), "%I:%M %p")
        return parsed.strftime("%I-%M_%p")


def prompt_count(label: str) -> int:
    while True:
        try:
            value = int(input(f"How many {label} records (0-{MAX_RECORDS})? ").strip())
            if 0 <= value <= MAX_RECORDS:
                return value
        except ValueError:
            pass
        print(f"[ERROR] Enter a whole number between 0 and {MAX_RECORDS}.")


def file_digest(path: Path) -> str:
    """Hash a file in chunks so large images do not fill memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def known_digests(folder: Path, extensions: set[str]) -> set[str]:
    """Hashes of existing records in this patient's date/modality folder."""
    hashes = set()
    if folder.is_dir():
        for item in folder.iterdir():
            if item.is_file() and item.suffix.lower() in extensions:
                hashes.add(file_digest(item))
    return hashes


def save_unique_bytes(folder: Path, name: str, data: bytes,
                      hashes: set[str]) -> Path | None:
    """Save a new record; return None when its contents already exist."""
    digest = hashlib.sha256(data).hexdigest()
    if digest in hashes:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    destination = unique_path(folder, name)
    destination.write_bytes(data)
    hashes.add(digest)
    return destination


def unique_path(directory: Path, name: str) -> Path:
    """Avoid overwriting existing patient data."""
    target = directory / name
    if not target.exists():
        return target
    for i in range(2, 10000):
        candidate = directory / f"{target.stem}-{i}{target.suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Too many duplicates of {name}")


def import_external(source: Path, destination: Path) -> dict[str, int]:
    """Copy files from source/{xray,ekg,blood test} without deleting originals."""
    folders = {
        "xrays": ("xrays", "xray", "x-ray"),
        "ekgs": ("ekgs", "ekg", "ecg"),
        "blood_tests": ("blood_tests", "blood test", "blood tests", "blood_test", "bloodtest"),
    }
    totals = {}
    if not source.is_dir():
        raise FileNotFoundError(f"External source directory not found: {source}")
    for target_name, aliases in folders.items():
        target_dir = destination / target_name
        matches = [p for p in source.iterdir() if p.is_dir() and p.name.lower() in aliases]
        count = 0
        accepted = {".json"} if target_name == "blood_tests" else EXTENSIONS
        hashes = known_digests(target_dir, accepted)
        if not matches:
            print(f"[WARN] Missing {target_name} folder in {source}")
        for folder in matches:
            for item in sorted(folder.iterdir()):
                if not item.is_file() or item.is_symlink():
                    continue
                if item.suffix.lower() not in ({".json"} if target_name == "blood_tests" else EXTENSIONS):
                    continue
                if target_name == "blood_tests":
                    try:
                        json.loads(item.read_text(encoding="utf-8-sig"))
                    except (OSError, ValueError) as error:
                        print(f"[WARN] Skipping invalid JSON {item.name}: {error}")
                        continue
                if file_digest(item) in hashes:
                    print(f"[SKIP] Identical {target_name} record already exists: {item.name}")
                    continue
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = unique_path(target_dir, item.name)
                shutil.copy2(item, target_path)
                hashes.add(file_digest(target_path))
                count += 1
        totals[target_name] = count
        print(f"[INFO] Copied {count} {target_name} file(s)")
    return totals


def commons_query(parameters: dict) -> dict:
    url = API + "?" + urlencode({"format": "json", "formatversion": "2", **parameters})
    request = Request(url, headers={"User-Agent": AGENT})
    with urlopen(request, timeout=40) as response:
        return json.load(response)


def download_image(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": AGENT})
    with urlopen(request, timeout=50) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Image exceeds 20 MB limit")
        data = response.read(MAX_DOWNLOAD_BYTES + 1)
        if len(data) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Image exceeds 20 MB limit")
        return data


def internet_images(kind: str, count: int, target: Path) -> int:
    """Download up to count public Wikimedia medical illustration images."""
    if count == 0:
        return 0
    search = '"chest x-ray" pneumonia' if kind == "xrays" else '"electrocardiogram" myocardial infarction'
    saved, offset, attribution = 0, 0, []
    hashes = known_digests(target, EXTENSIONS)
    seen = set()
    while saved < count and offset < 500:
        try:
            response = commons_query({
                "action": "query", "generator": "search", "gsrsearch": search,
                "gsrnamespace": "6", "gsrlimit": "50", "gsroffset": str(offset),
                "prop": "imageinfo", "iiprop": "url|mime|extmetadata", "iiurlwidth": "1600"
            })
        except (OSError, ValueError, HTTPError, URLError) as error:
            print(f"[WARN] Wikimedia search failed: {error}")
            break
        pages = response.get("query", {}).get("pages", [])
        if not pages:
            break
        for page in pages:
            if saved >= count:
                break
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            info = info_list[0]
            url = info.get("thumburl") or info.get("url", "")
            title = page.get("title", "")
            extension = Path(title).suffix.lower()
            if not url or extension not in {".png", ".jpg", ".jpeg", ".webp"} or url in seen:
                continue
            seen.add(url)
            try:
                data = download_image(url)
                # Basic signature validation without Pillow.
                if data.startswith(b"\x89PNG\r\n\x1a\n"):
                    extension = ".png"
                elif data.startswith(b"\xff\xd8\xff"):
                    extension = ".jpg"
                elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
                    extension = ".webp"
                else:
                    raise ValueError("Not a supported image file")
                path = save_unique_bytes(target, f"{kind}-{saved+1:03d}{extension}", data, hashes)
                if path is None:
                    print(f"[SKIP] Identical {kind} image already exists: {title}")
                    continue
                saved += 1
                attribution.append({
                    "file": path.name, "source_page": info.get("descriptionurl", ""),
                    "original_url": info.get("url", ""), "commons_title": title,
                    "license": info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", ""),
                    "educational_only": True,
                })
                print(f"[INFO] Downloaded {kind} {saved}/{count}: {path.name}")
                time.sleep(0.5)
            except (OSError, ValueError, HTTPError, URLError) as error:
                print(f"[WARN] Skipping {title}: {error}")
        offset += 50
        time.sleep(1)
    if attribution:
        source_file = target / "sources.csv"
        already_exists = source_file.is_file() and source_file.stat().st_size > 0
        with source_file.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(attribution[0]))
            if not already_exists:
                writer.writeheader()
            writer.writerows(attribution)
    if saved < count:
        print(f"[WARN] Only {saved} of {count} {kind} images were available.")
    return saved


def educational_blood(count: int, target: Path) -> int:
    """Generate synthetic blood-test records, NOT internet clinical data."""
    hashes = known_digests(target, {".json"})
    added = 0
    for index in range(count):
        template = BLOOD_EXAMPLES[index % len(BLOOD_EXAMPLES)]
        record = {**template, "educational_only": True,
                  "disclaimer": "Synthetic educational example, not a real patient's results."}
        payload = (json.dumps(record, indent=2) + "\n").encode("utf-8")
        path = save_unique_bytes(target, f"blood-test-{index+1:03d}.json", payload, hashes)
        if path is None:
            print(f"[SKIP] Identical synthetic blood-test record already exists: {template['title']}")
        else:
            added += 1
    print(f"[INFO] Created {added} new synthetic educational blood-test record(s)")
    return added


def main() -> int:
    print("\nMedGemma Patient Data Importer")
    print("Educational data is for software testing, not diagnosis.\n")
    root = output_root()
    patient, patient_name, dob_digits = prompt_patient_id()
    date = prompt_date()
    exam_time = prompt_exam_time()
    destination = root / "patient_records" / patient / date / exam_time
    print(f"\nDestination: {destination}")
    while True:
        choice = input("Source: [1] External device/folder  [2] Educational internet data: ").strip()
        if choice in {"1", "2"}:
            break
        print("[ERROR] Choose 1 or 2")
    source = None
    counts = None
    if choice == "1":
        default_source = configured_directory("BASE_INPUT_DIR")
        entered = input(f"External source directory [{default_source}]: ").strip().strip('"')
        source = Path(entered).expanduser() if entered else default_source
        if not source.is_dir():
            raise FileNotFoundError(f"External directory not found: {source}")
        print("Expected subfolders: xray/, ekg/, blood test/ (plural names also supported).")
    else:
        counts = (prompt_count("X-ray"), prompt_count("EKG"), prompt_count("blood-test"))
        print("[INFO] X-ray and EKG examples come from Wikimedia Commons.")
        print("[INFO] Blood tests are SYNTHETIC educational JSON, not downloaded clinical records.")
    date_of_birth = datetime.strptime(dob_digits, "%m%d%Y").strftime("%B %d, %Y")
    exam_date = datetime.strptime(date, "%Y-%m-%d").strftime("%B %d, %Y")
    exam_clock = datetime.strptime(exam_time, "%I-%M_%p").strftime("%I:%M %p")
    print("\nPlease confirm these exam details:")
    print(f"  Patient: {patient_name}")
    print(f"  Date of birth: {date_of_birth}")
    print(f"  Date of exam: {exam_date}")
    print(f"  Time of exam: {exam_clock}")
    if input("Proceed? [y/N]: ").strip().lower() not in {"y", "yes"}:
        print("[INFO] Cancelled; no files created.")
        return 0
    if destination.exists() and any(destination.iterdir()):
        print("[INFO] Patient/date/time already exists; merging only new files (SHA-256 duplicate check).")
    for folder in ("xrays", "ekgs", "blood_tests"):
        (destination / folder).mkdir(parents=True, exist_ok=True)
    if choice == "1":
        results = import_external(source, destination)
    else:
        xray, ekg, blood = counts
        results = {
            "xrays": internet_images("xrays", xray, destination / "xrays"),
            "ekgs": internet_images("ekgs", ekg, destination / "ekgs"),
            "blood_tests": educational_blood(blood, destination / "blood_tests"),
        }
    print("\n[COMPLETE]", destination)
    for name, count in results.items():
        print(f"  {name}: {count} record(s)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyboardInterrupt) as error:
        print(f"\n[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
