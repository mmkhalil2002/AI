"""Download or create educational blood-test JSON records.

The script creates a ``blood-test-data`` directory beside itself. Each record
is stored in a separate JSON file and has a ``title`` naming the represented
abnormal pattern. No third-party Python packages are required.

Optional .env settings:
    BLOOD_TEST_OUTPUT_DIR=blood-test-data
    BLOOD_TEST_RECORD_COUNT=5
    BLOOD_TEST_SOURCE_URL=

When BLOOD_TEST_SOURCE_URL is blank, the script writes its built-in synthetic
educational records. If a URL is supplied, it downloads a JSON list or an
object containing a ``records`` list. Never use these records for diagnosis.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent


def load_env() -> dict[str, str]:
    """Read simple KEY=VALUE settings from .env beside this script."""
    values: dict[str, str] = {}
    env_path = SCRIPT_DIR / ".env"
    if not env_path.is_file():
        return values
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


ENV = load_env()
OUTPUT_SETTING = os.environ.get(
    "BLOOD_TEST_OUTPUT_DIR", ENV.get("BLOOD_TEST_OUTPUT_DIR", "blood-test-data")
)
OUTPUT_DIR = Path(OUTPUT_SETTING).expanduser()
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = SCRIPT_DIR / OUTPUT_DIR

SOURCE_URL = os.environ.get(
    "BLOOD_TEST_SOURCE_URL", ENV.get("BLOOD_TEST_SOURCE_URL", "")
).strip()


def record_limit() -> int:
    value = os.environ.get(
        "BLOOD_TEST_RECORD_COUNT", ENV.get("BLOOD_TEST_RECORD_COUNT", "5")
    )
    try:
        number = int(value)
        if 1 <= number <= 1000:
            return number
    except ValueError:
        pass
    print(f"Invalid BLOOD_TEST_RECORD_COUNT={value!r}; using 5.")
    return 5


SYNTHETIC_RECORDS = [
    {
        "title": "Iron-Deficiency Anemia Pattern",
        "educational_only": True,
        "results": [
            {"test": "Hemoglobin", "value": 8.7, "unit": "g/dL", "reference_range": "12.0-16.0", "flag": "LOW"},
            {"test": "MCV", "value": 68, "unit": "fL", "reference_range": "80-100", "flag": "LOW"},
            {"test": "Ferritin", "value": 7, "unit": "ng/mL", "reference_range": "15-150", "flag": "LOW"},
            {"test": "Transferrin saturation", "value": 5, "unit": "%", "reference_range": "20-50", "flag": "LOW"},
        ],
    },
    {
        "title": "Bacterial Infection and Inflammation Pattern",
        "educational_only": True,
        "results": [
            {"test": "WBC", "value": 17.8, "unit": "x10^3/uL", "reference_range": "4.0-11.0", "flag": "HIGH"},
            {"test": "Neutrophils", "value": 86, "unit": "%", "reference_range": "40-70", "flag": "HIGH"},
            {"test": "CRP", "value": 112, "unit": "mg/L", "reference_range": "<10", "flag": "HIGH"},
        ],
    },
    {
        "title": "Poorly Controlled Diabetes Pattern",
        "educational_only": True,
        "results": [
            {"test": "Fasting glucose", "value": 238, "unit": "mg/dL", "reference_range": "70-99", "flag": "HIGH"},
            {"test": "Hemoglobin A1c", "value": 9.4, "unit": "%", "reference_range": "<5.7", "flag": "HIGH"},
            {"test": "Triglycerides", "value": 248, "unit": "mg/dL", "reference_range": "<150", "flag": "HIGH"},
        ],
    },
    {
        "title": "Kidney Impairment Pattern",
        "educational_only": True,
        "results": [
            {"test": "Creatinine", "value": 2.4, "unit": "mg/dL", "reference_range": "0.6-1.3", "flag": "HIGH"},
            {"test": "eGFR", "value": 29, "unit": "mL/min/1.73m2", "reference_range": ">=60", "flag": "LOW"},
            {"test": "BUN", "value": 44, "unit": "mg/dL", "reference_range": "7-20", "flag": "HIGH"},
            {"test": "Potassium", "value": 5.6, "unit": "mmol/L", "reference_range": "3.5-5.1", "flag": "HIGH"},
        ],
    },
    {
        "title": "Hepatocellular Liver Injury Pattern",
        "educational_only": True,
        "results": [
            {"test": "ALT", "value": 286, "unit": "U/L", "reference_range": "7-56", "flag": "HIGH"},
            {"test": "AST", "value": 192, "unit": "U/L", "reference_range": "10-40", "flag": "HIGH"},
            {"test": "Total bilirubin", "value": 2.1, "unit": "mg/dL", "reference_range": "0.1-1.2", "flag": "HIGH"},
            {"test": "Albumin", "value": 3.3, "unit": "g/dL", "reference_range": "3.5-5.0", "flag": "LOW"},
        ],
    },
]


def download_records(url: str) -> list[dict]:
    """Download and decode a JSON list of blood-test records."""
    request = Request(url, headers={"User-Agent": "EducationalBloodTestDownloader/1.0"})
    with urlopen(request, timeout=60) as response:
        data = json.load(response)
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError("Downloaded JSON must be a list or contain a 'records' list.")
    return records


def validate_record(record: object, position: int) -> dict:
    """Require a title and a non-empty result list in every record."""
    if not isinstance(record, dict):
        raise ValueError(f"Record {position} is not a JSON object.")
    title = record.get("title")
    results = record.get("results")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"Record {position} has no valid title.")
    if not isinstance(results, list) or not results:
        raise ValueError(f"Record {position} has no results list.")
    clean = dict(record)
    clean["title"] = title.strip()
    clean.setdefault("educational_only", True)
    clean.setdefault(
        "disclaimer",
        "Educational sample only; not for clinical diagnosis or treatment.",
    )
    return clean


def safe_filename(title: str) -> str:
    """Convert a record title into a Windows-safe lowercase filename."""
    name = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()
    return name[:80] or "untitled-record"


def main() -> int:
    try:
        if SOURCE_URL:
            print(f"Downloading blood-test JSON from: {SOURCE_URL}")
            records = download_records(SOURCE_URL)
            source = SOURCE_URL
        else:
            print("No BLOOD_TEST_SOURCE_URL configured; using synthetic educational records.")
            records = SYNTHETIC_RECORDS
            source = "built-in synthetic educational catalog"

        records = [validate_record(item, index) for index, item in enumerate(records, 1)]
        records = records[:record_limit()]
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        saved_records = []
        for index, record in enumerate(records, 1):
            filename = f"{index:02d}-{safe_filename(record['title'])}.json"
            destination = OUTPUT_DIR / filename
            destination.write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            saved_records.append({"title": record["title"], "file": filename})
            print(f"Created: {filename} | title: {record['title']}")

        index_data = {
            "source": source,
            "record_count": len(saved_records),
            "educational_only": True,
            "records": saved_records,
        }
        (OUTPUT_DIR / "index.json").write_text(
            json.dumps(index_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved {len(saved_records)} record(s) in:\n{OUTPUT_DIR}")
        return 0
    except (HTTPError, URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"\nDownload failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
