# Medical AI GUI — Single-File MedGemma Application

## Directory layout and Codex working directory (mandatory)

- Launch Codex **from `Project/md/`**, the directory containing this `AGENTS.md` and `README.md`.
- Determine the project root from the location of this Markdown file (`md/..`), **not from an assumed installation path**. In Codex's initial `md/` working directory, `../python/` is the Python source/output directory and `../config/` contains the launch scripts.
- Read the original reference Python file from `../python/client-med-gemma-2w-data-v2(1).py` if present; if it is not present, report the missing reference instead of inventing or silently replacing it.
- Write the **only new Python application** directly to `../python/client_medgemma_patients.py`. Do not generate it in `md/` and do not leave a duplicate `.py` file there.
- Preserve `../python/.env`, `../python/patient_records/`, and the original Python source. Never move, overwrite, or automatically create patient records.
- The running GUI must resolve its own location using `Path(__file__).resolve().parent`, then load `.env` and validate `patient_records` relative to that location. It must not use Codex's working directory as the data root.
- Use the Ollama URL from `../python/.env`; it may be local or remote. Do not hardcode an IP.
- Test the resulting file using `cd ../python` followed by `python client_medgemma_patients.py` on Windows or `python3 client_medgemma_patients.py` on Ubuntu. The existing launchers in `../config/` can also be used, but because they choose the first `.py` file, verify that they select the intended GUI.

## 1. Overview

This project is a **minimal extension of the existing Tkinter program** `client-med-gemma-2w-data-v2(1).py`. It adds patient selection and patient-specific loading while retaining the original image galleries, doctor prompt, MedGemma connection, and AI response display.

**Current scope:** one application Python file, `client_medgemma_patients.py`. **No Qwen, MCP, agent, tool-routing engine, database, or extra Python modules.** The **Actions** area is present but intentionally **empty**. Qwen-based decision-making may be added later as an optional extension within the same file; it is **not implemented in this phase**.

The original reference program and existing `.env` must remain unchanged.

## 2. Architecture

```text
Windows PC — client_medgemma_patients.py

  Patient selector
        |
        v
  patient_records/<patient_id>/
       |-- xrays/        -> X-ray gallery
       |-- ekgs/         -> EKG gallery
       `-- blood_tests/  -> Blood-test list
        |
        v
  Doctor selects data and types a request
        |
        v
  Existing Send button
        |
        v
  POST /api/generate -----------------> Configured Ollama server
  (question + selected blood data       MedGemma 1.5
   + original selected images)                |
        |<------------------------------------'
        v
  Existing AI response panel

  Actions: [empty placeholder]
```

All GUI, patient loading, and HTTP logic runs in the **same Python file** on the Windows PC. MedGemma runs on the **Ollama server configured in `python/.env`**, which may be local or remote.

## 3. Patient directory hierarchy

Each immediate subdirectory of `patient_records/` is a patient ID. Names can contain letters and numbers; they do not have to match `P0001`.

```text
Project/
├── config/
│   ├── run.bat
│   └── run.sh
├── md/
│   ├── AGENTS.md
│   └── README.md
└── python/
    ├── client-med-gemma-2w-data-v2(1).py  # original reference, if present
    ├── client_medgemma_patients.py         # generated GUI
    ├── .env                                # existing settings
    └── patient_records/
        ├── MohamedLjalil/
        │   ├── xrays/
        │   │   ├── xray01.png
        │   │   └── xray02.jpg
        │   ├── ekgs/
        │   │   └── ekg01.png
        │   └── blood_tests/
        │       └── blood01.json
        └── P0002/
            ├── xrays/
            ├── ekgs/
            └── blood_tests/
```

No `patients.json`, mapping table, or patient metadata file is required. The folder name identifies the patient. Blood-test `.json` files contain the medical test records and are retained.

Blood-test files may use the original format:

```json
{"title": "WBC", "results": "7.4 x10^3/uL"}
```

Or structured results:

```json
{"panel": "CBC", "results": [{"test": "WBC", "value": 7.4, "unit": "10^3/uL", "reference_range": "4.0-11.0"}]}
```

Empty modality folders are acceptable. The GUI should report missing or invalid data without crashing. Data from different patients must never be mixed.

### Required patient directory validation at startup

The application must look for `patient_records` **inside the same `python` directory as `client_medgemma_patients.py`**, unless `PATIENT_RECORDS_DIR` explicitly specifies a different path in `python/.env`.

The GUI code determines the Python directory automatically:

```python
from pathlib import Path
PYTHON_DIR = Path(__file__).resolve().parent
```

Load `python/.env` relative to `PYTHON_DIR`. The default patient records path is `PYTHON_DIR / "patient_records"`; any relative `PATIENT_RECORDS_DIR` setting is also resolved relative to `PYTHON_DIR`. This works regardless of the installation drive, operating system, or launch location (`config/run.bat` or `config/run.sh`).

**Before showing the GUI**, the Python program must check that the resolved patient records root exists, is a directory, and is readable. If it is missing or inaccessible, print an explicit error containing the actual full path and exit with a nonzero status. Do not create a replacement folder or continue with an empty GUI. An existing but empty `patient_records` root should instead show "No patient directories found". Missing individual modality subfolders (`xrays`, `ekgs`, `blood_tests`) remain nonfatal.

Example error (illustrative path only):

```text
[ERROR] Patient records directory not found or not accessible: <resolved path>
[ACTION] Restore the patient_records directory or correct PATIENT_RECORDS_DIR in python/.env.
[STOP] Application terminated.
```

This behavior must be implemented in **the Python application**, not the launcher scripts. It is a mandatory requirement for Codex when generating `client_medgemma_patients.py`.

## 4. How to use the GUI

1. Launch `client_medgemma_patients.py`.
2. Choose a patient in the new **Patient** dropdown. The program loads that patient's X-rays, EKGs, and blood tests into the existing panels.
3. Click the X-ray thumbnails, EKG thumbnails, and/or blood-test rows you want included. Clicking an item again deselects it.
4. Enter the doctor's request in the existing prompt box.
5. Click **Send**. The application sends the prompt and **only the selected patient data** directly to MedGemma.
6. Read the generated answer in the existing AI Response panel. Review all findings clinically.

When switching patients, the application clears previous selections, prior response/context, and displayed records. It must also prevent an earlier in-flight response from appearing under a different patient.

The **Actions** section remains **empty** in this version. The existing **Send** button continues to function independently.

## 5. Data sent to MedGemma

The application reuses the original `/api/generate` implementation:

- Doctor's text request is sent in the prompt.
- Selected blood-test results, including available units and reference ranges, are included in the prompt.
- Selected X-ray and EKG **original image bytes** are base64-encoded and sent in the `images` array. Display thumbnails are **never** uploaded.
- Unselected records and files from other patients are not sent.
- Network requests run in a background worker so the Tkinter GUI stays responsive.

The existing connection/model check (`/api/tags`), timeout handling, status messages, and optional voice playback should be preserved.

## 6. Configuration

Keep the existing `.env` file. The new patient-directory setting is optional; when omitted, the application uses `patient_records` beside the script.

```ini
PATIENT_RECORDS_DIR=patient_records
OLLAMA_URL=http://YOUR_OLLAMA_HOST:11434
MEDGEMMA_MODEL=medgemma1.5:latest
```

The URL above is a **placeholder**. Keep the actual local or remote server address from your existing `.env`. Other original settings, such as `NUM_PREDICT`, `TEMPERATURE`, `REPEAT_PENALTY`, `REPEAT_LAST_N`, `THINK`, `GENERATION_TIMEOUT`, and image display dimensions, continue to apply.

The GUI must not automatically install MedGemma. It must already be available on the configured Ollama server, whether local or remote.

## 7. Installation and execution

From Windows Command Prompt, run the launcher from `config/` (or run Python directly from `python/`):

```cmd
cd /d "C:\Users\Public\mkhalil\AI\AI\chatbot-med-gemma-without MCP\config"
run.bat
```

The launcher selects the first `.py` file in the sibling `python/` directory. Ensure the intended GUI is that file, or run `python client_medgemma_patients.py` directly from `python/`.

The reference program can also install missing supported packages automatically. `pyttsx3` is optional if voice output is not needed.

On Ubuntu, install Python, Tkinter, and Pillow as appropriate for the system, then run from `config/`:

```bash
bash run.sh
```

Or run `python3 client_medgemma_patients.py` directly from `python/`. The launcher selects the first `.py` file; confirm that it is the intended GUI.

Speech support is platform-dependent. A graphical desktop session is required for Tkinter.

**Note:** `client_medgemma_patients.py` is the output Codex should create from `AGENTS.md`; it is not supplied by this README.

## 8. Instructions for Codex

Place this README and the simplified `AGENTS.md` in `Project/md/`, with the original Python source in `Project/python/`. Open a terminal in `Project/md/` and run `codex`. In Codex, use:

> Read AGENTS.md and README.md from the current md/ directory and read ../python/client-med-gemma-2w-data-v2(1).py. Modify the existing design and write exactly one new application file directly to ../python/client_medgemma_patients.py. Preserve the GUI, add patient selection and patient-specific loading, enforce fatal startup validation of the patient_records root relative to the Python script, keep the Send-to-MedGemma flow, and leave Actions empty. Do not implement Qwen, MCP, or separate Python modules. Do not overwrite the original program, .env, or patient data. Run available tests and report what was actually verified.

If older Markdown documents in the folder describe a multi-module Qwen or agent architecture, **the simplified `AGENTS.md` takes precedence**.

## 9. Verification checklist

- Missing or inaccessible `python/patient_records/` produces a fatal startup error with the full path; no directory is created.
- The same check works when launched from `config/` or after moving the project.
- Patient dropdown discovers both ordinary names and `P0002`-style IDs.
- Switching patients refreshes all three panels and clears prior selections and answers.
- Images load from the active patient's `xrays/` and `ekgs/` folders.
- Blood-test JSON loads even when it uses `panel` instead of `title`.
- Selected originals, not thumbnails, are sent to MedGemma.
- Only the active patient's selected data appears in the outgoing request.
- The response appears in the original GUI, with meaningful error handling for unreachable Ollama, invalid files, and timeouts.
- The Actions section is visible but contains no controls.
- Exactly one new application `.py` file is generated.

Use synthetic or de-identified test data. **MedGemma is a research/clinician-review aid, not an autonomous diagnostic system.** Plain HTTP is unencrypted; do not transmit identifiable patient data over an unsecured connection. For real clinical use, implement appropriate authenticated, encrypted transport and organizational privacy controls.

## 10. Future extension: Qwen (not part of current build)

If flexible doctor-directed multi-step decision-making is later required, add Qwen calls and a bounded, validated Python execution loop **inside the same application file**. Qwen can interpret the doctor's requested sequence, MedGemma can analyze medical content, and Python should enforce patient isolation, data-selection rules, and numeric comparisons. This future extension must not be assumed present in the current version.
