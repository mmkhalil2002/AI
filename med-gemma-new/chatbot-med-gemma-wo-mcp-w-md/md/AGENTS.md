# AGENTS.md — Single-File MedGemma GUI (Simplified Codex Instructions)

## Directory layout and Codex working directory (mandatory)

- Launch Codex **from `Project/md/`**, the directory containing this `AGENTS.md` and `README.md`.
- Determine the project root from the location of this Markdown file (`md/..`), **not from an assumed installation path**. In Codex's initial `md/` working directory, `../python/` is the Python source/output directory and `../config/` contains the launch scripts.
- Read the original reference Python file from `../python/client-med-gemma-2w-data-v2(1).py` if present; if it is not present, report the missing reference instead of inventing or silently replacing it.
- Write the **only new Python application** directly to `../python/client_medgemma_patients.py`. Do not generate it in `md/` and do not leave a duplicate `.py` file there.
- Preserve `../python/.env`, `../python/patient_records/`, and the original Python source. Never move, overwrite, or automatically create patient records.
- The running GUI must resolve its own location using `Path(__file__).resolve().parent`, then load `.env` and validate `patient_records` relative to that location. It must not use Codex's working directory as the data root.
- Use the Ollama URL from `../python/.env`; it may be local or remote. Do not hardcode an IP.
- Test the resulting file using `cd ../python` followed by `python client_medgemma_patients.py` on Windows or `python3 client_medgemma_patients.py` on Ubuntu. The launchers in `../config/` explicitly start `../python/client_medgemma_patients.py`.

## 1. Goal — keep it simple

**Modify the attached existing program `client-med-gemma-2w-data-v2(1).py` rather than designing a new application.** Deliver **ONE Python application file only**, named `client_medgemma_patients.py`. Keep the original Python file unchanged as a backup/reference. All patient discovery, GUI controls, image and JSON loading, Ollama HTTP calls, and response display must be implemented **inside this one file**.

**Do not implement** Qwen, agents, MCP, tool calling, medical_tools.py, medical_agent.py, ollama_client.py, data_loader.py, data_models.py, databases, web services, or other Python modules. Do not create a multi-module architecture. Do not install AI models locally. This is a direct GUI-to-MedGemma application.

## 2. Required workflow

1. **Select patient:** Add a patient drop-down near the top of the existing Tkinter GUI. On startup, scan the `patient_records` folder and populate the list with its immediate patient subdirectory names. Names can be `P0001`, `MohamedLjalil`, `Patient_A`, etc.; **do not require a `P####` pattern**.
2. **Filter by date (optional):** Provide inclusive start and end date fields in `MMDDYY` format. Apply the range to X-ray, EKG, and blood-test panels for the selected patient. Read image dates from dated filenames or subfolder names, and blood-test dates from JSON date fields or dated filenames/subfolders. Blank bounds are open-ended; both blank shows all records. When a range is active, omit records with no recognizable date and show a clear empty-state message. Applying a range clears old selections and response context.
3. **Load patient data:** When a patient is selected, clear all previous patient selections, galleries, blood-test records, and old response/context. Load that patient's X-ray images, EKG images, and blood-test JSON from `data/xrays/`, `data/ekgs/`, and `data/blood_tests/`. Display them in the existing X-ray, EKG, and blood-test panels. Keep the existing image thumbnail selection/deselection and blood-test row selection behavior.
4. **Doctor's request:** The doctor types a question in the existing prompt field. Preserve the existing Send button.
5. **Send to MedGemma:** On Send, submit the doctor's request and **only the selected records belonging to the active patient and date range** directly to the configured MedGemma model on the Ollama server specified in `.env` (local or remote). Use the existing `/api/generate` request implementation and `.env` settings. Include selected blood-test values and reference ranges in the prompt. Send selected X-ray and EKG **original image bytes** encoded in base64; never send resized thumbnails. Keep network calls off the GUI thread.
6. **Show answer:** Display MedGemma's answer in the existing AI Response panel, with existing status/error handling. Preserve optional voice playback if already working. Indicate that the output is AI-generated and requires clinician review.

**Actions area:** Include an empty, labeled **Actions** section in the GUI as a placeholder. **No action buttons, tool workflows, routing, or agent logic** are to be implemented now. The existing **Send** button is outside this placeholder and must work.

## 3. Patient directory structure

The patient directory name **is the patient ID**. Example:

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
        │   └── data/
        │       ├── xrays/
        │       │   ├── xray01.png
        │       │   └── xray02.jpg
        │       ├── ekgs/
        │       │   └── ekg01.png
        │       └── blood_tests/
        │           └── blood01.json
        └── P0002/
            └── data/
                ├── xrays/
                ├── ekgs/
                └── blood_tests/
```

- Add support for optional `.env` key `PATIENT_RECORDS_DIR=patient_records`; if absent, use the `patient_records` folder beside the Python script. Relative paths are relative to the script directory.
- Discover only immediate patient directories, excluding hidden/system folders. Allow ordinary folder names with spaces and letters. Do not infer patient identity from image names or dated folders.
- Inside each patient's `data/` folder, display supported image formats in `xrays/` and `ekgs/`, and load `.json` files from `blood_tests/`. Support both the original `{"title": "...", "results": ...}` format and structured CBC-style JSON such as `{"panel": "CBC", "results": [{"test": "WBC", "value": 7.4, "unit": "10^3/uL"}]}`. Do not discard records simply because `title` is absent.
- Missing or empty modality folders are allowed; display a clear message, not an exception. Report invalid images/JSON without crashing the GUI.
- Do **not** create or require `patients.json`, `index.json`, `record.json`, patient metadata, or a database. Blood-test JSON files are the actual clinical-content inputs and must remain intact.
- Never mix records from different patients. When switching patients, clear selections and the displayed response. Capture a snapshot of the active patient and selected file paths before starting each request; prevent patient switching during an in-flight request or discard its result if the patient has changed. Validate that resolved file paths stay inside the selected patient directory, and avoid following symlinks outside it.
- Date input fields accept six digits as `MMDDYY`; the end date is inclusive. A recognized date in JSON takes precedence over a filename or folder date. Records without dates remain visible when no range is applied and are excluded during date filtering.

### Mandatory startup check: patient_records directory

**This check belongs in `client_medgemma_patients.py`, not in `run.bat`, `run.sh`, or the installer.** At the very beginning of application startup, before constructing or displaying the Tkinter GUI or loading patient data:

1. Resolve the GUI script's own directory with `PYTHON_DIR = Path(__file__).resolve().parent` (import `Path` from `pathlib`). This is the directory containing the `.py` file, regardless of the shell's current working directory or the installation location.
2. Load `.env` from `PYTHON_DIR / ".env"` using the application's existing configuration-reading method. Do not hardcode an absolute Windows or Ubuntu path.
3. Read optional `PATIENT_RECORDS_DIR`, defaulting to `patient_records`. Resolve relative values **against `PYTHON_DIR`**. Thus the default is `PYTHON_DIR / "patient_records"`; do not use `D:` or any other assumed drive. If the existing setting is an absolute path, honor that explicitly configured path.
4. Check that the resolved patient records root **exists, is a directory, and is readable**. If not, show a clear fatal startup error identifying the full resolved path, and terminate with a nonzero exit status. Log/print the error so it remains visible when launched by `config/run.bat` or `config/run.sh`; a Tkinter message box may additionally be used if a graphical session is available.
5. **Never create the missing directory**, fall back to a different folder, or silently launch an empty GUI. Do not confuse a missing `patient_records` root with an existing root that contains zero patients: the latter may show an explicit "No patient directories found" message without crashing.
6. Missing or empty `data/`, `xrays/`, `ekgs/`, or `blood_tests/` folders **inside an existing patient directory** remain nonfatal as already specified.

Required error example (the path is computed at runtime, not hardcoded):

```text
[ERROR] Patient records directory not found or not accessible: <resolved path>
[ACTION] Restore the patient_records directory alongside the GUI, or correct PATIENT_RECORDS_DIR in python/.env.
[STOP] Application terminated.
```

Test startup with the patient records root present, missing, a regular file instead of a directory, and an unreadable directory where permissions permit. Also test launching from `config/` and after moving the entire project to a different location. No virtual environment is required or created by this application.

## 4. Preserve the existing GUI and MedGemma code

Study the entire attached `client-med-gemma-2w-data-v2(1).py` before editing. **Reuse and minimally adapt** its existing functions and styling, especially:

- Tkinter GUI layout, prompt input, Send button, response area, status text, model/server controls, and voice toggle.
- X-ray and EKG thumbnail galleries, scrollbars, selection colors, and selection toggling.
- Blood-test selectable records and display formatting.
- `encode_image()`, `generate()`, `/api/tags` model check, `/api/generate` call, asynchronous request worker, and existing timeout/error handling.
- Existing `.env` keys, including `OLLAMA_URL`, `MEDGEMMA_MODEL`, image preview sizes, `NUM_PREDICT`, `TEMPERATURE`, `REPEAT_PENALTY`, `REPEAT_LAST_N`, `THINK`, and `GENERATION_TIMEOUT`.

**Change only what is necessary** to replace the previous latest-dated-folder data discovery with active-patient-folder loading. Existing `Data > Load/Delete` logic for legacy dated folders is **not required in this simplified version**: remove or disable it rather than automatically assigning data to a patient. Do not delete or move any source files. Keep the current layout unless a small adjustment is needed for the patient selector and empty Actions section.

The original script sends one combined prompt and selected images directly to MedGemma. **Keep that same simple request model**; no conditional analysis, automatic AI tool selection, or separate medical functions. The doctor controls which records are included by selecting them in the GUI.

## 5. Configuration and dependencies

Use the existing `.env` unchanged. If it does not already include the patient directory key, default correctly without modifying `.env`. Document this optional setting:

```ini
PATIENT_RECORDS_DIR=patient_records
OLLAMA_URL=http://YOUR_OLLAMA_HOST:11434
MEDGEMMA_MODEL=medgemma1.5:latest
```

The values above are examples, not replacements for the user's current settings. **Never hardcode a server IP.** Do not automatically install Qwen or MedGemma on the GUI computer. Prefer existing packages from the reference client (Pillow, pyttsx3 when supported); do not add unnecessary dependencies or build a new installer framework.

## 6. Acceptance checks

- One new `.py` file only, with detailed function docstrings and useful inline comments.
- Missing or inaccessible `python/patient_records/` produces a clear fatal startup error before the GUI opens; the application never creates the folder. Existing but empty patient records root is handled separately.
- Patient selector lists both `MohamedLjalil` and `P0002` when those directories exist.
- Selecting each patient loads only their own X-rays, EKGs, and blood tests.
- Switching patients clears all previous selections and the previous response.
- EKG gallery correctly loads images from `patient_records/<patient_id>/data/ekgs/` (not from legacy `ekg-data-MMDDYY`).
- An inclusive `MMDDYY` range filters X-rays, EKGs, and blood-test records; applying or clearing the range clears stale selections and response context.
- A CBC JSON without a `title` field is displayed and can be selected.
- Doctor can select one or more images/blood records, type a question, and press Send.
- The outgoing MedGemma request contains only selected patient data, with original image resolution retained.
- MedGemma answer appears in the existing response panel. Handle missing data, bad JSON, unavailable Ollama, and timeouts gracefully.
- **Actions section is visible and EMPTY.** No extra Python modules, no Qwen, no MCP.
- Test syntax/imports and patient switching with synthetic files; test network payload using a mock or reachable Ollama server. Report explicitly whether the GUI was visually tested and whether live MedGemma was contacted.

## 7. Codex execution instructions

From the `md/` directory, read this `AGENTS.md`, `README.md`, and the original Python script in `../python/`. Implement the single-file modification autonomously, making reasonable decisions without routine clarification. **Do not treat older Markdown files in the directory as additional implementation requirements** when they describe the previous multi-module Qwen/agent/MCP architecture; this document supersedes them. Preserve the original script, `.env`, and all patient data. At completion, report the one created Python file, test results, and any remaining issues.

From the `python/` directory, run the resulting program on Windows with:

```cmd
python client_medgemma_patients.py
```

For Ubuntu, use `python3 client_medgemma_patients.py` after installing Tkinter and the applicable Python dependencies. Voice support may be unavailable or platform-dependent.

**Safety:** Use synthetic or appropriately de-identified data for development. MedGemma output is research assistance, not an independently validated diagnosis. Plain HTTP to remote Ollama is not encrypted; do not send identifiable clinical data over an unsecured connection.
