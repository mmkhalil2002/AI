# Medical AI GUI — Single-File MedGemma Application

## Directory layout and Codex working directory (mandatory)

The project root is the directory containing the three sibling folders `config/`, `md/`, and `python/`. The name `Project/` in the diagrams below is illustrative; the project may be installed anywhere. Keep Markdown documentation in `md/`, launch scripts in `config/`, and the application, `.env`, and patient records in `python/`.

- Launch Codex **from `Project/md/`**, the directory containing this `AGENTS.md` and `README.md`.
- Determine the project root from the location of this Markdown file (`md/..`), **not from an assumed installation path**. In Codex's initial `md/` working directory, `../python/` is the Python source/output directory and `../config/` contains the launch scripts.
- Read the original reference Python file from `../python/client-med-gemma-2w-data-v2(1).py` if present; if it is not present, report the missing reference instead of inventing or silently replacing it.
- Write the **only new Python application** directly to `../python/client_medgemma_patients.py`. Do not generate it in `md/` and do not leave a duplicate `.py` file there.
- Preserve `../python/.env`, `../python/patient_records/`, and the original Python source. Never move, overwrite, or automatically create patient records.
- The running GUI must resolve its own location using `Path(__file__).resolve().parent`, then load `.env` and validate `patient_records` relative to that location. It must not use Codex's working directory as the data root.
- Use the Ollama URL from `../python/.env`; it may be local or remote. Do not hardcode an IP.
- Run the GUI from `python/` with `python client_medgemma_patients.py` on Windows or `python3 client_medgemma_patients.py` on Ubuntu. The launchers in `../config/` explicitly start this named application.

## 1. Overview

This project is a **minimal extension of the existing Tkinter program** `client-med-gemma-2w-data-v2(1).py`. It adds patient selection and patient-specific loading while retaining the original image galleries, doctor prompt, MedGemma connection, and AI response display.

**Current scope:** one application Python file, `client_medgemma_patients.py`. **No Qwen, MCP, agent, tool-routing engine, database, or extra Python modules.** There is **no Actions area** in the current GUI. Qwen-based decision-making may be added later as an optional extension within the same file; it is **not implemented in this phase**.

The original reference program and existing `.env` must remain unchanged.

## 2. Architecture

```text
Windows PC — client_medgemma_patients.py

  Patient selector
        |
        v
  patient_records/<patient_id>/data/
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

Each immediate subdirectory of `patient_records/` is a patient ID. Modality files are stored under that patient's `data/` folder. Names can contain letters and numbers; they do not have to match `P0001`.

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

The application must look for `patient_records` **inside the same `python` directory as `client_medgemma_patients.py`**, with modality folders under each patient's `data/`, unless `PATIENT_RECORDS_DIR` explicitly specifies a different path in `python/.env`.

The GUI code determines the Python directory automatically:

```python
from pathlib import Path
PYTHON_DIR = Path(__file__).resolve().parent
```

Load `python/.env` relative to `PYTHON_DIR`. The default patient records path is `PYTHON_DIR / "patient_records"`; any relative `PATIENT_RECORDS_DIR` setting is also resolved relative to `PYTHON_DIR`. This works regardless of the installation drive, operating system, or launch location (`config/run.bat` or `config/run.sh`).

**Before showing the GUI**, the Python program must check that the resolved patient records root exists, is a directory, and is readable. If it is missing or inaccessible, print an explicit error containing the actual full path and exit with a nonzero status. Do not create a replacement folder or continue with an empty GUI. An existing but empty `patient_records` root should instead show "No patient directories found". Missing `data/` or individual modality subfolders (`xrays`, `ekgs`, `blood_tests`) remain nonfatal.

Example error (illustrative path only):

```text
[ERROR] Patient records directory not found or not accessible: <resolved path>
[ACTION] Restore the patient_records directory or correct PATIENT_RECORDS_DIR in python/.env.
[STOP] Application terminated.
```

This behavior must be implemented in **the Python application**, not the launcher scripts. It is a mandatory requirement for Codex when generating `client_medgemma_patients.py`.

## 4. How to use the GUI

1. Launch `client_medgemma_patients.py` with the command for your operating system in section 8. Server/model settings remain in `.env`; they are not shown as GUI fields. Connection errors appear as status/error messages.
2. Select a patient ID from the dropdown. Check the active-patient name before choosing records. The application does not infer that a patient or filename is synthetic.
3. Optionally enter start and end dates in `MMDDYY` format and click **Apply range**. Bounds are inclusive, and either bound may be left blank. Dates are read from image/file or folder names and blood-test JSON date fields. Undated records are excluded while a range is active; **Clear range** restores all records.
4. Select only the X-ray, EKG, and blood-test records you want to include. Nothing is selected automatically. Selected items are highlighted or checked without displaying numerical counts. Keep image inspection and selection/deselection functionality.
5. Type the doctor's request in the initially blank **Your Question** box. If no records are selected, the request is text-only.
6. Click **Send**. The app snapshots the active patient, date range, and selected paths, checks the configured server/model, and sends one request in the background. **Cancel** immediately stops accepting the result locally and closes the response where possible; the Ollama server may continue its computation.
7. Review the MedGemma answer in the single **AI Response** area. The GUI has no **Response** or **Original** tabs. Any internal original-response record needed for export/review may be retained without showing a second response pane.
8. Use **Export TXT + HTML + JSON** to save deliberately to an existing folder outside `patient_records`. The versioned JSON bundle records both draft versions, completion/review state, model/time, and patient-relative references to inputs actually sent; it does not embed source images. Choose that folder under **Saved report history** to filter and reopen reports for the active patient. Reopening never sends a request.

When switching patients, the application clears previous selections, prior response/context, and displayed records. It must also prevent an earlier in-flight response from appearing under a different patient.

The **Actions** section is removed entirely. The existing **Send to MedGemma** button continues to function.

## 4A. Approved GUI layout (mandatory for regeneration)

The following overrides any legacy UI instructions elsewhere in this README while retaining the same single-file patient workflow:

- **Top:** one compact `Patient Selection and Date Range` row containing the Patient ID dropdown followed immediately by From/To dates, Apply Range, Clear Range, then the active-patient indicator **on the same physical line at ordinary desktop widths**. Do not split patient selection and date filtering into separate header rows or panels, or put the active-patient indicator between the menu and filters. A second row is allowed only as a narrow-window fallback that prevents clipping. Input dates use `MMDDYY`; formatted display may show `MM/DD/YYYY` if conversion is correct. Do not display Server URL, Model, or Check Connection widgets. Ollama settings still come from `.env`.
- **Left (~35% width):** `AI Clinical Assistant Center`; **blank** `Your Question` box on startup; `Send to MedGemma` and Voice toggle; one large `AI Response` box. **No Actions section** and **no Response/Original tabs**. No default question/prompt text.
- **Right (~65% width):** **XRAY**, **EKG/ECG**, **BLOOD TEST** in three **vertically stacked** panels, approximately 35%/35%/30% of available height, with scrolling and click selection preserved. **No Medical Image Datasets heading**.
- **Panel labels:** show the active date range in each panel title, e.g. `XRAY (01/01/2026 - 12/31/2026)`. If no range is applied, show that modality's recognized minimum-to-maximum available dates instead of `All dates`; if every record is undated, show `Undated records`. For one open bound, show only the applicable limit. The displayed date range must match the actual filtering.
- **No image/record counts anywhere in the GUI**, including modality headings and selection status. Use visual selection indicators instead of `0/3 selected` or `3 image(s)`.
- **Blood-test alignment:** render all blood-test table headings and rows in a fixed-width font with fixed-width columns for test, value, unit, reference range, and flag. Headers and their values must align vertically.
- Preserve patient isolation, date filtering, image/JSON loading, direct Ollama requests, original image bytes, optional voice, errors, and clinical review safeguards. Hide or omit any legacy report preset, input summary, original/draft tabs, or Actions placeholder that conflicts with this layout. Advanced export/history logic may remain if already implemented, but must not reintroduce removed GUI panels.

## 5. Review, saving, and history

- A response must complete successfully before it can be acknowledged or exported as a completed report. Failed, timed-out, cancelled, empty, or truncated output is not a completed report.
- Only one **AI Response** text area is shown; do not render separate original/draft tabs. If the application supports editing and review acknowledgement, preserve the underlying review safeguards without adding a second visible response pane.
- **Export TXT + HTML + JSON** is an explicit save action. Pick an existing output directory outside `patient_records`; the app refuses a report destination inside patient source data. Keep report files in a user-approved writable location.
- To reopen reports, select **Choose history folder…**, choose the folder that contains exported JSON bundles, select a patient, then choose a report and click **Reopen selected**. The list is filtered to the active patient. Malformed, unsupported, or other-patient bundles are skipped. Reopened reports are not added to a new model prompt.
- Exported review status means only that the user acknowledged the displayed draft locally. It does not identify or authenticate the reviewer and is not a clinical signature.

## 6. Data sent to MedGemma

The application reuses the original `/api/generate` implementation:

- Doctor's text request is sent in the prompt.
- Selected blood-test results, including available units and reference ranges, are included in the prompt.
- Selected X-ray and EKG **original image bytes** are base64-encoded and sent in the `images` array. Display thumbnails are **never** uploaded.
- Unselected records and files from other patients are not sent.
- Network requests run in a background worker so the Tkinter GUI stays responsive.

The existing connection/model check (`/api/tags`), timeout handling, status messages, and optional voice playback should be preserved.

## 7. Configuration

Keep the existing `.env` file. The new patient-directory setting is optional; when omitted, the application uses `patient_records` beside the script.

```ini
PATIENT_RECORDS_DIR=patient_records
OLLAMA_URL=http://YOUR_OLLAMA_HOST:11434
MEDGEMMA_MODEL=medgemma1.5:latest
```

The URL above is a **placeholder**. Keep the actual local or remote server address from your existing `.env`. Other original settings, such as `NUM_PREDICT`, `TEMPERATURE`, `REPEAT_PENALTY`, `REPEAT_LAST_N`, `THINK`, `GENERATION_TIMEOUT`, and image display dimensions, continue to apply.

The GUI must not automatically install MedGemma. It must already be available on the configured Ollama server, whether local or remote.

## 8. Installation and execution

Both launchers use the project root based on their own location and explicitly invoke `python/client_medgemma_patients.py`; they do not select an arbitrary Python file. The direct commands below run the same application from the `python/` directory.

From Windows Command Prompt, run the launcher from `config/` (or run Python directly from `python/`):

```cmd
cd /d "C:\Users\Public\mkhalil\AI\AI\chatbot-med-gemma-without MCP\config"
run.bat
```

To run the app directly:

```cmd
cd /d "C:\Users\Public\mkhalil\AI\AI\chatbot-med-gemma-without MCP\python"
python client_medgemma_patients.py
```

The Windows launcher explicitly starts `python/client_medgemma_patients.py` and checks that its selected Python interpreter is Python 3.9 or newer with Tkinter. If the `python` command is a Windows Store alias, it tries `py -3`; if neither is usable, install Python with Tcl/Tk support or turn off the alias under Windows **App execution aliases**. Pillow is required for image previews. `pyttsx3` is optional for voice playback.

On Ubuntu, install Python, Tkinter, and Pillow as appropriate for the system, then run from `config/`:

```bash
bash run.sh
```

Or run `python3 client_medgemma_patients.py` directly from `python/`. The shell launcher explicitly starts the named GUI and checks Python 3.9+ with Tkinter.

Speech support is platform-dependent. A graphical desktop session is required for Tkinter.

### Troubleshooting

- **Patient records directory missing or inaccessible:** startup stops before opening the GUI and prints the resolved path. Restore the existing `python/patient_records/` directory or correct `PATIENT_RECORDS_DIR` in `python/.env`. A relative value is resolved beside the application script. The app never creates this directory. An existing, readable but empty directory is different and displays “No patient directories found.”
- **Ollama is offline or intermittent:** patient records remain browsable, but generation is not ready until the configured server responds. Check `OLLAMA_URL`, server availability, and firewall/network access, then retry generation or restart the GUI; no Check Connection button is displayed. A successful GUI or mocked check does not guarantee the service will remain reachable. The app keeps the configured endpoint; it does not switch servers.
- **Model is missing:** confirm `MEDGEMMA_MODEL` names a model installed on the configured server. The GUI does not install or download models; model choice remains configured through `.env`, not a visible dropdown.
- **Timeout, cancel, or incomplete response:** full-resolution images may take longer. Check server/GPU load and the selected input count, then retry explicitly if appropriate. Cancel stops local acceptance; it does not establish that Ollama stopped computing.
- **Bad image or blood-test JSON:** that file is reported/skipped while the remaining patient records stay available. Correct the source file only through the normal data-owner workflow; the app does not rewrite source records.
- **Cannot save or reopen a report:** choose an existing writable folder outside `patient_records`. For history, point the history control to the same folder used for export. Reopening is patient-filtered and does not contact the model.
- **Voice unavailable:** voice is optional and platform-dependent. The text report remains available.

## 9. Instructions for Codex

For maintenance, start Codex in this `md/` directory and read `AGENTS.md`, this README, and `VERIFICATION.md` before changing the application. The current GUI is `../python/client_medgemma_patients.py`; keep application logic in that file and preserve the existing `.env` and source patient data. Use temporary synthetic fixtures for development checks, and record which checks were mocked, visually inspected, or run against a live synthetic model. Do not claim clinical accuracy from software tests.

The existing Codex instructions in `AGENTS.md` describe the original one-file patient-GUI build. Follow the active task scope when maintaining later features, while keeping the single-file/direct-Ollama constraints and the approved simplified GUI layout.

## 10. Verification checklist

- Missing or inaccessible `python/patient_records/` produces a fatal startup error with the full path; no directory is created. Each patient's modalities are beneath `data/`.
- The same check works when launched from `config/` or after moving the project.
- Patient dropdown discovers both ordinary names and `P0002`-style IDs.
- Switching patients refreshes all three panels and clears prior selections and answers.
- Images load from the active patient's `data/xrays/` and `data/ekgs/` folders.
- Blood-test JSON loads even when it uses `panel` instead of `title`.
- Selected originals, not thumbnails, are sent to MedGemma.
- Only the active patient's selected data appears in the outgoing request.
- The response appears in the original GUI, with meaningful error handling for unreachable Ollama, invalid files, and timeouts.
- The Actions section is absent; Response/Original tabs, Medical Image Datasets heading, visible model/server controls, and image counts are absent.
- At ordinary desktop width, the patient menu and From/To/Apply/Clear controls appear together in one top row; they do not occupy separate header bands.
- Exactly one new application `.py` file is generated.

Use synthetic or de-identified test data. **MedGemma is a research/clinician-review aid, not an autonomous diagnostic system.** Plain HTTP is unencrypted; do not transmit identifiable patient data over an unsecured connection. For real clinical use, implement appropriate authenticated, encrypted transport and organizational privacy controls.

## 11. Future extension: Qwen (not part of current build)

If flexible doctor-directed multi-step decision-making is later required, add Qwen calls and a bounded, validated Python execution loop **inside the same application file**. Qwen can interpret the doctor's requested sequence, MedGemma can analyze medical content, and Python should enforce patient isolation, data-selection rules, and numeric comparisons. This future extension must not be assumed present in the current version.
