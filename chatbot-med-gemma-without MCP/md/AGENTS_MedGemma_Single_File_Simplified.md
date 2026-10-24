# AGENTS.md — Single-File MedGemma GUI (Simplified Codex Instructions)

## 1. Goal — keep it simple

**Modify the attached existing program `client-med-gemma-2w-data-v2(1).py` rather than designing a new application.** Deliver **ONE Python application file only**, named `client_medgemma_patients.py`. Keep the original Python file unchanged as a backup/reference. All patient discovery, GUI controls, image and JSON loading, Ollama HTTP calls, and response display must be implemented **inside this one file**.

**Do not implement** Qwen, agents, MCP, tool calling, medical_tools.py, medical_agent.py, ollama_client.py, data_loader.py, data_models.py, databases, web services, or other Python modules. Do not create a multi-module architecture. Do not install AI models locally. This is a direct GUI-to-MedGemma application.

## 2. Required workflow

1. **Select patient:** Add a patient drop-down near the top of the existing Tkinter GUI. On startup, scan the `patient_records` folder and populate the list with its immediate patient subdirectory names. Names can be `P0001`, `MohamedLjalil`, `Patient_A`, etc.; **do not require a `P####` pattern**.
2. **Load patient data:** When a patient is selected, clear all previous patient selections, galleries, blood-test records, and old response/context. Load that patient's X-ray images, EKG images, and blood-test JSON from the three subfolders shown below. Display them in the existing X-ray, EKG, and blood-test panels. Keep the existing image thumbnail selection/deselection and blood-test row selection behavior.
3. **Doctor's request:** The doctor types a question in the existing prompt field. Preserve the existing Send button.
4. **Send to MedGemma:** On Send, submit the doctor's request and **only the selected records belonging to the active patient** directly to the configured MedGemma model on remote Ollama. Use the existing `/api/generate` request implementation and `.env` settings. Include selected blood-test values and reference ranges in the prompt. Send selected X-ray and EKG **original image bytes** encoded in base64; never send resized thumbnails. Keep network calls off the GUI thread.
5. **Show answer:** Display MedGemma's answer in the existing AI Response panel, with existing status/error handling. Preserve optional voice playback if already working. Indicate that the output is AI-generated and requires clinician review.

**Actions area:** Include an empty, labeled **Actions** section in the GUI as a placeholder. **No action buttons, tool workflows, routing, or agent logic** are to be implemented now. The existing **Send** button is outside this placeholder and must work.

## 3. Patient directory structure

The patient directory name **is the patient ID**. Example:

```text
medical-ai/
├── AGENTS.md
├── client-med-gemma-2w-data-v2(1).py   # original reference: do not overwrite
├── client_medgemma_patients.py          # ONLY new Python module
├── .env                                 # existing settings: do not overwrite
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

- Add support for optional `.env` key `PATIENT_RECORDS_DIR=patient_records`; if absent, use the `patient_records` folder beside the Python script. Relative paths are relative to the script directory.
- Discover only immediate patient directories, excluding hidden/system folders. Allow ordinary folder names with spaces and letters. Do not infer patient identity from image names or dated folders.
- In `xrays/` and `ekgs/`, display supported image formats already recognized by the original script. In `blood_tests/`, load `.json` files; support both the original `{"title": "...", "results": ...}` format and structured CBC-style JSON such as `{"panel": "CBC", "results": [{"test": "WBC", "value": 7.4, "unit": "10^3/uL"}]}`. Do not discard records simply because `title` is absent.
- Missing or empty modality folders are allowed; display a clear message, not an exception. Report invalid images/JSON without crashing the GUI.
- Do **not** create or require `patients.json`, `index.json`, `record.json`, patient metadata, or a database. Blood-test JSON files are the actual clinical-content inputs and must remain intact.
- Never mix records from different patients. When switching patients, clear selections and the displayed response. Capture a snapshot of the active patient and selected file paths before starting each request; prevent patient switching during an in-flight request or discard its result if the patient has changed. Validate that resolved file paths stay inside the selected patient directory, and avoid following symlinks outside it.

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
OLLAMA_URL=http://YOUR_REMOTE_SERVER:11434
MEDGEMMA_MODEL=medgemma1.5:latest
```

The values above are examples, not replacements for the user's current settings. **Never hardcode a server IP.** Do not install Qwen or MedGemma on the GUI computer. Prefer existing packages from the reference client (Pillow, pyttsx3 when supported); do not add unnecessary dependencies or build a new installer framework.

## 6. Acceptance checks

- One new `.py` file only, with detailed function docstrings and useful inline comments.
- Patient selector lists both `MohamedLjalil` and `P0002` when those directories exist.
- Selecting each patient loads only their own X-rays, EKGs, and blood tests.
- Switching patients clears all previous selections and the previous response.
- EKG gallery correctly loads images from `patient_records/<patient_id>/ekgs/` (not from legacy `ekg-data-MMDDYY`).
- A CBC JSON without a `title` field is displayed and can be selected.
- Doctor can select one or more images/blood records, type a question, and press Send.
- The outgoing MedGemma request contains only selected patient data, with original image resolution retained.
- MedGemma answer appears in the existing response panel. Handle missing data, bad JSON, unavailable Ollama, and timeouts gracefully.
- **Actions section is visible and EMPTY.** No extra Python modules, no Qwen, no MCP.
- Test syntax/imports and patient switching with synthetic files; test network payload using a mock or reachable Ollama server. Report explicitly whether the GUI was visually tested and whether live MedGemma was contacted.

## 7. Codex execution instructions

Read this `AGENTS.md` and the original Python script. Implement the single-file modification autonomously, making reasonable decisions without routine clarification. **Do not treat older Markdown files in the directory as additional implementation requirements** when they describe the previous multi-module Qwen/agent/MCP architecture; this document supersedes them. Preserve the original script, `.env`, and all patient data. At completion, report the one created Python file, test results, and any remaining issues.

Run the resulting program on Windows with:

```cmd
python client_medgemma_patients.py
```

For Ubuntu, use `python3 client_medgemma_patients.py` after installing Tkinter and the applicable Python dependencies. Voice support may be unavailable or platform-dependent.

**Safety:** Use synthetic or appropriately de-identified data for development. MedGemma output is research assistance, not an independently validated diagnosis. Plain HTTP to remote Ollama is not encrypted; do not send identifiable clinical data over an unsecured connection.
