# MedGemma patient GUI

The application is `python/client_medgemma_patients.py`, a single-file Tkinter client that sends selected records directly to the Ollama server configured in `python/.env`. No Qwen, MCP, agents, database, or additional application modules are used.

## Run

From the project root on Windows:

```powershell
python python/client_medgemma_patients.py
```

On Ubuntu:

```bash
python3 python/client_medgemma_patients.py
```

Alternatively use `config/run.bat` or `bash config/run.sh`. Python 3.9+, Tkinter, and Pillow are required for image viewing; pyttsx3 is optional for voice. A graphical desktop is required. Restart the application after source changes.

## Configuration and records

Keep the existing `python/.env`. Settings include `OLLAMA_URL`, `MEDGEMMA_MODEL`, optional `PATIENT_RECORDS_DIR`, `NUM_PREDICT`, `TEMPERATURE`, `REPEAT_PENALTY`, `REPEAT_LAST_N`, `THINK`, and `GENERATION_TIMEOUT`. The model must already exist on the configured Ollama server; the GUI does not install models.

The script resolves `.env` and relative patient-record paths beside itself, independently of the working directory. The default records root is `python/patient_records`. Startup stops if the root is missing or unreadable; the app never creates it. Empty or missing modality folders inside a patient are nonfatal.

```text
patient_records/
  <first>-<last>-<MMDDYYYY>/
    <date>/
      <time>/
        xrays/        # X-ray image files
        ct_scan/      # CT scan image files
        ekgs/         # supported image files
        blood_tests/  # JSON lab records
```

The legacy `<patient>/data/{xrays,ekgs,blood_tests}/` layout is also supported. Images support PNG, JPEG, BMP, GIF, WebP, and TIFF. Blood tests support structured results lists and mapping-style results, with either a title or panel name. Blood tests currently load JSON records, not scanned lab images.

## Patient and display controls (new target design)

1. Enter First name, Last name, and an eight-digit DOB (`MMDDYYYY`), then select **Search**. The newest snapshot loads; **Date & Time** immediately after Search lets you choose one other snapshot. Do not merge records across different times.
2. The **left** side retains the AI Clinical Assistant Center, blank question input, Ask AI, Voice, Close GUI, and one AI Response panel. The **right** side holds the medical records.
3. The **Display** dropdown has four independent ON/OFF toggles: **X-ray**, **CT Scan**, **EKG/ECG**, and **Blood Test**. All are initially ON. The enabled panels appear in that order, **stacked vertically**, and divide the record-display height evenly: four = 25% each, three = one-third each, two = 50% each, one = 100%. Zero enabled shows an empty records area. Hiding a panel preserves its loaded data and selections.
4. Each panel scrolls independently, supports selecting/deselecting individual items, and shows the active snapshot date and time. The active patient uses a teal heading and purple details. Blood-test table rows remain aligned in fixed-width columns.
5. The **Zoom** dropdown contains five independent ON/OFF toggles. See below. **Split/Merge applies only to Selection Zoom.**

## CT scan support

Use the exact folder name **`ct_scan`** under each patient snapshot, or under the legacy `data/` folder. The CT section loads supported raster images and preserves the original image bytes for MedGemma requests. CT scans belong to the same active patient and snapshot as the other records; a missing CT folder is allowed. DICOM support is not assumed unless explicitly implemented.

## Independent responses

Select records, enter a question, then click Ask AI. The app freezes the selected patient, paths, question, and blood data before background work begins.

- Each selected X-ray, CT scan, or EKG image is sent in its own independent `/api/generate` request with exactly one original image, its modality, and the doctor's question. Thumbnails are never uploaded. Image calls contain no blood-test data or other selected images.
- Selected blood records are grouped by source JSON file and each file is analyzed separately, without image attachments.
- Calls run sequentially. Responses appear in separate sections such as `X-RAY - image-name`, `CT SCAN - image-name`, `ECG/EKG - image-name`, and `BLOOD TEST - record-name`, without filename extensions. Each response requests Findings and Limitations.
- Each completed image or blood response updates its matching open zoom window as it arrives. The main combined report is finalized after all calls succeed. If a later call fails or is cancelled, earlier responses may remain in zoom windows, but no completed aggregate report is created.
- Cancel stops accepting results and closes the active connection where possible. It cannot guarantee that Ollama stops computing. Remaining calls are not started after cancellation. Retry uses the current prompt and selections.

Only selected active-patient records are sent. Patient changes are prevented during a request and obsolete callbacks are ignored. Source image and JSON files are not modified.

## Zoom dropdown and viewer windows (new target design)

All five Zoom options begin OFF and act as independent ON/OFF toggles. Closing a modality-specific viewer turns its option OFF. Switching patients or snapshots closes stale viewers.

| Zoom option | Viewer behavior |
|---|---|
| **1. Selection Zoom** | Existing selected-record zoom only; **Split** opens separate windows for selected items and **Merge** navigates selected items in one viewer. Deselection closes its matching Split window. |
| **2. X-ray Zoom** | One separate viewer for **all loaded X-ray images** of the active patient/snapshot, even if unselected. |
| **3. CT Scan Zoom** | One separate viewer for **all loaded CT images/slices** of the active patient/snapshot, even if unselected. |
| **4. EKG/ECG Zoom** | One separate viewer for **all loaded EKG images** of the active patient/snapshot, even if unselected. |
| **5. Blood Test Zoom** | One separate formatted viewer for **all loaded blood-test records** of the active patient/snapshot, even if unselected. |

The four modality-specific viewers each provide **Previous**, **Next**, **+**, **−**, **Fit**, and **Reset**. Previous/Next navigate records within that modality only. Image viewers start at original pixel size, support scrollbars, use Fit to display the whole current image, and use Reset to restore 100% scale. Blood Test Zoom displays formatted results, units, ranges, and statuses; +/− adjust text size, Fit makes the record readable within available width, and Reset restores default fonts. **Split/Merge does not affect modality-specific viewers.**

Image viewers retain the filename heading, image dimensions in the window title, and a maximum initial size of 1400 × 900. Blood test viewers use readable formatted sections (15 pt body, 16 pt test headings, 18 pt record headings, explicit High/Low in red). Each viewer has a scrollable approximately six-line AI response area showing only the currently displayed record's response, with no Diagnosis label. Responses follow the record during Previous/Next navigation and are cleared when a new request begins.

## Errors and verification

Missing models, unavailable Ollama, invalid files, timeouts, and incomplete output are reported through existing status/error handling. Model and endpoint settings come from `.env`; no server is silently substituted. Full-resolution analysis and sequential requests may take time.

See `md/VERIFICATION.md` for current checks and historical results. Syntax and mocked checks do not establish clinical accuracy. AI output requires clinician review.

## Implementation status and Codex handoff

This README describes the **requested target design**, not proof that the current Python GUI already implements the four-panel layout or new zoom options. The existing source and historical `VERIFICATION.md` must be checked before marking features complete. Update the single Python application in place, preserve existing behavior, do not modify patient data or `.env`, and record new tests honestly. The accompanying `AGENTS.md` is the consolidated implementation specification.
