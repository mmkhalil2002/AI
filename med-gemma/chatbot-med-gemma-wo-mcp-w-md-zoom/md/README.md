# MedGemma patient GUI

The application is `python/client_medgemma_patients.py`, a single-file Tkinter client that sends selected records directly to the Ollama server entered in the connection controls. No Qwen, MCP, agents, database, or additional application modules are used.

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

Keep the existing `python/.env`. Settings include `MEDGEMMA_MODEL`, optional `PATIENT_RECORDS_DIR`, `NUM_PREDICT`, `TEMPERATURE`, `REPEAT_PENALTY`, `REPEAT_LAST_N`, `THINK`, and `GENERATION_TIMEOUT`. The model must already exist on the selected Ollama server; the GUI does not install models. `OLLAMA_URL` in `.env` or the process environment is ignored.

### IP address connection controls

The IP address field and yellow **Connect** button appear directly beside **Date & Time** in the main window. The address starts blank. Enter only the server IP address; the app constructs the URL by adding HTTP and port `11434` automatically and contacts the server. On success, the field shows only the IP address. Explicit ports and full URLs remain supported. For example, `192.168.1.20` becomes `http://192.168.1.20:11434`.

If the field is blank or contains only spaces when **Connect** is pressed, the app attempts `http://127.0.0.1:11434`. On success, it displays the default IP `127.0.0.1` and shows green **Disconnect**. On failure, it leaves the field blank and shows red **Fail**. The address comes from this field, independently of `.env`, and is not saved between app restarts.

The connection check runs in the background against `/api/tags`. The button shows yellow **Connecting...** during the check, green **Disconnect** when the server responds successfully, or red **Fail** with an error message if the check fails. A reachable server can still have a missing model; model readiness is reported separately. The address is read-only while connected. Click **Disconnect** to cancel any active analysis, clear the IP address field, and restore yellow **Connect**. Connect again before submitting another question. Disconnect clears the client's selected endpoint; Ollama itself keeps running. Editing an unconnected address resets the button to yellow **Connect**. Connection messages appear below the header controls. The server is checked only when Connect is pressed.

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

The legacy `<patient>/data/{xrays,ct_scan,ekgs,blood_tests}/` layout is also supported. Images support PNG, JPEG, BMP, GIF, WebP, and TIFF. Blood tests support structured results lists and mapping-style results, with either a title or panel name. Blood tests currently load JSON records, not scanned lab images.

### Record window definitions

The first setting in `python/.env` is:

```env
# Folder names in panel order; restart after changing.
WINDOW_DEFINTIONS=xrays,ct_scan,ekgs,blood_tests
```

Use exactly `WINDOW_DEFINTIONS`. Enter a plain sequence of up to four names; no mapping syntax is required. Each name is both the exact record-folder name and the label used for its panel, Display button, Zoom button, and Zoom window. Names must match the section folders in `patient_records`, including case. The first three positions retain the existing image-panel behavior; the fourth retains JSON lab rendering. Clinical analysis retains the existing modality roles for these positions.

For example, `WINDOW_DEFINTIONS=scans,slices,traces,labs` reads folders named `scans`, `slices`, `traces`, and `labs`, and uses those names throughout the controls. Names do not rename files or folders on disk.

A shorter list omits trailing panels and their corresponding buttons. Empty comma-separated positions omit middle panels without shifting later slots: `WINDOW_DEFINTIONS=xrays,,ekgs,blood_tests` omits the second panel. An empty value omits every panel. Omitted sections are not loaded or submitted. Duplicate names, path separators, dot-prefixed names, and more than four positions are rejected. Missing `WINDOW_DEFINTIONS` defaults to the four standard names. Environment variables take precedence over `.env`.

All included panels start ON. Their Display buttons remain interactive, showing green ON or brown OFF. The old `XRAY_ON`, `CTSCAN_ON`, `EKG_ON`, and `BLOOD_TEST_ON` settings were removed from `.env` and are ignored. Restart after configuration changes.

### Patient search labels

Immediately after `WINDOW_DEFINTIONS` in `python/.env`, configure the search labels:

```env
FIRST_NAME_LABEL=First name
LAST_NAME_LABEL=Last name
DOD_LABEL=DOD (MMDDYYYY)
```

These settings change the three field captions only. Missing or blank values use the defaults shown above. Environment variables take precedence. Restart to apply changes. The date field continues to search the date component of the patient folder using a valid `MMDDYYYY` date; changing its caption does not change matching or validation.

## Patient and display controls

1. Enter any combination of First name, Last name, and DOB, then select **Search**. At least one field is required; blank fields are ignored. Supplied names match exactly, case-insensitively, with surrounding spaces trimmed. If DOB is supplied, it must be a valid eight-digit `MMDDYYYY` date. One match loads directly; multiple matches open a patient chooser requiring an explicit selection (Load selected patient, Enter, or double-click). Cancel keeps the current patient. No match or invalid input leaves the current patient unchanged. Searching is blocked during active analysis. The newest snapshot of the selected patient loads; **Date & Time** immediately after Search lets you choose one other snapshot. Do not merge records across different times. The search area has no Patient Date and Time heading.
2. The **left** side retains the AI Clinical Assistant Center, blank question input, Ask AI, Voice, Close GUI, and one AI Response panel. The left question and response column initially occupies **25%** of the workspace width; the **right medical-record display occupies 75%**. The divider remains draggable.
3. With all four sections defined, nine independent toggle buttons appear beneath the Search row, with extra spacing between the Selection, modality-Zoom, and Display clusters but without group labels. **Zoom Select Out/In** and Merge/Split appear together; the other four Zoom toggles appear next; then the four Display toggles (**X-ray**, **CT Scan**, **EKG/ECG**, and **Blood Test**). The Zoom Select and modality controls initially read **Out**; after activation, they read **In**. Zoom Select and default Merge are orange initially and dark blue when active. The modality controls use yellow while inactive and purple while active. Only sections included in `WINDOW_DEFINTIONS` have Display and Zoom buttons; labels match the configured folder names. Included Display toggles start ON, show **ON** in green, and show **OFF** in brown when manually switched off. With all panels on, the record display uses a **2-by-2 grid**: **X-ray | CT Scan** in the top row and **EKG/ECG | Blood Test** in the bottom row. When a panel is switched off, the remaining visible panels stack vertically and divide the complete records area equally: three panels each use one-third of the height, two each use one-half of the height, and one uses the full area. Hiding a panel preserves its loaded data and selections. Zero enabled shows an empty records area.
4. Each panel scrolls independently, supports selecting/deselecting individual items, and shows the active snapshot date and time. The active patient uses a teal heading and purple details. Blood-test table rows remain aligned in fixed-width columns.
   X-ray and CT Scan galleries show two images per row with all four panels visible, and four images per row when one to three panels are visible. EKG remains one image per row.
5. The five **Zoom** toggles report ON/OFF state. **Merge/Split applies only to Selection Zoom** and defaults to Merge.

## CT scan support

The second configured name (default **`ct_scan`**) identifies the CT folder under each patient snapshot or the legacy `data/` folder. The CT section loads supported raster images and preserves the original image bytes for MedGemma requests. CT scans belong to the same active patient and snapshot as the other records; a missing CT folder is allowed. DICOM support is not assumed unless explicitly implemented.

## Independent responses

Select records, enter a question, then click Ask AI. The app freezes the selected patient, paths, question, and blood data before background work begins.

- Each selected X-ray, CT scan, or EKG image is sent in its own independent `/api/generate` request with exactly one original image, its modality, and the doctor's question. Thumbnails are never uploaded. Image calls contain no blood-test data or other selected images.
- Selected blood records are grouped by source JSON file and each file is analyzed separately, without image attachments.
- Calls run sequentially. Responses appear in separate sections such as `X-RAY - image-name`, `CT SCAN - image-name`, `ECG/EKG - image-name`, and `BLOOD TEST - record-name`, without filename extensions. Each response requests Findings and Limitations.
- Each completed image or blood response updates its matching open zoom window as it arrives. The main combined report is finalized after all calls succeed. If a later call fails or is cancelled, earlier responses may remain in zoom windows, but no completed aggregate report is created.
- Cancel stops accepting results and closes the active connection where possible. It cannot guarantee that Ollama stops computing. Remaining calls are not started after cancellation. Retry uses the current prompt and selections.

Only selected active-patient records are sent. Patient changes are prevented during a request and obsolete callbacks are ignored. Source image and JSON files are not modified.

## Zoom toggles and viewer windows

All five Zoom options begin OFF and act as independent ON/OFF toggles. The title-bar Close (X) action is disabled in every zoom viewer; use its corresponding Zoom toggle to close a modality-specific viewer. Switching patients or snapshots closes stale viewers.

| Zoom option | Viewer behavior |
|---|---|
| **1. Selection Zoom** | Existing selected-record zoom only; **Split** opens separate windows for selected items and **Merge** navigates selected images in one viewer; selected blood files retain separate formatted windows. Deselection closes its matching Split window. |
| **2. X-ray Zoom** | One separate viewer for **all loaded X-ray images** of the active patient/snapshot, even if unselected. |
| **3. CT Scan Zoom** | One separate viewer for **all loaded CT images/slices** of the active patient/snapshot, even if unselected. |
| **4. EKG/ECG Zoom** | One separate viewer for **all loaded EKG images** of the active patient/snapshot, even if unselected. |
| **5. Blood Test Zoom** | One separate formatted viewer for **all loaded blood-test records** of the active patient/snapshot, even if unselected. |

The four modality-specific viewers each provide **Previous**, **Next**, **+**, **−**, **Fit**, and **Actual Size** for images. Previous/Next navigate records within that modality only. Image viewers start at original pixel size, support scrollbars, use Fit to display the whole current image, and use Actual Size to restore 100% scale; the shared image viewer also states the current actual-size percentage. Blood Test Zoom displays formatted results, units, ranges, and statuses; +/− adjust text size, Fit makes the record readable within available width, and Reset restores default fonts. **Split/Merge does not affect modality-specific viewers.**

Image viewers retain the filename heading, image dimensions in the window title, and a maximum initial size of 1400 × 900. Blood test viewers use readable formatted sections (15 pt body, 16 pt test headings, 18 pt record headings, explicit High/Low in red). Each viewer has a scrollable approximately six-line AI response area showing only the currently displayed record's response, with no Diagnosis label. Responses follow the record during Previous/Next navigation and are cleared when a new request begins.

## Errors and verification

Missing models, unavailable Ollama, invalid files, timeouts, and incomplete output are reported through existing status/error handling. The model setting comes from `.env`; the endpoint comes from the connection controls, with localhost used for blank input and the default IP displayed after a successful connection. Full-resolution analysis and sequential requests may take time.

See `md/VERIFICATION.md` for current checks and historical results. Syntax and mocked checks do not establish clinical accuracy. AI output requires clinician review.

## Implementation status and Codex handoff

The current Python source includes CT support, a responsive record grid, Display and Zoom menus, and the initial 25%/75% workspace split. Synthetic GUI checks passed for earlier Display combinations and independent modality viewers. User-reported crashes remain unresolved; startup checks did not reproduce them. Full visual review and live record generation remain unverified. See the new verification entry separately from historical results. Update the single Python application in place, preserve existing behavior, do not modify patient data; preserve `.env` except for explicitly requested configuration changes, and record new tests honestly. The accompanying `AGENTS.md` is the consolidated implementation specification.
