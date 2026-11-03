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
        xrays/        # supported image files
        ekgs/         # supported image files
        blood_tests/  # JSON lab records
```

The legacy `<patient>/data/{xrays,ekgs,blood_tests}/` layout is also supported. Images support PNG, JPEG, BMP, GIF, WebP, and TIFF. Blood tests support structured results lists and mapping-style results, with either a title or panel name. Blood tests currently load JSON records, not scanned lab images.

## Patient and panel controls

1. Enter First name, Last name, and a real eight-digit DOB (`MMDDYYYY`), then click Search. Folder matching is case-insensitive. No patient is loaded at startup.
2. The newest available snapshot loads after a successful search. Use the Date & Time button, immediately after Search, to load another snapshot; records from different snapshots are not merged. Dates and times appear in the modality headings. The redundant Loaded label is hidden.
3. The active patient display uses two lines: a teal heading and purple patient details.
4. Beside Zoom, use X-ray On/Off, EKG On/Off, and Blood Test On/Off to show or hide panels. On is green; Off is red. All start On. Hiding preserves contents and selections.
5. Visible panels stack vertically and share the records area equally: one fills it, two split it in half, and three each receive one-third. All panels can be hidden.

The left assistant area contains a blank question input, Ask AI, Voice, Close GUI, and one visible AI Response area. Server/model controls, Actions, and separate original/draft tabs are hidden in the streamlined layout. Export, review, and history helpers remain in the source but their controls are hidden in this layout.

## Independent responses

Select records, enter a question, then click Ask AI. The app freezes the selected patient, paths, question, and blood data before background work begins.

- Each selected image is sent in its own independent `/api/generate` request with exactly one original image, its modality, and the doctor's question. Thumbnails are never uploaded. Image calls contain no blood-test data or other selected images.
- Selected blood records are grouped by source JSON file and each file is analyzed separately, without image attachments.
- Calls run sequentially. Responses appear in separate sections such as `X-RAY - image-name`, `ECG/EKG - image-name`, and `BLOOD TEST - record-name`, without filename extensions. Each response requests Findings and Limitations.
- Each completed image or blood response updates its matching open zoom window as it arrives. The main combined report is finalized after all calls succeed. If a later call fails or is cancelled, earlier responses may remain in zoom windows, but no completed aggregate report is created.
- Cancel stops accepting results and closes the active connection where possible. It cannot guarantee that Ollama stops computing. Remaining calls are not started after cancellation. Retry uses the current prompt and selections.

Only selected active-patient records are sent. Patient changes are prevented during a request and obsolete callbacks are ignored. Source image and JSON files are not modified.

## Zoom windows

The initial Zoom out label means selection windows are disabled. Clicking it changes the button to blue Zoom in and enables them. Use the adjacent Split/Merge button to choose the layout: Split gives every selected image its own window; Merge puts all selected X-ray and EKG images into one window with Previous, Next, +, −, Fit, and Reset. Previous and Next move through the selected images; +/− change the current image scale; Fit shows the complete image; Reset restores its original 100% pixel size. Split image windows provide +, −, Fit, and Reset independently. Selected blood-test JSON files continue to open in their readable record windows. Deselecting closes the matching window, and disabling zoom closes selection windows.

Image windows show a large bold white filename label on dark teal. Dimensions remain in the window title; the duplicate size label in the middle has been removed. Images remain at natural pixel size with scrollbars, and image windows are capped at 1400 by 900 pixels. A separate detail viewer also supports fit, zoom, and previous/next navigation.

Blood-test windows format each test into readable sections with result, unit, reference range, and status, plus record metadata. Text is 15 pt, test headings 16 pt, and record headings 18 pt. High/Low status is red. The source JSON remains unchanged.

A compact, scrollable response panel sits at the bottom of each viewer. It has no Diagnosis heading, uses an approximately six-line height, and shows only the matching record's response. A window opened after completion also shows the cached response. Navigating the detail viewer changes the response to match the displayed image. Starting a new request clears previous cached responses and displays a waiting message.

## Errors and verification

Missing models, unavailable Ollama, invalid files, timeouts, and incomplete output are reported through existing status/error handling. Model and endpoint settings come from `.env`; no server is silently substituted. Full-resolution analysis and sequential requests may take time.

See `md/VERIFICATION.md` for current checks and historical results. Syntax and mocked checks do not establish clinical accuracy. AI output requires clinician review.
