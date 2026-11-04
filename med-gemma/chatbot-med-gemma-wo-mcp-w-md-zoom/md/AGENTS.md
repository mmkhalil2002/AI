# Maintenance instructions: single-file MedGemma GUI

## Scope and paths

The project root is the parent of `md/`. Keep application logic in `python/client_medgemma_patients.py`, Markdown in `md/`, and launchers in `config/`. `python/README.md` mirrors `md/README.md` for users working beside the script.

Read `README.md`, `VERIFICATION.md`, and the current Python source before maintenance. Treat the current source and the user's latest instructions as the implementation baseline. If rebuilding without source, use this file and README as the behavioral specification; they are not an executable generator or an exact source backup. If the original reference `python/client-med-gemma-2w-data-v2(1).py` exists, preserve it; do not invent a missing reference.

Deliver one application Python file. Do not introduce Qwen, agents, MCP, tool routing, databases, web services, or extra application modules. Preserve `.env`, source records, original reference files, and existing launchers. Never create, move, delete, or rewrite patient records as an application side effect. Do not install models automatically or hardcode server addresses.

## Configuration and isolation

Resolve the script directory with `Path(__file__).resolve().parent`. Read `.env` there; environment variables take precedence. Resolve relative `PATIENT_RECORDS_DIR` beside the script, defaulting to `patient_records`. Validate the root before constructing Tk; a missing, non-directory, or unreadable root is fatal and is never created. An existing empty root and missing patient modality folders are separate, nonfatal cases.

Use the existing Ollama configuration and standard-library HTTP implementation. Preserve Python 3.9+, Tkinter, Pillow image previews, optional pyttsx3, background work, cancellation, completion validation, final-answer filtering, and error handling. Never perform HTTP or heavy image work on the Tk event thread.

Search by First name, Last name, and validated eight-digit `MMDDYYYY` DOB, matching `<first>-<last>-<MMDDYYYY>` folders case-insensitively. Do not load a patient at startup. Support `<patient>/<date>/<time>/{xrays,ekgs,blood_tests}/` plus the legacy `<patient>/data/` layout. Load the newest snapshot after search. The Date & Time button loads one snapshot without merging different times. The picker shows dates and time children, with Load selected date & time and Close controls. Preserve source path confinement, patient isolation, and frozen request inputs. Prevent patient switching in flight or discard stale results.

## Required interface — four dynamically sized record panels (target design)

- Keep the existing **left AI Clinical Assistant Center** with blank question input, Ask AI, Voice, Close GUI, and one AI Response area. Keep the **right medical-record display**. Preserve existing overall proportions and header controls unless necessary to fit the new menus. Do not introduce Actions, visible model/server controls, or visible original/draft tabs.
- The right medical-record display has **four vertically stacked panels in this order**: **X-RAY**, **CT SCAN**, **EKG/ECG**, **BLOOD TEST**. Each enabled panel gets **exactly 1/N of the available record-display height**, where N is the number enabled. For four enabled: 25% each; three: 33.33% each; two: 50% each; one: 100%; zero: show a neutral empty records area. Recalculate on toggle and window resize. Each panel has its own scrolling, selection, and snapshot date/time heading.
- Replace the separate X-ray/EKG/Blood Test top-level On/Off buttons with one **Display** dropdown menu containing four independent checkbutton-style ON/OFF toggles: **X-ray**, **CT Scan**, **EKG/ECG**, **Blood Test**. All start ON. Use clear green ON / red OFF indicators where the toolkit supports it. Hiding a panel must not clear its selection or records; hidden records are not implicitly selected or submitted. Switching patients or snapshots refreshes all four panels consistently.
- Add a **Zoom** dropdown menu containing **five independent ON/OFF toggle options**: **(1) Selection Zoom**, **(2) X-ray Zoom**, **(3) CT Scan Zoom**, **(4) EKG/ECG Zoom**, **(5) Blood Test Zoom**. Toggle state must be explicit and synchronized with its viewer window(s). Turning an option OFF closes only its corresponding windows. Patient/snapshot changes close stale viewer windows and prevent cross-patient content.
- **Selection Zoom** retains the existing behavior: it opens only selected record items. The **Split/Merge** button applies **only to Selection Zoom**; Split opens separate selected-item viewers, Merge navigates selected items in a shared viewer. Preserve selection/deselection lifecycle, per-record AI responses, and blood-test formatted presentation.
- Each modality-specific Zoom option opens a **separate, independently controlled viewer window** showing **all loaded records of that modality for the active patient and selected snapshot**, regardless of whether individual records are selected or its main panel is hidden. Do not combine modalities in one viewer. For image modalities, show original/natural pixel size initially (scrollbars as needed), not thumbnails. Provide **Previous**, **Next**, **+**, **−**, **Fit**, and **Reset** controls. Previous/Next navigate all records of that modality; +/− adjust zoom; Fit fits the current item to the viewer; Reset restores 100% natural pixel size. Preserve source path and correct per-item response as navigation changes. Empty modality shows a friendly no-records message. Selection Zoom Split/Merge must not affect these viewers.
- **Blood Test Zoom** uses the same Previous/Next/+/−/Fit/Reset controls on all loaded blood-test JSON records. Present formatted lab fields, not a raw JSON dump. +/− adjust text scaling; Fit sizes text to available width legibly; Reset restores the default text sizes. Preserve original results, units, ranges, and High/Low status without inventing values.
- Keep patient search by first name, last name, and validated `MMDDYYYY` DOB. Keep **Date & Time** immediately after Search, active patient heading (teal) and details (purple), and the hidden redundant date/time label. Do not load a patient on startup. Existing blood rows remain aligned using fixed-width formatting.

## CT scan record support

- Add the **`ct_scan`** folder as a fourth modality under each patient snapshot: `<patient>/<date>/<time>/ct_scan/`. Also support `<patient>/data/ct_scan/` in the legacy layout. Use this exact canonical folder name; do not silently substitute `ct_scans` or `ct_scsan`.
- Read CT image files from the patient snapshot using the same safe path confinement and supported raster-image formats as X-rays. Preserve original image bytes for analysis and image dimensions for display. Do not modify, move, or generate source files. DICOM support is optional and must not be claimed unless implemented and tested.
- Include CT records in loading, date/time selection, per-modality selection, independent analysis, viewer response matching, and error handling. CT images must remain associated with the correct patient and snapshot. Missing CT folders are nonfatal.

## Independent analysis

Use one independent `/api/generate` call per selected X-ray, CT scan, or EKG image, sequentially, sending only that original image's base64 bytes, modality, and the doctor's question. Do not send thumbnails, other images, or blood data in that call. Analyze selected blood JSON separately, grouping selected records by source file with one text-only call per file. With no selected records, preserve text-only question handling.

Request clear Findings and Limitations for each input without hidden reasoning or unsupported claims. Label aggregate response sections with modality and source filename stem, omitting extensions. Keep each completed response associated with its exact source path, not its display name. Do not merge or compare image findings automatically.

Publish individual responses to matching zoom windows on the Tk thread as they complete. Finalize the aggregate report only after all calls succeed. Failed/cancelled runs are not completed aggregate reports; earlier individual responses may remain visible. Cancel prevents remaining calls and ignores stale callbacks. Retry explicitly uses current inputs. Preserve frozen patient references, source bytes, and existing persistence safeguards.

## Zoom and response panels — implementation details

- Implement the five Zoom dropdown toggles and their lifecycles exactly as described under **Required interface**. Selection Zoom is initially OFF; modality-specific Zoom options are initially OFF. A dropdown option is a toggle, not a one-shot command. Closing a viewer using its window Close button updates its corresponding toggle OFF; for Selection Zoom, closing one Split child removes that child without incorrectly disabling other selected-item windows.
- The Selection Zoom Split viewer provides +, −, Fit, Reset for each selected image; Merge provides Previous, Next, +, −, Fit, Reset. For blood records, preserve formatted record views. Fit shows the whole image; Reset restores 100% original pixels. Never use Split/Merge for X-ray Zoom, CT Scan Zoom, EKG Zoom, or Blood Test Zoom.
- Image windows show natural-size images with scrollbars and a maximum 1400 by 900 initial window. Show dimensions in the title and a large 18 pt bold white filename heading on dark teal, without a duplicate middle image-size label.
- Blood JSON viewers show metadata and each test's result, unit, reference range, and status; default body text 15 pt, test headings 16 pt bold, record headings 18 pt. Highlight explicitly High/Low status in red. Do not invent reference ranges or alter source values.
- Each viewer has a bottom scrollable response panel about six lines tall, with no Diagnosis label. Display only the response associated with the exact current source path, including after reopening or navigating. A new request clears cached responses; if none exists show a waiting/no-response message.

## Validation and delivery

Run syntax and diff checks. For request changes, use synthetic or mocked inputs to check one-image payloads (including CT), blood isolation, source-path matching, cancellation, and stale-result rejection. Visually check all Display toggle combinations, five Zoom toggle states, Split/Merge isolation, viewer navigation, and layout when possible and report any unverified behavior honestly. Do not claim clinical accuracy or live service testing from mocks. Record new results separately from historical checks in VERIFICATION.md.

Run from the project root with `python python/client_medgemma_patients.py` on Windows or `python3 python/client_medgemma_patients.py` on Ubuntu, or use the named launcher in config/. Derive project paths from the files rather than an installation-specific location. Keep both READMEs consistent with implemented behavior.

## Implementation status

This document specifies the requested **target behavior**; it does not claim these changes are already implemented or tested. Historical VERIFICATION.md results must not be presented as tests of the new CT/display/zoom design.
