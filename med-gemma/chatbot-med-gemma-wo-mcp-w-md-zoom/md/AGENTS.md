# Maintenance instructions: single-file MedGemma GUI

## Scope and paths

The project root is the parent of `md/`. Keep application logic in `python/client_medgemma_patients.py`, Markdown in `md/`, and launchers in `config/`. `python/README.md` mirrors `md/README.md` for users working beside the script.

Read `README.md`, `VERIFICATION.md`, and the current Python source before maintenance. Treat the current source and the user's latest instructions as the implementation baseline. If rebuilding without source, use this file and README as the behavioral specification; they are not an executable generator or an exact source backup. If the original reference `python/client-med-gemma-2w-data-v2(1).py` exists, preserve it; do not invent a missing reference.

Deliver one application Python file. Do not introduce Qwen, agents, MCP, tool routing, databases, web services, or extra application modules. Preserve `.env`, source records, original reference files, and existing launchers. Never create, move, delete, or rewrite patient records as an application side effect. Do not install models automatically or hardcode server addresses.

## Configuration and isolation

Resolve the script directory with `Path(__file__).resolve().parent`. Read `.env` there; environment variables take precedence. Resolve relative `PATIENT_RECORDS_DIR` beside the script, defaulting to `patient_records`. Validate the root before constructing Tk; a missing, non-directory, or unreadable root is fatal and is never created. An existing empty root and missing patient modality folders are separate, nonfatal cases.

Use the existing Ollama configuration and standard-library HTTP implementation. Preserve Python 3.9+, Tkinter, Pillow image previews, optional pyttsx3, background work, cancellation, completion validation, final-answer filtering, and error handling. Never perform HTTP or heavy image work on the Tk event thread.

Search by First name, Last name, and validated eight-digit `MMDDYYYY` DOB, matching `<first>-<last>-<MMDDYYYY>` folders case-insensitively. Do not load a patient at startup. Support `<patient>/<date>/<time>/{xrays,ekgs,blood_tests}/` plus the legacy `<patient>/data/` layout. Load the newest snapshot after search. Choose date & time loads one snapshot without merging different times. The picker shows dates and time children, with Load selected date & time and Close controls. Preserve source path confinement, patient isolation, and frozen request inputs. Prevent patient switching in flight or discard stale results.

## Current interface

- Top: patient search, Zoom, X-ray On/Off, EKG On/Off, Blood Test On/Off, Choose date & time, and active patient. Window toggles sit beside Zoom, are green On and red Off, and preserve selections when hidden. Active patient uses a teal heading and purple details on two lines. Hide the redundant Loaded label.
- Left: AI Clinical Assistant Center, a blank question input, Ask AI, Voice, Close GUI, and one AI Response area. No Actions, visible model/server controls, or visible original/draft tabs. Existing advanced review/export/history helpers may remain hidden.
- Right: selected visible XRAY, EKG/ECG, and BLOOD TEST panels stack vertically and divide the available records area equally. One fills the area; two get half each; three get one-third each. All can be hidden. Keep independent scrolling and selection. Modality headings show the loaded snapshot date and time, not date ranges.
- Main blood-test rows remain aligned in a fixed-width font. Preserve title-less structured CBC and mapping-style JSON support. The current blood-test loader reads JSON, not scanned lab images.

## Independent analysis

Use one independent `/api/generate` call per selected image, sequentially, sending only that original image's base64 bytes, modality, and the doctor's question. Do not send thumbnails, other images, or blood data in that call. Analyze selected blood JSON separately, grouping selected records by source file with one text-only call per file. With no selected records, preserve text-only question handling.

Request clear Findings and Limitations for each input without hidden reasoning or unsupported claims. Label aggregate response sections with modality and source filename stem, omitting extensions. Keep each completed response associated with its exact source path, not its display name. Do not merge or compare image findings automatically.

Publish individual responses to matching zoom windows on the Tk thread as they complete. Finalize the aggregate report only after all calls succeed. Failed/cancelled runs are not completed aggregate reports; earlier individual responses may remain visible. Cancel prevents remaining calls and ignores stale callbacks. Retry explicitly uses current inputs. Preserve frozen patient references, source bytes, and existing persistence safeguards.

## Zoom and response panels

Zoom starts disabled with the label Zoom out. Enabling changes it to blue Zoom in. Apply the same selection-window lifecycle to X-rays, EKGs, and blood-test JSON records: open selected sources, close deselected sources, and close selection windows when zoom is disabled or the patient changes.

Image windows display natural-size images with scrollbars and a maximum 1400 by 900 window. Keep dimensions in the title, with no duplicate middle image-size label. Use a large 18 pt bold white filename heading on dark teal. Preserve the separate detail viewer's fit/zoom/navigation behavior.

Blood JSON opens in an enlarged formatted record viewer, not raw JSON text. Display metadata plus each test's result, unit, reference range, and status. Use 15 pt body text, 16 pt bold test headings, and 18 pt record headings; highlight explicit High/Low status in red. Do not invent ranges or change source values.

Each viewer has a bottom scrollable response panel, approximately six lines tall, without a Diagnosis label. Populate only the corresponding source's response, including when opening after generation. Detail-viewer navigation must switch the response too. Clear cached responses on a new request and show a waiting message where no corresponding response exists.

## Validation and delivery

Run syntax and diff checks. For request changes, use synthetic or mocked inputs to check one-image payloads, blood isolation, source-path matching, cancellation, and stale-result rejection. Visually check layout when possible and report any unverified behavior honestly. Do not claim clinical accuracy or live service testing from mocks. Record new results separately from historical checks in VERIFICATION.md.

Run from the project root with `python python/client_medgemma_patients.py` on Windows or `python3 python/client_medgemma_patients.py` on Ubuntu, or use the named launcher in config/. Derive project paths from the files rather than an installation-specific location. Keep both READMEs consistent with implemented behavior.
