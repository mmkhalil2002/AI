# Coding instructions: preserve the full AI Clinical Assistant Center GUI

## Authoritative implementation

`python/client_medgemma_patients_mcp.py` is the working GUI and must be repaired in place. It is a full Tkinter application, not a prototype. Do not replace it with a short script, a generic dashboard, a filename list, or Text-widget image rendering.

The approved interface is a full-width `Patient Selection and Date Range` bar above a responsive two-column workspace: approximately 35% left for the assistant controls and response, 65% right for patient records. The right side simultaneously shows framed horizontal X-ray tiles, a wide EKG gallery, and selectable blood-test records. The `python/README` describes the same behavior.

## Required functions and boundaries

Preserve the existing functions and call paths, including `patient_records_root`, `discover_patients`, `patient_snapshots`, `choose_snapshot`, `select_patient`, `select_snapshot`, `record_client`, `read_record_bytes`, `read_blood_record`, `patient_files`, `load_images`, `select_image`, `update_gallery_scrollbars`, `open_image_viewer`, `load_blood_tests`, `generate`, `run_request`, `send_question`, `finish_request`, and `close_window`.

Keep `mcp_client.py` separate and imported through `from mcp_client import MCPRecords`. Keep the three read-only modality-server entry points separate. The GUI owns Tk presentation, navigation, selection, prompt construction, Ollama calls, report review, voice, and shutdown. MCP owns stdio lifecycle and record retrieval only.

## Layout and behavior

- Top bar: Patient ID dropdown on the left; `Choose date & time`, loaded date/time, and active-patient label on the right.
- Left: red/orange `AI Clinical Assistant Center` title; `Your Question`; multiline prompt; `Send to MedGemma`, `Voice: OFF`, and `Close`; concise status; large scrollable `AI Response` output.
- Right: visible `XRAY`, `EKG/ECG`, and `BLOOD TEST` panels with actual snapshot date/time headings. Use pale borders/light surfaces, teal headings and date, purple time.
- Retain horizontal and vertical gallery scrollbars, framed selectable X-ray tiles, wide readable EKGs, image viewer zoom/navigation, and Tk image references/caches.
- Exclude `sources.csv` and unsupported files before tile construction. Never show loose metadata text in place of gallery tiles.
- Blood tests require red titles and explicit selection markers, then aligned Test, Value, Unit, Reference range, and Flag results. Preserve existing normalization and flag rules.
- `Choose date & time` opens the expandable snapshot dialog. Its Close closes only the dialog; main Close calls `close_window`.

## Safety, paths, and requests

Resolve `.env` and relative `PATIENT_RECORDS_DIR` beside the GUI. Honor OS environment precedence. Support direct modality folders, legacy `data/<modality>`, and dated snapshots. Pass the absolute root to child processes.

Only explicitly selected records may be sent. Preserve selected X-rays before EKGs and submit original image bytes, never thumbnails. Blood inference must use clinical result content, not display-only metadata. Keep immutable request snapshots, request IDs, stale-result rejection, UI-thread Tk updates, bounded queues/caches, cancellation semantics, and MCP cleanup.

Servers must preserve path confinement, hidden/link checks, modality checks, allowed extensions, and the 64 MiB limit. Their contracts are `list_records(relative_folder)` and `read_record(relative_path)` returning JSON fields `paths`, `base64`, or `record`.

## Validation

Compile changed Python. With de-identified records, exercise at least two patients, snapshot switching, gallery selection, blood selection/details, image viewer, Voice fallback, Send error/success, stale callbacks, and Close cleanup. Confirm sources.csv is excluded and a populated window preserves the two-column gallery layout. Do not run installers, uninstallers, remote inference, or use real patient records solely for validation.
