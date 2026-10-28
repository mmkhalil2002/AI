# Instructions for coding agents

## Scope and structure

These instructions cover this MedGemma project. Preserve the existing `python/` and `config/` directories. The user-facing README describes the current implementation; inspect current source before modifying behavior.

- Keep the GUI in `python/client_medgemma_patients_mcp.py`.
- Keep MCP transport/lifecycle code in `python/mcp_client.py`.
- Keep three separate servers in `python/mcp_xray_server.py`, `python/mcp_ekg_server.py`, and `python/mcp_blood_server.py`.
- Keep setup, launcher and cleanup scripts in `config/`.
- Do not introduce another GUI, additional orchestration modules, or a virtual environment without an explicit request.
- Qwen is excluded from the current runtime. Keep inference directed to configured MedGemma through Ollama unless requested otherwise.

## Read before changing

Read the affected files, their imports/callers, relevant `.env` key handling and README. Do not print `.env` values, patient contents, prompts or model responses in diagnostic output. Retain the extensive function-header descriptions, examples and inline comments; update examples when signatures or behavior change.

A documentation-only request must leave executable syntax unchanged. Verify this by comparing parsed Python syntax trees as well as compiling the result. Comments/examples must describe real behavior; label illustrative values and required GUI context.

## Component boundaries

The GUI owns patient/snapshot navigation, selection, display, direct Ollama requests, report review/export and voice. `record_client()` lazily creates/reuses `MCPRecords`. The GUI's `read_record_bytes()` and `read_blood_record()` wrappers delegate to that instance.

`MCPRecords` owns child-process stdio transports, the dedicated asyncio thread/loop, initialized sessions, synchronized tool requests and cleanup requests. Server selection comes from the modality path component. Preserve `xrays`, `ekgs`, and `blood_tests` as session keys.

The servers are read-only record providers. Each exposes `list_records(relative_folder)` and `read_record(relative_path)` through the official MCP SDK. Keep response contracts compatible: `paths` for listing, `base64` for images, `record` for parsed blood JSON. Return JSON text as expected by the current client decoder. Do not add model calls or file-write tools to the servers.

## Paths and records

- `PYTHON_DIR` is the resolved GUI-script directory, not the shell working directory.
- Resolve relative `PATIENT_RECORDS_DIR` beside the GUI; honor environment-over-file precedence.
- Pass the absolute root to subprocesses using `PATIENT_RECORDS_DIR`.
- Preserve direct patient/modality directories, existing `data/<modality>` fallback behavior and date/time snapshots.
- Do not invent patient mapping files or require a database.
- Preserve path containment, hidden-path checks, wrong-modality rejection, supported-extension checks, link/junction restrictions and the 64 MiB server read limit.
- Never broaden one record request to arbitrary files or overwrite patient data.

## GUI, requests and models

Preserve the existing panels and selection behavior, patient/date-time picker, image viewer, prompt/report workspace and main Close button. Close must use `close_window`, including unsaved-edit handling and existing cleanup. Do not bypass it with an unrelated direct destroy callback for the main window.

Keep Tk widget operations on the main thread. Background workers must use `post_ui`/queue delivery. Decode Pillow thumbnails off-thread and construct Tk images on-thread. Preserve bounded caches/queues and stale-thumbnail generation checks.

Keep immutable request snapshots and request-ID guards so patient changes, cancellation and old callbacks cannot corrupt current output. Upload only explicit selections, using original image bytes; retain X-ray-before-EKG ordering and blood-record identity. Display resizing must not silently alter uploaded source data.

Keep original AI reports separate from editable drafts. Invalidate local review acknowledgement after edits. Preserve safe export validation, HTML escaping, schema checks and per-file atomic writes outside patient records. Do not describe local acknowledgement as a signature.

Keep cancellation semantics honest: local cancellation does not prove remote inference stopped. Do not display hidden reasoning fields as final report content. Preserve malformed/truncated-response checks and user-visible error categories.

## Dependencies and configuration

Use Python 3.10+ for the MCP application. Retain `mcp>=1.20,<2` unless deliberately migrating and validating both client and servers. Keep Pillow for images and optional pyttsx3 voice support. Do not install standard-library modules with pip.

Keep the `.env` contract in sync with its actual allowlist and readers. `GENERATION_TIMEOUT` is the recognized inference-timeout key; do not document unused aliases as active settings. Remote/local Ollama comes from `OLLAMA_URL`; avoid hardcoded host addresses.

Updated init calls `config/generate_requirements.py`; inspect that helper when modifying dependency generation. The root requirements manifest is installed, with the application copy under `python/`. Verify generated files contain real newline-separated requirement lines and are accepted by pip's requirement parser. Preserve additional existing dependencies. Do not claim arbitrary new imports are discovered by a fixed dependency list.

Preserve no-venv installation behavior and explicit optional local-model prompts. Do not execute init/uninit or download models merely to document or inspect scripts. Do not remove Python, user packages, models, settings or patient files during cleanup. Keep confirmation for optional global Codex removal and `.venv` deletion.

## Validation and delivery

Use checks appropriate to the change:

- Compile changed Python files. For comment-only edits, compare ASTs.
- Check shell changes with `bash -n`; do not claim Windows execution without actually testing it.
- For MCP changes, exercise real stdio sessions with temporary synthetic records: listing, image byte round-trip, blood JSON, traversal and wrong-modality rejection; request cleanup afterward.
- For requirements changes, generate in a temporary `python/` + `config/` tree, verify preserved extra dependencies, repeatability and actual line contents.
- For GUI layout changes, inspect a rendered window when available. Otherwise state that appearance is unverified. Preserve unrelated function behavior.
- Avoid live patient data and remote inference in automated checks. Do not claim medical accuracy from transport tests.

Keep changes scoped to the request. Update README with material behavior/setup changes. Report what changed, checks performed and relevant untested items. Deliver exact filenames and maintain the user's existing directory layout.
