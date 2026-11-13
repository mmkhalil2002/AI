# Coding instructions: preserve the full AI Clinical Assistant Center GUI

## Authoritative implementation

The existing MedGemma GUI implementation must be preserved and repaired in place. It is a full Tkinter application, not a prototype. Do not replace it with a short script, a generic dashboard, a filename list, or Text-widget image rendering.

The approved interface is a full-width unlabeled patient-search bar above a responsive two-column workspace: approximately 35% left for the assistant controls and response, 65% right for patient records. The right side shows framed X-ray, CT scan, and EKG galleries plus selectable blood-test records. CT uses its configured record-section folder name and the same supported raster formats as X-rays; an omitted CT window is allowed. This document is the sole authoritative written specification.

## Required functions and boundaries

Preserve the existing functions and call paths, including `patient_records_root`, `discover_patients`, `patient_snapshots`, `choose_snapshot`, `select_patient`, `select_snapshot`, `record_client`, `read_record_bytes`, `read_blood_record`, `patient_files`, `load_images`, `select_image`, `update_gallery_scrollbars`, `open_image_viewer`, `load_blood_tests`, `generate`, `run_request`, `send_question`, `finish_request`, and `close_window`.

Keep `mcp_client.py` separate from the GUI and retain its MCPRecords interface. Keep the four read-only modality-server entry points separate. The GUI owns Tk presentation, navigation, selection, prompt construction, Ollama calls, report review, voice, and shutdown. MCP owns stdio lifecycle and record retrieval only.

## Layout and behavior

- Configure the three patient-search captions with FIRST_NAME_LABEL, LAST_NAME_LABEL, and DOD_LABEL in python/.env, immediately after the window mapping. Their defaults are first name, last name, and DOD (MMDDYYYY), respectively. Missing or blank values use these defaults; operating-system variables override file values. Apply changes after restarting. These settings change captions only: the third input still searches date of birth using MMDDYYYY, and patient matching, validation, and folder conventions retain their existing behavior.

- `WINDOW_SERVER_MAP` at the start of python/.env maps the configured window/record section names to independent unique MCP server IDs. Propagate IDs to server identity and route sections through this table. Window names are arbitrary valid folder names. Use one window-name:server-ID or window-name:ON/OFF:server-ID table. Entries also accept window-name:ON/OFF:server-ID (case-insensitive). The two-field format means ON. OFF entries are removed before assigning handler positions, exactly as if absent: later entries shift forward. OFF entries have no panel, buttons, records or MCP connection. All OFF is allowed and shows no record panels. Enabled panels start ON. Select record handlers by entry position: X-ray, CT, EKG, then blood-test JSON. Keep the positional order; omit trailing entries for fewer windows. Blood tests require a fourth entry. Do not use a separate handler table.

- Match the configured window names to the actual patient-record section folders. The current dataset uses the existing four-section mapping for xrays, ct_scan, ekgs, and blood_tests with unique server IDs. Removing a window entry removes its panel, Zoom control, ON/OFF control and MCP connection. Do not substitute example folder names for the dataset's existing names.

- Beside Date & time, show a narrow IP field and yellow Connect button. Pressing Connect with a blank IP fills the field with localhost; connect to the chosen IP on Ollama port 11434 without requiring a separately configured Ollama URL. Check connectivity asynchronously; success shows green Disconnect, failure shows red Fail and allows retry. Clicking Disconnect cancels any current local request, closes its established HTTP response, clears the active server and IP field, marks the server offline and restores yellow Connect without checking a server. Block new inference until Connect succeeds again; a blank IP uses localhost only after an explicit Connect. Editing IP resets yellow Connect.

- Top bar: First name, last name, and DOB (`MMDDYYYY`) fields plus `Search`; any nonempty combination is accepted and multiple matches require more fields. A successful search loads the newest patient snapshot, while an invalid or missing patient shows an error dialog. Existing patient folders use two-digit years, so search maps the entered four-digit year to the matching `YY` folder suffix. `Date & time` follows the search controls. At the far-right edge, a two-line block displays `Active patient` in teal above the selected name and DOB in coral.
- Left: red/orange `AI Clinical Assistant Center` title; `Your Question`; multiline prompt; `Ask AI`, `Voice: OFF`, and `Close`; concise status; large scrollable `AI Response` output.
- Right: only configured panels, labeled with their arbitrary window names, with actual snapshot date/time headings. Use pale borders/light surfaces, teal headings and date, purple time.
- Retain horizontal and vertical gallery scrollbars, framed selectable X-ray tiles, wide readable EKGs, and Tk image references/caches. Zoom starts in its `Zoom out` state. `Merge` shares one selected-image viewer with Previous, Next, +, −, Fit, and Reset; `Split` opens a separate viewer for each selected X-ray/EKG image with +, −, Fit, and Reset only. Fit shows the whole image; Reset restores original 100% pixel size.
- Exclude `sources.csv` and unsupported files before tile construction. Never show loose metadata text in place of gallery tiles.
- Blood tests require red titles and explicit selection markers, then aligned Test, Value, Unit, Reference range, and Flag results. Preserve existing normalization and flag rules.
- `Date & time` opens the expandable snapshot dialog. Its Close closes only the dialog; main Close calls `close_window`.

## Safety, paths, and requests

The project-root requirements.txt is the Python dependency manifest consumed by the Windows and Ubuntu setup scripts. Keep it aligned with the application's third-party imports and compatible with the existing python/requirements.txt. Tkinter is provided by the Python distribution or operating system. Updating dependency files does not itself authorize installing packages or running setup scripts.

Only entries in `WINDOW_SERVER_MAP` produce panels, Zoom and ON/OFF controls. All configured panels start ON. Entries also accept window-name:ON/OFF:server-ID (case-insensitive). The two-field format means ON. OFF entries are removed before assigning handler positions, exactly as if absent: later entries shift forward. OFF entries have no panel, buttons, records or MCP connection. All OFF is allowed and shows no record panels. Enabled panels start ON. Keep interactive ON buttons green and OFF buttons muted gray (`#b0bec5`) with dark text (`#263238`); toggles remain usable.

Resolve `.env` and relative `PATIENT_RECORDS_DIR` beside the GUI. Honor OS environment precedence. Support direct modality folders, legacy `data/<modality>`, and dated snapshots. Pass the absolute root to child processes.

Only explicitly selected records may be sent. Preserve selected X-rays, CT scans, then EKGs and submit original image bytes, never thumbnails. Blood inference must use clinical result content, not display-only metadata. Keep immutable request snapshots, request IDs, stale-result rejection, UI-thread Tk updates, bounded queues/caches, cancellation semantics, and MCP cleanup.

Servers must preserve path confinement, hidden/link checks, modality checks, allowed extensions, and the 64 MiB limit. Their contracts are `list_records(relative_folder)` and `read_record(relative_path)` returning JSON fields `paths`, `base64`, or `record`.

## Validation

Compile changed Python. With de-identified records, exercise at least two patients, snapshot switching, gallery selection, blood selection/details, image viewer, Voice fallback, Send error/success, stale callbacks, and Close cleanup. Confirm sources.csv is excluded and a populated window preserves the two-column gallery layout. Do not run installers, uninstallers, remote inference, or use real patient records solely for validation.

Use `py config/verify_window_mapping.py` for synthetic record routing, MCP identity/cleanup, section confinement and 1–4 window control checks. Use `py config/verify_connection_controls.py` for simulated connection transitions, active-response cancellation, disconnected-request blocking and reconnect/failure checks. Keep this written specification synchronized with the application behavior.


## Single-document maintenance policy

- This document, located at **md/AGENTS.md**, is the **only Markdown instruction file** for this chatbot. Do not create a second root-level AGENTS.md or other instruction Markdown files.
- All requirements and future feature descriptions belong in this file. Keep it descriptive: Markdown headings, paragraphs, and lists only. Do not embed source code, fenced code blocks, program templates, installation commands, package lists, or generated-file contents.
- The application implementation remains in its existing Python files. This document describes desired behavior; it is not an executable generator and is not a byte-for-byte recovery copy.
- Do not require a generators directory, a code-generation utility, a second AGENTS.md, or any dependency manifest as an input to interpreting this document.
- When asked to change the application, Codex should inspect the actual project files, preserve working functionality, implement the requested behavior, and update the relevant descriptive section here.
- If an application file is missing, Codex should recreate it from the written requirements and surviving project files, then validate it. Do not claim an exact restoration is possible from prose alone.
- If a needed detail is not documented or cannot be established from existing files, ask for clarification rather than inventing application behavior.

## Application responsibilities

### Patient search and snapshots

- Accept any nonempty combination of first name, last name, and an eight-digit date of birth (DOB, MMDDYYYY). Match all supplied fields exactly, ignoring name case and outer whitespace. Reject blank, invalid, unmatched, or ambiguous searches; ask for more fields when several patients match. Show the selected patient folder identity independently of the search fields.
- Recognize existing patient folder naming conventions with two-digit birth years while accepting a four-digit year from the user.
- On successful search, isolate the chosen patient's data, clear prior selections and stale results, and load the newest valid dated snapshot.
- The Date & time dialog lists available patient snapshots by date and time. Selecting a time loads only that snapshot; closing the dialog does not close the main GUI.
- Preserve the current patient's context while navigating records, and prevent accidental cross-patient record exposure.

### Patient-record presentation

- The record workspace contains only the configured active panels. X-ray and CT scan panels display framed image thumbnails; EKG panels favor wide, readable image previews; blood-test panels display structured results.
- Keep each image selectable, visually indicate its selected state, and provide horizontal and vertical scrolling where needed.
- Maintain image references and bounded caches so previews remain visible without excessive memory use.
- Image Zoom controls support merged viewing and separate viewing. Merged mode allows previous/next navigation across selected images. Split mode gives each selected image its own controls. Fit scales the whole image to the viewer; Reset restores its original pixel scale.
- Ignore unsupported file formats, hidden files, and sources.csv. Never replace the image gallery with a filename-only listing.
- Blood tests display their title, selection indicator, test name, value, unit, reference range, and flag. Preserve existing result parsing, normalization, and abnormal-flag rules.

### Record-section mapping and controls

- Read the section-to-server mapping from the GUI's existing configuration. Each enabled entry identifies a patient-record folder and a unique MCP server identity.
- The configured order determines the record handler: X-ray, CT scan, EKG, then blood-test JSON. Disabled entries are removed before assigning handler positions; later enabled entries move forward.
- Support both mappings with an implicit enabled state and mappings with explicit ON/OFF state. Interpret state without regard to letter case.
- A disabled entry must have no panel, no records, no Zoom control, no ON/OFF button, and no MCP connection. Zero enabled entries is valid.
- Active panels begin enabled and retain interactive ON/OFF controls. ON is green; OFF is muted gray with dark text. Do not confuse a temporarily hidden active panel with a disabled mapping entry.
- Panel names must match actual record-section folders; do not silently rename or substitute folder names.

### Ollama connection and inference

- Display an IP input next to Date & time, with a yellow Connect control. When the user explicitly connects with a blank field, populate localhost.
- Check connection asynchronously without freezing the interface. Successful connection changes the control to green Disconnect. Failed connection shows red Fail and permits retry.
- Disconnect cancels the active request, closes its HTTP response, clears the active server and IP field, and returns to the disconnected yellow Connect state. Editing the IP also invalidates the previous connection state.
- Do not allow inference until a connection has been explicitly established. Do not silently reconnect after disconnection.
- Send the clinician's question and only explicitly selected patient records. Submit original image bytes, not display thumbnails. Include blood-test clinical result content, not merely labels or display metadata.
- Preserve selection order for image inputs: X-ray, CT scan, then EKG. Capture immutable request inputs and identify each request so callbacks from an older request cannot overwrite a newer response.
- Run network work off the GUI thread, post widget updates to the GUI thread, and maintain cancellation, timeout, and cleanup behavior.
- Display elapsed request time as MM:SS and the server-reported received output token count at completion. Sum output counts across separate reports; show unavailable when the server omits counts, and pending during generation.
- Keep the AI response scrollable and preserve report review and voice controls. AI-generated medical output requires clinician review.

### MCP client and servers

- Keep MCP transport separate from Tkinter presentation and from direct Ollama inference. The GUI delegates record listing and reading to the MCP client.
- Maintain distinct read-only record handlers for X-ray, CT scan, EKG, and blood tests. Each handler enforces its assigned record type and configured folder.
- Record listing accepts a patient-root-relative folder and returns eligible record paths. Record reading accepts a patient-root-relative path and returns original image content or parsed clinical JSON as appropriate.
- Validate path containment and reject traversal, hidden files, symbolic links, junctions, unsupported extensions, and inappropriate modality access. Preserve the 64 MiB record-size limit.
- Resolve patient-record paths relative to the GUI when configured as relative paths, while honoring operating-system environment overrides. Support existing direct modality folders, legacy data-wrapped folders, and date/time snapshot folders.
- Start MCP connections only for enabled sections and close them on shutdown. Avoid leaking server processes or file handles.

### Report, voice, and shutdown

- Preserve clinician review of generated reports and existing safeguards for unsaved edits when changing patients or snapshots.
- Keep voice playback optional, with a clear ON/OFF state and graceful handling when speech playback is unavailable.
- The main Close action must cancel active work, stop voice, close image viewers, release MCP resources, and close the application cleanly.
- Snapshot-dialog Close must affect only the snapshot dialog.

## Codex implementation and recovery rules

- Work from this single descriptive specification and the current project files. Never treat prose as executable code.
- Before changing an existing implementation, inspect the actual functions and their call paths. Preserve features not explicitly requested to change.
- When rebuilding a missing file, use existing interfaces and behavior as evidence, then verify the result. The written requirements alone cannot guarantee identical reconstruction.
- Do not run installation, uninstallation, model downloads, remote inference, or real patient-record tests solely to validate a code edit.
- After a requested change, run applicable syntax checks and targeted tests with synthetic or de-identified records. Review patient isolation, section mapping, image selection, blood-test formatting, connection state, cancellation, stale callbacks, and shutdown.
- Keep this file free of Python code, generated source blocks, generator instructions, package dependencies, and duplicate instructions in other Markdown files.
