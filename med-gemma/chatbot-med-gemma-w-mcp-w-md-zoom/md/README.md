# AI Clinical Assistant Center with MCP patient records

This Tkinter desktop application reviews selected patient X-rays, CT scans, EKGs, and blood-test JSON records, then submits only that selected evidence and a clinician question to MedGemma through Ollama. Its main implementation is `python/client_medgemma_patients_mcp.py`; preserve the full GUI rather than replacing it with a simplified prototype. Qwen is excluded.

## Interface

The window has an unlabeled patient-search control bar above a responsive two-column workspace. The left side contains the red/orange **AI Clinical Assistant Center** heading, **Your Question** input, **Ask AI**, **Voice: OFF**, **Close**, status, and scrollable **AI Response**. The right side supports these four record handlers; only configured windows are displayed:

- **XRAY** â€” framed, selectable image tiles with horizontal and vertical scrollbars.
- **CT SCAN** â€” framed, selectable CT image tiles with the same supported image formats as X-rays.
- **EKG/ECG** â€” wide, readable selectable images with scrolling.
- **BLOOD TEST** â€” selectable records with red title/marker and Test, Value, Unit, Reference range, and Flag columns.

The top bar searches patients by **First name**, **Last name**, and **DOB (MMDDYYYY)**. **Search** validates the values, maps the four-digit year to the existing two-digit patient-folder suffix, loads the newest snapshot for a matching patient, and shows an error dialog when no match exists. **Date & time** then selects a different snapshot for that active patient. At the far right, a two-line active-patient block shows `Active patient` in teal and the selected name plus DOB in coral. Snapshot selection supports direct folders, dated snapshots, and legacy `data/<modality>` layouts.

Selected X-ray, CT, and EKG records can be inspected with **Zoom**. In **Merge** mode, all selected images share one viewer with **Previous**, **Next**, **+**, **âˆ’**, **Fit**, and **Reset**. In **Split** mode, each selected image opens in its own viewer with **+**, **âˆ’**, **Fit**, and **Reset**. **Fit** shows the whole image in the window; **Reset** returns it to 100% pixel size.

## Components

| File | Responsibility |
|---|---|
| `python/client_medgemma_patients_mcp.py` | GUI, navigation, galleries, selection, Ollama inference, report lifecycle, voice, and cleanup |
| `python/mcp_client.py` | `MCPRecords`, local stdio server lifecycle, calls, and cleanup |
| `python/mcp_xray_server.py` | Read-only X-ray MCP server |
| `python/mcp_ct_scan_server.py` | Read-only CT-scan MCP server |
| `python/mcp_ekg_server.py` | Read-only EKG MCP server |
| `python/mcp_blood_server.py` | Read-only blood-test MCP server |

MCP servers run automatically as local child processes. No MCP TCP ports or manual server terminals are needed. They return root-relative `paths`, image `base64`, or blood `record` JSON.

## Records and configuration

Immediately after `WINDOW_SERVER_MAP` in `python/.env`, configure the search-field labels:

```dotenv
FIRST_NAME_LABEL=first name
LAST_NAME_LABEL=last name
DOD_LABEL=DOD (MMDDYYYY)
```

These are also the defaults when a value is missing or blank. OS environment variables override file values. Restart to apply changes. The captions are editable; the third field still matches date of birth in MMDDYYYY format.

`WINDOW_SERVER_MAP` controls which record windows exist. Use `window_name:ON/OFF:ServerID`; the state is case-insensitive, and the older `window_name:ServerID` format means ON. OFF entries are treated as absent: they create no panel, Zoom button, visibility button or MCP connection. All-OFF is allowed. Enabled panels start ON and can be hidden or shown with their visibility buttons.

Keep `.env` beside the GUI. Relevant settings include `MEDGEMMA_MODEL`, `PATIENT_RECORDS_DIR`, `GENERATION_TIMEOUT`, `NUM_PREDICT`, `XRAY_WIDTH`, `XRAY_HEIGHT`, `CT_SCAN_WIDTH`, `CT_SCAN_HEIGHT`, `EKG_WIDTH`, and `EKG_HEIGHT`. The window/record section names and independent MCP server IDs are configured by `WINDOW_SERVER_MAP` at the beginning of `.env`. Relative patient paths resolve beside the GUI; OS environment values override `.env`.

Records use patient folders with `xrays`, `ct_scan`, `ekgs`, and `blood_tests` modality directories. A missing `ct_scan` folder is allowed. Images may be PNG, JPG/JPEG, BMP, GIF, TIF/TIFF, or WebP; blood tests are JSON. `sources.csv` and unsupported files are never displayable or submittable evidence.

Only explicitly selected original records are sent to MedGemma. Selected X-rays precede CT scans, which precede EKGs. Thumbnails are display copies only. Generated content requires clinician review; this application is not an authentication system or a medical decision maker.

## Run

Use Python 3.10+ with Tkinter. The root `requirements.txt` is the dependency file used by the setup scripts; it includes MCP, Pillow for images, and pyttsx3 for voice. Tkinter comes from Python or the operating system rather than pip.

```bat
py -m pip install -r requirements.txt
py python\client_medgemma_patients_mcp.py
```

Use **Close** to invoke normal cancellation and MCP cleanup. Closing locally does not prove that remote Ollama inference has stopped.

Enter the Ollama IP beside **Date & time**, then press yellow **Connect**. When the IP field is blank, pressing Connect fills it with `localhost` and connects to `http://localhost:11434`, without OLLAMA_URL configuration. Success shows green **Disconnect**; failure shows red **Fail** (click to retry). Clicking **Disconnect** disconnects this client from the current Ollama server: it cancels the active local request, closes its established HTTP response, clears the active server and IP field, and restores yellow **Connect**. New AI requests are blocked until **Connect** succeeds again. Disconnect does not stop the Ollama service; remote computation may continue after the client closes its response. Editing the IP also restores yellow **Connect**.

## Window names and MCP server IDs

At the beginning of `python/.env`, use this single table:

```dotenv
WINDOW_SERVER_MAP=xrays:ON:ID001,ct_scan:ON:ID002,ekgs:ON:ID003,blood_tests:ON:ID004
```

Both the window name and server ID in each pair can be changed. Window names are arbitrary valid folder names, including spaces and Unicode, and must match the patient's record-section folders. Changing the table does not rename any folders. Server IDs are independent, unique values containing letters, digits, underscores or hyphens; change an ID in this table only. The client verifies the advertised server identity when connecting.

Entry position after removing OFF entries selects the existing record handler:

| Position | Handler | Default folder/name | Default server ID |
|---|---|---|---|
| 1 | X-ray images | xrays | ID001 |
| 2 | CT images | ct_scan | ID002 |
| 3 | EKG/ECG images | ekgs | ID003 |
| 4 | Blood-test JSON | blood_tests | ID004 |

Define one to four entries. OFF entries are removed before assigning handlers, so later entries shift forward exactly as if the OFF entries had been deleted. Blood-test JSON requires a fourth enabled entry. For example, setting xrays to OFF in the four-entry definition makes ct_scan use the X-ray handler, ekgs use CT, and blood_tests use the EKG image handler.

Four visible panels use a 2-by-2 grid in entry order. One, two or three visible panels stack vertically across the record area with equal height. Hiding a panel with its session ON/OFF button redistributes the space without changing handler assignments. Configuration changes require restarting the application.

For example, `patient_records/<patient>/<date>/<time>/xrays/chest.png` routes to `ID001`. Direct `<patient>/<section>` and legacy `<patient>/data/<section>` layouts also work.

Example changing the first window name and ID (requires a matching `Chest Images` folder):

```dotenv
WINDOW_SERVER_MAP=Chest Images:ON:IMAGES_A,ct_scan:ON:ID002,ekgs:ON:ID003,blood_tests:ON:ID004
```

Restart after editing. The OS `WINDOW_SERVER_MAP` variable overrides the file. Individual `*_SERVER_ID` settings and a separate handler table are not used.

Run `py config/verify_window_mapping.py` for synthetic routing, renamed server identities, cleanup, confinement and 1-4 window control checks. Run `py config/verify_connection_controls.py` for simulated Connect/Disconnect and cancellation checks without network or patient data.

