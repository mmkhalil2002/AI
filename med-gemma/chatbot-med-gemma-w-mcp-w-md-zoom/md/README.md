# AI Clinical Assistant Center with MCP patient records

This Tkinter desktop application reviews selected patient X-rays, CT scans, EKGs, and blood-test JSON records, then submits only that selected evidence and a clinician question to MedGemma through Ollama. Its main implementation is `python/client_medgemma_patients_mcp.py`; preserve the full GUI rather than replacing it with a simplified prototype. Qwen is excluded.

## Interface

The window has an unlabeled patient-search control bar above a responsive two-column workspace. The left side contains the red/orange **AI Clinical Assistant Center** heading, **Your Question** input, **Ask AI**, **Voice: OFF**, **Close**, status, and scrollable **AI Response**. The right side supports these four record handlers; only configured windows are displayed:

- **XRAY** — framed, selectable image tiles with horizontal and vertical scrollbars.
- **CT SCAN** — framed, selectable CT image tiles with the same supported image formats as X-rays.
- **EKG/ECG** — wide, readable selectable images with scrolling.
- **BLOOD TEST** — selectable records with red title/marker and Test, Value, Unit, Reference range, and Flag columns.

The top bar searches patients by **First name**, **Last name**, and **DOB (MMDDYYYY)**. **Search** validates the values, maps the four-digit year to the existing two-digit patient-folder suffix, loads the newest snapshot for a matching patient, and shows an error dialog when no match exists. **Date & time** then selects a different snapshot for that active patient. At the far right, a two-line active-patient block shows `Active patient` in teal and the selected name plus DOB in coral. Snapshot selection supports direct folders, dated snapshots, and legacy `data/<modality>` layouts.

Selected X-ray, CT, and EKG records can be inspected with **Zoom**. In **Merge** mode, all selected images share one viewer with **Previous**, **Next**, **+**, **−**, **Fit**, and **Reset**. In **Split** mode, each selected image opens in its own viewer with **+**, **−**, **Fit**, and **Reset**. **Fit** shows the whole image in the window; **Reset** returns it to 100% pixel size.

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

`WINDOW_SERVER_MAP` controls which record windows exist. Three entries create three Zoom buttons, three ON/OFF buttons and three panels. All configured panels start ON. Use the buttons to hide/show them during the session; there are no startup ON/OFF flags in `.env`. Omitted sections are not loaded and their servers are not started.

Keep `.env` beside the GUI. Relevant settings include `MEDGEMMA_MODEL`, `PATIENT_RECORDS_DIR`, `GENERATION_TIMEOUT`, `NUM_PREDICT`, `XRAY_WIDTH`, `XRAY_HEIGHT`, `CT_SCAN_WIDTH`, `CT_SCAN_HEIGHT`, `EKG_WIDTH`, and `EKG_HEIGHT`. The window/record section names and independent MCP server IDs are configured by `WINDOW_SERVER_MAP` at the beginning of `.env`. Relative patient paths resolve beside the GUI; OS environment values override `.env`.

Records use patient folders with `xrays`, `ct_scan`, `ekgs`, and `blood_tests` modality directories. A missing `ct_scan` folder is allowed. Images may be PNG, JPG/JPEG, BMP, GIF, TIF/TIFF, or WebP; blood tests are JSON. `sources.csv` and unsupported files are never displayable or submittable evidence.

Only explicitly selected original records are sent to MedGemma. Selected X-rays precede CT scans, which precede EKGs. Thumbnails are display copies only. Generated content requires clinician review; this application is not an authentication system or a medical decision maker.

## Run

Use Python 3.10+ with Tkinter and the requirements in `python/requirements.txt`:

```bat
py -m pip install -r python\requirements.txt
py python\client_medgemma_patients_mcp.py
```

Use **Close** to invoke normal cancellation and MCP cleanup. Closing locally does not prove that remote Ollama inference has stopped.

Enter the Ollama IP beside **Date & time**, then press yellow **Connect**. When the IP field is blank, pressing Connect fills it with `localhost` and connects to `http://localhost:11434`, without OLLAMA_URL configuration. Success shows green **Disconnect**; failure shows red **Fail** (click to retry). Clicking **Disconnect** disconnects this client from the current Ollama server: it cancels the active local request, closes its established HTTP response, clears the active server and IP field, and restores yellow **Connect**. New AI requests are blocked until **Connect** succeeds again. Disconnect does not stop the Ollama service; remote computation may continue after the client closes its response. Editing the IP also restores yellow **Connect**.

## Window names and MCP server IDs

At the beginning of `python/.env`:

```dotenv
WINDOW_SERVER_MAP=xrays:ID001,ct_scan:ID002,ekgs:ID003,blood_tests:ID004
SERVER_HANDLERS=ID001:xrays,ID002:ct_scan,ID003:ekgs,ID004:blood_tests
```

| Window / patient-record section | MCP server identity |
|---|---|
| xrays | ID001 |
| ct_scan | ID002 |
| ekgs | ID003 |
| blood_tests | ID004 |

For example, `patient_records/<patient>/<date>/<time>/xrays/chest.png` routes to server `ID001`. Direct patient sections and `data/<section>` also work. Window names are arbitrary valid folder names, including spaces and Unicode; they must match the section folder names. To keep existing folders, use their current names in the table. Configured names appear in gallery headings; server IDs are independent of window names and folders, and are verified during MCP initialization. IDs must be unique and contain only letters, digits, underscores or hyphens. Define between one and four windows using any subset of the configured servers. Names are not predefined and table order does not affect routing. The separate `SERVER_HANDLERS=ID001:xrays,ID002:ct_scan,ID003:ekgs,ID004:blood_tests` table assigns each ID to the existing image/JSON record handler; change this too when changing IDs. The four record handlers remain fixed; this does not add new record types. The OS `WINDOW_SERVER_MAP` variable overrides `.env`. Restart the app after changing the table. Old individual `*_SERVER_ID` settings have been replaced by this table.

Three-window example (three Zoom controls and three ON/OFF controls):

```dotenv
WINDOW_SERVER_MAP=xrays:ID001,ekgs:ID003,blood_tests:ID004
```

Run `py config/verify_window_mapping.py` to check arbitrary-name routing, local MCP identities/cleanup, section confinement and 1–4 window controls with synthetic records and hidden Tk windows.

Run `py config/verify_connection_controls.py` to check Connect, Disconnect, response cancellation, blocked requests while disconnected, explicit reconnection and failed connections without network requests or patient data.
