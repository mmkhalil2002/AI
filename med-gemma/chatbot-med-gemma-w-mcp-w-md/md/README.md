# AI Clinical Assistant Center with MCP patient records

This project is a Tkinter desktop application for reviewing selected patient X-rays, EKGs, and blood-test JSON records and submitting selected evidence plus a clinician question to MedGemma through Ollama. The main implementation is `python/client_medgemma_patients_mcp.py`; it is a full existing GUI and must be preserved rather than replaced with a simplified prototype. Qwen is excluded.

## Interface

The window has an unlabeled patient-search control bar above a responsive two-column workspace. The left side contains the red/orange **AI Clinical Assistant Center** heading, **Your Question** input, **Send to MedGemma**, **Voice: OFF**, **Close**, status, and scrollable **AI Response**. The right side has three simultaneous panels:

- **XRAY** — framed, selectable, horizontally arranged image tiles with scrollbars.
- **EKG/ECG** — wide readable selectable images with scrolling.
- **BLOOD TEST** — selectable records with red title/marker and Test, Value, Unit, Reference range, and Flag columns.

The top bar searches patients by **First name**, **Last name**, and **DOB (MMDDYY)**. **Search** validates the values, loads the newest snapshot for a matching patient, and shows an error dialog when no match exists. **Choose date & time** then selects a different snapshot for that active patient. At the far right, a two-line active-patient block shows `Active patient` in teal and the selected name plus DOB in coral. Snapshot selection supports direct folders, dated snapshots, and legacy `data/<modality>` layouts.

## Components

| File | Responsibility |
|---|---|
| `python/client_medgemma_patients_mcp.py` | GUI, navigation, galleries, selection, Ollama inference, report lifecycle, voice, and cleanup |
| `python/mcp_client.py` | `MCPRecords`, local stdio server lifecycle, calls, and cleanup |
| `python/mcp_xray_server.py` | Read-only X-ray MCP server |
| `python/mcp_ekg_server.py` | Read-only EKG MCP server |
| `python/mcp_blood_server.py` | Read-only blood-test MCP server |

MCP servers run automatically as local child processes. No MCP TCP ports or manual server terminals are needed. They return root-relative `paths`, image `base64`, or blood `record` JSON.

## Records and configuration

Keep `.env` beside the GUI. Relevant settings include `OLLAMA_URL`, `MEDGEMMA_MODEL`, `PATIENT_RECORDS_DIR`, `GENERATION_TIMEOUT`, `NUM_PREDICT`, `XRAY_WIDTH`, `XRAY_HEIGHT`, `EKG_WIDTH`, and `EKG_HEIGHT`. The local MCP server IDs are `XRAY_SERVER_ID=xrays`, `EKG_SERVER_ID=ekgs`, and `BLOODTEST_SERVER_ID=blood_tests`; leave them at these defaults unless the local server configuration changes. Relative patient paths resolve beside the GUI; OS environment values override `.env`.

Records use patient folders with `xrays`, `ekgs`, and `blood_tests` modality directories. Images may be PNG, JPG/JPEG, BMP, GIF, TIF/TIFF, or WebP; blood tests are JSON. `sources.csv` and unsupported files are never displayable or submittable evidence.

Only explicitly selected original records are sent to MedGemma. Selected X-rays precede EKGs. Thumbnails are display copies only. Generated content requires clinician review; this application is not an authentication system or a medical decision maker.

## Run

Use Python 3.10+ with Tkinter and the requirements in `python/requirements.txt`:

```bat
py -m pip install -r python\requirements.txt
py python\client_medgemma_patients_mcp.py
```

Use **Close** to invoke normal cancellation and MCP cleanup. Closing locally does not prove that remote Ollama inference has stopped.
