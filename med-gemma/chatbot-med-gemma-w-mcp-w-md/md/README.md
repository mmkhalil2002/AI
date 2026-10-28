# MedGemma GUI with MCP patient-record servers

A Windows/Ubuntu desktop application for selecting patient X-rays, EKGs and blood-test results, submitting selected information and a doctor's prompt to Ollama-hosted MedGemma, and reviewing the returned report.

The GUI imports a separate MCP client module. That client starts three local read-only MCP servers. MedGemma runs on the Ollama host configured in `.env`. Qwen is not used in this version. Outputs require clinician review; use educational or de-identified records.

## Project structure

Place this README and AGENTS.md at the project root:

```text
project/
├── AGENTS.md
├── README.md
├── requirements.txt
├── config/
│   ├── init.bat
│   ├── init.sh
│   ├── run.bat
│   ├── run.sh
│   ├── uninit.bat
│   ├── uninit.sh
│   └── generate_requirements.py
└── python/
    ├── client_medgemma_patients_mcp.py
    ├── mcp_client.py
    ├── mcp_xray_server.py
    ├── mcp_ekg_server.py
    ├── mcp_blood_server.py
    ├── requirements.txt
    ├── .env
    └── patient_records/
        └── P0001/
            ├── xrays/
            ├── ekgs/
            └── blood_tests/
```

Date/time snapshots such as `P0001/2026-10-26/05-23_AM/xrays/` are also supported. Use matching `ekgs/` and `blood_tests/` under the same time folder. The picker offers snapshots containing recognized modality directories. Existing `<patient>/data/<modality>` layouts remain supported; when present outside snapshot mode, that layout takes precedence over direct modality folders.

## Component responsibilities

| File | Responsibility |
|---|---|
| `client_medgemma_patients_mcp.py` | Tkinter GUI, patient/snapshot navigation, selection, display, Ollama requests, report review/export and optional voice |
| `mcp_client.py` | `MCPRecords`, server startup, stdio sessions, tool calls, image decoding from Base64, and shutdown requests |
| `mcp_xray_server.py` | Lists and reads X-ray image files under `xrays` |
| `mcp_ekg_server.py` | Lists and reads EKG image files under `ekgs` |
| `mcp_blood_server.py` | Lists and reads blood-test JSON under `blood_tests` |

The GUI and imported MCP client run in one Python process. Each server runs as its own child process using the same interpreter. No MCP TCP ports or manually opened server terminals are required. Directory navigation remains in the GUI; record listing/content retrieval uses MCP. Report export is performed by the GUI, not by MCP servers.

## Setup and launch

Use Python 3.10 or newer with Tkinter. Keep the exact Python filenames shown above so imports and server launches work.

From the project root on Windows:

```bat
config\init.bat
config\run.bat
```

On Ubuntu:

```bash
bash config/init.sh
bash config/run.sh
```

The updated init scripts call `config/generate_requirements.py` and then install requirements. Keep that helper alongside init. The root requirements manifest is used for installation; the helper is intended to maintain an application copy in `python/` too. Inspect generated requirements if installation reports invalid requirement lines. These scripts use the existing Python installation without creating a virtual environment. Init retains removal of an existing project-root `.venv`. Ubuntu init uses `pip --user --break-system-packages`, which can affect compatibility with other Python software.

Node/npm and Codex setup are retained by init, although the running GUI itself uses Python. Local Ollama/MedGemma setup is optional; choose remote hosting by configuring `.env`. Init does not launch the GUI or the MCP servers.

To install the application dependencies directly and bypass init:

```bat
py -m pip install -r requirements.txt
py python\client_medgemma_patients_mcp.py
```

For Ubuntu, use the Python installation's supported pip method; Tkinter may require `sudo apt install python3-tk`. Launch with `python3 python/client_medgemma_patients_mcp.py`.

Expected application requirements:

```text
mcp>=1.20,<2
Pillow>=10
pyttsx3>=2.90
```

The MCP upper bound preserves compatibility with the v1 `ClientSession` and `FastMCP` APIs. Voice is optional at runtime and may need operating-system speech components. Standard-library modules do not need pip installation.

## Configuration

Keep `.env` beside the GUI inside `python/`:

```dotenv
OLLAMA_URL=http://127.0.0.1:11434
MEDGEMMA_MODEL=medgemma1.5:latest
PATIENT_RECORDS_DIR=patient_records
NUM_PREDICT=4048
GENERATION_TIMEOUT=1200
TEMPERATURE=0.2
REPEAT_PENALTY=1.2
REPEAT_LAST_N=256
THINK=false
XRAY_WIDTH=256
XRAY_HEIGHT=256
EKG_WIDTH=800
EKG_HEIGHT=256
WINDOW_WIDTH=2400
WINDOW_HEIGHT=850
```

`127.0.0.1` means Ollama on the GUI computer. For remote Ollama, use that server's reachable address. Use the exact installed model name. `THINK` is sent only when advertised model capabilities support it; do not assume every model honors the option.

`PYTHON_DIR = Path(__file__).resolve().parent` identifies the GUI folder. `patient_records_root()` resolves `PATIENT_RECORDS_DIR` relative to that folder, validates access and returns the absolute path. Existing OS environment variables take precedence over `.env`. The GUI passes the resolved records root to MCP child processes; servers do not independently parse the GUI's `.env`.

Use `GENERATION_TIMEOUT`; the earlier `GENERATE_TIMEOUT` and `MODEL_LIST_TIMEOUT` names are not recognized settings in this GUI. No additional MCP settings are required. The records directory must already exist.

## Example record retrieval

The following illustrates calls inside the running GUI, where `record_client()` is defined:

```python
xray_bytes = record_client().read_record_bytes(xray_path)
ekg_bytes = record_client().read_record_bytes(ekg_path)
blood_data = record_client().read_blood_record(blood_path)
```

For `P0001/xrays/chest.png`, `MCPRecords.record_location()` selects `xrays` and returns a root-relative path. The client calls `self.sessions['xrays'].call_tool('read_record', arguments)`. The corresponding server reads the image and returns Base64; the client decodes it into original bytes. EKG retrieval works identically through the `ekgs` session. Blood retrieval uses the `blood_tests` session and returns the parsed JSON value.

All servers expose the following tools:

| Tool | Arguments | JSON text response |
|---|---|---|
| `list_records` | `{"relative_folder":"P0001/xrays"}` | `{"paths":["P0001/xrays/chest.png"]}` |
| Image `read_record` | `{"relative_path":"P0001/xrays/chest.png"}` | `{"base64":"..."}` |
| Blood `read_record` | `{"relative_path":"P0001/blood_tests/cbc.json"}` | `{"record": ...}` |

Substitute `ekgs` or `blood_tests` as appropriate for list requests. Routing uses modality folder names, not image recognition. Patient records are files, not a database. Servers retrieve data; they do not perform medical interpretation.

## GUI workflow

1. Start the GUI and select a patient. Choose a date/time when snapshots are available.
2. Inspect the image galleries and blood-test rows; select the intended records.
3. Enter the doctor's request and press **Send to MedGemma**.
4. The GUI freezes request inputs, retrieves original image bytes through MCP and sends an Ollama HTTP request. Selected blood records are included as JSON text. X-rays precede EKGs in image order.
5. Review the final output. The original AI draft and editable version are tracked separately. Export reports explicitly to a safe folder outside patient records.
6. Use **Close** to invoke the existing cleanup handler and handle unsaved edits before exit.

The patient/date picker has Patient ID, expandable date/time rows, **Load selected date & time**, and **Close**. The main GUI also has a **Close** button. Existing detailed function comments and examples explain the callbacks.

## Limits and troubleshooting

- Servers reject wrong-modality paths, prohibited links, unsupported extensions and files larger than 64 MiB. Images support PNG, JPG/JPEG, BMP, GIF, TIF/TIFF and WebP; blood records use JSON.
- The MCP client starts all three servers when first constructed. It waits up to 30 seconds for readiness and up to 60 seconds for a tool result. Calls are serialized across modalities.
- MCP I/O has its own event loop/thread, but synchronous retrieval blocks its caller; some GUI record loading may briefly pause the window.
- Missing `mcp_client`: keep `mcp_client.py` beside the GUI. Missing server: keep all three server scripts in that same folder.
- Missing patient root: correct `PATIENT_RECORDS_DIR` or restore the directory. Changing the launch directory does not change the GUI-relative records root.
- Ollama connection failure: verify the address, service availability, model installation and remote reachability.
- Cancel closes a local response and rejects later output; it does not guarantee that remote inference has stopped. MCP close requests asynchronous cleanup without joining background threads.
- This application is not an authentication system. Patient/modality path checks do not implement user-level access control or database permissions.

## Cleanup and validation

Run `config\uninit.bat` or `bash config/uninit.sh` for optional cleanup. They ask before removing global Codex or a project `.venv` and preserve Python packages, source, settings, Ollama/models and patient records. Close the GUI first; uninit does not stop active servers.

Source syntax, MCP record communication/path rejection, and selected setup checks were exercised during development. Final Windows appearance, real installer execution and the user's Ollama host still require local verification.
