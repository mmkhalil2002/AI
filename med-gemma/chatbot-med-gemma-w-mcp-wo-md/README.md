# Medical MCP + MedGemma — Installation and Run Guide

This guide describes the Windows and Ubuntu installers, the Python configuration routine, and the Medical MCP client/server components. The **installer package** contains `install_windows.bat`, `install_ubuntu.sh`, and `config.py`. The MCP application files must be placed in the same project directory separately.

## 1. What each file/routine does

| File / routine | Brief description |
|---|---|
| `install_windows.bat` | Checks for Python 3.11+, installs Python 3.12 using `winget` if missing, creates `.venv`, upgrades pip, and runs `config.py`. |
| `install_ubuntu.sh` | Installs Ubuntu Python packages with `apt`, checks for Python 3.11+, creates `.venv`, upgrades pip, and runs `config.py`. |
| `config.py` → `config()` | Installs `mcp[cli]`, `openai`, `requests`, `python-dotenv`, and `Pillow` into the active Python environment; verifies their imports. |
| `medical_client.py` → `run_agent()` | Accepts a doctor's question and selected X-ray/EKG/blood data, connects to MCP servers, discovers tools, asks OpenAI to select tools, invokes them, and returns the final text. |
| `medical_client.py` → `save_image()` | Saves a selected PIL image or image file as a complete PNG in a temporary local session directory. |
| `medical_client.py` → `main()` | Parses command-line arguments and runs `run_agent()` for terminal-based testing. |
| `xray_server.py` → `analyze_xray()` | Loads the selected X-ray PNG, encodes the complete image as Base64, and requests MedGemma analysis. MCP port **8001**. |
| `ekg_server.py` → `analyze_ekg()` | Loads the selected EKG PNG, encodes the complete image as Base64, and requests MedGemma analysis. MCP port **8002**. |
| `blood_server.py` → `analyze_blood_test()` | Loads selected blood-test JSON and sends the values in a text prompt to MedGemma. MCP port **8003**. |
| `common.py` → `session_file()` | Validates the local session ID and permitted filename, then locates the data saved by the client. |
| `ollama_config.py` → `generate()` | Reads MedGemma settings from `.env` and calls Ollama's `/api/generate` endpoint with text and optional images. |
| `.env` | Holds the Ollama IP/port, MedGemma model name, generation timeout, and generation parameters. |

The MCP servers and client run on the **same computer** in this prototype because they share a local temporary session directory. Ollama/MedGemma can run on another computer reachable through the IP in `.env`.

## 2. Recommended project directory

```text
medical_mcp/
├── install_windows.bat
├── install_ubuntu.sh
├── config.py
├── README.md
├── medical_client.py
├── xray_server.py
├── ekg_server.py
├── blood_server.py
├── common.py
├── ollama_config.py
├── .env
└── .venv/                 # created by installer
```

## 3. Configure MedGemma

Put a file named exactly `.env` in the project folder. Relevant settings from your supplied configuration are:

```ini
OLLAMA_URL=http://10.0.81.107:11434
MEDGEMMA_MODEL=medgemma1.5:latest
GENERATE_TIMEOUT=1800
NUM_PREDICT=2048
TEMPERATURE=0.2
REPEAT_PENALTY=1.2
REPEAT_LAST_N=256
```

`OLLAMA_URL` is the **base address**; `ollama_config.py` appends `/api/generate`. Verify the address and model name match your actual Ollama server. If Ollama runs remotely, the machine hosting it must permit network access from the MCP server computer.

## 4. Install on Windows

**Prerequisites:** Windows Command Prompt, internet access, and `winget` if Python is not already installed. `winget` may require a normal interactive install/approval. Python 3.11 or newer is required.

1. Put `install_windows.bat` and `config.py` in the project folder.
2. Open **Command Prompt** in that folder.
3. Run:

```bat
install_windows.bat
```

The script checks/installs Python, creates `.venv`, upgrades pip, and calls `config.py`. It does not install Ollama or MedGemma. If Python was just installed but cannot be found, open a new terminal and rerun the installer.

### Start the servers on Windows

Open **three separate Command Prompt windows** in the project folder and leave them running:

```bat
.venv\Scripts\python.exe xray_server.py
```

```bat
.venv\Scripts\python.exe ekg_server.py
```

```bat
.venv\Scripts\python.exe blood_server.py
```

In a **fourth** Command Prompt window, configure the OpenAI API key for that session and run a test:

```bat
set OPENAI_API_KEY=YOUR_API_KEY
.venv\Scripts\python.exe medical_client.py --xray "C:\medical\xray.png" --question "Review this chest X-ray"
```

Replace the example file path with a real test image. To test blood results:

```bat
.venv\Scripts\python.exe medical_client.py --blood "C:\medical\blood.json" --question "Review these blood-test values"
```

## 5. Install on Ubuntu

**Prerequisites:** Internet access and `sudo` permission to install system packages. The default Ubuntu repositories must provide Python **3.11 or newer**; the installer stops otherwise.

1. Put `install_ubuntu.sh` and `config.py` in the project folder.
2. Open a terminal in that folder.
3. Run:

```bash
chmod +x install_ubuntu.sh
./install_ubuntu.sh
```

The script installs Python/pip/venv system packages, creates `.venv`, upgrades pip, and runs `config.py`.

### Start the servers on Ubuntu

Open **three separate terminal windows** in the project folder:

```bash
.venv/bin/python xray_server.py
```

```bash
.venv/bin/python ekg_server.py
```

```bash
.venv/bin/python blood_server.py
```

In a **fourth** terminal, set the API key and run the client:

```bash
export OPENAI_API_KEY='YOUR_API_KEY'
.venv/bin/python medical_client.py --xray /path/to/xray.png --question 'Review this chest X-ray'
```

For blood-test JSON:

```bash
.venv/bin/python medical_client.py --blood /path/to/blood.json --question 'Review these blood-test values'
```

## 6. How a request flows

1. `medical_client.py` receives the doctor's question and selected complete data.
2. It saves the selected image(s)/blood data to a temporary local session folder.
3. It connects to the appropriate MCP endpoints: `127.0.0.1:8001/mcp`, `:8002/mcp`, and/or `:8003/mcp`.
4. It calls `list_tools()` and sends the available tool descriptions to the OpenAI API.
5. OpenAI returns a function selection; the Python client routes it with `call_tool()`.
6. The selected MCP server loads the data and calls `ollama_config.generate()`.
7. `generate()` sends the complete image or blood data to the configured Ollama/MedGemma server.
8. The textual findings return through MCP to the client, which can send them to OpenAI for further tool selection or summarization.
9. The client returns the final response and removes its temporary session files.

## 7. Troubleshooting

- **`python` or `py` not found (Windows):** rerun `install_windows.bat`; if just installed, open a fresh Command Prompt.
- **Ubuntu reports Python version too old:** install a supported Python version using your approved Ubuntu distribution/repository procedure, then rerun; the installer does not add third-party repositories.
- **`ModuleNotFoundError`:** run the installer again and use `.venv`'s Python rather than system Python.
- **MCP connection refused:** ensure all three server processes are running and ports 8001–8003 are free.
- **Ollama connection refused/timeouts:** check `.env`, server reachability, firewall, and Ollama service status. On a permitted network, test `curl http://10.0.81.107:11434/api/tags`.
- **Ollama model not found:** check `ollama list` on the Ollama host and match `MEDGEMMA_MODEL` exactly.
- **OpenAI authentication failure:** configure `OPENAI_API_KEY` in the **client's** terminal; it is separate from ChatGPT Plus.
- **No selected data:** provide at least one image or blood JSON input to the client.

## 8. Privacy and limitations

This is an **educational integration prototype**, not a validated clinical diagnostic system. The complete images/blood data are sent to the Ollama host identified by `.env` (which may be remote). The OpenAI API receives the doctor's question and MedGemma's **textual findings**, potentially containing sensitive information; image pixels are not sent directly to OpenAI by this architecture. Use synthetic/non-identifiable test data until security, authorization, encryption, retention, and clinical validation requirements are met. The installers install Python dependencies only; they do **not** install the Ollama service, download MedGemma, or configure network security.
