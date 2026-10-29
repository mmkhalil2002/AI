# MedGemma MCP setup — existing python/ and config/ layout

Extract into your existing project root. Keep this layout:

```text
project/
  requirements.txt
  README.md
  config/
    init.bat
    init.sh
    run.bat
    run.sh
    uninit.bat
    generate_requirements.py
    README.txt
  python/
    client_medgemma_patients_mcp.py   (keep existing file)
    mcp_client.py                     (keep existing file)
    mcp_xray_server.py
    mcp_ekg_server.py
    mcp_blood_server.py
    requirements.txt
    .env                              (keep existing file)
    patient_records/                  (keep existing directory)
```

GUI, MCP client, .env and patient data are not included; retain your copies in python/. No files need to move out of python/.

Windows: run config\init.bat, then config\run.bat.
Ubuntu: run bash config/init.sh, then bash config/run.sh.

The original init scripts generated requirements only when missing. Updated init calls config/generate_requirements.py on every run. The helper requires the five application Python files in python/, refreshes the root requirements.txt and mirrors it into python/requirements.txt. It preserves additional requirements and sets mcp>=1.20,<2, Pillow>=10, and pyttsx3>=2.90. This is an explicit dependency list for this application, not a general import scanner. pipreqs is not needed for generation. Init installs the root requirements file.

Python 3.10+ is required. No virtual environment is created; existing init behavior removing project-root .venv is retained. Ubuntu includes Tkinter and uses pip --user --break-system-packages. Node/npm and Codex setup is retained; local Ollama/MedGemma setup remains optional. Qwen download prompts are removed. Reopen your terminal after init for PATH changes.

Keep python/.env and its OLLAMA_URL and MEDGEMMA_MODEL settings. PATIENT_RECORDS_DIR=patient_records resolves to python/patient_records. The GUI passes the absolute records directory to the servers. No MCP port settings are required.

Init does not start the application. Run launches the GUI in python/; its separate MCP client starts all three stdio servers when needed. Closing the GUI requests server shutdown.

Windows uninit retains confirmation prompts and preserves installed Python packages, settings, source, requirements and patient files. It can optionally remove global Codex and a leftover .venv. No Ubuntu uninit was supplied or added.

Validation: requirement generation and repeatability checked for this directory layout; shell syntax and ZIP integrity checked. Windows execution and actual installations were not tested here.
