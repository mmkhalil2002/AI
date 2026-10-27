# MedGemma MCP servers

This package contains mcp_xray_server.py, mcp_ekg_server.py, mcp_blood_server.py, requirements.txt, and this README.

## Setup

Extract the files into the same folder as your updated GUI (client_medgemma_patients_mcp.py) and separate mcp_client.py. Those two client files are supplied separately and are not included in this ZIP.

Use Python 3.10 or newer. No virtual environment is required.

Windows installation:

    py -m pip install -r requirements.txt

Windows launch:

    py client_medgemma_patients_mcp.py

Ubuntu: install Tkinter with sudo apt install python3-tk, install requirements using your Python installation's supported pip method, and run python3 client_medgemma_patients_mcp.py.

## Configuration and records

Keep your existing .env beside the GUI. Set PATIENT_RECORDS_DIR=patient_records and retain your actual OLLAMA_URL and MEDGEMMA_MODEL values. The GUI resolves a relative records directory beside its Python file and passes the absolute path to the servers. It must already exist.

Store each patient's files under:

    patient_records/<PatientID>/xrays/
    patient_records/<PatientID>/ekgs/
    patient_records/<PatientID>/blood_tests/

X-ray and EKG folders contain supported image files; blood_tests contains JSON files. Existing GUI date/time snapshot layouts also remain supported.

## Operation

Run only the GUI. Its separate MCP client starts all three servers automatically on the first record request, using the same Python interpreter. No server terminals or TCP ports are needed. Do not start these stdio servers manually for normal GUI use.

Each server exposes list_records and read_record tools. The X-ray server serves xrays, the EKG server serves ekgs, and the blood server serves blood_tests. Images return as Base64; blood records return parsed JSON. The servers check patient/modality paths, links, extensions and a 64 MiB read limit. They expose no write tools and perform no medical inference.

The GUI sends selected data directly to MedGemma through Ollama. Qwen is not called. Closing the GUI requests MCP shutdown. Startup/record errors are returned to the GUI; synchronous record calls can briefly pause the interface.

The MCP SDK is pinned below version 2 for the ClientSession/FastMCP APIs used by these scripts. pyttsx3 supports optional voice output; it may require operating-system speech components.

Use educational or de-identified data. AI output requires clinician review.
