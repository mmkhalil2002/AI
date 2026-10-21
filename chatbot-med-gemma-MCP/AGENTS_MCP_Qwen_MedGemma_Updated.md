# MedGemma Medical AI Agent — Codex Project Instructions

## 1. Purpose and working method

Build a complete Python desktop medical-data research application from the existing reference client `client-med-gemma-2w-data-v2.py`. Treat that file and the supplied `.env` as **references**, not disposable files. Do not replace working behaviors without equivalent tested implementations. Use synthetic/de-identified test data only; the application is for research and clinician review, not autonomous medical diagnosis or treatment.

Codex should implement, install dependencies, test, and document the project in the current project directory. Work incrementally: first preserve/refactor the GUI, then build and test MCP servers, then connect the central agent. Report what has and has not been verified. Never claim to have tested the GUI visually if no display is available.

## 2. Existing reference implementation

Read `client-med-gemma-2w-data-v2.py` completely before making changes. The reference currently provides:

- Tkinter UI with connection/model controls, doctor's question, AI response, X-ray and EKG image galleries, and a selectable blood-test list.
- X-ray multi-selection (blue) and EKG multi-selection (orange), with scrollable previews. Original image bytes are sent, not resized thumbnail bytes.
- Blood-test JSON records selected/deselected by clicking individual lines. Records with a `title` field are displayed as title and remaining JSON; the existing request sends the selected record's `results` field.
- `Data > Load > X-Ray / EKG/ECG / Blood Test` imports dated folders from `DISK`; `Data > Delete` deletes only local imported copies after confirmation, never source-drive folders.
- Local dated folders `xray-data-MMDDYY`, `ekg-data-MMDDYY`, `bloodtest-data-MMDDYY`; source aliases supported in existing code. `IMAGE_DIR` defaults to `data`.
- Ollama server/model selection and connectivity check via `/api/tags`; generation via `/api/generate`; timeout, sampling, and model options from `.env`.
- Optional Windows text-to-speech, asynchronous inference, status reporting, and three response headings: X-RAY, ECG/EKG, BLOOD TEST.

Preserve these behaviors unless the new architecture explicitly supersedes them. Existing UI currently places prompt/response on the **left** (35%) and data on the **right** (65%). The target layout below intentionally places **data on the left** and prompt/response on the **right**; make this change while retaining the styling and functionality.

## 3. Target GUI: five panels and Actions section

Implement a Tkinter/ttk GUI with:

1. **X-ray panel**: scrollable gallery, multiple selection/deselection, thumbnail previews, original image path/bytes preserved, visible selected count.
2. **EKG/ECG panel**: same selection behavior; support long waveform images and horizontal/vertical scrolling.
3. **Blood Test panel**: one selectable row per valid JSON record, clear test title and result display, parsed record preserved, selected count, invalid-record reporting.
4. **Doctor's Prompt panel**: multiline input, Send button, conversation/follow-up support, clear/reset conversation control.
5. **AI Response panel**: scrollable report with source/tool provenance, progress/errors, copy capability, explicit AI-generated/clinician-review notice.

Use approximately 65% width for medical-data panels on the **left**, and 35% for prompt and response on the **right**. Within the medical-data area, stack X-ray (~40% height), EKG (~40%), and blood tests (~20%). Make dividers adjustable. Keep a top row for Ollama URL, model name, and connection check. Preserve the existing light clinical-dashboard color palette and voice toggle where supported.

### Actions section — REQUIRED

Add a clearly labeled **Actions** area, preferably beneath the doctor's prompt or as an adjacent compact toolbar. Provide these actionable controls:

- **Analyze Selected X-ray** — use only currently selected X-ray originals, through the X-ray MCP tool and MedGemma.
- **Analyze Selected EKG** — use only currently selected EKG originals, through the EKG MCP tool and MedGemma.
- **Analyze Selected Blood Tests** — use only selected blood-test records, through the Blood Test MCP tool and MedGemma.
- **Analyze All Selected Data** — analyze selected data from all three modalities and combine results in clearly separated sections; skip unselected modalities.
- **Run Doctor's Request (Agent)** — send the doctor's prompt and selected-data manifest to the central agent, which selects and sequences MCP tools based on the request.
- **Clear Selections** — deselect all X-ray, EKG, and blood-test items without deleting source files.
- **Cancel/Stop** — cancel a pending request when technically possible; otherwise stop waiting and mark the result canceled without corrupting UI state.

Show a concise execution trace/status such as `Blood Test MCP → MedGemma → Agent condition check → X-ray MCP → MedGemma → Final report`, without exposing hidden model reasoning. Disable or safely debounce duplicate actions while a request is active. Never silently process unselected records. When a user explicitly requests a modality without any selection, ask them to select data or clearly explain the missing input.

### Action semantics and routing

- Explicit single-modality buttons invoke only the corresponding tool, not all three tools.
- `Analyze All Selected Data` invokes each selected modality and synthesizes a clearly labeled summary; do not fabricate findings for missing modalities.
- `Run Doctor's Request (Agent)` lets the central agent choose tools from the selected manifest. The agent may invoke zero, one, or multiple tools, sequentially as needed.
- For conditions involving numeric results (e.g., WBC above a reference limit), extract the structured numeric value and reference interval and perform the comparison **deterministically in Python**. Do not rely solely on a language model's assertion for the threshold comparison. If the reference interval or unit is unavailable or ambiguous, report this and ask for clarification rather than guessing.
- Example: `Analyze selected blood tests first. If WBC exceeds its supplied upper reference limit, analyze the selected chest X-ray. Summarize both findings.` The blood-test server must call MedGemma and return structured values plus its analysis before the central agent checks the condition. Only then call the X-ray tool if the condition is satisfied and an X-ray is selected.
- Allow doctor-directed follow-up requests using conversation history. Preserve the selected-data snapshot for each submitted request; do not inadvertently reuse stale selections after they change.
- Every result should identify the data modality, tools used, any missing data, uncertainties, and that clinician review is required.

## 4. Architecture and generated project files

Create or refactor into:

```text
medgemma-ai-agent/
├── AGENTS.md
├── README.md
├── client_gui.py
├── medical_agent.py
├── mcp_client_manager.py
├── data_models.py
├── ollama_client.py
├── requirements.txt
├── .env.example
├── .gitignore
├── config.bat
├── config.sh
├── mcp_servers/
│   ├── __init__.py
│   ├── blood_test_server.py
│   ├── xray_server.py
│   ├── ekg_server.py
│   └── medgemma_server.py
├── tests/
│   ├── test_data_loading.py
│   ├── test_actions.py
│   ├── test_mcp_tools.py
│   ├── test_agent_routing.py
│   └── test_ollama_client.py
└── sample-data/  # synthetic only
```

`client_gui.py` handles presentation and selections; `medical_agent.py` orchestrates agent decisions and multi-turn context; `data_models.py` defines typed selection and result structures; `ollama_client.py` handles connection, requests, timeouts, and response parsing. MCP servers expose documented tools and call the shared MedGemma/Ollama integration where required. Avoid circular calls between MCP servers; if a MedGemma MCP server is exposed, use it through a clearly defined client/tool path rather than having servers recursively invoke one another.

Use **Qwen3:8B through the remote Ollama server** as the central agent's reasoning and tool-selection model; do not use OpenAI Agents SDK, ChatGPT, or the OpenAI API. The central-agent **Python orchestration code runs on the Windows GUI machine**, and communicates with Qwen over Ollama's HTTP API. Implement a tool-calling loop that presents validated MCP tool definitions to Qwen, executes only authorized tool calls through local MCP clients, returns tool results to Qwen, and stops safely on limits/errors. Use the official Python MCP SDK (or a compatible, documented MCP implementation) to create **three mandatory, runnable MCP servers**: X-ray, EKG, and Blood Test. The MCP servers run as separate local processes on the GUI machine, preferably via stdio transport; the central agent connects as an MCP client. The optional MedGemma MCP server must not replace or bypass these three required servers. Provide a deterministic Python fallback for explicit action buttons and numeric comparisons. Test tool-call compatibility with the installed Qwen model; do not assume it works without verification. Provide mock-agent/testing mode with no running model required.

## 5. MCP server contracts

**Blood Test MCP:** Accept selected structured blood-test records, preserve test names, values, units, and reference intervals, and **call MedGemma before returning** its interpretation. Return both structured data and interpretation, with missing/ambiguous values flagged. Do not substitute title strings for test results.

**X-ray MCP:** Accept selected original X-ray image bytes (or a safe internal reference resolved to original bytes), send full-resolution images to MedGemma, and return findings associated with each selected image. Do not send GUI thumbnails.

**EKG MCP:** Accept selected original EKG image bytes and return MedGemma's interpretation, clearly identified as EKG/ECG. Preserve original resolution independently of the displayed thumbnail.

**MedGemma MCP (optional):** Provide a reusable model-analysis interface if needed by the architecture, while keeping model connection logic in `ollama_client.py`. Use the installed model selected from `.env`/GUI and handle unavailable models and server errors.

Provide independently launchable MCP server entry points, tool names/descriptions, input schemas, typed outputs, startup/shutdown handling, and local stdio connection tests. The installation scripts must install MCP dependencies and the README must explain how the GUI starts/connects to the MCP server processes. Use structured tool inputs and outputs, type validation, and explicit failure reporting. The agent must not invent tool results or silently claim analysis succeeded.

## 6. Environment and configuration

Read the supplied `.env` for *configuration names and reference values*, but do not overwrite an existing `.env`, commit credentials, or copy private configuration into source control. Generate `.env.example` with placeholders and comments. Supported reference keys include:

```ini
AGENT_PROVIDER=ollama
AGENT_OLLAMA_URL=http://10.0.81.107:11434
AGENT_MODEL=qwen3:8b
OLLAMA_URL=http://10.0.81.107:11434
MEDGEMMA_MODEL=medgemma1.5:latest
IMAGE_DIR=data
DISK=E:
XRAY_WIDTH=256
XRAY_HEIGHT=256
EKG_WIDTH=800
EKG_HEIGHT=256
WINDOW_WIDTH=2400
WINDOW_HEIGHT=850
GENERATION_TIMEOUT=1200
NUM_PREDICT=2048
TEMPERATURE=0.2
REPEAT_PENALTY=1.2
REPEAT_LAST_N=256
THINK=false
```

Both Qwen3:8B and MedGemma run on **the same remote Windows or Ubuntu AI server**, served by Ollama at the address configured in `.env`. The example IP is illustrative of the supplied reference: always honor the actual `.env` and never hardcode this IP. Do not install or require either model on the Windows GUI machine. The three MCP servers, MCP client manager, GUI, and central-agent Python code all run on the GUI machine. Configure `AGENT_OLLAMA_URL` and `OLLAMA_URL` independently even when they have the same value. Check `/api/tags` for both named models, distinguish connectivity/model-not-found errors, and never silently fall back to localhost. Ensure the remote Ollama service is securely reachable; for real patient data, use an encrypted and access-controlled connection. Its `NUM_PREDICT` appears more than once, so document a single unambiguous value in the new example and retain existing `.env` unless the user approves cleanup. The source client uses `GENERATION_TIMEOUT`, whereas the reference also has `GENERATE_TIMEOUT`; document and support the intended canonical name. Check Ollama/model compatibility with `THINK` and do not assume all models accept that parameter.

The reference client currently uses `/api/generate`. Preserve compatibility with its payload or implement `/api/chat` with tests and documented migration. Use model, temperature, token limit, repetition settings, and timeouts from configuration. Keep HTTP calls off the Tkinter UI thread.

## 7. Loading, selection, and safety

- Retain `Data > Load` for X-ray, EKG, and Blood Test, and `Data > Delete` for **local imported copies only**, with confirmation. Never delete the source folders under `DISK`.
- Support the canonical dated directory patterns and existing accepted aliases. Validate MMDDYY dates. Handle missing folders, invalid JSON, corrupt images, and empty selections clearly.
- Keep original image bytes unchanged for PNG/JPEG; convert other supported formats to full-resolution PNG only when needed by Ollama. Preserve separate thumbnails for UI display.
- Blood-test JSON may contain a `title` plus `results`, or structured named results. Normalize these formats without dropping numeric values, units, or ranges.
- Do not log raw patient data, model credentials, or image bytes. Do not use OpenAI APIs. Medical data is sent from the GUI machine to the configured remote Ollama server for MedGemma analysis; the central Qwen model on that same remote server may receive the doctor's prompt, selected-data manifest, and relevant tool results. Document these data flows, minimize unnecessary exposure, and use authenticated encrypted transport for real patient data.
- Do not give definitive diagnosis, treatment instructions, or automated clinical decisions. Label findings as AI-generated research assistance requiring clinician review.

## 8. Build sequence for Codex

1. **Inspect** the existing `.py` and `.env` reference files and summarize retained behavior and requested differences.
2. **GUI phase:** Create/refactor `client_gui.py` with five panels and Actions section, using mock responses initially. Test data loading, selection, and actions without an API key.
3. **MedGemma phase:** Extract the working Ollama connection into `ollama_client.py`; verify `/api/tags` and image/text request handling using mocks first, then optional live server tests if reachable.
4. **MCP phase:** Implement each server and tool contract with clear function-level and inline comments; test them individually using synthetic data and mocked MedGemma.
5. **Agent phase:** Implement `medical_agent.py` with remote Qwen3:8B tool calling, a local MCP client manager, dynamic tool choice, deterministic numerical condition checks, multi-turn context, and integration with the GUI's `Run Doctor's Request (Agent)` action.
6. **Packaging phase:** Create `config.bat`, `config.sh`, `requirements.txt`, `.env.example`, `.gitignore`, and a detailed README with Windows and Ubuntu setup, usage, architecture, and troubleshooting.
7. **Verification phase:** Run automated tests, syntax checks, import checks, MCP server startup and tool-discovery tests, and GUI startup checks when a display is available. Test the Qwen tool-call loop with mocked responses and, when reachable, with the actual remote Qwen model. Report any blocked live integration tests without claiming success.

## 9. Tests and acceptance criteria

Write tests for folder discovery and date validation; image selection/deselection; original image bytes versus thumbnail bytes; blood-test JSON normalization and reference ranges; all seven Actions controls; no-selection behavior; correct routing for single-modality and all-selected actions; blood-first conditional WBC workflow; no X-ray call when WBC is not above the supplied upper bound; errors when reference ranges are missing; Ollama timeouts and invalid model; cancel/debounce behavior; and multi-turn follow-ups.

Use only synthetic X-ray/EKG placeholders and synthetic blood results in tests. Verify the application does not contact OpenAI APIs, and that Qwen receives only the minimum context required for orchestration. Document exactly which data crosses network boundaries.

## 10. Documentation and final report

For every function, add a header comment/docstring describing purpose, parameters, return values, side effects, and errors. Add meaningful inline comments for non-obvious code. The README must explain each module, all Actions buttons, how the agent chooses MCP tools, how to configure Ollama, how to install Qwen3:8B on the existing remote Ollama server and configure both remote models, how to start the GUI and servers, and how to test on Windows and Ubuntu.

### Required README architecture diagram and walkthroughs

Codex MUST include the following in the generated `README.md` (not just mention that they exist):

1. **Architecture diagram:** Provide a readable Mermaid flowchart **and** a plain-text ASCII fallback. Draw two distinct machine boundaries: **Windows GUI PC** (`client_gui.py`, `medical_agent.py`, `mcp_client_manager.py`, `ollama_client.py`, and the three `mcp_servers/*.py` modules) and **remote AI server** (Ollama hosting `qwen3:8b` and `medgemma1.5:latest`). Show network requests from the agent to Qwen and from each local MCP server to MedGemma; show the returned results and final GUI response. Explicitly show the three mandatory local MCP server processes and their MCP client connection to the central agent; state that there are no OpenAI API calls and no AI models installed on the GUI PC.
2. **Component explanations:** For every generated Python file, explain its purpose, main functions/classes, what calls it, what it calls, and which machine executes it. Explain the MCP tool contracts, local stdio transport, process lifecycle, and how the central agent discovers and invokes each MCP tool.
3. **End-to-end data flow:** Number the steps from selecting X-ray/EKG/blood data and entering a doctor's prompt, through Qwen's tool choice, validated MCP tool invocation, remote MedGemma inference, tool result return, and final answer displayed in the GUI. Distinguish the explicit Actions buttons (direct calls to the corresponding MCP tools) from the Agent action (Qwen-selected MCP tool calls).
4. **Worked examples:** Include (a) X-ray-only analysis, (b) all-selected-modalities analysis, and (c) a conditional workflow: read the selected blood test first, compare WBC against its **provided** upper reference limit using Python, and analyze the selected X-ray only if WBC is above that limit. Explain what happens when data or reference limits are missing. Show example requests and sample *synthetic* outcomes, not real clinical claims.
5. **Installation and execution:** Give copyable commands for Windows and Ubuntu setup scripts; remote Ollama installation and `ollama pull qwen3:8b` (and MedGemma availability); `.env` examples with `AGENT_OLLAMA_URL`, `AGENT_MODEL`, `OLLAMA_URL`, and `MEDGEMMA_MODEL`; how to verify `/api/tags`, start and verify all three MCP servers (or their GUI-managed subprocesses), start `client_gui.py`, run tests, and troubleshoot server unreachable, model missing, timeout, and insufficient memory. Never suggest the models must be installed on the Windows GUI PC.
6. **Data handling and safeguards:** Document precisely what information is sent over the network to each model, that plain HTTP is not encrypted, how to use secure transport for real patient data, the limits of model-generated medical interpretations, and clinician review requirements.
7. **MCP operations and maintenance:** Explain how to change the remote server IP or model names in `.env` without changing Python source code, and how to add a new MCP server/tool for another modality, register it with the MCP client manager, and test tool discovery.

Make the README understandable to a developer who did not build the application. Ensure the diagram and written explanations agree with the actual implementation.


At completion, list created/modified files, dependency installation results, test commands and outcomes, whether the GUI was visually tested, whether a live Ollama model was contacted, any unresolved issues, and the next action required from the user.
