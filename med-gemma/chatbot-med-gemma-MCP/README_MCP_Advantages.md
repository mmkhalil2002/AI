# Why We Use MCP in the MedGemma Medical AI Agent

## 1. Purpose

This document explains why the project uses **Model Context Protocol (MCP)** rather than having the central agent call ordinary Python medical-analysis functions directly. It complements the project's main `README.md` and `AGENTS.md`.

The system is a **research and clinician-review prototype**, not an autonomous medical diagnosis or treatment system. Use synthetic or appropriately de-identified data for development and testing.

## 2. System architecture

```text
WINDOWS GUI PC
+---------------------------------------------------------+
| Doctor's Tkinter GUI                                    |
|   - Select X-ray, EKG and blood-test data               |
|   - Enter instructions / view responses                 |
|                       |                                 |
|                       v                                 |
| Central Agent (Python orchestration code)              |
|   - Calls remote Qwen3:8B for tool selection             |
|   - Validates selections, permissions and tool results  |
|                       |                                 |
|                       v                                 |
| MCP Client Manager                                      |
|   +----------------+----------------+----------------+  |
|   | X-ray MCP      | EKG MCP        | Blood-test MCP |  |
|   | server         | server         | server         |  |
|   +----------------+----------------+----------------+  |
|                       |                                 |
+-----------------------|---------------------------------+
                        |
                        | Network requests to Ollama
                        v
REMOTE AI SERVER (IP/port defined in .env)
+---------------------------------------------------------+
| Ollama                                                  |
|   - Qwen3:8B: reasoning / tool selection                 |
|   - MedGemma 1.5: medical-data interpretation            |
+---------------------------------------------------------+
```

**Important distinction:** The MCP servers and agent code run on the Windows PC. Qwen and MedGemma run on the same remote Ollama server. Qwen does not speak MCP directly: the **Python agent/MCP client** exposes tool descriptions to Qwen, receives tool-call requests, validates them, and invokes the appropriate MCP server.

## 3. Main advantages of MCP

### A. Standard tool interface

MCP defines a common way for a client to discover and invoke tools, including their names, descriptions and input schemas. The agent does not need a different custom integration protocol for each medical tool.

### B. Easier expansion

Start with three MCP servers:

- **X-ray MCP server:** Receives selected original X-ray image data and requests MedGemma analysis.
- **EKG MCP server:** Receives selected original EKG/ECG image data and requests MedGemma analysis.
- **Blood-test MCP server:** Receives selected laboratory records and requests MedGemma interpretation.

Later, additional servers could expose MRI, CT, ultrasound or approved clinical-information tools. Each still requires integration, access controls, testing and validation; MCP does not make new tools automatically clinically reliable.

### C. Independent development and testing

Each MCP server can be started and tested separately from the GUI and Qwen. For example, the X-ray server can be tested with synthetic image data and a mocked MedGemma response to verify its input validation, tool schema and error handling.

### D. Reuse with other compatible agents

An MCP server can potentially be used by another MCP-compatible client without rewriting its tool interface. Switching from Qwen to another reasoning model generally affects the central agent's model adapter and tool-calling behavior, not the medical MCP servers.

### E. Separation of responsibilities

- **GUI:** Displays data and gathers doctor requests.
- **Qwen:** Proposes which tool to call and in what order.
- **Agent Python code:** Applies safety constraints, verifies selections, executes the workflow and handles results.
- **MCP servers:** Offer well-defined medical-analysis capabilities.
- **MedGemma:** Interprets medical images and laboratory information.

This separation can make the system easier to maintain as it grows.

## 4. Example: conditional analysis

Doctor's request:

> Analyze the selected blood tests first. If WBC is above the supplied laboratory upper reference limit, analyze the selected chest X-ray and summarize the findings.

1. The GUI sends the doctor's request and selected-data references to the central Python agent.
2. The agent asks remote Qwen to choose the next available tool.
3. Qwen requests the blood-test tool; the Python MCP client calls the local Blood-test MCP server.
4. The Blood-test MCP server sends the selected blood-test data to remote MedGemma and returns a structured result.
5. **Python**, not the language model alone, compares the numeric WBC result with the **supplied** reference limit, accounting for units and missing values.
6. If the condition is satisfied and a chest X-ray is selected, the agent invokes the X-ray MCP server; otherwise, it skips that step.
7. The agent prepares a response identifying which analyses were performed and which were skipped.

MCP standardizes the tool invocation and results exchange; it does **not** perform reasoning or medical interpretation by itself.

## 5. MCP versus ordinary Python functions

| Consideration | MCP servers | Direct Python functions |
|---|---|---|
| Tool interface | Standard protocol and schemas | Application-specific interface |
| Discovery | MCP client can list available tools | Usually manually registered |
| Independent processes | Natural fit | Requires additional design |
| Reuse across compatible applications | Easier | May need wrappers or adapters |
| Adding new tool types | Consistent integration pattern | Simpler for a small single-process app |
| Initial development and debugging | More moving parts | Usually simpler |
| Performance overhead | Additional local communication | Usually lower |

**Why choose MCP here?** The project is intended to support multiple medical tools and potentially expand over time. The reusable interface and independent testing justify the additional complexity. For a fixed, small, single-machine prototype, ordinary Python functions would also work.

## 6. Limitations and responsibilities

MCP is **not** a medical validation framework, security boundary, or accuracy guarantee. The application must still:

- Restrict tools to doctor-selected data and authorized operations.
- Validate tool inputs and outputs and handle unavailable data.
- Use deterministic code for numeric conditions and reference-range comparisons.
- Avoid making clinical claims without appropriate expert validation.
- Protect medical data in transit to the remote server with suitable authentication, encryption and access controls; plain HTTP alone does not provide transport encryption.
- Log errors without unnecessarily exposing patient information.
- Test Qwen's tool selection and the MCP client/server communication independently.

## 7. Deployment and configuration

Both models can be served by one remote Ollama instance:

```env
AGENT_PROVIDER=ollama
AGENT_OLLAMA_URL=http://10.0.81.107:11434
AGENT_MODEL=qwen3:8b

OLLAMA_URL=http://10.0.81.107:11434
MEDGEMMA_MODEL=medgemma1.5:latest
```

The IP above is an **example from the project's configuration**. Always use the actual values in `.env`; do not hardcode the address. For production medical data, replace or protect the plain-HTTP connection with an appropriate secure transport.

## 8. Relationship to other project documents

- **`AGENTS.md`** tells Codex what to implement and test.
- **`README.md`** should explain installation, execution, architecture, configuration and troubleshooting.
- **`README_MCP_Advantages.md` (this document)** explains the design rationale, benefits, tradeoffs and an end-to-end MCP example.

Keep this document alongside `AGENTS.md` in the project directory and link to it from the main `README.md`.
