"""
FILE: blood_server.py
PURPOSE: Expose an MCP tool for Blood test review using MedGemma through Ollama.

DATA FLOW:
    GUI -> MCP client -> this server -> Ollama host configured in .env
    Ollama -> this server -> MCP client -> GUI / orchestration model

This MCP server listens ONLY on loopback port 8003. The Ollama service may
run on another machine; its IP address and port come from .env.
Selected medical data is retrieved from a client-created local session folder.
This is an educational prototype, not a clinically validated application.
"""

# Standard library: parse and serialize complete blood-test JSON records.
import json

# FastMCP registers Python functions as tools and hosts the MCP endpoint.
from mcp.server.fastmcp import FastMCP

# session_file checks the request ID and finds the locally selected data.
from common import session_file

# generate loads .env configuration and performs the real HTTP call to Ollama.
from ollama_config import generate

# Create the local MCP service. This is NOT the MedGemma server address.
# The client connects to http://127.0.0.1:8003/mcp.
mcp = FastMCP('Blood test Server', host='127.0.0.1', port=8003)


# ============================================================================
# ROUTINE: analyze_blood_test(session_id, question)
# PURPOSE: Ask local/remote Ollama MedGemma to review selected Blood test data.
# INPUTS:
#   session_id: trusted client-generated 32-character hex request identifier.
#   question:   natural-language question typed by the doctor in the GUI.
# OUTPUT:
#   str: textual observations produced by the configured MedGemma model.
# PROCESSING:
#   1. Resolve the selected data from the client-created session folder.
#   2. Read the entire selected record / image without truncation.
#   3. Prepare a task-specific prompt including the doctor's question.
#   4. Call generate(), which reads Ollama IP, model, timeout, and options
#      from .env and posts to the /api/generate REST endpoint.
#   5. Return MedGemma's textual response as the MCP tool result.
# ERRORS:
#   Missing data, invalid session, HTTP failure, or malformed Ollama replies
#   raise exceptions; the MCP client must handle and report them.
# SECURITY:
#   This example assumes trusted processes on the same computer. Session
#   IDs alone are not authorization; do not expose it to untrusted users.
# ============================================================================
@mcp.tool()
def analyze_blood_test(session_id: str, question: str) -> str:
    """Analyze selected laboratory values, units, and ranges with MedGemma."""
    # 1. Validate the request session and locate the selected JSON file.
    #    The trusted MCP client saved this file before invoking the tool.
    data_path = session_file(session_id, 'blood.json')

    # 2. Read the COMPLETE JSON file as UTF-8 text. This preserves any
    #    non-English labels and avoids dropping selected records.
    raw_json = data_path.read_text(encoding='utf-8')

    # 3. Parse JSON to catch invalid formatting and obtain Python values.
    #    The parsed object may be a dict, a list, or nested lab records.
    blood_data = json.loads(raw_json)

    # 4. Convert data back to readable JSON, preserving units and reference
    #    ranges exactly as supplied. Do not invent missing ranges.
    blood_text = json.dumps(blood_data, ensure_ascii=False, indent=2)

    # 5. Build a task-specific prompt with the doctor's question and the
    #    selected laboratory values. No image is needed for this tool.
    prompt = (
        'Educational laboratory review. Compare values only with supplied '
        'reference ranges and explain uncertainty.\n'
        f'Doctor question: {question}\n'
        f'Blood-test data:\n{blood_text}'
    )

    # 6. Call Ollama through the shared .env-aware helper. The complete
    #    laboratory data is sent as prompt text to the MedGemma model.
    findings = generate(prompt)

    # 7. Return textual findings to the MCP client for display or orchestration.
    return findings


# ============================================================================
# ROUTINE: MAIN / MCP SERVER STARTUP
# PURPOSE: Start the MCP Streamable HTTP service when run as a Python script.
# INPUTS: None; uses the FastMCP configuration above.
# OUTPUT: Long-running HTTP listener at http://127.0.0.1:8003/mcp.
# PROCESSING:
#   1. Confirm this file is executed directly, not imported.
#   2. Start the Streamable HTTP transport.
#   3. Keep serving MCP requests until the process is stopped.
# ============================================================================
if __name__ == '__main__':
    # This blocks the current terminal while the server handles MCP requests.
    mcp.run(transport='streamable-http')
