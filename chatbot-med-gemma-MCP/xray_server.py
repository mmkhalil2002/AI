"""
FILE: xray_server.py
PURPOSE: Expose an MCP tool for X-ray review using MedGemma through Ollama.

DATA FLOW:
    GUI -> MCP client -> this server -> Ollama host configured in .env
    Ollama -> this server -> MCP client -> GUI / orchestration model

This MCP server listens ONLY on loopback port 8001. The Ollama service may
run on another machine; its IP address and port come from .env.
Selected medical data is retrieved from a client-created local session folder.
This is an educational prototype, not a clinically validated application.
"""

# Standard library: encode binary image bytes for Ollama's JSON API.
import base64

# FastMCP registers Python functions as tools and hosts the MCP endpoint.
from mcp.server.fastmcp import FastMCP

# session_file checks the request ID and finds the locally selected data.
from common import session_file

# generate loads .env configuration and performs the real HTTP call to Ollama.
from ollama_config import generate

# Create the local MCP service. This is NOT the MedGemma server address.
# The client connects to http://127.0.0.1:8001/mcp.
mcp = FastMCP('X-ray Server', host='127.0.0.1', port=8001)


# ============================================================================
# ROUTINE: analyze_xray(session_id, question)
# PURPOSE: Ask local/remote Ollama MedGemma to review selected X-ray data.
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
def analyze_xray(session_id: str, question: str) -> str:
    """Analyze selected X-ray image using MedGemma; use for radiographic images and chest X-ray findings."""
    # 1. Validate the session ID and locate the selected image in the local
    #    session folder. This prevents the model from supplying a file path.
    image_path = session_file(session_id, 'xray.png')

    # 2. Read the FULL PNG image as bytes; no cropping or resizing occurs here.
    #    The GUI/client is responsible for saving the intended selected image.
    image_bytes = image_path.read_bytes()

    # 3. Base64-encode all image bytes, then convert the bytes to ASCII text.
    #    Ollama expects image content as Base64 strings in its 'images' field.
    image_base64 = base64.b64encode(image_bytes).decode('ascii')

    # 4. Build the prompt. It identifies the analysis domain, explicitly asks
    #    for uncertainty, and includes the doctor's exact original question.
    #    Prompt text alone does not guarantee clinically reliable results.
    prompt = (
        'Educational X-ray review. Describe visible findings and limitations. '
        'Do not invent measurements or claim a definitive diagnosis.\n'
        f'Doctor question: {question}'
    )

    # 5. Send BOTH the prompt and the COMPLETE image to Ollama/MedGemma.
    #    generate() reads OLLAMA_URL and MEDGEMMA_MODEL from the .env file.
    #    Its return value is the model-generated text (not a simulated result).
    findings = generate(prompt, images=[image_base64])

    # 6. Give the textual findings back to the MCP client. The client decides
    #    whether to display them or pass them to the orchestration model.
    return findings


# ============================================================================
# ROUTINE: MAIN / MCP SERVER STARTUP
# PURPOSE: Start the MCP Streamable HTTP service when run as a Python script.
# INPUTS: None; uses the FastMCP configuration above.
# OUTPUT: Long-running HTTP listener at http://127.0.0.1:8001/mcp.
# PROCESSING:
#   1. Confirm this file is executed directly, not imported.
#   2. Start the Streamable HTTP transport.
#   3. Keep serving MCP requests until the process is stopped.
# ============================================================================
if __name__ == '__main__':
    # This blocks the current terminal while the server handles MCP requests.
    mcp.run(transport='streamable-http')
