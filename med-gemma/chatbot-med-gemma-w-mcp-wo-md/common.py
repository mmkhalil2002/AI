"""Shared local-session file lookup used by all three MCP servers."""

# re validates identifiers; tempfile and Path build the local data directory.
import re
import tempfile
from pathlib import Path

# Must match the location used by the MCP client when saving selected data.
ROOT = Path(tempfile.gettempdir()) / 'medical_mcp_sessions'


# ============================================================================
# ROUTINE: session_file(session_id, filename)
# PURPOSE: Locate one selected image or blood-test JSON file for this request.
# INPUTS: session_id (32 lowercase hexadecimal characters), filename (allowed
#         name: xray.png, ekg.png, or blood.json).
# OUTPUT: Path to an existing file; otherwise raises an exception.
# STEPS: Validate ID, validate filename, build path, confirm file exists.
# NOTE: This is path validation, NOT access control or patient authentication.
# ============================================================================
def session_file(session_id: str, filename: str) -> Path:
    """Return a validated path to a selected medical-data file."""
    # 1. Reject malformed session IDs before constructing a filesystem path.
    if not re.fullmatch(r'[0-9a-f]{32}', session_id):
        raise ValueError('Invalid session ID')

    # 2. Accept only the three expected filenames; block arbitrary traversal.
    if filename not in {'xray.png', 'ekg.png', 'blood.json'}:
        raise ValueError('Unsupported medical-data filename')

    # 3. Build the path inside the current local session's directory.
    path = ROOT / session_id / filename

    # 4. Confirm that the client saved the selected data before this tool call.
    if not path.is_file():
        raise FileNotFoundError('Selected data is not available for this session')

    # 5. Return the path so the caller can read the full file contents.
    return path
