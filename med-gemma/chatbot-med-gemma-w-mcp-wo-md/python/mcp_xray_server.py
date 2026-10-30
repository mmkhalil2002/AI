"""Read-only xrays MCP server. Started automatically by client.py."""
import base64
import json
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

MODALITY = 'xrays'
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff', '.webp'}
BASE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('PATIENT_RECORDS_DIR', 'patient_records')).expanduser()
if not ROOT.is_absolute():
    ROOT = BASE / ROOT
ROOT = ROOT.resolve()
# The GUI supplies this identifier from XRAY_SERVER_ID in .env when it starts
# the child process. The folder modality remains fixed for access control.
mcp = FastMCP(os.environ.get('MCP_SERVER_ID', 'xrays'))

def confined(relative, file=False):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or len(rel.parts) < 2:
        raise ValueError('Expected patient-relative record path.')
    if any(part.startswith('.') for part in rel.parts):
        raise ValueError('Hidden paths are not allowed.')
    parts = rel.parts[1:]
    if parts.count(MODALITY) != 1 or any(p in ('xrays','ekgs','blood_tests') and p != MODALITY for p in parts):
        raise ValueError('Wrong modality.')
    if file and MODALITY not in rel.parts[:-1]:
        raise ValueError('Expected a file inside the modality directory.')
    path = ROOT
    for part in rel.parts:
        path = path / part
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            raise ValueError('Linked paths are not allowed.')
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(ROOT / rel.parts[0]):
        raise ValueError('Path escapes patient directory.')
    if file and (not resolved.is_file() or resolved.suffix.lower() not in EXTENSIONS):
        raise ValueError('Unsupported record file.')
    return resolved

@mcp.tool()
def list_records(relative_folder: str) -> str:
    """List supported files within one patient modality folder."""
    folder = confined(relative_folder)
    if folder.name != MODALITY or not folder.is_dir():
        raise ValueError('Expected a modality directory.')
    paths = []
    for path in sorted(folder.rglob('*'), key=lambda p: str(p).casefold()):
        try:
            relative = path.relative_to(ROOT).as_posix()
            safe = confined(relative, file=True)
            if getattr(safe.stat(), 'st_file_attributes', 0) & 6:
                continue
            paths.append(relative)
        except (OSError, ValueError):
            continue
    return json.dumps({'paths': paths})

@mcp.tool()
def read_record(relative_path: str) -> str:
    """Read a supported record, confined to its patient and modality."""
    path = confined(relative_path, file=True)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('Record exceeds the 64 MiB size limit.')
    if MODALITY == 'blood_tests':
        return json.dumps({'record': json.loads(path.read_text(encoding='utf-8-sig'))})
    return json.dumps({'base64': base64.b64encode(path.read_bytes()).decode('ascii')})

if __name__ == '__main__':
    mcp.run(transport='stdio')
