"""Generate/refresh project requirements using this MedGemma application's dependencies.
Preserve additional existing requirements. No package installation occurs here.
"""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
source = root / 'python'
required = ['client_medgemma_patients_mcp.py', 'mcp_client.py', 'mcp_xray_server.py', 'mcp_ekg_server.py', 'mcp_blood_server.py']
missing = [name for name in required if not (source / name).is_file()]
if missing:
    sys.exit('[ERROR] Missing application files in ' + str(source) + ': ' + ', '.join(missing))
path = root / 'requirements.txt'
existing = path.read_text(encoding='utf-8-sig').splitlines() if path.exists() else []
# Explicit versions are required: automatic import scanners cannot infer SDK
# compatibility constraints such as mcp<2. Keep unrelated existing dependencies.
known = {'mcp', 'pillow', 'pyttsx3'}
retained = [line for line in existing if not re.match(r'^(mcp|pillow|pyttsx3)(?=[\s<>=!~;\[]|$)', line.strip(), re.I)]
text = '\n'.join(retained + ['# MedGemma GUI + separate MCP client and servers', 'mcp>=1.20,<2', 'Pillow>=10', 'pyttsx3>=2.90']) + '\n'
# Avoid accumulating the generated heading when init is rerun.
heading = '# MedGemma GUI + separate MCP client and servers\n'
text = text.replace(heading, '')
text = heading + text.lstrip('\n')
path.write_text(text, encoding='utf-8')
# Keep the application copy synchronized with the root installation manifest.
if source != root:
    (source / 'requirements.txt').write_text(text, encoding='utf-8')
print('[INFO] Requirements refreshed: ' + str(path))
print('[INFO] Application directory: ' + str(source))
