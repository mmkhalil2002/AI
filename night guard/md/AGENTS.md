# Night Guard — Real-Time Camera Server and Windows Client — Codex Instructions

## Objective
Implement a cross-platform Python server (Ubuntu 24.04 or Windows 11) that streams live video from **1–4 USB cameras** to a separate Windows 11 Python GUI over a local network. The GUI displays four fixed panels in two columns and two rows, with a reserved control area. **Do not record videos, save snapshots, or create a captures directory.** This file is a specification, not executable code.

## Project structure (required)
```text
project_root/
├── requirements.txt            # At project root, NOT config/
├── md/
│   └── AGENTS.md
├── config/
│   ├── init.sh                 # Ubuntu 24.04 setup
│   ├── init.bat                # Windows setup
└── python/
    ├── night_guard_server.py        # Cross-platform server
    ├── night_guard_client.py           # Windows GUI client
    ├── server.env             # Server-only configuration
    └── client.env             # Windows GUI-only configuration
```
All paths must be resolved relative to the Python script or init script location, not the current working directory. Do not generate a `captures/` directory or any recording/snapshot code. Do not require a virtual environment on Ubuntu.

## Configuration: two separate files
**Use `python/server.env` for the server and `python/client.env` for the GUI.** Do not require a combined `.env`. Neither program should load the other's configuration. Do not overwrite existing configuration during initialization. Supply documented sample values if files are absent, and clearly explain how to customize them. Never commit private settings or secrets.

Example `python/server.env` (Ubuntu camera paths are illustrative; actual paths must be verified):
```ini
CAMERA_COUNT=4
FRONT_CAMERA=/dev/video0
BACK_CAMERA=/dev/video2
LEFT_CAMERA=/dev/video4
RIGHT_CAMERA=/dev/video6
WIDTH=640
HEIGHT=480
FPS=15
JPEG_QUALITY=80
SERVER_HOST=127.0.0.1
SERVER_PORT=14095
CONTROL_PORT=14098
```
On Windows, camera identifiers are OpenCV device indices (`0`, `1`, `2`, `3`) instead of Linux `/dev/video*` paths. `CAMERA_COUNT` from 1 through 4 enables the first N positions in the fixed order FRONT, BACK, LEFT SIDE, RIGHT SIDE. No attempt should be made to open disabled cameras. `CONTROL_PORT` is reserved for future use; do not open it or advertise a control service until implemented.

Example `python/client.env`:
```ini
SERVER_URL=http://192.168.12.133:14095
GUI_TITLE=Four USB Camera Monitor
GUI_WIDTH=1400
GUI_HEIGHT=900
CONTROL_PANEL_HEIGHT=200
```
`SERVER_URL` is an example LAN address, **not** a known current address. The user must configure it to the reachable server IP. For remote LAN use, change `SERVER_HOST` to the server's LAN interface or `0.0.0.0` and restrict firewall access to trusted devices. Do not expose streams to the public internet.

## Video server: python/night_guard_server.py
1. Run on Ubuntu 24.04 and Windows 11, using the appropriate OpenCV capture backend (V4L2 on Linux, DirectShow or supported alternative on Windows).
2. Run one capture worker per enabled camera; a failed camera must not block the others. Continuously keep the latest JPEG-encoded frame in memory only.
3. Provide HTTP MJPEG endpoints `GET /stream/front`, `/stream/back`, `/stream/left`, `/stream/right` and `GET /status` with enabled/connected/offline state for all four positions. Status responses must distinguish disabled cameras from disconnected ones.
4. Stream frames in real time to the GUI with reasonable latency and bounded memory use; drop stale frames rather than accumulate queues. Retry disconnected cameras periodically.
5. No snapshot endpoints, files, recording, or disk output. Shut down camera handles and workers cleanly.
6. Keep compatibility with `python night_guard_server.py -v 2 -p 14095 -c 14098`; `-p` overrides `SERVER_PORT`, `-c` refers to reserved `CONTROL_PORT`, and `-v` controls logging verbosity. Explain that control port is not listening yet.
7. Read only `python/server.env`, with meaningful validation and error messages. Default to localhost binding for safety.

## Windows GUI: python/night_guard_client.py
- Runs on Windows 11, connects via `SERVER_URL` from `python/client.env`, and **never opens local USB cameras**.
- Use PySide6 or another maintained desktop GUI toolkit. Perform network stream reading and reconnection off the GUI thread.
- Always display the same four panels in a 2 × 2 layout:
```text
+--------------------------+--------------------------+
| FRONT                    | BACK                     |
|       Live video         |       Live video         |
+--------------------------+--------------------------+
| LEFT SIDE                | RIGHT SIDE               |
|       Live video         |       Live video         |
+--------------------------+--------------------------+
| RESERVED CONTROL AREA                               |
| Controls will be defined later                       |
+-----------------------------------------------------+
```
- Preserve aspect ratio when resizing. Disabled positions display `Not configured`; offline cameras display `Camera offline`; unavailable server displays a clear connection status. Never freeze the UI during network failure.
- The reserved control area is a visible, empty placeholder. **Do not invent buttons or functionality.** Stop worker threads and network requests on GUI exit.

## Initialization and dependencies
- `requirements.txt` is at **project root**, not under `config/`. Include Python dependencies needed for the server and GUI. Clearly document platform-specific system packages such as `v4l-utils` on Ubuntu.
- `config/init.sh` runs on Ubuntu 24.04 and installs required system/Python dependencies without creating a virtual environment. `config/init.bat` runs on Windows 11. Both must locate `../requirements.txt` relative to their own directory. Both programs must locate their own `.env`-format configuration beside the Python script, regardless of current working directory.
- Preserve any existing useful initialization functionality (including Python/Node/Codex/Ollama setup) when updating previously supplied init scripts; do not silently remove it. Never start the server or GUI automatically during initialization.

## Run commands (from project root)
```bash
bash config/init.sh
python3 python/night_guard_server.py -v 2 -p 14095 -c 14098
```
```bat
config\init.bat
python python\night_guard_server.py -v 2 -p 14095 -c 14098
python python\night_guard_client.py
```
Run the GUI on Windows; the server can run on either OS. Ensure the client is configured to connect to the correct LAN address and port.

## Acceptance criteria
- [ ] Supports 1, 2, 3, or 4 USB cameras with OS-appropriate identifiers in `server.env`.
- [ ] Server streams live video without writing images or video files.
- [ ] Windows GUI shows fixed FRONT/BACK/LEFT SIDE/RIGHT SIDE panels in a 2 × 2 layout.
- [ ] Reserved control region is present but has no implemented controls.
- [ ] Server and GUI read separate environment files; no combined `.env` required.
- [ ] `requirements.txt` is at project root; init scripts remain under `config/`.
- [ ] Reconnects after camera/network interruptions; disabled cameras do not cause errors.
- [ ] No `captures/` directory and no recording/snapshot functionality.
- [ ] Existing init setup behavior is preserved, with changes documented.
- [ ] Cross-platform server and Windows GUI are tested for syntax/startup; real USB camera operation must be verified on actual hardware.

## Coding guidance
Use clear function docstrings with examples for important functions. Do not assume a camera path exists. Log actionable failures without exposing secrets. Summarize changed files and exact test commands after implementation.
