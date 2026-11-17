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
WIDTH=0
HEIGHT=0
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
GUI_WIDTH=880
GUI_HEIGHT=620
WIDTH=480
HEIGHT=320
CONTROL_PANEL_WIDTH=200
```
`SERVER_URL` is an example LAN address, **not** a known current address. The user must configure it to the reachable server IP. For remote LAN use, change `SERVER_HOST` to the server's LAN interface or `0.0.0.0` and restrict firewall access to trusted devices. Do not expose streams to the public internet.

## Video server: python/night_guard_server.py
When `WIDTH=0` and `HEIGHT=0` (also the defaults when omitted), query all enabled cameras using Linux `v4l2-ctl --list-formats-ext` and choose the lowest resolution supported by every enabled camera. Choose the smallest pixel area, breaking ties by width then height. Include discrete sizes and continuous/stepwise ranges; do not combine incompatible width/height pairs. Apply the selected dimensions to all camera workers, choosing an advertised pixel format for each camera. Report the selected dimensions at startup. Fail startup with an actionable message if any enabled camera cannot be queried, no common resolution exists, or v4l-utils is missing. Disabled cameras must not be queried. Automatic selection is Linux-only; Windows servers must configure positive dimensions. If both dimensions are positive, request those exact dimensions for every camera without automatic mode discovery. Reject a mix of zero and positive values. Verify captured frame dimensions and mark a camera offline/retry if it delivers a different resolution; do not resize frames to hide unsupported capture settings. Client panel dimensions are independent.
1. Run on Ubuntu 24.04 and Windows 11, using the appropriate OpenCV capture backend (V4L2 on Linux, DirectShow or supported alternative on Windows).
2. Run one capture worker per enabled camera; a failed camera must not block the others. Continuously keep the latest JPEG-encoded frame in memory only.
3. Provide HTTP MJPEG endpoints `GET /stream/front`, `/stream/back`, `/stream/left`, `/stream/right` and `GET /status` with enabled/connected/offline state for all four positions. Status responses must distinguish disabled cameras from disconnected ones.
4. Stream frames in real time to the GUI with reasonable latency and bounded memory use; drop stale frames rather than accumulate queues. Retry disconnected cameras periodically.
5. No snapshot endpoints, files, recording, or disk output. Shut down camera handles and workers cleanly.
6. Keep compatibility with `python night_guard_server.py -v 2 -p 14095 -c 14098`; `-p` overrides `SERVER_PORT`, `-c` refers to reserved `CONTROL_PORT`, and `-v` controls logging verbosity. Explain that control port is not listening yet.
7. Read only `python/server.env`, with meaningful validation and error messages. Default to localhost binding for safety.

## Windows GUI: python/night_guard_client.py
- Split camera windows omit all four arrow toolbar buttons (left, right, up, down). They keep zoom, fit, actual-size controls and scrollbars. The merged viewer retains its arrow buttons.
- Add an `Exit` button at the far right of the connection row, with flexible space separating it from `Connect`. Clicking it closes the client and zoom window and stops network workers through the normal shutdown handler.
- Each of the four main camera image panels uses `WIDTH` and `HEIGHT` from `client.env`, defaulting to 480 × 320 pixels when omitted, with aspect-preserving image fitting. Validate both as positive integers. These client settings are separate from the capture resolution settings in `server.env`. Default overall GUI size is 880 × 620. The zoom window continues to show native captured image dimensions at 1×.
- Open the zoom window at 1280 × 960 pixels, allowing manual resizing. Keep an icon-only toolbar at the top matching the supplied reference, in this exact order: previous camera (left arrow), next camera (right arrow), pan up, pan down, fit image, actual image size (1×), zoom in, zoom out, save, copy, print. Use compact rounded black buttons with white line icons, approximately 25×25 px and 7 px spacing. Save/copy/print are visibly disabled placeholders because media export, snapshots and recording are forbidden. Retain tooltips and accessible names without visible button text.
- Main-window images must fit their four panel bounds while preserving aspect ratio. The zoom window displays native image pixel dimensions at 1× and enlarges the entire image with magnification; horizontal/vertical scrollbars appear as needed. Do not crop magnified images to fit the window.
- Put a yellow `Zoom Out` button beside `Connect` in the connection row, followed by a yellow `Merge` button. Zoom opens a separate resizable window displaying one full-size live camera image at a time, starting with FRONT, and changes to green `Zoom In`. Left/right arrow buttons and keyboard arrow keys select the previous/next camera in the order FRONT, BACK, LEFT SIDE, RIGHT SIDE, wrapping at the ends. Display the current camera name and position. Each camera retains its own magnification while switching. Clicking Zoom again closes all zoom windows and resets the controls. Reuse the existing streams; clear all zoom images on disconnect and close all zoom windows when the client exits.
- The layout button labels the current state: yellow `Merge` by default, blue `Split` when separated. Clicking yellow `Merge` opens one separate live zoom window per camera and hides the merged viewer. Clicking blue `Split` closes separate windows and restores the original single viewer, preserving magnification and the previously selected camera. Each separate window retains its camera's toolbar, magnification, and scrollbars; all arrow controls are omitted because each window shows a fixed camera. The layout button can open the views if closed. Closing one separate window leaves the others open; closing the last resets Zoom and the layout button to yellow `Merge`.
- Reserve a blank sidebar on the left for future controls, with width set by `CONTROL_PANEL_WIDTH` (default 200 pixels). Put the connection row and four camera panels to its right. Do not show the startup instruction `Enter the server IP address and press Connect.` or a bottom control placeholder.
- Size the server address field to fit its displayed address rather than stretching across the window. The default `Connect` button is yellow with dark text, including after disconnecting.
- Clicking the green `Connected` button disconnects and clears the address and camera images immediately, returning the button to `Connect`. Connecting with an empty address uses `SERVER_URL` from `client.env` (or `http://127.0.0.1:14095` if absent) and displays the resolved address in the field.
- Provide an editable server IP / URL field above the camera panels, prefilled from `client.env`. Start with a `Connect` button; connect when clicked or Enter is pressed. Show `Connecting...` during the request, green `Connected` after a successful server status response, and red `Failed` on connection failure. Allow changing the address and retrying without blocking the GUI. A bare IP uses the configured video port, or 14095 by default.
- Runs on Windows 11, connects via `SERVER_URL` from `python/client.env`, and **never opens local USB cameras**.
- Use PySide6 or another maintained desktop GUI toolkit. Perform network stream reading and reconnection off the GUI thread.
- Always display the same four panels in a 2 × 2 layout:
```text
+----------+--------------------------+--------------------------+
|          | FRONT                    | BACK                     |
|          |       Live video         |       Live video         |
|          +--------------------------+--------------------------+
|          | LEFT SIDE                | RIGHT SIDE               |
|          |       Live video         |       Live video         |
+----------+--------------------------+--------------------------+
```
- Preserve aspect ratio when resizing. Disabled positions display `Not configured`; offline cameras display `Camera offline`; unavailable server displays a clear connection status. Never freeze the UI during network failure.
- Leave the sidebar empty for future controls. Stop worker threads and network requests on GUI exit.

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
- [ ] Left control region is reserved for future controls; Zoom and Split/Merge sit beside Connect.
- [ ] Server and GUI read separate environment files; no combined `.env` required.
- [ ] `requirements.txt` is at project root; init scripts remain under `config/`.
- [ ] Reconnects after camera/network interruptions; disabled cameras do not cause errors.
- [ ] No `captures/` directory and no recording/snapshot functionality.
- [ ] Existing init setup behavior is preserved, with changes documented.
- [ ] Cross-platform server and Windows GUI are tested for syntax/startup; real USB camera operation must be verified on actual hardware.

## Coding guidance
Use clear function docstrings with examples for important functions. Do not assume a camera path exists. Log actionable failures without exposing secrets. Summarize changed files and exact test commands after implementation.



## Zoom toolbar acceptance criteria (reference-image update)
- [ ] All 11 icon-only buttons appear across the top in the reference order; no visible labels.
- [ ] Previous/next camera, pan up/down, fit-to-window, native 1×, and zoom ± operate on the selected live stream.
- [ ] Save, copy, and print appear disabled; no disk image output, clipboard image export, or printing.
- [ ] Keyboard Left/Right and per-camera zoom levels still work; image scrollbars remain available.
- [ ] Toolbar uses inline vector icons, so no external icon files or fonts are required.

