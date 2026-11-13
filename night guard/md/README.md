# Night Guard — Live Camera Server and Windows Client

## Overview

This project is designed to stream **live video from 1–4 USB cameras** connected to a server running **Ubuntu 24.04 or Windows 11**. A separate **Windows 11 GUI client** displays the video over a trusted local network.

The GUI always contains four video positions in a **2 × 2 layout**, labeled **FRONT**, **BACK**, **LEFT SIDE**, and **RIGHT SIDE**, followed by a **reserved control area**. Controls will be specified later.

**Current scope: live streaming only.** No recording, snapshots, or `captures/` directory. The control port is reserved and is not an active service.

> **Implementation status:** This README documents the intended project architecture and usage defined in `md/AGENTS.md`. It does not confirm that the Python applications or initialization scripts have been implemented or tested.

## Project layout

```text
project_root/
├── requirements.txt            # Python dependencies (project root)
├── md/
│   ├── AGENTS.md               # Implementation specification for Codex
│   └── README.md               # This guide
├── config/
│   ├── init.sh                 # Ubuntu 24.04 initialization
│   └── init.bat                # Windows 11 initialization
└── python/
    ├── night_guard_server.py        # Server: Ubuntu or Windows
    ├── night_guard_client.py           # GUI client: Windows
    ├── server.env             # Server and USB camera configuration
    └── client.env             # GUI and server connection configuration
```

The two environment files belong in `python/`, **not** `config/`. Scripts must locate files relative to their own locations, not the shell's current directory.

## How it works

1. Connect 1–4 USB cameras to the server machine.
2. Configure the camera identifiers and network listening address in `python/server.env`.
3. Start `python/night_guard_server.py` on Ubuntu 24.04 or Windows 11.
4. Set the server's reachable LAN URL in `python/client.env` on the Windows GUI machine.
5. Start `python/night_guard_client.py` on Windows 11. The GUI receives live streams over HTTP; it does not access USB cameras itself.

### GUI layout

```text
+----------------------------+----------------------------+
| FRONT                      | BACK                       |
|        Live video          |        Live video          |
+----------------------------+----------------------------+
| LEFT SIDE                  | RIGHT SIDE                 |
|        Live video          |        Live video          |
+----------------------------+----------------------------+
| RESERVED CONTROL AREA                                   |
| Controls to be defined later                            |
+---------------------------------------------------------+
```

The first `CAMERA_COUNT` positions are enabled in this order: FRONT, BACK, LEFT SIDE, RIGHT SIDE. Unused positions display **Not configured**. A disconnected camera displays **Camera offline**. The GUI must remain responsive if the server or network is unavailable.

## Server configuration: `python/server.env`

Example for **Ubuntu** (verify the actual `/dev/video*` assignments):

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

For a **Windows server**, replace Linux device paths with OpenCV camera indices, such as `FRONT_CAMERA=0`, `BACK_CAMERA=1`, `LEFT_CAMERA=2`, and `RIGHT_CAMERA=3`. Camera indices are machine-specific.

`SERVER_HOST=127.0.0.1` allows connections only from the server machine. For a GUI on a different computer, bind the server to a suitable LAN interface or set `SERVER_HOST=0.0.0.0`, then restrict firewall access to trusted LAN clients. Do not expose the stream publicly.

`CONTROL_PORT` is reserved for future functionality; the current design does **not** open a listener on that port.

### Find Ubuntu camera devices

```bash
sudo apt install -y v4l-utils
v4l2-ctl --list-devices
ls -l /dev/video*
```

A single USB camera can expose multiple `/dev/video*` nodes. Choose its video-capture node rather than assuming each successive device number represents a different camera.

## GUI configuration: `python/client.env`

```ini
SERVER_URL=http://192.168.12.133:14095
GUI_TITLE=Four USB Camera Monitor
GUI_WIDTH=1400
GUI_HEIGHT=900
CONTROL_PANEL_HEIGHT=200
```

The URL above is **only an example**. Replace it with the server's actual reachable IP address and configured port. Keep server-only settings in `server.env` and client-only settings in `client.env`.

## Initialization

Run the platform-specific initializer from the project root:

**Ubuntu 24.04:**

```bash
bash config/init.sh
```

**Windows 11 (Command Prompt):**

```bat
config\init.bat
```

The initializers should install dependencies from **`../requirements.txt` relative to the `config/` directory**. The Ubuntu initializer must not require a virtual environment. Updates to existing init scripts must preserve previously useful setup steps (including Python, Node.js, Codex, and Ollama) unless explicitly changed. Initialization must not automatically launch the server or GUI, and it must not overwrite existing environment settings.

## Start the applications

**Ubuntu server:**

```bash
python3 python/night_guard_server.py -v 2 -p 14095 -c 14098
```

**Windows server (alternative):**

```bat
python python\night_guard_server.py -v 2 -p 14095 -c 14098
```

**Windows GUI client:**

```bat
python python\night_guard_client.py
```

The command-line options preserve the earlier server invocation style: `-v` sets logging verbosity, `-p` overrides the video server port, and `-c` identifies the **reserved**, inactive control port.

## Expected server interfaces

| Endpoint | Purpose |
|---|---|
| `GET /stream/front` | Live FRONT MJPEG stream |
| `GET /stream/back` | Live BACK MJPEG stream |
| `GET /stream/left` | Live LEFT SIDE MJPEG stream |
| `GET /stream/right` | Live RIGHT SIDE MJPEG stream |
| `GET /status` | Camera enabled/connected/offline status |

These are **implementation requirements**, not verified existing endpoints. The server should retain only the latest frame in memory, discard stale frames, reconnect failed cameras, and avoid writing media to disk.

## Troubleshooting

- **No cameras detected:** Check USB connections, `v4l2-ctl --list-devices`, permissions, and the camera paths in `server.env`.
- **Wrong camera shown:** Confirm the mapping of physical cameras to `/dev/video*` nodes or Windows camera indices.
- **GUI cannot connect:** Confirm the server is running, `SERVER_URL` matches the server's reachable IP/port, the bind address permits LAN access, and the firewall allows the video port.
- **One camera disconnected:** Other streams should continue; the affected camera should be shown as offline and retried automatically.
- **Slow video:** Reduce `WIDTH`, `HEIGHT`, `FPS`, or `JPEG_QUALITY`; verify available USB bandwidth and LAN performance.

## Implementation and verification

Use `md/AGENTS.md` as the authoritative implementation specification. After implementation, verify server startup on Ubuntu and Windows, GUI startup on Windows, all 1–4 camera configurations, live streaming, camera/network reconnection, and the reserved control area. Physical USB camera operation requires testing on real hardware.

No recording or snapshot features should be introduced until explicitly requested.
