# Night Guard camera server

The server captures live video from one to four USB cameras on Ubuntu 24.04 or Windows 11. It uses OpenCV with V4L2 on Linux and DirectShow on Windows, and serves MJPEG over HTTP to the [desktop client](README-client.md).

## How capture and streaming work

- `CAMERA_COUNT` enables the first one to four positions in this order: FRONT, BACK, LEFT SIDE, RIGHT SIDE. Each enabled camera has its own capture worker.
- Each worker encodes frames as JPEG and retains only its latest frame in memory. Stream readers can skip frames when capture outpaces delivery. The service provides live video only.
- If a camera fails to open, read, or encode a frame, the worker clears its last frame, reports it offline, releases the camera, and retries after two seconds. Other camera workers keep running.
- On shutdown, the server signals the workers to stop and closes the HTTP listener. Capture workers release their camera handles as they exit.

## Setup and startup

From the project root, run the initializer for your server OS. Before running the launcher, configure the camera mapping and resolution in `python/server.env` as described below. On Windows, use positive dimensions supported by your cameras. Run Windows setup as Administrator.

**Ubuntu 24.04**

The Ubuntu run script installs and starts a systemd service and enables it at boot.

```bash
bash config/init-server-ubuntu.sh
bash config/run-server-ubuntu.sh
```

**Windows 11 (Command Prompt)**

```bat
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\config\init-server-windows.ps1
config\run-server-windows.bat
```

The initializers use the root `requirements.txt`, preserve existing configuration, and leave application startup to you. Ubuntu setup uses the system Python environment. The server reads `server.env` from beside its Python script. `-v` sets logging verbosity (0 for errors, 1 for info, 2 for debug), `-p` overrides the HTTP port, and `-c` sets the reserved control port value.

## Server configuration

Example for **Ubuntu** (verify the actual `/dev/video*` assignments):

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

For a **Windows server**, use OpenCV camera indices, such as `FRONT_CAMERA=0`, `BACK_CAMERA=1`, `LEFT_CAMERA=2`, and `RIGHT_CAMERA=3`. Camera indices are machine-specific. Also set `WIDTH` and `HEIGHT` to positive values supported by your cameras.

`SERVER_HOST=127.0.0.1` allows connections only from the server machine. For a GUI on a different computer, bind the server to a suitable LAN interface or set `SERVER_HOST=0.0.0.0`, then restrict firewall access to trusted LAN clients. Do not expose the stream publicly.

`CONTROL_PORT` is reserved for future functionality; the current design does **not** open a listener on that port.

### Find Ubuntu camera devices

```bash
sudo apt install -y v4l-utils
v4l2-ctl --list-devices
ls -l /dev/video*
```

A single USB camera can expose multiple `/dev/video*` nodes. Choose its video-capture node rather than assuming each successive device number represents a different camera.

### Capture resolution

Set both `WIDTH` and `HEIGHT` to positive values to request those exact dimensions on every enabled camera. A worker takes its camera offline and retries if it receives a frame of a different size.

On Linux, set both values to zero to query enabled cameras with `v4l2-ctl` and automatically select their smallest common resolution by pixel area, breaking ties by width and then height. For example, if one camera supports 160 × 120 and 640 × 480 and another supports 320 × 240 and 640 × 480, the server selects 640 × 480. Startup fails if a camera cannot be queried, `v4l-utils` is missing, or the cameras share no resolution. Startup logs the selected dimensions at info level.

## HTTP endpoints

| Endpoint | Purpose |
|---|---|
| `GET /stream/front` | Live FRONT MJPEG stream |
| `GET /stream/back` | Live BACK MJPEG stream |
| `GET /stream/left` | Live LEFT SIDE MJPEG stream |
| `GET /stream/right` | Live RIGHT SIDE MJPEG stream |
| `GET /status` | Camera enabled/connected/offline status |

`/status` returns JSON keyed by `front`, `back`, `left`, and `right`. Each entry has `enabled` and `connected` booleans and a `state` of `disabled`, `offline`, or `connected`. A disabled stream returns HTTP 404. An enabled camera that is offline can accept a stream request and resumes delivery when fresh frames become available.

## Troubleshooting

- **No cameras detected:** Check USB connections, `v4l2-ctl --list-devices`, permissions, and the camera paths in `server.env`.
- **Wrong camera shown:** Confirm the mapping of physical cameras to `/dev/video*` nodes or Windows camera indices.
- **One camera disconnected:** Other streams continue while the affected camera reports offline and retries automatically.
- **Slow video:** Request a smaller supported resolution by setting positive `WIDTH` and `HEIGHT` values, or reduce `FPS` or `JPEG_QUALITY`. Verify available USB bandwidth and LAN performance. Both dimensions set to zero select the lowest common supported resolution across enabled cameras on Linux.

[Installer and firewall details](../config/README.txt)

## Verification

From the project root, run `python config/verify_apps.py` after installing the dependencies. This suite uses simulated cameras and an offscreen GUI. Physical camera operation and platform initialization require verification on the target machines. See [AGENTS.md](AGENTS.md) for the implementation specification.
