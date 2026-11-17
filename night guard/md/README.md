# Night Guard — Live Camera Server and Windows Client

## Overview

This project is designed to stream **live video from 1–4 USB cameras** connected to a server running **Ubuntu 24.04 or Windows 11**. A separate **Windows 11 GUI client** displays the video over a trusted local network.

The GUI has an empty left sidebar reserved for future controls. To its right are the connection row, with Zoom and Split beside Connect, and four video positions in a **2 × 2 layout**, labeled **FRONT**, **BACK**, **LEFT SIDE**, and **RIGHT SIDE**. No startup instruction text is shown.

**Current scope: live streaming only.** No recording, snapshots, or `captures/` directory. The control port is reserved and is not an active service.

> **Implementation status:** The Python server, PySide6 client, separate configuration files, dependencies, and initialization scripts are implemented. Hardware-free integration checks passed on Windows, including offscreen GUI startup. Physical USB camera operation, Ubuntu startup, and initialization scripts still require verification on their target systems.

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
+----------+----------------------------+----------------------------+
|          | FRONT                      | BACK                       |
|          |        Live video          |        Live video          |
|          +----------------------------+----------------------------+
|          | LEFT SIDE                  | RIGHT SIDE                 |
|          |        Live video          |        Live video          |
+----------+----------------------------+----------------------------+
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

WIDTH=0
HEIGHT=0
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
GUI_WIDTH=880
GUI_HEIGHT=620
WIDTH=480
HEIGHT=320
CONTROL_PANEL_WIDTH=200
```

The URL above is **only an example**. Replace it with the server's actual reachable IP address and configured port. Keep server-only settings in `server.env` and client-only settings in `client.env`.

The client opens with a **Server IP / URL** field prefilled from `client.env` and a **Connect** button. Enter an IP address (optionally with a port, such as `192.168.1.100:14095`) or a complete HTTP URL, then click **Connect** or press Enter. A bare IP uses the port configured in `SERVER_URL`, or 14095 by default. The button shows **Connecting...**, then green **Connected** when the server responds or red **Failed** if connection fails. You can edit the address and click again to retry. Changes in this field apply to the current session. A connected server can still report individual cameras offline.

The address field sizes itself to fit the address. The default **Connect** button is yellow. Click the green **Connected** button to disconnect and clear the address and camera panels. The button returns to yellow **Connect**. Clicking **Connect** with an empty address uses the default `SERVER_URL` from `client.env` (or `http://127.0.0.1:14095` if absent) and displays that address in the field.

## Camera zoom

Split camera windows omit all arrow buttons. Use their scrollbars to move around magnified images. The merged zoom window retains its camera-navigation and pan arrows.

The **Exit** button at the far right of the connection row closes the client, closes any zoom window, and stops its network workers.

Each main camera panel uses `WIDTH` and `HEIGHT` from `python/client.env`, defaulting to 480 × 320 pixels when omitted. The default overall GUI is 880 × 620, configured separately by `GUI_WIDTH` and `GUI_HEIGHT`. In `python/server.env`, setting both `WIDTH=0` and `HEIGHT=0` makes the Linux server query all enabled cameras and request their lowest common supported resolution (smallest pixel area, then width and height). For example, if one camera supports 160×120 and 640×480 while another supports 320×240 and 640×480, the common choice is 640×480. This requires `v4l-utils` and readable modes for every enabled camera; startup reports an error if a camera cannot be queried or there is no common resolution. Disabled cameras are excluded. Set both server values positive to request that exact resolution for all cameras; Windows servers must use positive dimensions. A camera delivering different dimensions is reported offline and retried. Frames are not resized on the server. Client panel sizes do not change capture resolution or native-size zoom display.

The main window scales each image to fit its camera panel while preserving aspect ratio. The zoom window displays the image at its actual pixel dimensions at 1×. Magnification enlarges the entire image; scrollbars let you explore any parts that extend outside the window. The zoom window opens at 1280 × 960 pixels and can be resized manually. Across the top is a compact black-and-white icon toolbar matching the supplied reference: **previous**, **next**, **pan up**, **pan down**, **fit image**, **actual size (1×)**, **zoom in**, **zoom out**, **save**, **copy**, **print**. The final three icons are intentionally disabled to preserve the no-snapshots/no-export policy. Hover over a control for its description.

The yellow **Zoom Out** button beside **Connect** opens a separate resizable window showing one full-size camera image at a time, starting with **FRONT**, and changes to green **Zoom In**. Use **←** and **→** buttons or keyboard arrow keys to switch between **FRONT**, **BACK**, **LEFT SIDE**, and **RIGHT SIDE**; navigation wraps at either end. The current camera name and position appear above the image. **+** and **−** magnify the selected image from 1× to 8×, and each camera retains its zoom level while switching. Click the main Zoom button again to close all zoom windows and return to yellow **Zoom Out**. Zoom is digital and uses the existing live streams.

The layout button beside Zoom shows the current state: yellow **Merge** by default. Click it to open each camera in its own window; the button changes to blue **Split**. Click **Split** to restore the original single zoom viewer, preserving camera magnifications and the previously selected camera. Each separate window has its own zoom and scrolling controls. The layout button also opens the views if Zoom is closed. You can close an individual camera window; closing the last resets the controls to yellow **Merge**. Disconnect clears images in all windows, and Exit closes all windows and stops the network workers.

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

These endpoints are implemented. Hardware-free checks verified MJPEG streaming, status responses, and disabled-camera responses. The server retains only the latest frame in memory and reconnects failed cameras without writing media to disk.

## Troubleshooting

- **No cameras detected:** Check USB connections, `v4l2-ctl --list-devices`, permissions, and the camera paths in `server.env`.
- **Wrong camera shown:** Confirm the mapping of physical cameras to `/dev/video*` nodes or Windows camera indices.
- **GUI cannot connect:** Confirm the server is running, `SERVER_URL` matches the server's reachable IP/port, the bind address permits LAN access, and the firewall allows the video port.
- **One camera disconnected:** Other streams should continue; the affected camera should be shown as offline and retried automatically.
- **Slow video:** Request a smaller supported resolution by setting positive `WIDTH` and `HEIGHT` values, or reduce `FPS` or `JPEG_QUALITY`; verify available USB bandwidth and LAN performance. Both dimensions set to zero select the lowest common supported resolution across enabled cameras on Linux.

## Implementation and verification

Run the hardware-free checks from the project root after installing dependencies:

```bash
python config/verify_apps.py
```

This checks simulated camera counts 1–4, disabled devices, MJPEG and status endpoints, camera reconnection, offscreen GUI frames and status messages, server failure, and worker shutdown. It does not access physical cameras. Example configurations default to Windows camera indices and localhost; customize the camera paths for Ubuntu and the bind address and client URL for LAN use.

Use `md/AGENTS.md` as the authoritative implementation specification. After implementation, verify server startup on Ubuntu and Windows, GUI startup on Windows, all 1–4 camera configurations, live streaming, camera/network reconnection, and the reserved control area. Physical USB camera operation requires testing on real hardware.

No recording or snapshot features should be introduced until explicitly requested.



### Updated zoom toolbar controls

| Icon order | Action |
|---|---|
| 1–2 | Previous / next camera (wrap around) |
| 3–4 | Scroll current image up / down when enlarged |
| 5 | Fit entire image within the zoom viewport |
| 6 | Display at native 1× size |
| 7–8 | Zoom in / zoom out (1×–8× for manual zoom) |
| 9–11 | Save / copy / print: visible but disabled (no image export) |

The toolbar uses bundled inline SVG icons through PySide6.QtSvg. No additional icon files are needed. To test without camera hardware on Windows, set `QT_QPA_PLATFORM=offscreen` before launching, or run normally with a reachable camera server.
