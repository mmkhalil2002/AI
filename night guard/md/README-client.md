# Night Guard desktop client

The desktop client displays live video from the [Night Guard server](README-server.md) over HTTP. The server manages all USB cameras. The client reserves an empty left sidebar for future controls.

## Setup and startup

From the project root in an elevated Windows Command Prompt:

```bat
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\config\init-client-windows.ps1
config\run-client-windows.bat
```

On Ubuntu 24.04 desktop, run the launcher as the desktop user whenever you want to open Night Guard.

```bash
bash config/init-client-ubuntu.sh
bash config/run-client-ubuntu.sh
```

The initializer installs the shared dependencies from the root `requirements.txt`, preserves existing configuration, and leaves application startup to you. The client reads `client.env` from beside its Python script.

## Client configuration

```ini
SERVER_URL=http://192.168.12.133:14095
GUI_TITLE=Four USB Camera Monitor
GUI_WIDTH=880
GUI_HEIGHT=620
WIDTH=480
HEIGHT=320
CONTROL_PANEL_WIDTH=200
```

The URL above is **only an example**. Replace it with the server's actual reachable IP address and configured port. See the [server guide](README-server.md) for LAN setup.

The client opens with a **Server IP / URL** field prefilled from `client.env` and a **Connect** button. Enter an IP address (optionally with a port, such as `192.168.1.100:14095`) or a complete HTTP URL, then click **Connect** or press Enter. A bare IP uses the port configured in `SERVER_URL`, or 14095 by default. The button shows **Connecting...**, then green **Connected** when the server responds or red **Failed** if connection fails. You can edit the address and click again to retry. Changes in this field apply to the current session. A connected server can still report individual cameras offline.

The address field sizes itself to fit the address. The default **Connect** button is yellow. Click the green **Connected** button to disconnect and clear the address and camera panels. The button returns to yellow **Connect**. Clicking **Connect** with an empty address uses the default `SERVER_URL` from `client.env` (or `http://127.0.0.1:14095` if absent) and displays that address in the field.

## GUI layout

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

## Camera zoom

Split camera windows omit all arrow buttons. Use their scrollbars to move around magnified images. The merged zoom window retains its camera-navigation and pan arrows.

The **Exit** button at the far right of the connection row closes the client, closes any zoom window, and stops its network workers.

Each main camera panel uses `WIDTH` and `HEIGHT` from `python/client.env`, defaulting to 480 × 320 pixels. The default overall GUI is 880 × 620, configured separately by `GUI_WIDTH` and `GUI_HEIGHT`. Panel sizes are independent of the server capture resolution and the native-size zoom display.

The main window scales each image to fit its camera panel while preserving aspect ratio. The zoom window displays the image at its actual pixel dimensions at 1×. Magnification enlarges the entire image; scrollbars let you explore any parts that extend outside the window. The zoom window opens at 1280 × 960 pixels and can be resized manually. Across the top is a compact black-and-white icon toolbar matching the supplied reference: **previous**, **next**, **pan up**, **pan down**, **fit image**, **actual size (1×)**, **zoom in**, **zoom out**, **save**, **copy**, **print**. The final three icons are intentionally disabled to preserve the no-snapshots/no-export policy. Hover over a control for its description.

The yellow **Zoom Out** button beside **Connect** opens a separate resizable window showing one full-size camera image at a time, starting with **FRONT**, and changes to green **Zoom In**. Use **←** and **→** buttons or keyboard arrow keys to switch between **FRONT**, **BACK**, **LEFT SIDE**, and **RIGHT SIDE**; navigation wraps at either end. The current camera name and position appear above the image. **+** and **−** magnify the selected image from 1× to 8×, and each camera retains its zoom level while switching. Click the main Zoom button again to close all zoom windows and return to yellow **Zoom Out**. Zoom is digital and uses the existing live streams.

The layout button beside Zoom shows the current state: yellow **Merge** by default. Click it to open each camera in its own window; the button changes to blue **Split**. Click **Split** to restore the original single zoom viewer, preserving camera magnifications and the previously selected camera. Each separate window has its own zoom and scrolling controls. The layout button also opens the views if Zoom is closed. You can close an individual camera window; closing the last resets the controls to yellow **Merge**. Disconnect clears images in all windows, and Exit closes all windows and stops the network workers.

### Zoom toolbar controls

| Icon order | Action |
|---|---|
| 1–2 | Previous / next camera (wrap around) |
| 3–4 | Scroll current image up / down when enlarged |
| 5 | Fit entire image within the zoom viewport |
| 6 | Display at native 1× size |
| 7–8 | Zoom in / zoom out (1×–8× for manual zoom) |
| 9–11 | Save / copy / print: visible but disabled (no image export) |

The toolbar uses bundled inline SVG icons through PySide6.QtSvg. No additional icon files are needed. To test without camera hardware on Windows, set `QT_QPA_PLATFORM=offscreen` before launching, or run normally with a reachable camera server.

## Troubleshooting

- **GUI cannot connect:** Confirm the server is running, `SERVER_URL` matches the server's reachable IP/port, the bind address permits LAN access, and the firewall allows the video port.

[Installer and firewall details](../config/README.txt)

## Verification

From the project root, run `python config/verify_apps.py` after installing the dependencies. This suite uses simulated cameras and an offscreen GUI. Physical camera operation and platform initialization require verification on the target machines. See [AGENTS.md](AGENTS.md) for the implementation specification.
