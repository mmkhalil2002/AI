Night Guard setup and launch

Run from the project root. Windows client and server setup support a standard
PowerShell session. Use an elevated session to configure Windows Firewall.

Windows 11 client
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\config\init-client-windows.ps1

Windows 11 server
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\config\init-server-windows.ps1

Ubuntu 24.04 desktop client
  bash config/init-client-ubuntu.sh

Ubuntu 24.04 server
  bash config/init-server-ubuntu.sh

The installers regenerate requirements-client.txt or requirements-server.txt
at the project root before installing Python dependencies. They create their
own .env file if absent and configure TCP firewall rules. Windows setup
configures firewall rules when run as Administrator. Server rules allow inbound traffic
on SERVER_PORT. Client rules allow outbound traffic to the SERVER_URL port.
The generated configurations use TCP 14095. An HTTP or HTTPS URL without an
explicit port uses 80 or 443, respectively. Set the port in your configuration before
running setup. The reserved control port has no listener or firewall rule.

Windows server rules apply to the local subnet on Private and Domain profiles.
For Ubuntu server setup, optionally set NIGHT_GUARD_LAN_CIDR to restrict the
source subnet, for example:
  NIGHT_GUARD_LAN_CIDR=192.168.1.0/24 bash config/init-server-ubuntu.sh
The default Ubuntu server rule allows any source. If UFW is inactive, setup saves
the rule for the next time UFW is enabled. On Ubuntu, remove any previous Night
Guard rule for an obsolete port or subnet with sudo ufw status numbered and
sudo ufw delete NUMBER when UFW is active.

Set SERVER_HOST to the server LAN address for remote access and set the client
SERVER_URL to the corresponding HTTP URL. The installer does not start the app.
Run the launcher for your machine after configuring its .env file:

  Windows client: config\run-client-windows.bat
  Windows server: config\run-server-windows.bat
  Ubuntu client:  bash config/run-client-ubuntu.sh
  Ubuntu server:  bash config/run-server-ubuntu.sh

Both clients and the Windows server start when you run their launchers.
The Ubuntu server launcher starts its service and enables it at every boot.
Run that launcher as the account that installed the server packages.

  journalctl -u night-guard-server.service
  bash config/debug-server-ubuntu.sh
  bash config/stop-server-ubuntu.sh
  bash config/restart-server-ubuntu.sh
  sudo systemctl disable --now night-guard-server.service

If you enabled the Ubuntu client with the earlier launcher, turn off the
old sign-in service once as the desktop user:

  systemctl --user disable --now night-guard-client.service

Setup installs only Night Guard dependencies and firewall tooling.

Camera preview tests

Run a script from the project root with `py -3.12` on Windows or `python3`
on Ubuntu:

  python/test_1_camera.py
  python/test_2_cameras.py
  python/test_3_cameras.py
  python/test_4_cameras.py

Each script opens one window per requested camera, with an error message in
any window whose camera cannot stream. Press Q or Esc to close the previews.
The previews run simultaneously by default on Windows and Ubuntu. Use
--sequential to cycle through the cameras one at a time. In that mode,
inactive windows display the last frame until their next turn.
The tests detect cameras at startup. Use --list-cameras to see the detected
devices and their identifiers. To override discovery, pass the devices for
that test after its script name. Ubuntu needs v4l-utils for discovery and a
desktop session or X11 forwarding for previews.

Uninitialize Night Guard

Run the matching command from the project root. Close any manually launched
client or server first. On Windows, use an elevated PowerShell session. On
Ubuntu, run as the account that set up the app. The scripts use sudo as needed.

  bash config/uninit-client-ubuntu.sh
  bash config/uninit-server-ubuntu.sh
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\config\uninit-client-windows.ps1
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\config\uninit-server-windows.ps1

Each script removes its role's firewall rules. The Ubuntu server script also
stops and disables night-guard-server.service, removes its unit file, and
reloads systemd. The Ubuntu client script also removes the earlier user
service if installed. Project files, .env settings, and shared Python and
system packages are preserved.
