Night Guard setup and launch

Run from the project root. On Windows, use an elevated PowerShell session.

Windows 11 client
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\config\init-client-windows.ps1

Windows 11 server
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\config\init-server-windows.ps1

Ubuntu 24.04 desktop client
  bash config/init-client-ubuntu.sh

Ubuntu 24.04 server
  bash config/init-server-ubuntu.sh

The role-specific installers use requirements.txt, create their own .env file
if absent, and configure TCP firewall rules. Server rules allow inbound traffic
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
  sudo systemctl disable --now night-guard-server.service

If you enabled the Ubuntu client with the earlier launcher, turn off the
old sign-in service once as the desktop user:

  systemctl --user disable --now night-guard-client.service

Setup installs only Night Guard dependencies and firewall tooling.
