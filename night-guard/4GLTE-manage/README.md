# 4GLTE — Ubuntu 24.04 / Raspberry Pi 5 / Sixfab Quectel EC25

This package includes the unified `4GLTE.md` Codex specification, `README.md`, and **all 13 Bash scripts (12 LTE + 1 Wi-Fi)** from the LTE management, install/reset, and full-status workflows.

## Setup

```bash
unzip 4GLTE-complete.zip
cd 4GLTE-complete
chmod +x lte-*.sh wifi-add.sh
sudo ./lte-install.sh
```

**Do not remove NetworkManager when connected over SSH.** The uninstall script preserves NetworkManager and Ethernet, but will interrupt cellular connections. A USB power or hub problem cannot be fixed by reinstalling software.

## Commands

| Command | Purpose |
|---|---|
| `sudo ./lte-install.sh` | Install LTE packages and check detection |
| `sudo ./lte-uninstall.sh` | Purge selected LTE packages after confirmation |
| `./lte-verify-uninstall.sh` | Check that uninstall completed and Ethernet was preserved (read-only) |
| `./lte-verify-install.sh` | Check LTE packages, ModemManager, EC25/SIM detection and data readiness (read-only) |
| `sudo ./lte-status.sh` | Show modem, SIM and network status |
| `sudo ./lte-full-status.sh` | Show detailed modem and SIM information (redacted) |
| `sudo ./lte-enable.sh` | Enable modem |
| `sudo ./lte-disable.sh` | Disable modem |
| `sudo ./lte-connect.sh fast.t-mobile.com` | Attempt LTE data connection using T-Mobile APN |
| `sudo ./lte-disconnect.sh` | Disconnect LTE data profile |
| `sudo ./lte-diagnostics.sh` | Inspect USB, drivers, network interfaces and errors |
| `lte-common.sh` | Helper sourced by other scripts; do not run directly |

The modem index is discovered automatically, rather than assuming `239`. `lte-connect.sh` requires a working cellular data interface; serial `/dev/ttyUSB*` ports alone do not establish LTE internet connectivity. Do not assume the connection is operational until verified.

## Verify uninstall and reinstall

```bash
sudo ./lte-uninstall.sh
./lte-verify-uninstall.sh
sudo ./lte-install.sh
./lte-verify-install.sh
```

The verification scripts display PASS/FAIL/WARN results. A PASS from `lte-verify-install.sh` verifies the installed software; it **does not guarantee LTE internet access**. USB data interface readiness may still be incomplete, and previous USB over-current events require hardware troubleshooting.

## Regenerate with Codex

Ask Codex: **Read `4GLTE.md`, implement/update all 13 Bash scripts and `README.md`, then validate each Bash script with `bash -n`.** The included scripts are the previously developed implementations; the Markdown specification is the desired contract for future updates. They have not been tested on your Raspberry Pi hardware.

## Verify

```bash
for file in lte-*.sh; do bash -n "$file" || exit 1; done
sudo ./lte-full-status.sh
sudo ./lte-diagnostics.sh
```

Do not publish unredacted `mmcli` output because it may contain SIM identifiers, IMEI, or a telephone number.


## Configure built-in or USB Wi-Fi — `wifi-add.sh`

The same package also includes a universal interactive Wi-Fi configuration utility. It detects available Wi-Fi adapters using `iw dev`, presents a numbered selection menu, and prompts for the network name (SSID) and hidden password. It works with built-in Raspberry Pi Wi-Fi (`wlan0`, when detected) and USB Wi-Fi adapters (`wlx...`), without assuming either name.

```bash
sudo apt install -y iw python3-yaml
sudo ./wifi-add.sh
```

It backs up the relevant Netplan YAML file, preserves Ethernet and other Wi-Fi interface entries, and uses `netplan try --timeout 60` to allow rollback. Run it at a local console if possible: applying Netplan may temporarily interrupt SSH. The Wi-Fi password is stored in the root-readable Netplan file as required for reconnecting; do not share that file publicly. The script checks for an assigned IPv4 address, which does not by itself prove internet access.

For Codex, `4GLTE.md` now specifies **all 13 Bash scripts** (12 LTE and 1 Wi-Fi) plus this README.


### Automatic WPA service repair in `wifi-add.sh`

The Wi-Fi tool now checks and repairs a failed Netplan WPA service for the **selected adapter only**. It stops that adapter's WPA service, removes a stale per-interface control socket only when it is not owned by a running process, and restarts the service if needed. If another live process owns the socket, it stops safely with a diagnostic instead of killing processes or disrupting other network interfaces. After configuration, it waits briefly for DHCP and reports association, IPv4 address, and relevant service errors.

Run: `sudo ./wifi-add.sh` (or `./wifi-add.sh --help`). An accepted Netplan configuration alone does not establish a working Wi-Fi connection.

### Improved handling of an occupied WPA socket

`wifi-add.sh` now tries the selected adapter first and only attempts a targeted repair if its Netplan WPA service fails. It stops that service, waits for socket release, checks for active owners, removes a **stale** socket only when safe, and restarts the service. If another process still manages the selected adapter, the script reports the owner instead of killing it or disrupting other Wi-Fi/Ethernet/LTE connections. A successful Netplan confirmation is not proof of a successful Wi-Fi connection; inspect the final link and IPv4 status.

### Global WPA socket conflict: automatic interface-specific release

The updated `wifi-add.sh` can recognize a socket owned by the system-wide D-Bus `wpa_supplicant -u` process (as observed on the Ralink RT5370 USB adapter). It attempts to release **only the selected interface** via the WPA D-Bus API, or via the optional global `wpa_cli` socket, then starts that adapter's Netplan WPA service. It never kills the global daemon or removes a live socket, and it refuses takeover when another manager controls the interface. Success requires an active WPA service, association with an access point, and an IPv4 address. The script is not yet hardware-tested.


## Persistent Wi-Fi interface names (wlan1, wlan2, ...)

`wifi-rename.sh` is an additional Bash script. It lets the operator select a
currently detected Wi-Fi adapter and assign a persistent name `wlan1`, `wlan2`,
`wlan3`, etc., matched to the adapter's MAC address using Netplan `match:
macaddress` and `set-name`. Preserve the Raspberry Pi built-in `wlan0` and
never rename Ethernet or LTE interfaces. Reject name collisions, ambiguous
Netplan definitions, and invalid names. Back up the original Netplan YAML,
validate with `netplan generate`, and restore the original on validation failure.
Do not restart networking or rename live interfaces; the new name takes effect
only after reboot. The script must be runnable as
`sudo ./wifi-rename.sh` (interactive) or
`sudo ./wifi-rename.sh wlx000f0241f1aa wlan1` (explicit).

`wifi-add.sh` should offer optional persistent renaming after its Wi-Fi
configuration and connection checks, invoking `wifi-rename.sh` for the selected
adapter. This is optional; it should not affect the current session. If the
adapter is moved to another USB port, its MAC-based name remains stable.
After reboot, `iw dev`, `ip -br link`, and `networkctl status wlan1` verify the
name and link. Hardware-specific renaming may fail if a driver does not support
renaming or if other naming policies conflict; diagnose rather than forcing a
live rename.

### Persistent Wi-Fi naming fix (Ubuntu 24.04)

Run `sudo ./wifi-rename.sh` to assign a USB Wi-Fi adapter a stable `wlan1`, `wlan2`, etc. The script now uses a MAC-matched systemd `.link` file plus a renamed Netplan Wi-Fi key. Do **not** put `match:`/`set-name:` inside a Netplan `wifis:` entry with the `networkd` renderer; that combination fails validation. It backs up the original Netplan file, validates with `netplan generate`, rolls back on error, and does not modify running connections. Reboot to activate, then check `ip -br link`, `iw dev`, and `networkctl status wlan1`. Existing built-in `wlan0` remains unchanged. If an earlier failed rename left `match:` or `set-name:` in a Wi-Fi YAML entry, restore the original backup before retrying.


### One-step Wi-Fi setup and persistent naming

Run `sudo ./wifi-add.sh`. Select any detected onboard/USB adapter, enter SSID and hidden password, confirm the Netplan change, and optionally assign the selected USB adapter a persistent `wlan1`/`wlan2`/etc. name **directly within the same script**. No separate `wifi-rename.sh` run is necessary. `wlan0` is preserved. The rename takes effect only after reboot; verify with `ip -br link` and `iw dev`. The script backs up and validates Netplan, and rolls back failed naming changes. `wifi-rename.sh` remains available as an optional standalone command.


## Automatic Wi-Fi connection verification (November 2026)

`wifi-add.sh` must verify the selected built-in or USB Wi-Fi adapter after Netplan configuration and targeted WPA conflict recovery. It must print the Netplan mapping, `netplan-wpa-<interface>.service` state, `iw dev <interface> link` association, `ip -br addr` and global IPv4 address, `networkctl status`, selected WPA service errors, and filtered `systemd-networkd` DHCP events. Wait up to 40 seconds for DHCP; distinguish failed WPA service, unassociated Wi-Fi, associated Wi-Fi without IPv4, and successful association with an IPv4 address. Do not report configuration acceptance as successful connectivity. A renamed interface needs reboot; after reboot the new interface name must match the Netplan key. Do not restart all networking, change default routes, kill other WPA processes, or print Wi-Fi passwords. Internet reachability is not implied by DHCP success.


## Raspberry Pi built-in Wi-Fi after reboot (wifi-add.sh)

On Raspberry Pi hardware, `wifi-add.sh` loads the `brcmfmac` kernel module and ensures `/etc/modules-load.d/brcmfmac.conf` contains `brcmfmac` so the onboard Broadcom Wi-Fi is requested at subsequent boots. It checks whether `wlan0` exists, prints its IPv4 status, and reports Broadcom kernel messages if the interface remains missing. It does not rename `wlan0`, change the USB `wlan1` naming rule, or modify Ethernet/LTE. The `systemd-modules-load.service` is normally static; it should not be forcibly enabled. After reboot, check `iw dev` and `ip -4 -br addr`. A missing interface despite the module loading requires further firmware/boot diagnostics; driver loading alone does not guarantee Wi-Fi or DHCP connectivity.
