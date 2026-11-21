# 4GLTE — Unified Master Specification for Ubuntu 24.04

## Purpose and Codex instructions

This **single Markdown file is the complete source of requirements** for generating and maintaining the Ubuntu LTE management project. It replaces the previous separate LTE management and reset/install Markdown specifications. **The Bash scripts are not embedded or executed by this file**: Codex must generate them from the instructions below. No existing `.sh` files are required.

**Give Codex this instruction:**

> Read `4GLTE.md` completely. Generate or update every Bash script and `README.md` specified here in the same directory. Follow all safety, behavior, and CLI requirements. Add a descriptive header comment to every script. Run `bash -n` on all generated scripts and fix syntax errors. Do not modify the modem USB composition or unrelated network interfaces.

Codex should treat this file as authoritative, generate actual Bash implementations rather than placeholders, and preserve this file unchanged. If existing scripts exist, update them consistently with this specification. Do not delete unrelated project files.

## Target hardware and observed state

- Raspberry Pi 5 running Ubuntu 24.04; Sixfab Quectel EC25 LTE modem, USB vendor/product ID `2c7c:0125`.
- ModemManager already recognized `[Quectel] EC25` and an active T-Mobile USA SIM (operator ID `310260`).
- The modem was observed as `state: disabled`, `power state: on`, with AT port `/dev/ttyUSB2` and serial ports `/dev/ttyUSB0`–`/dev/ttyUSB3`.
- A cellular data interface such as `wwan0`, `usb0`, or `/dev/cdc-wdm0` was **not** observed; LTE internet connectivity is unverified.
- A previous modem index was `239`, but modem indices can change. Never hardcode that index.
- Earlier USB logs showed hub over-current events and repeated USB disconnects. A previous power check returned `throttled=0x80000` (historical undervoltage). Software reinstall cannot fix a defective or unpowered hub, cabling, or power instability.
- Never include actual IMSI, ICCID, IMEI, telephone number, or other private identifiers in examples, default output, or logs.

## Complete list of deliverables

Generate **12 Bash scripts plus `README.md`**. Keep them together with `4GLTE.md`.

| File | Function | Example command |
|---|---|---|
| `lte-common.sh` | Shared defaults, dependency checks, modem discovery, safe output helpers | Sourced by scripts; do not execute directly |
| `lte-status.sh` | Concise, read-only modem, SIM, LTE, IP and network status | `sudo ./lte-status.sh` |
| `lte-full-status.sh` | Detailed, read-only ModemManager modem and SIM reports with sensitive identifiers redacted | `sudo ./lte-full-status.sh` |
| `lte-enable.sh` | Enable modem without initiating a data connection | `sudo ./lte-enable.sh` |
| `lte-disable.sh` | Disable modem, interrupting LTE service | `sudo ./lte-disable.sh` |
| `lte-connect.sh` | Create/update dedicated NetworkManager cellular profile and activate LTE data | `sudo ./lte-connect.sh fast.t-mobile.com` |
| `lte-disconnect.sh` | Deactivate the dedicated cellular profile, leaving modem enabled | `sudo ./lte-disconnect.sh` |
| `lte-diagnostics.sh` | Read-only USB, driver, modem, networking and power-error diagnostics | `sudo ./lte-diagnostics.sh` |
| `lte-uninstall.sh` | Safely remove selected LTE user-space packages, preserving Ethernet and NetworkManager | `sudo ./lte-uninstall.sh` |
| `lte-install.sh` | Install/reinstall LTE user-space packages and verify modem/driver detection | `sudo ./lte-install.sh` |
| `lte-verify-uninstall.sh` | Read-only confirmation that LTE reset packages are absent and ModemManager is stopped, while Ethernet is preserved | `./lte-verify-uninstall.sh` |
| `lte-verify-install.sh` | Read-only confirmation of installed LTE packages, service, kernel drivers, EC25/SIM detection and data-interface readiness | `./lte-verify-install.sh` |
| `README.md` | Complete installation, reset, usage, configuration, diagnostics and troubleshooting guide | Read as documentation |

## Requirements common to all scripts

1. Use `#!/usr/bin/env bash`, `set -euo pipefail`, standard Ubuntu Bash, and no Python/virtual environment.
2. **Begin every `.sh` file** with a multi-line comment header specifying `Script`, `Function`, `Usage`, `Requirements`, and `Notes / side effects`. The common helper must explicitly say `Source this file; do not run it directly`.
3. Scripts other than `lte-common.sh` must source the helper relative to their own script location, regardless of working directory. All scripts must support `--help` / `-h` to print purpose, syntax, examples and side effects without making changes; helper may document its usage only.
4. Check required commands and permissions; show clear errors and exit nonzero on failure. Use safe quoting and meaningful return codes. Read-only commands should not require root if unnecessary, but documented examples may use `sudo`.
5. Discover the modem ID from `mmcli -L` on every run. Support a numeric `MODEM_ID` environment override. If multiple modems are present, choose the EC25 only if it can be identified unambiguously; otherwise request an explicit `MODEM_ID`. Handle zero modems and changed IDs safely.
6. Support `LTE_CONNECTION` (default `NightGuard-LTE`) and `LTE_APN` (default `fast.t-mobile.com`). A positional APN to `lte-connect.sh` overrides the environment default. Reject invalid APNs and extra arguments.
7. Do not expose private modem or SIM identifiers in routine output, shell tracing, logs or error reports. Redact at least `imei`, `imsi`, `iccid`, `equipment id`, `device id`, `own`/phone number, and any sensitive SIM identifiers. If displaying complete raw output is requested in future, require a clearly separate opt-in; default must remain redacted.
8. Distinguish **USB detected**, **SIM detected**, **modem enabled**, **network registered**, **data bearer connected**, **network interface present**, and **usable IP connection**. Never claim internet connectivity based only on `mmcli -L` or an active SIM.
9. Do not change modem USB composition, send undocumented AT commands, reload kernel drivers, reboot, or alter unrelated interfaces automatically. Avoid `apt autoremove` and destructive cleanup of unrelated network configuration.
10. Scripts should be idempotent when practical, with explicit success/failure messages. Do not make Ethernet or Wi-Fi connectivity dependent on LTE.
11. Provide friendly guidance when the EC25 exposes only `ttyUSB*` ports and no USB networking data interface. Do not promise package installation alone will create `wwan0`.
12. Do not hardcode a device path, SIM ID, modem index, IP address, or interface name. Use the current system state.

## Detailed script specifications

### `lte-common.sh`

- Define safe defaults for `MODEM_ID`, `LTE_CONNECTION`, `LTE_APN`.
- Implement `need <command>` to check dependencies with helpful package guidance.
- Implement `modem_id` using `mmcli -L`; support `MODEM_ID` override and validate numeric input. Handle multiple modems safely.
- Provide reusable safe/redacted reporting utilities. Never execute network changes just by sourcing the helper.
- Provide helper(s) for `--help`, logging and root checks, without making read-only scripts require root.

### `lte-status.sh`

- Read-only summary: modem manufacturer/model, state, power state, access technology, registration, signal quality, operator ID/name where available, SIM presence and lock status, and relevant port/interface names.
- Display `ip -br link`, `nmcli device status` and active state of the configured LTE profile, when commands exist.
- If connected, report cellular interface, assigned IP and route information without assuming the default route uses LTE.
- Mask personal identifiers; distinguish missing network interface from unregistered SIM.

### `lte-full-status.sh`

- Read-only, detailed view matching the **structure** of `mmcli -m <id>` and `mmcli -i <sim-id>` output, with headings `LTE MODEM INFORMATION` and `LTE SIM INFORMATION`.
- Automatically detect the modem ID and extract the actual primary SIM path from the modem report; never assume the SIM ID equals the modem ID.
- Preserve useful ModemManager fields (manufacturer, model, firmware, supported/current radio modes, drivers, primary/secondary ports, modem state, power, network registration, SIM active, operator ID) while redacting IMEI/IMSI/ICCID, phone number and other identifiers.
- Display a clear message if no SIM is present. Never enable or connect the modem.

### `lte-enable.sh`

- Discover modem; execute `mmcli -m "$id" --enable` unless already enabled; print updated state.
- Do not start data or change Ethernet. Explain that registration and a data connection are separate steps.

### `lte-disable.sh`

- Warn that LTE service will be interrupted. Discover modem; run `mmcli -m "$id" --disable` unless already disabled.
- Do not modify Ethernet/Wi-Fi or remove saved profiles.

### `lte-connect.sh`

- Accept zero or one positional APN; default to `LTE_APN` / `fast.t-mobile.com`; validate APN syntax.
- Require `mmcli` and `nmcli`; discover and enable the modem, then check registration when possible.
- Create or update **only** a dedicated NetworkManager GSM connection named `LTE_CONNECTION`, using `type gsm`, `ifname '*'`, and `gsm.apn` equal to selected APN. Do not overwrite other profiles.
- Set `connection.autoconnect no`, `ipv4.route-metric 700`, `ipv6.route-metric 700`. Explain these metrics reduce preference for LTE but do not guarantee Ethernet priority if Ethernet route metrics are higher; never intentionally replace Ethernet's default route.
- Attempt `nmcli --wait 45 connection up "$LTE_CONNECTION"`. On success, verify profile state and report assigned cellular IP/interface; only call the internet usable if a real data test succeeds.
- On failure, report likely causes: SIM/network registration, APN, missing QMI/MBIM/ECM interface, USB instability. Suggest `lte-diagnostics.sh`; do **not** change modem USB composition automatically.

### `lte-disconnect.sh`

- If the named profile is active, run `nmcli connection down "$LTE_CONNECTION"`; otherwise report already disconnected and exit successfully.
- Leave modem enabled, preserve Ethernet/Wi-Fi, and do not delete the profile.

### `lte-diagnostics.sh`

- Read-only output of `mmcli -L`, sanitized modem information, `lsusb -d 2c7c:`, `lsusb -t`, `/dev/ttyUSB*`, `/dev/cdc-wdm*`, `ip -br link`, `nmcli device status`, and relevant kernel modules: `option`, `usbserial`, `usb_wwan`, `qmi_wwan`, `cdc_wdm`, `cdc_mbim`, `cdc_ether`.
- Report whether cellular data interfaces exist (e.g. `wwan*`, `usb*`, `enx*`, `cdc-wdm*`) without assuming every USB Ethernet device is the modem; correlate USB device topology/driver where possible.
- Show a limited recent selection of kernel messages relevant to USB over-current, undervoltage, disconnect/reset, Quectel, QMI/MBIM and UVC. Avoid dumping secrets or unrelated logs.
- Explain how to check USB hub power and cabling. Do not change settings.

### `lte-uninstall.sh`

- Intended for **controlled clean reset of LTE user-space software only**, not a factory reset of the modem.
- Require root and display an explicit warning/confirmation before removing packages. Support `--yes` for noninteractive confirmation; `--help` must be read-only.
- Stop LTE cellular sessions if practical, then stop/disable ModemManager only as part of the removal process.
- Target `modemmanager` and `libqmi-utils` for package removal/purge if installed. **Do not remove or purge `network-manager`, `netplan.io`, `systemd-networkd`, `usbutils`, Linux kernel modules/packages, firmware, or any package whose removal would break Ethernet/SSH.**
- Before any `apt purge`, perform and inspect an APT simulated removal (`apt-get -s purge ...`); abort if the transaction would remove `network-manager`, `netplan.io`, `systemd`, `openssh-server`, or other critical networking packages. Keep this safeguard even with `--yes`.
- Do not run `apt autoremove`; do not delete NetworkManager connection profiles, Ethernet settings, SIM files, modem firmware, or USB driver modules.
- Tell the user to run `lte-install.sh` next. Explain that LTE service will be unavailable during the reset and that this does not fix USB hardware instability.

### `lte-install.sh`

- Require root. Run `apt-get update` and install/reinstall `modemmanager`, `network-manager`, `libqmi-utils`, and `usbutils` from Ubuntu repositories. Avoid uninstalling/reconfiguring the existing network stack; preserve Ethernet/SSH.
- Enable/start ModemManager with `systemctl enable --now ModemManager`; report whether NetworkManager is available without unnecessarily changing its management of existing interfaces.
- Verify `mmcli -L`, Quectel USB ID `2c7c:0125`, current serial ports, `modinfo option`, `modinfo qmi_wwan` (or note module unavailable), and data interface candidates `/dev/cdc-wdm*` and `ip -br link`.
- If the modem is detected but only serial ports are present, report **partial success** and explain that QMI/MBIM/ECM USB composition/driver investigation is still required; do not falsely claim LTE data connectivity.
- Do not automatically enable/connect LTE, change APN, modify USB composition, or reboot.


### `lte-verify-uninstall.sh`

- Read-only; must never remove packages, stop services, or alter network settings.
- Check that `modemmanager`, `libqmi-utils`, `libmbim-utils`, and `usb-modeswitch` are not installed; `rc` residual configuration should be a warning, not an installed package.
- Check that `mmcli` is absent and ModemManager is inactive; preserve NetworkManager and report Ethernet link state.
- EC25 USB detection and `/dev/ttyUSB*` remaining are **informational**, not failures: Linux kernel drivers were intentionally preserved.
- Print `[PASS]`, `[FAIL]`, `[WARN]`, totals, and exit nonzero when required removal checks fail. Support `--help`.

### `lte-verify-install.sh`

- Read-only; never enable the modem, establish data, install packages, change routes or drivers.
- Verify all packages installed by the current `lte-install.sh`: `modemmanager`, `network-manager`, `libqmi-utils`, `libmbim-utils`, `usb-modeswitch`, `usbutils`. Check `mmcli`, `nmcli`, `lsusb`, `ip`, ModemManager active status, and available kernel modules (`option`, `qmi_wwan`).
- Detect EC25 USB ID `2c7c:0125`, serial ports, ModemManager modem detection, and SIM path without printing IMEI/IMSI/ICCID or telephone numbers.
- Check for `/dev/cdc-wdm*` and possible cellular network interfaces. Treat missing cellular data interface as a warning about connectivity readiness, not as proof that package installation failed.
- Display `[PASS]`, `[FAIL]`, `[WARN]` totals and nonzero exit code on required software failures. Explain that a passing installation verification **does not prove a usable 4G internet connection**. Support `--help`.

## `README.md` requirements

Document all scripts, arguments, environment variables, permissions, package prerequisites, installation/reset workflow, examples, troubleshooting, and security considerations. Include:

```bash
sudo apt update
sudo apt install -y modemmanager network-manager libqmi-utils usbutils
sudo systemctl enable --now ModemManager
chmod +x lte-*.sh
```

Normal operation:

```bash
sudo ./lte-status.sh
sudo ./lte-full-status.sh
sudo ./lte-diagnostics.sh
sudo ./lte-enable.sh
sudo ./lte-connect.sh fast.t-mobile.com
sudo ./lte-disconnect.sh
sudo ./lte-disable.sh
```

Clean reinstall (prefer running from Ethernet or local console):

```bash
sudo ./lte-uninstall.sh
./lte-verify-uninstall.sh
sudo ./lte-install.sh
./lte-verify-install.sh
sudo ./lte-diagnostics.sh
```

Explain that `sudo env MODEM_ID=239 ./lte-status.sh` can select a specific modem, but `239` is only an example and may change. Show `sudo env LTE_CONNECTION=NightGuard-LTE LTE_APN=fast.t-mobile.com ./lte-connect.sh` as an example. Explain `bash -n lte-*.sh` checks syntax only, not actual modem operation.

Include notes that T-Mobile APN `fast.t-mobile.com` is an example and should match the active SIM plan, and that the modem's SIM can be detected even if data connectivity is unavailable. Describe safe redaction of sensitive identifiers. Include the earlier USB hub over-current/disconnect and undervoltage warnings as known hardware concerns; suggest a properly powered USB hub and appropriate Raspberry Pi 5 USB-C power supply.

## Validation and acceptance criteria

Codex must verify:

1. All **12** `.sh` files plus `README.md` are generated alongside this single `4GLTE.md`.
2. Every Bash file starts with the required explanatory comment header and has functional `--help` (except the sourced helper).
3. `bash -n lte-*.sh` passes; scripts have executable permissions.
4. No hardcoded modem ID `239`, SIM ID, private identifier, USB serial device index or LTE IP address.
5. `lte-status.sh`, `lte-full-status.sh`, and `lte-diagnostics.sh` are read-only.
6. Enable/disable/connect/disconnect act only on the intended LTE modem/profile and preserve Ethernet/Wi-Fi.
7. Uninstall includes explicit confirmation and APT simulation safeguards, never purges NetworkManager or removes kernel drivers.
8. Install verifies software and hardware detection, but does not falsely claim LTE internet is operational.
9. A missing USB data interface produces actionable guidance rather than silent success.
10. `README.md` covers all 13 scripts and the full reset/reinstall and connection workflow.
11. Both verification scripts are read-only, report PASS/FAIL/WARN and exit nonzero for required software check failures.
12. Document that **real Raspberry Pi/EC25 hardware testing is still necessary**; syntax checks alone are not functional verification.




## Universal Wi-Fi adapter configuration — `wifi-add.sh`

Codex must also generate and maintain **`wifi-add.sh`** in the same directory as the LTE scripts. This is a separate Wi-Fi utility and must not depend on an LTE modem, SIM, or LTE scripts. It must work on Ubuntu 24.04 with either a Raspberry Pi built-in Wi-Fi adapter (for example `wlan0`) **or any detected USB Wi-Fi adapter** (for example `wlx000f0241f1aa`). Never hardcode either interface name.

Requirements:

1. Begin with a clear Bash comment header explaining purpose, requirements, and `sudo ./wifi-add.sh` usage. Support `--help` without root privileges.
2. Require root only when configuring, and use `iw dev` to discover all detected Wi-Fi interfaces. On a Raspberry Pi, optionally attempt `modprobe brcmfmac` before discovery; failure must not prevent USB adapters from being used.
3. Display a numbered menu with each Wi-Fi interface and identify USB interfaces by inspecting `/sys/class/net/<interface>/device` paths. Let the user select the interface by number; validate the selection.
4. Interactively prompt for an SSID and a **hidden password** (`read -s`). Allow an empty password for an open network. Never echo the password or place it in command-line arguments or shell history.
5. Use Netplan and `python3-yaml` to update only the selected Wi-Fi interface. If it is already defined in an existing `/etc/netplan/*.yaml`, modify that file's Wi-Fi entry while preserving Ethernet, other Wi-Fi adapters, and unrelated configuration; otherwise create `/etc/netplan/90-wifi-configure.yaml`. Back up an existing file before modifying it. Apply restrictive `0600` permissions to the resulting file because it contains a Wi-Fi credential.
6. Validate with `netplan generate`, then use `netplan try --timeout 60` for rollback on lost connectivity. If validation or confirmation fails, restore the original file (or delete a newly created file). Warn that remote SSH sessions may still be interrupted and recommend running from a local console.
7. After confirmation, display `ip -br addr show dev <selected-interface>` and `iw dev <selected-interface> link`. Report whether an IPv4 address was assigned; do not claim successful internet access solely from an IP address.
8. Avoid altering Ethernet routes or disabling the LTE modem. Do not log or print stored Wi-Fi credentials. The existing NetworkManager/Netplan renderer configuration must be respected; do not switch renderers globally.
9. Document required tools (`bash`, `ip`, `iw`, `netplan`, `python3`, `python3-yaml`) and include instructions to install missing packages with `sudo apt install -y iw python3-yaml` where appropriate.
10. Validate with `bash -n wifi-add.sh` and test interface discovery/selection on systems with one or multiple Wi-Fi adapters. Hardware connection tests are required before claiming successful operation.

Usage:

```bash
sudo apt install -y iw python3-yaml
chmod +x wifi-add.sh
sudo ./wifi-add.sh
```

Typical menu on this Raspberry Pi (examples only; discover interfaces dynamically):

```text
Available Wi-Fi interfaces:
  1) wlan0 (onboard/other)
  2) wlx000f0241f1aa (USB)
Select Wi-Fi interface number:
Wi-Fi SSID (network name):
Wi-Fi password (hidden):
Save and try this Wi-Fi configuration? [y/N]:
```

**Codex acceptance criteria:** The project now contains **13 Bash files** (12 LTE scripts and one universal Wi-Fi configuration script), plus `4GLTE.md` and `README.md`. The README must document the Wi-Fi tool and both Wi-Fi types, and `bash -n` must pass for all scripts.

**This is the only Markdown specification required to regenerate the project.**


## Mandatory Wi-Fi service self-repair for `wifi-add.sh`

The universal Wi-Fi utility must automatically handle the observed Ubuntu 24.04 Netplan WPA-supplicant startup failure: `Failed to initialize control interface '/run/wpa_supplicant'`, potentially caused by a stale `/run/wpa_supplicant/<selected-interface>` socket or another active supplicant. Before applying a selected interface's configuration, stop **only** `netplan-wpa-<selected-interface>.service`, check the matching socket, and remove it **only if** no live process owns or manages it (check `pgrep -af wpa_supplicant`, `fuser` if installed, and `ss` if available). Never kill a competing supplicant or remove an actively owned socket; print a targeted error instead. Reset failed state for that service only. Following Netplan confirmation, check the selected interface's WPA service; if inactive, perform one targeted repair and restart. Wait briefly for association/DHCP and show the service status, `iw link`, IPv4 address, and filtered failure logs. Netplan acceptance does not prove Wi-Fi connectivity. Preserve all other Wi-Fi interfaces, Ethernet, LTE, and routes. Support `--help` without sudo. The filename is `wifi-add.sh` (not `wifi-configure.sh`).

### WPA socket conflict enhancement (required behavior)

`wifi-add.sh` must not fail prematurely merely because `fuser` reports an occupied WPA socket. After `netplan try` has applied the selected adapter configuration, inspect the **selected** `netplan-wpa-<interface>.service` state. If failed, stop that service only, wait up to five seconds for its socket to be released, inspect the socket using `ss`, `fuser`, and optionally `wpa_cli`, and remove the selected adapter's socket only when **no live process owns it** and no independent supplicant manages that adapter. Reset the failed state, restart the selected service, and wait for association/DHCP. Never kill arbitrary WPA processes, delete live sockets, restart all networking, or change Ethernet/LTE/other Wi-Fi services. If a competing manager still owns the interface, report the process and fail safely; no automatic takeover is permitted. A service being active does not prove association, and association does not prove IPv4 connectivity.

### Global WPA supplicant ownership recovery (required; November 2026)

When `fuser -v /run/wpa_supplicant/<interface>` shows the socket is owned by the system-wide `wpa_supplicant -u -s -O DIR=/run/wpa_supplicant GROUP=netdev`, the Wi-Fi script MUST NOT simply stop with a socket conflict. It must identify the actual socket owner using `fuser` and `/proc/<pid>/cmdline`; verify it is the global D-Bus-enabled `wpa_supplicant -u` process; avoid taking over an interface actively managed by NetworkManager; and request **interface-specific removal** through `busctl --system call ... fi.w1.wpa_supplicant1 GetInterface s <interface>` followed by `RemoveInterface o <returned-object-path>`. If supported, an alternative is `wpa_cli -g /run/wpa_supplicant/global interface_remove <interface>`. Check the selected socket is no longer owned before starting only `netplan-wpa-<interface>.service`. Do not kill the global daemon, stop other Wi-Fi services, remove active sockets, or restart Ethernet/LTE. If the API refuses the request, fail safely with clear diagnostics. Confirm WPA service active, wireless association, and IPv4 before reporting success.


## Persistent Wi-Fi interface names (wlan1, wlan2, ...) — corrected Ubuntu 24.04 implementation

`wifi-rename.sh` renames USB Wi-Fi adapters persistently using a MAC-address-matched **systemd `.link` file**, e.g. `/etc/systemd/network/10-wifi-000f0241f1aa.link`, with `[Match] MACAddress=...` and `[Link] NamePolicy= / Name=wlan1`. **Do not use Netplan `match:` or `set-name:` under `network.wifis` with the networkd backend**: Ubuntu 24.04 rejects it with `networkd backend does not support wifi with match:, only by interface name`.

The script updates the existing Netplan Wi-Fi mapping key from the old interface name (`wlx...`) to the new (`wlan1`), leaving `access-points`, DHCP, Ethernet, LTE, and other adapters intact. It must back up the original YAML, refuse collisions, validate using `netplan generate`, and restore both YAML and `.link` files if validation fails. It must not call `netplan apply`, restart networking, or rename interfaces live. Reboot to activate; verify with `iw dev`, `ip -br link`, and `networkctl status wlan1`. Keep built-in `wlan0` unchanged. If another earlier `.link` naming policy takes precedence or the driver does not support renaming, report it rather than claiming success. The existing `wifi-add.sh` should continue offering the optional `wifi-rename.sh` call.

Examples: `sudo ./wifi-rename.sh` or `sudo ./wifi-rename.sh wlx000f0241f1aa wlan1`.

**Acceptance:** `bash -n` passes for all 14 Bash scripts, `netplan generate` succeeds after rename preparation, and actual renaming is confirmed only after reboot.


## Integrated persistent Wi-Fi renaming in wifi-add.sh (latest)

`wifi-add.sh` MUST include the complete persistent renaming implementation internally, not depend on `wifi-rename.sh`. After configuring and testing the selected Wi-Fi adapter, offer to assign `wlan1`, `wlan2`, etc. Preserve built-in `wlan0`. Match the selected adapter by MAC using a `/etc/systemd/network/10-wifi-<MAC>.link` file containing `[Match] MACAddress=...` and `[Link] NamePolicy=` / `Name=wlanN`. In the Netplan YAML, rename the Wi-Fi dictionary key from the current interface to the chosen `wlanN` **without** using `match:` or `set-name:` (unsupported with the networkd Wi-Fi backend). Check for name and policy collisions, back up Netplan, validate with `netplan generate`, roll back both the Netplan change and `.link` file if validation fails, and never rename the active interface in-place. Reboot required. Retain `wifi-rename.sh` as an optional standalone utility for compatibility, but `wifi-add.sh` must work independently.


## Automatic Wi-Fi connection verification (November 2026)

`wifi-add.sh` must verify the selected built-in or USB Wi-Fi adapter after Netplan configuration and targeted WPA conflict recovery. It must print the Netplan mapping, `netplan-wpa-<interface>.service` state, `iw dev <interface> link` association, `ip -br addr` and global IPv4 address, `networkctl status`, selected WPA service errors, and filtered `systemd-networkd` DHCP events. Wait up to 40 seconds for DHCP; distinguish failed WPA service, unassociated Wi-Fi, associated Wi-Fi without IPv4, and successful association with an IPv4 address. Do not report configuration acceptance as successful connectivity. A renamed interface needs reboot; after reboot the new interface name must match the Netplan key. Do not restart all networking, change default routes, kill other WPA processes, or print Wi-Fi passwords. Internet reachability is not implied by DHCP success.


## Raspberry Pi built-in Wi-Fi after reboot (wifi-add.sh)

On Raspberry Pi hardware, `wifi-add.sh` loads the `brcmfmac` kernel module and ensures `/etc/modules-load.d/brcmfmac.conf` contains `brcmfmac` so the onboard Broadcom Wi-Fi is requested at subsequent boots. It checks whether `wlan0` exists, prints its IPv4 status, and reports Broadcom kernel messages if the interface remains missing. It does not rename `wlan0`, change the USB `wlan1` naming rule, or modify Ethernet/LTE. The `systemd-modules-load.service` is normally static; it should not be forcibly enabled. After reboot, check `iw dev` and `ip -4 -br addr`. A missing interface despite the module loading requires further firmware/boot diagnostics; driver loading alone does not guarantee Wi-Fi or DHCP connectivity.
