#!/usr/bin/env bash
# Read-only camera and server diagnostics. Run as the account that starts Night Guard.
set -uo pipefail
shopt -s nullglob
export LC_ALL=C

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unit=night-guard-server.service

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash config/debug-server-ubuntu.sh [video-port]'
    echo 'Reports USB topology, V4L2 modes, kernel messages, service logs, and local HTTP status.'
    exit 0
fi
if [[ $(uname -s) != Linux ]]; then
    echo 'Run this diagnostic on the Ubuntu machine with the cameras attached.' >&2
    exit 1
fi

# Read only SERVER_PORT with the same dotenv parser as the server. Never source
# the .env file as shell code. A positional port overrides it for a server run with -p.
port=${1:-}
if [[ -z $port ]] && command -v python3 >/dev/null; then
    port="$(python3 -c 'import sys; from dotenv import dotenv_values; print(dotenv_values(sys.argv[1]).get("SERVER_PORT") or 14095)' "$root/python/server.env" 2>/dev/null)"
fi
port=${port:-14095}
if [[ $# -gt 1 || ! $port =~ ^[0-9]{1,5}$ ]] || (( 10#$port < 1 || 10#$port > 65535 )); then
    echo 'Provide one TCP port from 1 through 65535.' >&2
    exit 2
fi
port=$((10#$port))

section() { printf '\n=== %s ===\n' "$1"; }

section 'Host and permissions'
# uname prints the running kernel. id shows whether this session already has
# the video group. Device ACLs can also grant access without video membership.
uname -sr
id

section 'USB devices and topology'
# lsusb lists devices that successfully enumerated. lsusb -t shows the parent
# hub/root bus, loaded drivers (look for uvcvideo), and negotiated link speeds.
# Compare upstream paths as well as port numbers when testing multiple hubs.
if command -v lsusb >/dev/null; then
    lsusb
    lsusb -t
else
    echo 'lsusb is unavailable. Ubuntu provides it in the usbutils package.'
fi

section 'V4L2 device nodes and stable names'
# A physical webcam can expose multiple /dev/video nodes, including metadata.
# The by-id and by-path links show which node each stable name currently targets.
nodes=(/dev/video[0-9]*)
if (( ${#nodes[@]} )); then
    ls -l -- "${nodes[@]}"
else
    echo 'No /dev/video* nodes were found. Check whether lsusb sees the cameras and uvcvideo is bound.'
fi
for directory in /dev/v4l/by-id /dev/v4l/by-path; do
    if [[ -d $directory ]]; then
        ls -l -- "$directory"
    fi
done

section 'V4L2 capabilities and advertised modes'
# --list-devices groups nodes by device. --all reports capabilities, controls,
# and the current/default format visible to v4l2-ctl. --list-formats-ext reports
# advertised formats (such as MJPG and YUYV), sizes, and frame intervals.
# These queries do not capture frames or confirm the live server's exact mode.
if command -v v4l2-ctl >/dev/null; then
    timeout 10s v4l2-ctl --list-devices
    for node in "${nodes[@]}"; do
        printf '\n--- %s ---\n' "$node"
        if [[ ! -r $node || ! -w $node ]]; then
            echo 'Current account lacks read or write access. Compare its groups with the device owner and service user.'
        fi
        timeout 8s v4l2-ctl -d "$node" --all ||
            echo 'Capability query failed or timed out.'
        timeout 8s v4l2-ctl -d "$node" --list-formats-ext ||
            echo 'Mode query failed or timed out. This can also happen on a metadata node.'
    done
else
    echo 'v4l2-ctl is unavailable. Ubuntu provides it in the v4l-utils package.'
fi

section 'Recent USB and camera kernel messages'
# sudo dmesg -T | grep -Ei 'uvc|bandwidth|usb.*reset|usb.*error|not enough' | tail -50
# dmesg reads the kernel ring buffer. -T renders approximate wall-clock times.
# grep keeps USB/UVC warnings of interest. tail limits the report to 50 lines.
# Capture dmesg first so permission failures are distinguishable from no matches.
if messages="$(sudo dmesg -T)"; then
    relevant="$(printf '%s\n' "$messages" |
        grep -Ei 'uvc|bandwidth|usb.*reset|usb.*error|not enough' | tail -50)"
    if [[ -n $relevant ]]; then
        printf '%s\n' "$relevant"
        if grep -Eiq 'not enough|insufficient|no space left|bandwidth.*(fail|exceed)' <<< "$relevant"; then
            echo 'Possible resource or bandwidth failure. Compare the timestamps with when the camera failed.'
        fi
        if grep -Eiq 'usb.*reset|usb.*error' <<< "$relevant"; then
            echo 'USB resets or errors were reported. Check whether they coincide with the failed camera start.'
        fi
    else
        echo 'No matching messages in the current kernel buffer.'
    fi
else
    echo 'Could not read kernel messages. Check sudo access and dmesg permissions.'
fi

section 'Processes using camera nodes'
# fuser -v lists processes with each node open. Night Guard itself is expected
# while it is streaming. Another capture process can make a camera busy.
if (( ${#nodes[@]} )) && command -v fuser >/dev/null; then
    if ! sudo fuser -v "${nodes[@]}"; then
        echo 'No camera owners were reported, or the process query failed.'
    fi
elif ! command -v fuser >/dev/null; then
    echo 'fuser is unavailable. Ubuntu provides it in the psmisc package.'
fi

section 'Night Guard service and local listener'
# systemctl show reports the current state and configured user/groups.
# journalctl prints the most recent messages for the unit in this boot.
if command -v systemctl >/dev/null; then
    systemctl show "$unit" -p LoadState -p ActiveState -p SubState -p MainPID -p User -p SupplementaryGroups
fi
if command -v journalctl >/dev/null; then
    sudo journalctl -b -u "$unit" -n 50 --no-pager ||
        echo 'Could not read the service journal.'
fi

# ss lists listening TCP sockets on the video port. curl tests the local
# /status endpoint, which distinguishes connected, offline, and disabled slots.
if command -v ss >/dev/null; then
    ss -ltn "sport = :$port"
fi
if command -v curl >/dev/null; then
    if status="$(curl --noproxy '*' --fail --silent --show-error --max-time 4 "http://127.0.0.1:$port/status")"; then
        printf 'Night Guard /status: %s\n' "$status"
        echo 'Connected confirms frame delivery to the server. A connected camera can still produce a dark image.'
    else
        echo "Could not reach /status on localhost:$port. Check the listener, bind address, and video port above."
    fi
else
    echo 'curl is unavailable. Install it to include the local /status check.'
fi
