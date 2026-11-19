#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
server="$root/python/night_guard_server.py"
unit=night-guard-server.service
if [[ ! -f $server ]]; then
    echo "Night Guard server source is missing: $server" >&2
    exit 1
fi

if [[ -f /etc/systemd/system/$unit ]]; then
    sudo systemctl stop "$unit"
fi

# Also stop servers started directly with Python from this checkout.
sudo python3 - "$server" <<'PYTHON'
import os
from pathlib import Path
import re
import select
import signal
import sys

target = Path(sys.argv[1]).resolve()
for process in Path("/proc").iterdir():
    if not process.name.isdecimal():
        continue
    try:
        pid = int(process.name)
        executable = (process / "exe").resolve(strict=True)
        if not re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", executable.name):
            continue
        arguments = (process / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        for argument in arguments[1:]:
            if argument.startswith((b"-c", b"-m")):
                break
            if argument.startswith(b"-"):
                continue
            script = process / "cwd" / os.fsdecode(argument)
            if script.resolve() == target:
                fd = os.pidfd_open(pid)
                try:
                    # Recheck after opening the descriptor in case the PID was reused.
                    if (process / "cmdline").read_bytes().rstrip(b"\0").split(b"\0") != arguments:
                        break
                    print(f"Stopping manually launched Night Guard server (PID {pid}).", flush=True)
                    signal.pidfd_send_signal(fd, signal.SIGTERM)
                    wait = select.poll()
                    wait.register(fd, select.POLLIN)
                    if not wait.poll(15000):
                        sys.exit(f"Night Guard server PID {pid} did not exit within 15 seconds.")
                finally:
                    os.close(fd)
            break
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
PYTHON

echo 'Night Guard server stopped.'
