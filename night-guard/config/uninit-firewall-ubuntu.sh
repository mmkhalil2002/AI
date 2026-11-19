#!/usr/bin/env bash

remove_night_guard_firewall_rules() {
    if ! command -v ufw >/dev/null; then
        return 0
    fi

    python3 - "$1" <<'PYTHON'
import shlex
import subprocess
import sys

comment = f"Night Guard {sys.argv[1]}"
rules = subprocess.check_output(["sudo", "ufw", "show", "added"], text=True)
for line in rules.splitlines():
    if not line.startswith("ufw allow ") or not line.endswith(f" comment '{comment}'"):
        continue
    command = shlex.split(line)
    if command[-2:] == ["comment", comment]:
        subprocess.run(["sudo", "ufw", "--force", "delete", *command[1:]], check=True)
PYTHON
}
