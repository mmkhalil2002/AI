#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ${SUDO_USER:-$(id -un)} == root ]]; then
    echo 'Run this script from the Ubuntu account that installed Night Guard.' >&2
    exit 1
fi

bash "$script_dir/stop-server-ubuntu.sh"
exec bash "$script_dir/run-server-ubuntu.sh"
