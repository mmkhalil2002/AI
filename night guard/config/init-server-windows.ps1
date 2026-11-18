$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this setup in PowerShell as Administrator to configure Windows Firewall.'
}

function Find-Python {
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $executable = & $launcher.Source -3.12 -c "import sys; print(sys.executable)"
        if ($LASTEXITCODE -eq 0) { return $executable }
    }

    $locations = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe')
        (Join-Path $env:ProgramFiles 'Python312\python.exe')
    )
    foreach ($location in $locations) {
        if (Test-Path -LiteralPath $location -PathType Leaf) { return $location }
    }

    $command = Get-Command python.exe -ErrorAction SilentlyContinue |
        Where-Object Source -NotLike '*\Microsoft\WindowsApps\*'
    if ($command) {
        $executable = & $command.Source -c "import sys; print(sys.executable) if sys.version_info >= (3, 10) else sys.exit(1)"
        if ($LASTEXITCODE -eq 0) { return $executable }
    }
}

$pythonExe = Find-Python
if (!$pythonExe) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (!$winget) {
        throw 'Install Python 3.12 or Windows App Installer (winget), then rerun setup.'
    }
    & $winget.Source install --id Python.Python.3.12 --exact --scope user --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
    $pythonExe = Find-Python
    if (!$pythonExe) { throw 'Open a new elevated PowerShell session and rerun setup to locate Python.' }
}

$root = Split-Path $PSScriptRoot -Parent
$requirements = Join-Path $root 'requirements.txt'
& $pythonExe -m pip install --user -c $requirements opencv-python python-dotenv
if ($LASTEXITCODE -ne 0) { throw 'Python package installation failed.' }

$envFile = Join-Path $root 'python/server.env'
$port = @'
import sys
from pathlib import Path

from dotenv import dotenv_values

path = Path(sys.argv[1])
defaults = """CAMERA_COUNT=4
FRONT_CAMERA=0
BACK_CAMERA=1
LEFT_CAMERA=2
RIGHT_CAMERA=3
WIDTH=640
HEIGHT=480
FPS=15
JPEG_QUALITY=80
SERVER_HOST=127.0.0.1
SERVER_PORT=14095
CONTROL_PORT=14098
"""
try:
    with path.open("x", encoding="utf-8") as config:
        config.write(defaults)
except FileExistsError:
    pass

try:
    settings = dotenv_values(path)
    port = int(settings.get("SERVER_PORT", 14095))
    if not 1 <= port <= 65535:
        raise ValueError("Port must be from 1 to 65535")
except (TypeError, ValueError) as exc:
    sys.exit(f"{path.name}: {exc}")
print(port)
'@ | & $pythonExe - $envFile
if ($LASTEXITCODE -ne 0) { throw 'Could not read the firewall port from the configuration.' }

$rule = @{
    Name = 'NightGuard-server-TCP'
    Enabled = 'True'
    Action = 'Allow'
    Protocol = 'TCP'
    Direction = 'Inbound'
    LocalPort = $port
    RemoteAddress = 'LocalSubnet'
    Profile = @('Domain', 'Private')
}
if (Get-NetFirewallRule -Name $rule.Name -ErrorAction SilentlyContinue) {
    Set-NetFirewallRule @rule | Out-Null
} else {
    New-NetFirewallRule @rule -DisplayName 'Night Guard server (TCP)' | Out-Null
}
Write-Host "Night Guard server setup complete (TCP port $port)."
Write-Host "Python executable: $pythonExe"
