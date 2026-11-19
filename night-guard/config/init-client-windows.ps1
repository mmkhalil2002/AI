$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
$isAdministrator = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

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
$requirements = Join-Path $root 'requirements-client.txt'
& $pythonExe (Join-Path $PSScriptRoot 'generate_requirements.py') client
if ($LASTEXITCODE -ne 0) { throw 'Requirements generation failed.' }
& $pythonExe -m pip install --user -r $requirements
if ($LASTEXITCODE -ne 0) { throw 'Python package installation failed.' }

$envFile = Join-Path $root 'python/client.env'
$port = @'
import sys
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

path = Path(sys.argv[1])
defaults = """SERVER_URL=http://127.0.0.1:14095
GUI_TITLE=Night Guard
GUI_WIDTH=880
GUI_HEIGHT=620
WIDTH=480
HEIGHT=320
CONTROL_PANEL_WIDTH=200
"""
try:
    with path.open("x", encoding="utf-8") as config:
        config.write(defaults)
except FileExistsError:
    pass

try:
    settings = dotenv_values(path)
    address = (settings.get("SERVER_URL") or "").strip() or "http://127.0.0.1:14095"
    explicit_scheme = "://" in address
    parsed = urlsplit(address if explicit_scheme else "http://" + address)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Invalid SERVER_URL")
    port = parsed.port
    if port is None:
        port = {"http": 80, "https": 443}[parsed.scheme] if explicit_scheme else 14095
    if not 1 <= port <= 65535:
        raise ValueError("Port must be from 1 to 65535")
except (TypeError, ValueError) as exc:
    sys.exit(f"{path.name}: {exc}")
print(port)
'@ | & $pythonExe - $envFile
if ($LASTEXITCODE -ne 0) { throw 'Could not read the firewall port from the configuration.' }

$rule = @{
    Name = 'NightGuard-client-TCP'
    Enabled = 'True'
    Action = 'Allow'
    Protocol = 'TCP'
    Direction = 'Outbound'
    RemotePort = $port
    Profile = 'Any'
}
if (!$isAdministrator) {
    Write-Host 'Client setup is running without Administrator privileges. Run setup as Administrator if your network requires an explicit outbound firewall rule.'
} elseif (Get-NetFirewallRule -Name $rule.Name -ErrorAction SilentlyContinue) {
    Set-NetFirewallRule @rule | Out-Null
} else {
    New-NetFirewallRule @rule -DisplayName 'Night Guard client (TCP)' | Out-Null
}
Write-Host "Night Guard client setup complete (TCP port $port)."
Write-Host "Python executable: $pythonExe"
