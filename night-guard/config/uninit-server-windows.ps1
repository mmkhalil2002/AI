#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'

Get-NetFirewallRule -PolicyStore PersistentStore |
    Where-Object Name -EQ 'NightGuard-server-TCP' |
    Remove-NetFirewallRule

Write-Host 'Night Guard server uninitialization complete.'
