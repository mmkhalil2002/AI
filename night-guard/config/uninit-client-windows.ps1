#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'

Get-NetFirewallRule -PolicyStore PersistentStore |
    Where-Object Name -EQ 'NightGuard-client-TCP' |
    Remove-NetFirewallRule

Write-Host 'Night Guard client uninitialization complete.'
