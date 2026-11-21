#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Script: lte-diagnostics.sh
# Function: Read-only diagnostics for EC25 USB, serial ports, drivers and networking.
# Usage: sudo ./lte-diagnostics.sh
# Note: Use when the modem is detected but LTE data or a WWAN interface is missing.
# -----------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/lte-common.sh"
echo '=== ModemManager ==='
if command -v mmcli >/dev/null; then mmcli -L || true; else echo 'mmcli missing'; fi
echo; echo '=== Quectel USB device ==='
if command -v lsusb >/dev/null; then lsusb -d 2c7c: || true; else echo 'lsusb missing'; fi
echo; echo '=== USB serial ports ==='
ls -l /dev/ttyUSB* 2>/dev/null || true
echo; echo '=== Network interfaces ==='
ip -br link
echo; echo '=== Relevant drivers ==='
lsmod | grep -E '(^| )(option|qmi_wwan|cdc_mbim|cdc_ether|usb_wwan|usbserial|cdc_wdm)( |$)' || true
echo; echo '=== NetworkManager ==='
if command -v nmcli >/dev/null; then nmcli device status; else echo 'nmcli missing'; fi
echo; echo '=== Recent USB/kernel events ==='
dmesg -T 2>/dev/null | grep -Ei '2c7c|quectel|qmi|mbim|wwan|usb.*disconnect|over.current' | tail -40 || echo 'Run with sudo for kernel logs'
echo; echo 'If the EC25 shows only ttyUSB ports, cellular data may require a USB networking driver or modem USB-mode configuration.'
