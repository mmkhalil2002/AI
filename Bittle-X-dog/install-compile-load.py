# ============================================================
# BITTLE X ESP32 LOCAL WINDOWS FIRMWARE TOOL
# ============================================================
#
# PURPOSE
# ------------------------------------------------------------
# This script prepares a complete local development environment
# for Petoi Bittle X ESP32 firmware.
#
# Everything is installed relative to the current directory:
#
#   current_folder/
#      bittle_env/
#         arduino-cli.exe
#         data/
#         firmware/
#
# Nothing is installed globally.
# No system PATH changes are required.
#
#
# ============================================================
# MANUAL PROCEDURE THIS SCRIPT AUTOMATES
# ============================================================
#
# STEP 1 — Create a local working folder
# ------------------------------------------------------------
# Manually, you would run:
#
#   mkdir bittle_env
#
# This folder stores Arduino CLI, firmware, board packages,
# and local configuration.
#
#
# STEP 2 — Download Arduino CLI
# ------------------------------------------------------------
# Manually, you would download:
#
#   https://downloads.arduino.cc/arduino-cli/arduino-cli_latest_Windows_64bit.zip
#
# Then extract it into:
#
#   bittle_env/
#
# After extraction you should have:
#
#   bittle_env/arduino-cli.exe
#
# Arduino CLI is the tool that compiles and uploads the firmware.
#
#
# STEP 3 — Create local Arduino CLI configuration
# ------------------------------------------------------------
# Manually, you would run:
#
#   bittle_env\arduino-cli.exe config init --config-file bittle_env\data\cli.yaml
#
# This creates a local configuration file instead of using the
# global Arduino configuration in your Windows user folder.
#
#
# STEP 4 — Install ESP32 board support locally
# ------------------------------------------------------------
# Manually, you would run:
#
#   bittle_env\arduino-cli.exe core update-index --config-file bittle_env\data\cli.yaml
#
# Then:
#
#   bittle_env\arduino-cli.exe core install esp32:esp32@2.0.12 --config-file bittle_env\data\cli.yaml
#
# This installs:
#
#   - ESP32 compiler
#   - ESP32 upload tools
#   - ESP32 board definitions
#   - ESP32 libraries
#
# For Bittle X / BiBoard ESP32, version 2.0.12 is commonly used
# because newer ESP32 cores can sometimes cause compatibility issues.
#
#
# STEP 5 — Download OpenCat ESP32 firmware
# ------------------------------------------------------------
# Manually, you would download or clone:
#
#   https://github.com/PetoiCamp/OpenCatEsp32-Quadruped-Robot
#
# The script downloads the ZIP version and extracts it into:
#
#   bittle_env/firmware/
#
# This firmware folder should contain the OpenCat ESP32 source code.
#
#
# STEP 6 — Modify firmware manually
# ------------------------------------------------------------
# Before compiling, you can open the firmware folder and insert
# your custom C++ code, such as:
#
#   - my_custom_walk()
#   - stand_neutral_custom()
#   - Wi-Fi command server
#   - handleWiFiCommand()
#
# This script does NOT edit your firmware automatically.
# It compiles whatever C++ code is currently in the firmware folder.
#
#
# STEP 7 — Compile firmware
# ------------------------------------------------------------
# Manually, you would run:
#
#   bittle_env\arduino-cli.exe compile --fqbn esp32:esp32:esp32 bittle_env\firmware --config-file bittle_env\data\cli.yaml
#
# This checks and builds your C++ firmware.
#
#
# STEP 8 — Connect Bittle X by USB
# ------------------------------------------------------------
# Use a real USB data cable.
#
# Windows should show a COM port such as:
#
#   COM3
#   COM5
#   COM7
#
# You can check it in:
#
#   Device Manager → Ports (COM & LPT)
#
#
# STEP 9 — Upload firmware to ESP32
# ------------------------------------------------------------
# Manually, you would run:
#
#   bittle_env\arduino-cli.exe upload -p COM3 --fqbn esp32:esp32:esp32 bittle_env\firmware --config-file bittle_env\data\cli.yaml
#
# Replace COM3 with your actual COM port.
#
#
# STEP 10 — Verify firmware
# ------------------------------------------------------------
# Manually, you can open serial monitor:
#
#   bittle_env\arduino-cli.exe monitor -p COM3 -c baudrate=115200
#
# If Wi-Fi code is included, you should see:
#
#   ESP32 IP Address: 192.168.x.x
#
#
# ============================================================
# SIMPLE MENU OPTIONS
# ============================================================
#
# 1 - Install Arduino CLI locally
# 2 - Install ESP32 board support locally
# 3 - Download firmware locally
# 4 - Compile firmware
# 5 - Upload firmware using USB
# 6 - Run all steps
#
# ============================================================
# ============================================================
# BITTLE X ESP32 LOCAL WINDOWS FIRMWARE TOOL
# ============================================================
#
# MENU:
#   1 - Install all necessary packages and directories
#   2 - Compile firmware
#   3 - Install/upload firmware to ESP32 using USB
#
# Everything is installed relative to the current folder:
#
#   current_folder/
#      bittle_env/
#         arduino-cli.exe
#         data/
#         firmware/
#
# ============================================================

import os
import sys
import zipfile
import shutil
import subprocess
import urllib.request

BASE_DIR = os.path.join(os.getcwd(), "bittle_env")
DATA_DIR = os.path.join(BASE_DIR, "data")
FIRMWARE_DIR = os.path.join(BASE_DIR, "firmware")

CLI_EXE = os.path.join(BASE_DIR, "arduino-cli.exe")
CLI_CONFIG = os.path.join(DATA_DIR, "cli.yaml")

ESP32_FQBN = "esp32:esp32:esp32"
ESP32_CORE = "esp32:esp32@2.0.12"

CLI_URL = "https://downloads.arduino.cc/arduino-cli/arduino-cli_latest_Windows_64bit.zip"
FIRMWARE_URL = "https://github.com/PetoiCamp/OpenCatEsp32-Quadruped-Robot/archive/refs/heads/main.zip"


def run_cmd(cmd):
    print("\nRUNNING:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def download_file(url, path):
    print("\nDownloading:")
    print(url)
    urllib.request.urlretrieve(url, path)
    print("Saved:", path)


def extract_zip(zip_path, output_dir):
    print("\nExtracting:", zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)


def install_all():
    """
    OPTION 1:
    Install all necessary packages and directories.

    Manual equivalent:
      mkdir bittle_env
      download arduino-cli.exe
      create bittle_env/data/cli.yaml
      install ESP32 core
      download OpenCat ESP32 firmware into bittle_env/firmware
      install pyserial for COM port detection
    """

    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Install Arduino CLI locally
    # --------------------------------------------------------
    if not os.path.exists(CLI_EXE):
        cli_zip = os.path.join(BASE_DIR, "arduino-cli.zip")
        download_file(CLI_URL, cli_zip)
        extract_zip(cli_zip, BASE_DIR)
    else:
        print("Arduino CLI already installed:", CLI_EXE)

    # --------------------------------------------------------
    # Create local Arduino config
    # --------------------------------------------------------
    run_cmd([
        CLI_EXE,
        "config",
        "init",
        "--config-file",
        CLI_CONFIG
    ])

    # --------------------------------------------------------
    # Install ESP32 board support locally
    # --------------------------------------------------------
    run_cmd([
        CLI_EXE,
        "core",
        "update-index",
        "--config-file",
        CLI_CONFIG
    ])

    run_cmd([
        CLI_EXE,
        "core",
        "install",
        ESP32_CORE,
        "--config-file",
        CLI_CONFIG
    ])

    # --------------------------------------------------------
    # Download firmware locally
    # --------------------------------------------------------
    if not os.path.exists(FIRMWARE_DIR):
        firmware_zip = os.path.join(BASE_DIR, "firmware.zip")
        download_file(FIRMWARE_URL, firmware_zip)
        extract_zip(firmware_zip, BASE_DIR)

        extracted_folder = None

        for name in os.listdir(BASE_DIR):
            full_path = os.path.join(BASE_DIR, name)
            if os.path.isdir(full_path) and name.startswith("OpenCatEsp32"):
                extracted_folder = full_path
                break

        if extracted_folder is None:
            raise RuntimeError("Could not find extracted firmware folder.")

        shutil.move(extracted_folder, FIRMWARE_DIR)
        print("Firmware installed locally:", FIRMWARE_DIR)
    else:
        print("Firmware already exists:", FIRMWARE_DIR)

    # --------------------------------------------------------
    # Install pyserial for COM port detection
    # --------------------------------------------------------
    try:
        import serial.tools.list_ports
        print("pyserial already installed.")
    except ImportError:
        print("Installing pyserial...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyserial"])

    print("\nINSTALLATION COMPLETE.")


def compile_firmware():
    """
    OPTION 2:
    Compile firmware.

    Manual equivalent:
      bittle_env\\arduino-cli.exe compile --fqbn esp32:esp32:esp32 bittle_env\\firmware --config-file bittle_env\\data\\cli.yaml
    """

    if not os.path.exists(CLI_EXE):
        raise RuntimeError("Arduino CLI not installed. Run option 1 first.")

    if not os.path.exists(FIRMWARE_DIR):
        raise RuntimeError("Firmware not found. Run option 1 first.")

    run_cmd([
        CLI_EXE,
        "compile",
        "--fqbn",
        ESP32_FQBN,
        FIRMWARE_DIR,
        "--config-file",
        CLI_CONFIG
    ])

    print("\nCOMPILE COMPLETE.")


def detect_com_port():
    try:
        import serial.tools.list_ports
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyserial"])
        import serial.tools.list_ports

    ports = list(serial.tools.list_ports.comports())

    if not ports:
        raise RuntimeError("No COM ports found. Check USB cable and ESP32 driver.")

    print("\nDetected COM ports:")
    for p in ports:
        print(f"  {p.device} - {p.description}")

    for p in ports:
        desc = p.description.lower()
        if "cp210" in desc or "ch340" in desc or "usb" in desc or "uart" in desc:
            print("Selected ESP32 port:", p.device)
            return p.device

    print("Could not identify ESP32 clearly. Using:", ports[0].device)
    return ports[0].device


def install_firmware():
    """
    OPTION 3:
    Install/upload firmware to ESP32 using USB.

    Manual equivalent:
      bittle_env\\arduino-cli.exe upload -p COM3 --fqbn esp32:esp32:esp32 bittle_env\\firmware --config-file bittle_env\\data\\cli.yaml
    """

    if not os.path.exists(CLI_EXE):
        raise RuntimeError("Arduino CLI not installed. Run option 1 first.")

    if not os.path.exists(FIRMWARE_DIR):
        raise RuntimeError("Firmware not found. Run option 1 first.")

    port = detect_com_port()

    run_cmd([
        CLI_EXE,
        "upload",
        "-p",
        port,
        "--fqbn",
        ESP32_FQBN,
        FIRMWARE_DIR,
        "--config-file",
        CLI_CONFIG
    ])

    print("\nFIRMWARE INSTALL/UPLOAD COMPLETE.")


def main():
    while True:
        print("\n======================================")
        print(" Bittle X ESP32 Firmware Tool")
        print("======================================")
        print("1 - Install all necessary packages and directories")
        print("2 - Compile firmware")
        print("3 - Install firmware to ESP32 using USB")
        print("0 - Exit")

        choice = input("Select option: ").strip()

        try:
            if choice == "1":
                install_all()
            elif choice == "2":
                compile_firmware()
            elif choice == "3":
                install_firmware()
            elif choice == "0":
                break
            else:
                print("Invalid option.")
        except Exception as e:
            print("\nERROR:")
            print(e)


if __name__ == "__main__":
    main()