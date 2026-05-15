# ============================================================
# BITTLE X / PETOI ESP32 LOCAL WINDOWS FIRMWARE TOOL
# ============================================================
#
# PURPOSE
# ------------------------------------------------------------
# This script automates:
#
#   1. Installing required Python packages automatically
#   2. Installing Arduino CLI locally
#   3. Installing ESP32 board support locally
#   4. Downloading OpenCat ESP32 firmware locally
#   5. Downloading required Arduino libraries
#   6. Compiling OpenCat ESP32 firmware
#   7. Uploading firmware to ESP32 over USB
#
# IMPORTANT FIXES INCLUDED:
# ------------------------------------------------------------
# 1. Installs MuVisionSensor3 locally into:
#
#      bittle_env/libraries/MuVisionSensor3
#
# 2. Installs WebSockets locally into:
#
#      bittle_env/libraries/WebSockets
#
#    This fixes:
#
#      fatal error: WebSocketsServer.h: No such file or directory
#
# 3. Compiles with:
#
#      --libraries bittle_env/libraries
#
# 4. Installs ArduinoJson using Arduino CLI:
#
#      arduino-cli lib install ArduinoJson
#
# 5. Uses ESP32 Huge APP partition scheme:
#
#      esp32:esp32:esp32:PartitionScheme=huge_app
#
#    This fixes:
#
#      Sketch uses 1762373 bytes (134%)
#      Error during build: text section exceeds available space in board
#
#    The default ESP32 partition allows about 1.3 MB for the app.
#    Huge APP allows about 3 MB for the app.
#
# 6. Menu input does NOT require pressing Enter.
#    You only press:
#
#      1
#      2
#      3
#      0
#
#
# ============================================================
# EXPECTED DIRECTORY HIERARCHY
# ============================================================
#
# Run this script from:
#
#   C:\Users\Public\mkhalil\AI\OpenCatEsp32
#
# After option 1 finishes:
#
#   OpenCatEsp32/
#   │
#   ├── install-compile-load.py
#   │
#   └── bittle_env/
#       │
#       ├── arduino-cli.exe
#       ├── data/
#       │   └── cli.yaml
#       │
#       ├── libraries/
#       │   ├── MuVisionSensor3/
#       │   │   └── MuVisionSensor.h
#       │   │
#       │   └── WebSockets/
#       │       └── src/
#       │           └── WebSocketsServer.h
#       │
#       └── OpenCatEsp32/
#           │
#           ├── OpenCatEsp32.ino
#           ├── src/
#           │   └── OpenCat.h
#           ├── SkillLibrary/
#           ├── ModuleTests/
#           ├── resource/
#           ├── serialMaster/
#           └── pyUI/
#
print("======================================")
print(" Bittle X ESP32 Firmware Tool")
print("======================================")
print("1 - Install all necessary packages, firmware, and libraries")
print("2 - Compile firmware only and generate .bin/.elf/.map files")
print("4 - Merge original bittle_env OpenCatEsp32.ino + current myOpenCatEsp32.ino into mergeOpenCatEsp32.ino")
print("5 - Copy mergeOpenCatEsp32.ino into bittle_env OpenCatEsp32.ino")
print("6 - Flash/upload compiled .bin files to ESP32 over USB")
print("0 - Exit")

choice = input("Select option: ").strip()
# ESP32 PLATFORM SUPPORT
# ============================================================
#
# This script currently compiles firmware for:
#
#   Classic ESP32 architecture
#
# using:
#
#   esp32:esp32
#
# and:
#
#   --chip esp32
#
# during USB flashing.
#
# ============================================================
# COMPATIBLE BOARDS
# ============================================================
#
# The generated firmware is typically compatible with:
#
#   - ESP32-WROOM-32
#   - ESP32 DevKit V1
#   - ESP32-WROVER
#   - NodeMCU-32S
#   - Petoi Bittle X ESP32 board
#   - Most generic ESP32 boards
#
# ============================================================
# NOT CURRENTLY TARGETED
# ============================================================
#
# The current compile/upload configuration does NOT specifically
# target:
#
#   - ESP32-S2
#   - ESP32-S3
#   - ESP32-C3
#   - ESP8266
#
# because those boards may use:
#
#   - Different CPU architectures
#   - Different flash layouts
#   - Different partition tables
#   - Different bootloader formats
#
# ============================================================
# GENERATED COMPILED FILES
# ============================================================
#
# The compile process generates firmware files such as:
#
#   OpenCatEsp32.ino.bin
#   OpenCatEsp32.ino.elf
#   OpenCatEsp32.ino.map
#   OpenCatEsp32.ino.bootloader.bin
#   OpenCatEsp32.ino.partitions.bin
#
# These files are copied into:
#
#   bittle_env\OpenCatEsp32
#
# after compilation.
#
# ============================================================
# DIRECT USB FLASHING
# ============================================================
#
# Option 6 uses:
#
#   esptool.py
#
# to directly flash the ESP32 over USB using:
#
#   0x1000   -> bootloader
#   0x8000   -> partition table
#   0x10000  -> application firmware
#
# ============================================================

import os
import sys
import shutil
import zipfile
import subprocess
import urllib.request
import re



# ============================================================
# SELF-INSTALL REQUIRED PYTHON PACKAGES
# ============================================================
#
# PURPOSE:
# ------------------------------------------------------------
# This section automatically installs required Python packages
# if they are missing.
#
# IMPORTANT:
# ------------------------------------------------------------
# The error:
#
#   serial.tools cannot be resolved
#
# usually means the Python package "pyserial" is missing from
# the current Python environment.
#
# The import name is:
#
#   serial
#
# But the pip package name is:
#
#   pyserial
#
# This script installs pyserial automatically and then imports:
#
#   serial.tools.list_ports
#
# when detecting the ESP32 COM port.
# ============================================================

def ensure_python_package(import_name, pip_name):
    """
    Checks whether a Python package can be imported.

    If the package is missing, the script installs it automatically
    using:

        python -m pip install <package>

    PARAMETERS:
    ------------------------------------------------------------
    import_name:
        The name used by Python import.

    pip_name:
        The name used by pip install.
    """

    try:
        __import__(import_name)
        print(f"{pip_name} already installed.")

    except ImportError:
        print(f"{pip_name} is missing.")
        print(f"Installing Python package: {pip_name}")

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            pip_name
        ])

        print(f"{pip_name} installed successfully.")


def install_required_python_packages():
    """
    Installs all Python packages required by this script.

    Current requirements:
    ------------------------------------------------------------
    pyserial:
        Required for:
            from serial.tools import list_ports

        Used by:
            detect_com_port()
    """

    ensure_python_package("serial", "pyserial")
    ensure_python_package("esptool", "esptool")


# Install required Python packages immediately when script starts.
install_required_python_packages()



# ============================================================
# BASE PATHS
# ============================================================

# ============================================================
# SCRIPT DIRECTORY
# ============================================================
#
# IMPORTANT:
# ------------------------------------------------------------
# Do NOT use only os.getcwd() here.
#
# os.getcwd() means:
#
#   "the folder where Python was launched from"
#
# That may be different from:
#
#   "the folder where this script file is stored"
#
# Example:
#   If you run:
#
#      python C:\Users\Public\mkhalil\AI\OpenCatEsp32\install-compile-load.py
#
#   from another directory, os.getcwd() may point to the other directory.
#
# Therefore, use __file__ so all paths are based on the actual
# location of this Python script.
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LAUNCH_DIR = os.getcwd()

BASE_DIR = SCRIPT_DIR

BITTLE_ENV_DIR = os.path.join(BASE_DIR, "bittle_env")
DATA_DIR = os.path.join(BITTLE_ENV_DIR, "data")

CLI_EXE = os.path.join(BITTLE_ENV_DIR, "arduino-cli.exe")
CLI_CONFIG = os.path.join(DATA_DIR, "cli.yaml")

PROJECT_DIR = os.path.join(BITTLE_ENV_DIR, "OpenCatEsp32")
INO_FILE = os.path.join(PROJECT_DIR, "OpenCatEsp32.ino")
SRC_HEADER_FILE = os.path.join(PROJECT_DIR, "src", "OpenCat.h")

LOCAL_LIBRARIES_DIR = os.path.join(BITTLE_ENV_DIR, "libraries")

# ============================================================
# DEBUG PATH PRINTING
# ============================================================
#
# PURPOSE:
# ------------------------------------------------------------
# Prints all important paths used by this script.
#
# WHY:
# ------------------------------------------------------------
# This helps you understand exactly where the script is running,
# where the firmware input files are expected, and where the
# generated mergeOpenCatEsp32.ino file is written.
# ============================================================

def print_path_debug_info():
    print("\n============================================================")
    print(" DEBUG PATH INFORMATION")
    print("============================================================")
    print("Python executable :", sys.executable)
    print("Script file       :", os.path.abspath(__file__))
    print("SCRIPT_DIR        :", SCRIPT_DIR)
    print("LAUNCH_DIR        :", LAUNCH_DIR)
    print("BASE_DIR          :", BASE_DIR)
    print("BITTLE_ENV_DIR    :", BITTLE_ENV_DIR)
    print("PROJECT_DIR       :", PROJECT_DIR)
    print("INO_FILE          :", INO_FILE)
    print("LOCAL_LIBRARIES   :", LOCAL_LIBRARIES_DIR)
    print("Original input   :", os.path.join(PROJECT_DIR, "OpenCatEsp32.ino"))
    print("Custom input     :", os.path.join(SCRIPT_DIR, "myOpenCatEsp32.ino"))
    print("Expected merge out:", os.path.join(SCRIPT_DIR, "mergeOpenCatEsp32.ino"))
    print("Backup target    :", os.path.join(PROJECT_DIR, "OpenCatEsp32.ino.org"))
    print("============================================================\n")

MU_VISION_LOCAL_DIR = os.path.join(
    LOCAL_LIBRARIES_DIR,
    "MuVisionSensor3"
)

MU_VISION_LOCAL_HEADER = os.path.join(
    MU_VISION_LOCAL_DIR,
    "MuVisionSensor.h"
)

WEBSOCKETS_LOCAL_DIR = os.path.join(
    LOCAL_LIBRARIES_DIR,
    "WebSockets"
)

WEBSOCKETS_LOCAL_HEADER_1 = os.path.join(
    WEBSOCKETS_LOCAL_DIR,
    "src",
    "WebSocketsServer.h"
)

WEBSOCKETS_LOCAL_HEADER_2 = os.path.join(
    WEBSOCKETS_LOCAL_DIR,
    "WebSocketsServer.h"
)


# ============================================================
# ESP32 CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# IMPORTANT:
# ------------------------------------------------------------
# The OpenCat ESP32 firmware with BLE, WiFi, WebSockets, and
# related modules can exceed the default ESP32 app partition.
#
# Default ESP32 app partition is around 1.3 MB.
# The error you saw was:
#
#   Sketch uses 1762373 bytes (134%) of program storage space.
#   Error during build: text section exceeds available space in board
#
# Therefore we use the Huge APP partition scheme.
# This gives approximately 3 MB for the application.
#
# Arduino CLI supports board menu options through the FQBN.
# For ESP32 this option is:
#
#   PartitionScheme=huge_app
#
# ------------------------------------------------------------
ESP32_FQBN = "esp32:esp32:esp32:PartitionScheme=huge_app"
ESP32_CORE = "esp32:esp32@2.0.12"

CLI_URL = (
    "https://downloads.arduino.cc/arduino-cli/"
    "arduino-cli_latest_Windows_64bit.zip"
)

FIRMWARE_URL = (
    "https://github.com/PetoiCamp/"
    "OpenCatEsp32-Quadruped-Robot/archive/refs/heads/main.zip"
)

MU_VISION_URLS = [
    "https://github.com/mu-opensource/MuVisionSensor3/archive/refs/heads/master.zip",
    "https://github.com/mu-opensource/MuVisionSensor3/archive/refs/heads/main.zip",
]

WEBSOCKETS_URLS = [
    "https://github.com/Links2004/arduinoWebSockets/archive/refs/heads/master.zip",
    "https://github.com/Links2004/arduinoWebSockets/archive/refs/heads/main.zip",
]

CLI_ZIP = os.path.join(BITTLE_ENV_DIR, "arduino-cli.zip")
FIRMWARE_ZIP = os.path.join(BITTLE_ENV_DIR, "OpenCatEsp32.zip")
MU_VISION_ZIP = os.path.join(BITTLE_ENV_DIR, "MuVisionSensor3.zip")
WEBSOCKETS_ZIP = os.path.join(BITTLE_ENV_DIR, "WebSockets.zip")


# ============================================================
# SINGLE-KEY MENU INPUT
# ============================================================

def get_menu_choice():
    """
    Reads one key without requiring Enter on Windows.
    Falls back to normal input() on non-Windows systems.
    """

    print("Select option: ", end="", flush=True)

    if os.name == "nt":
        import msvcrt

        key = msvcrt.getch()

        try:
            choice = key.decode("utf-8")
        except UnicodeDecodeError:
            choice = ""

        print(choice)

        return choice.strip()

    return input().strip()


# ============================================================
# RUN COMMAND WITH LIVE OUTPUT
# ============================================================

def run_cmd(cmd):
    print("\nRUNNING:")
    print(" ".join(cmd))

    print("\nOUTPUT:")
    print("--------------------------------------------------")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    print("--------------------------------------------------")

    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}"
        )


# ============================================================
# DOWNLOAD FILE
# ============================================================

def download_file(url, output_path):
    print("\nDownloading:")
    print(url)

    urllib.request.urlretrieve(url, output_path)

    print("Saved:", output_path)


# ============================================================
# DOWNLOAD FILE WITH FALLBACK URLS
# ============================================================

def download_file_with_fallback(urls, output_path):
    last_error = None

    for url in urls:
        try:
            download_file(url, output_path)
            return
        except Exception as e:
            last_error = e
            print("Download failed, trying next URL...")

    raise RuntimeError(
        f"All download attempts failed.\nLast error: {last_error}"
    )


# ============================================================
# EXTRACT ZIP
# ============================================================

def extract_zip(zip_path, output_dir):
    print("\nExtracting:")
    print(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)


# ============================================================
# FIND FILE RECURSIVELY
# ============================================================

def find_file(root_dir, filename):
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)

    return None


# ============================================================
# VERIFY FIRMWARE PROJECT
# ============================================================

def verify_firmware_project():
    if not os.path.isdir(PROJECT_DIR):
        raise RuntimeError(
            "Firmware project directory not found:\n"
            f"{PROJECT_DIR}\n\n"
            "Run option 1 to download the firmware."
        )

    if not os.path.isfile(INO_FILE):
        raise RuntimeError(
            "Firmware sketch not found:\n"
            f"{INO_FILE}\n\n"
            "Arduino requires the folder name and .ino filename to match."
        )

    if not os.path.isfile(SRC_HEADER_FILE):
        raise RuntimeError(
            "Required source header not found:\n"
            f"{SRC_HEADER_FILE}\n\n"
            "The full firmware source tree is missing.\n"
            "Delete bittle_env\\OpenCatEsp32 and rerun option 1."
        )

    print("\nFirmware sketch found successfully:")
    print("Sketch folder:", PROJECT_DIR)
    print("INO file     :", INO_FILE)
    print("Header file  :", SRC_HEADER_FILE)


# ============================================================
# VERIFY MUVISION SENSOR LIBRARY
# ============================================================

def verify_muvision_library():
    if not os.path.isfile(MU_VISION_LOCAL_HEADER):
        print("\nMuVisionSensor3 library is missing.")
        print("Trying to install it automatically...")
        install_muvision_library()

    if not os.path.isfile(MU_VISION_LOCAL_HEADER):
        raise RuntimeError(
            "MuVisionSensor library still not found:\n"
            f"{MU_VISION_LOCAL_HEADER}\n\n"
            "The compile will fail until MuVisionSensor.h exists."
        )

    print("\nMuVisionSensor3 library found:")
    print(MU_VISION_LOCAL_DIR)


# ============================================================
# VERIFY WEBSOCKETS LIBRARY
# ============================================================

def verify_websockets_library():
    """
    Verifies that WebSocketsServer.h exists locally.

    OpenCatEsp32/src/webServer.h includes:

        #include <WebSocketsServer.h>

    If Arduino CLI cannot find this header, the compile fails with:

        fatal error: WebSocketsServer.h: No such file or directory

    This function makes sure the Links2004 arduinoWebSockets
    library is installed into bittle_env/libraries/WebSockets.
    """

    if not (
        os.path.isfile(WEBSOCKETS_LOCAL_HEADER_1)
        or os.path.isfile(WEBSOCKETS_LOCAL_HEADER_2)
    ):
        print("\nWebSockets library is missing.")
        print("Trying to install it automatically...")
        install_websockets_library()

    if not (
        os.path.isfile(WEBSOCKETS_LOCAL_HEADER_1)
        or os.path.isfile(WEBSOCKETS_LOCAL_HEADER_2)
    ):
        raise RuntimeError(
            "WebSocketsServer.h still not found. Checked:\n"
            f"{WEBSOCKETS_LOCAL_HEADER_1}\n"
            f"{WEBSOCKETS_LOCAL_HEADER_2}\n\n"
            "The compile will fail until WebSocketsServer.h exists."
        )

    print("\nWebSockets library found:")
    print(WEBSOCKETS_LOCAL_DIR)


# ============================================================
# VERIFY HUGE APP PARTITION FILE
# ============================================================

def verify_huge_app_partition_file():
    """
    Checks whether the ESP32 core contains the huge_app partition file.

    The compile uses:

        PartitionScheme=huge_app

    The installed ESP32 core should include:

        tools/partitions/huge_app.csv

    If this file is missing, the ESP32 core installation is incomplete.
    """

    arduino15_dir = os.path.join(
        os.path.expanduser("~"),
        "AppData",
        "Local",
        "Arduino15"
    )

    partitions_dir = os.path.join(
        arduino15_dir,
        "packages",
        "esp32",
        "hardware",
        "esp32",
        "2.0.12",
        "tools",
        "partitions"
    )

    huge_app_file = os.path.join(partitions_dir, "huge_app.csv")

    if os.path.isfile(huge_app_file):
        print("\nHuge APP partition file found:")
        print(huge_app_file)
    else:
        print("\nWARNING:")
        print("Huge APP partition file was not found at the expected path:")
        print(huge_app_file)
        print("The compile may still work if Arduino CLI resolves it internally.")


# ============================================================
# DOWNLOAD OPEN CAT ESP32 FIRMWARE
# ============================================================

def download_firmware():
    if (
        os.path.isdir(PROJECT_DIR)
        and os.path.isfile(INO_FILE)
        and os.path.isfile(SRC_HEADER_FILE)
    ):
        print("\nFirmware already exists:")
        print(PROJECT_DIR)
        return

    if os.path.isdir(PROJECT_DIR):
        print("\nIncomplete firmware folder found. Removing:")
        print(PROJECT_DIR)
        shutil.rmtree(PROJECT_DIR)

    if not os.path.exists(FIRMWARE_ZIP):
        download_file(FIRMWARE_URL, FIRMWARE_ZIP)
    else:
        print("\nFirmware ZIP already exists:")
        print(FIRMWARE_ZIP)

    extract_zip(FIRMWARE_ZIP, BITTLE_ENV_DIR)

    extracted_folder = None

    for name in os.listdir(BITTLE_ENV_DIR):
        full_path = os.path.join(BITTLE_ENV_DIR, name)

        if (
            os.path.isdir(full_path)
            and name.startswith("OpenCatEsp32")
            and name != "OpenCatEsp32"
        ):
            extracted_folder = full_path
            break

    if extracted_folder is None:
        raise RuntimeError(
            "Firmware ZIP was extracted, but extracted OpenCat folder was not found."
        )

    os.rename(extracted_folder, PROJECT_DIR)

    print("\nFirmware downloaded successfully:")
    print(PROJECT_DIR)


# ============================================================
# INSTALL MUVISION SENSOR LIBRARY LOCALLY
# ============================================================

def install_muvision_library():
    os.makedirs(LOCAL_LIBRARIES_DIR, exist_ok=True)

    if os.path.isfile(MU_VISION_LOCAL_HEADER):
        print("\nMuVisionSensor3 library already installed:")
        print(MU_VISION_LOCAL_DIR)
        return

    if os.path.isdir(MU_VISION_LOCAL_DIR):
        print("\nIncomplete MuVisionSensor3 folder found. Removing:")
        print(MU_VISION_LOCAL_DIR)
        shutil.rmtree(MU_VISION_LOCAL_DIR)

    temp_extract_dir = os.path.join(BITTLE_ENV_DIR, "muvision_temp")

    if os.path.isdir(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)

    os.makedirs(temp_extract_dir, exist_ok=True)

    if os.path.exists(MU_VISION_ZIP):
        print("\nRemoving old MuVisionSensor3 ZIP:")
        print(MU_VISION_ZIP)
        os.remove(MU_VISION_ZIP)

    download_file_with_fallback(MU_VISION_URLS, MU_VISION_ZIP)

    extract_zip(MU_VISION_ZIP, temp_extract_dir)

    header_path = find_file(temp_extract_dir, "MuVisionSensor.h")

    if header_path is None:
        raise RuntimeError(
            "MuVisionSensor.h was not found inside the downloaded ZIP."
        )

    library_root = os.path.dirname(header_path)

    print("\nDetected MuVisionSensor library root:")
    print(library_root)

    shutil.copytree(library_root, MU_VISION_LOCAL_DIR)

    shutil.rmtree(temp_extract_dir)

    print("\nMuVisionSensor3 library installed locally:")
    print(MU_VISION_LOCAL_DIR)


# ============================================================
# INSTALL WEBSOCKETS LIBRARY LOCALLY
# ============================================================

def install_websockets_library():
    """
    Installs the Links2004 arduinoWebSockets library locally.

    The firmware includes:

        #include <WebSocketsServer.h>

    The required library is:

        https://github.com/Links2004/arduinoWebSockets

    It is installed locally into:

        bittle_env/libraries/WebSockets

    This keeps the build self-contained and avoids depending on
    whatever libraries happen to exist in the user's Documents/Arduino
    library folder.
    """

    os.makedirs(LOCAL_LIBRARIES_DIR, exist_ok=True)

    if (
        os.path.isfile(WEBSOCKETS_LOCAL_HEADER_1)
        or os.path.isfile(WEBSOCKETS_LOCAL_HEADER_2)
    ):
        print("\nWebSockets library already installed:")
        print(WEBSOCKETS_LOCAL_DIR)
        return

    if os.path.isdir(WEBSOCKETS_LOCAL_DIR):
        print("\nIncomplete WebSockets folder found. Removing:")
        print(WEBSOCKETS_LOCAL_DIR)
        shutil.rmtree(WEBSOCKETS_LOCAL_DIR)

    temp_extract_dir = os.path.join(BITTLE_ENV_DIR, "websockets_temp")

    if os.path.isdir(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)

    os.makedirs(temp_extract_dir, exist_ok=True)

    if os.path.exists(WEBSOCKETS_ZIP):
        print("\nRemoving old WebSockets ZIP:")
        print(WEBSOCKETS_ZIP)
        os.remove(WEBSOCKETS_ZIP)

    download_file_with_fallback(WEBSOCKETS_URLS, WEBSOCKETS_ZIP)

    extract_zip(WEBSOCKETS_ZIP, temp_extract_dir)

    header_path = find_file(temp_extract_dir, "WebSocketsServer.h")

    if header_path is None:
        raise RuntimeError(
            "WebSocketsServer.h was not found inside the downloaded ZIP."
        )

    # --------------------------------------------------------
    # For arduinoWebSockets, the header is normally under src/.
    # The library root is therefore the folder above src.
    # Example:
    #
    #   arduinoWebSockets-master/src/WebSocketsServer.h
    #
    # library_root should be:
    #
    #   arduinoWebSockets-master
    # --------------------------------------------------------

    header_dir = os.path.dirname(header_path)

    if os.path.basename(header_dir).lower() == "src":
        library_root = os.path.dirname(header_dir)
    else:
        library_root = header_dir

    print("\nDetected WebSockets library root:")
    print(library_root)

    shutil.copytree(library_root, WEBSOCKETS_LOCAL_DIR)

    shutil.rmtree(temp_extract_dir)

    print("\nWebSockets library installed locally:")
    print(WEBSOCKETS_LOCAL_DIR)


# ============================================================
# INSTALL REQUIRED ARDUINO LIBRARIES
# ============================================================

def install_arduino_libraries():
    """
    Installs Arduino libraries required by OpenCat ESP32 firmware.

    ArduinoJson is needed because:
        Seeed_Arduino_SSCMA.h includes <ArduinoJson.h>

    WebSockets is installed manually by install_websockets_library()
    because the firmware specifically needs:
        WebSocketsServer.h
    """

    run_cmd([
        CLI_EXE,
        "lib",
        "install",
        "ArduinoJson",
        "--config-file",
        CLI_CONFIG
    ])


# ============================================================
# OPTION 1
# INSTALL ALL TOOLS, FIRMWARE, AND LIBRARIES
# ============================================================

def install_all():
    os.makedirs(BITTLE_ENV_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOCAL_LIBRARIES_DIR, exist_ok=True)

    ensure_python_package("serial", "pyserial")

    # --------------------------------------------------------
    # Install Arduino CLI locally
    # --------------------------------------------------------

    if not os.path.exists(CLI_EXE):
        download_file(CLI_URL, CLI_ZIP)
        extract_zip(CLI_ZIP, BITTLE_ENV_DIR)
    else:
        print("\nArduino CLI already installed:")
        print(CLI_EXE)

    # --------------------------------------------------------
    # Create local Arduino configuration
    # --------------------------------------------------------

    run_cmd([
        CLI_EXE,
        "config",
        "init",
        "--config-file",
        CLI_CONFIG
    ])

    # --------------------------------------------------------
    # Update board index
    # --------------------------------------------------------

    run_cmd([
        CLI_EXE,
        "core",
        "update-index",
        "--config-file",
        CLI_CONFIG
    ])

    # --------------------------------------------------------
    # Install ESP32 board support
    # --------------------------------------------------------

    run_cmd([
        CLI_EXE,
        "core",
        "install",
        ESP32_CORE,
        "--config-file",
        CLI_CONFIG
    ])

    # --------------------------------------------------------
    # Install Arduino libraries managed by Arduino CLI
    # --------------------------------------------------------

    install_arduino_libraries()

    # --------------------------------------------------------
    # Download OpenCat ESP32 firmware
    # --------------------------------------------------------

    download_firmware()

    # --------------------------------------------------------
    # Download required MuVisionSensor3 library
    # --------------------------------------------------------

    install_muvision_library()

    # --------------------------------------------------------
    # Download required WebSockets library
    # --------------------------------------------------------

    install_websockets_library()

    # --------------------------------------------------------
    # Check Huge APP partition support
    # --------------------------------------------------------

    verify_huge_app_partition_file()

    print("\nINSTALLATION COMPLETE.")


# ============================================================
# OPTION 2
# COMPILE FIRMWARE
#
# OPTION 2 ONLY COMPILES:
# ------------------------------------------------------------
# This option builds OpenCatEsp32.ino and generates output files
# such as .bin, .elf, and .map. It does NOT flash the ESP32.
# Use option 6 if you want to upload the compiled .bin files.
# ============================================================

def compile_firmware():
    if not os.path.exists(CLI_EXE):
        print("\nArduino CLI is missing. Installing environment first...")
        install_all()

    verify_firmware_project()
    verify_muvision_library()
    verify_websockets_library()
    verify_huge_app_partition_file()

    print("\nUsing ESP32 board configuration:")
    print(ESP32_FQBN)
    print("\nThis uses the Huge APP partition scheme to avoid sketch-too-big errors.")

    run_cmd([
        CLI_EXE,
        "compile",
        "--verbose",
        "--fqbn",
        ESP32_FQBN,
        PROJECT_DIR,
        "--output-dir",
        PROJECT_DIR,
        "--libraries",
        LOCAL_LIBRARIES_DIR,
        "--config-file",
        CLI_CONFIG
    ])

    print("\nCOMPILE COMPLETE.")


# ============================================================
# DETECT COM PORT
# ============================================================

def detect_com_port():
    """
    Detects the ESP32 serial COM port.

    IMPORTANT:
    ------------------------------------------------------------
    This function requires pyserial.

    pyserial provides:

        serial.tools.list_ports

    If pyserial is missing, this function installs it automatically
    before trying again.
    """

    try:
        from serial.tools import list_ports

    except ImportError:
        print("\nserial.tools could not be imported.")
        print("Installing pyserial automatically...")
        ensure_python_package("serial", "pyserial")

        from serial.tools import list_ports

    ports = list(list_ports.comports())

    if not ports:
        raise RuntimeError(
            "No COM ports detected.\n"
            "Check USB cable, ESP32 power, and USB-to-UART driver."
        )

    print("\nDetected COM Ports:")

    for p in ports:
        print(f"  {p.device} - {p.description}")

    # --------------------------------------------------------
    # Prefer common ESP32 USB-to-serial adapters.
    # --------------------------------------------------------

    for p in ports:
        desc = p.description.lower()

        if (
            "cp210" in desc or
            "ch340" in desc or
            "usb" in desc or
            "uart" in desc or
            "silicon labs" in desc
        ):
            print("\nSelected ESP32 Port:", p.device)
            return p.device

    print("\nUsing first detected port:", ports[0].device)

    return ports[0].device


# ============================================================
# OPTION 3
# UPLOAD FIRMWARE
# ============================================================

def install_firmware():
    if not os.path.exists(CLI_EXE):
        print("\nArduino CLI is missing. Installing environment first...")
        install_all()

    verify_firmware_project()
    verify_muvision_library()
    verify_websockets_library()
    verify_huge_app_partition_file()

    port = detect_com_port()

    print("\nUsing ESP32 board configuration for upload:")
    print(ESP32_FQBN)

    run_cmd([
        CLI_EXE,
        "upload",
        "-p",
        port,
        "--fqbn",
        ESP32_FQBN,
        PROJECT_DIR,
        "--libraries",
        LOCAL_LIBRARIES_DIR,
        "--config-file",
        CLI_CONFIG
    ])

    print("\nFIRMWARE INSTALL/UPLOAD COMPLETE.")






# ============================================================
# CLEAN PREVIOUS MERGED CUSTOM SECTION
# ============================================================

def remove_previous_merged_custom_section(text):
    """
    PURPOSE:
    ------------------------------------------------------------
    Removes a previously appended custom merge section from
    OpenCatEsp32.ino before creating a new merged file.

    WHY:
    ------------------------------------------------------------
    If option 5 already copied mergeOpenCatEsp32.ino into:

        bittle_env\\OpenCatEsp32\\OpenCatEsp32.ino

    and then option 4 is run again, the original firmware file
    already contains the custom functions.

    Without this cleanup, option 4 appends the custom functions
    again, causing compile errors such as:

        redefinition of 'void my_custom_walk()'
        redefinition of 'void mysetupWiFiControl()'
        redefinition of 'void myloopWiFiCommand()'

    WHAT IT REMOVES:
    ------------------------------------------------------------
    Everything starting from the marker:

        // MERGED CUSTOM FUNCTIONS FROM myOpenCatEsp32.ino

    to the end of the file.

    WHAT IT KEEPS:
    ------------------------------------------------------------
    All original OpenCat code and comments before the merged
    section marker.
    """

    marker = "// MERGED CUSTOM FUNCTIONS FROM myOpenCatEsp32.ino"

    marker_index = text.find(marker)

    if marker_index == -1:
        return text

    # --------------------------------------------------------
    # Move back to the section banner before the marker so the
    # whole merged section header is removed cleanly.
    # --------------------------------------------------------

    banner = "// ============================================================"
    banner_index = text.rfind(banner, 0, marker_index)

    if banner_index == -1:
        cleanup_index = marker_index
    else:
        cleanup_index = banner_index

    print("\nPrevious merged custom section found.")
    print("Removing old merged custom section before creating a new merge.")

    return text[:cleanup_index].rstrip() + "\n"


# ============================================================
# REMOVE PREVIOUS CUSTOM SETUP/LOOP CALLS
# ============================================================

def remove_previous_custom_call_injections(text):
    """
    PURPOSE:
    ------------------------------------------------------------
    Removes previously injected custom calls from the original
    OpenCat setup() and loop() before injecting fresh calls.

    WHY:
    ------------------------------------------------------------
    If option 4 is run repeatedly, we do not want multiple copies
    of:

        mysetupWiFiControl();
        myloopWiFiCommand();

    inside setup() or loop().
    """

    text = text.replace(
        "\n\n"
        "  // Custom merged setup\n"
        "  // Starts Wi-Fi TCP command server.\n"
        "  mysetupWiFiControl();",
        ""
    )

    text = text.replace(
        "\n\n"
        "  // Custom merged setup\n"
        "  mysetupWiFiControl();",
        ""
    )

    text = text.replace(
        "\n\n"
        "  // Custom merged loop\n"
        "  // Checks for incoming Wi-Fi TCP commands.\n"
        "  myloopWiFiCommand();",
        ""
    )

    text = text.replace(
        "\n\n"
        "  // Custom merged loop\n"
        "  myloopWiFiCommand();",
        ""
    )

    return text




# ============================================================
# NORMALIZE CUSTOM FIRMWARE BEFORE MERGE
# ============================================================

def normalize_custom_firmware_text(text):
    """
    PURPOSE:
    ------------------------------------------------------------
    Normalize the custom myOpenCatEsp32.ino text before it is
    appended to the original OpenCat firmware.

    SERVO API FIX:
    ------------------------------------------------------------
    OpenCat ESP32 defines:

        setServoP(unsigned int p)

    That function accepts only ONE argument. It cannot be called
    as:

        setServoP(servoIndex, angle)

    Therefore this function replaces the custom writeServoAngle()
    wrapper with a correct OpenCat-compatible implementation:

        ESP_PWM:
            servo[servoIndex].write(angle)

        PCA9685/BiBoard2:
            pwm.writeAngle(servoIndex, angle)

    ORIGINAL CODE SAFETY:
    ------------------------------------------------------------
    This function is applied ONLY to myOpenCatEsp32.ino text.
    It does NOT rename or modify original OpenCat functions.
    """

    replacement_function = """void writeServoAngle(int servoIndex, int angle) {
  // ----------------------------------------------------------
  // OpenCat-compatible servo angle writer
  //
  // IMPORTANT:
  // Do NOT use:
  //
  //   setServoP(servoIndex, angle);
  //
  // because OpenCat ESP32 defines setServoP() as:
  //
  //   setServoP(unsigned int p)
  //
  // and it accepts only ONE argument.
  //
  // For ESP PWM boards, OpenCat exposes:
  //
  //   servo[index].write(angle)
  //
  // For PCA9685 boards, OpenCat exposes:
  //
  //   pwm.writeAngle(index, angle)
  //
  // ----------------------------------------------------------

  if (servoIndex < 0 || servoIndex >= PWM_NUM) {
    return;
  }

#ifdef ESP_PWM
  servo[servoIndex].write(angle);
#else
  pwm.writeAngle(servoIndex, angle);
#endif
}"""

    text = re.sub(
        r"void\s+writeServoAngle\s*\(\s*int\s+servoIndex\s*,\s*int\s+angle\s*\)\s*\{.*?\n\}",
        replacement_function,
        text,
        flags=re.DOTALL
    )

    text = text.replace(
        "setServo(servoIndex, angle);",
        "writeServoAngle(servoIndex, angle);"
    )

    text = text.replace(
        "setServoP(servoIndex, angle);",
        "writeServoAngle(servoIndex, angle);"
    )

    text = text.replace(
        "void startWiFiControl()",
        "void mysetupWiFiControl()"
    )

    text = text.replace(
        "startWiFiControl();",
        "mysetupWiFiControl();"
    )

    text = text.replace(
        "void handleWiFiCommand()",
        "void myloopWiFiCommand()"
    )

    text = text.replace(
        "handleWiFiCommand();",
        "myloopWiFiCommand();"
    )

    text = text.replace(
        "void stand_neutral_custom()",
        "void my_stand_neutral_custom()"
    )

    text = text.replace(
        "stand_neutral_custom();",
        "my_stand_neutral_custom();"
    )

    text = text.replace(
        "void stand_neutral()",
        "void my_stand_neutral()"
    )

    text = text.replace(
        "stand_neutral();",
        "my_stand_neutral();"
    )

    text = text.replace(
        "my_my_stand_neutral_custom",
        "my_stand_neutral_custom"
    )

    text = text.replace(
        "my_my_stand_neutral",
        "my_stand_neutral"
    )

    return text


# ============================================================
# OPTION 4
# MERGE ORIGINAL OpenCatEsp32.ino WITH CUSTOM myOpenCatEsp32.ino
# ============================================================

def merge_firmware_files():
    """
    PURPOSE
    ------------------------------------------------------------
    Merge TWO firmware files:

        1. Original OpenCat firmware:
             bittle_env\\OpenCatEsp32\\OpenCatEsp32.ino

        2. Custom firmware:
             myOpenCatEsp32.ino

    and create ONE output file in the current script directory:

        mergeOpenCatEsp32.ino

    IMPORTANT:
    ------------------------------------------------------------
    This function keeps comments from BOTH files and normalizes custom function names.

    It also keeps:

        #include <WiFi.h>

    from myOpenCatEsp32.ino because the custom firmware uses:

        WiFiServer
        WiFiClient
        WiFi.begin()

    The previous version removed #include <WiFi.h>, which caused
    the merged firmware to miss the Wi-Fi dependency.

    DUPLICATE PROTECTION:
    ------------------------------------------------------------
    If option 4 was already run before, and option 5 copied the
    merged file back into:

        bittle_env\\OpenCatEsp32\\OpenCatEsp32.ino

    then the original file may already contain a previous merged
    custom section.

    This routine removes the old merged custom section first, then
    creates a clean new merge.
    """

    print_path_debug_info()

    # --------------------------------------------------------
    # Original OpenCat firmware input file
    #
    # This file comes from bittle_env.
    # --------------------------------------------------------

    original_ino = os.path.join(
        PROJECT_DIR,
        "OpenCatEsp32.ino"
    )

    # --------------------------------------------------------
    # Custom firmware input file
    #
    # This file comes from the same directory as this script.
    # --------------------------------------------------------

    custom_ino = os.path.join(
        SCRIPT_DIR,
        "myOpenCatEsp32.ino"
    )

    # --------------------------------------------------------
    # Output merged firmware file
    #
    # This file is created in the same directory as this script.
    # --------------------------------------------------------

    merged_ino = os.path.join(
        SCRIPT_DIR,
        "mergeOpenCatEsp32.ino"
    )

    print("\n============================================================")
    print(" OPTION 4 - MERGE FIRMWARE FILES")
    print("============================================================")
    print("Original input :", os.path.abspath(original_ino))
    print("Custom input   :", os.path.abspath(custom_ino))
    print("Merged output  :", os.path.abspath(merged_ino))
    print("============================================================")

    # --------------------------------------------------------
    # Verify original firmware exists
    # --------------------------------------------------------

    if not os.path.isfile(original_ino):
        raise RuntimeError(
            "Original INO not found:\n"
            f"{os.path.abspath(original_ino)}\n\n"
            "Expected original firmware location:\n"
            "  bittle_env\\OpenCatEsp32\\OpenCatEsp32.ino\n\n"
            "Run option 1 first, or make sure bittle_env exists."
        )

    # --------------------------------------------------------
    # Verify custom firmware exists
    # --------------------------------------------------------

    if not os.path.isfile(custom_ino):
        raise RuntimeError(
            "Custom INO not found:\n"
            f"{os.path.abspath(custom_ino)}\n\n"
            "Expected custom firmware location:\n"
            "  myOpenCatEsp32.ino\n\n"
            "This file must be in the SAME directory as this Python script."
        )

    # --------------------------------------------------------
    # Read original firmware from bittle_env
    # --------------------------------------------------------

    print("\nReading original firmware from bittle_env...")

    with open(original_ino, "r", encoding="utf-8", errors="replace") as f:
        original_text = f.read()

    print("Original firmware size:", len(original_text), "characters")

    # --------------------------------------------------------
    # Remove old merged custom section if present.
    #
    # This prevents duplicate functions when option 4 is run
    # repeatedly.
    # --------------------------------------------------------

    original_text = remove_previous_merged_custom_section(
        original_text
    )

    original_text = remove_previous_custom_call_injections(
        original_text
    )

    # --------------------------------------------------------
    # Read custom firmware from script directory
    # --------------------------------------------------------

    print("\nReading custom firmware from script directory...")

    with open(custom_ino, "r", encoding="utf-8", errors="replace") as f:
        custom_text = f.read()

    print("Custom firmware size:", len(custom_text), "characters")

    # --------------------------------------------------------
    # Normalize custom function names before merge.
    #
    # This automatically applies:
    #   setServo                -> setServoP
    #   startWiFiControl        -> mysetupWiFiControl
    #   handleWiFiCommand       -> myloopWiFiCommand
    #   stand_neutral_custom    -> my_stand_neutral_custom
    # --------------------------------------------------------

    custom_text = normalize_custom_firmware_text(
        custom_text
    )

    # --------------------------------------------------------
    # Keep custom file content and comments.
    #
    # IMPORTANT:
    # Do NOT remove #include <WiFi.h>.
    #
    # The custom firmware owns the Wi-Fi dependency.
    # --------------------------------------------------------

    filtered_custom_text = custom_text

    # --------------------------------------------------------
    # Remove setup() and loop() wrappers from custom code.
    #
    # WHY:
    # The original OpenCatEsp32.ino already has setup() and loop().
    # Keeping setup()/loop() from the custom file would create
    # duplicate Arduino functions.
    #
    # IMPORTANT:
    # This keeps the custom helper functions, custom comments,
    # include lines such as #include <WiFi.h>, and section banners.
    # --------------------------------------------------------

    filtered_custom_text = remove_setup_and_loop_wrappers(
        filtered_custom_text
    )

    # --------------------------------------------------------
    # Inject custom setup call into original setup().
    # --------------------------------------------------------

    if "mysetupWiFiControl();" not in original_text:
        if "initRobot();" in original_text:
            original_text = original_text.replace(
                "initRobot();",
                "initRobot();\n\n"
                "  // Custom merged setup\n"
                "  // Starts Wi-Fi TCP command server.\n"
                "  mysetupWiFiControl();",
                1
            )

            print("\nInjected mysetupWiFiControl() after initRobot().")
        else:
            print("\nWARNING:")
            print("Could not find initRobot();")
            print("mysetupWiFiControl() was not automatically injected.")
    else:
        print("\nmysetupWiFiControl() already exists in original file.")

    # --------------------------------------------------------
    # Inject custom loop call into original loop().
    # --------------------------------------------------------

    if "myloopWiFiCommand();" not in original_text:
        if "reaction();" in original_text:
            original_text = original_text.replace(
                "reaction();",
                "reaction();\n\n"
                "  // Custom merged loop\n"
                "  // Checks for incoming Wi-Fi TCP commands.\n"
                "  myloopWiFiCommand();",
                1
            )

            print("Injected myloopWiFiCommand() after reaction().")
        else:
            print("\nWARNING:")
            print("Could not find reaction();")
            print("myloopWiFiCommand() was not automatically injected.")
    else:
        print("myloopWiFiCommand() already exists in original file.")

    # --------------------------------------------------------
    # Merge while preserving comments and formatting.
    #
    # The original OpenCat firmware stays first.
    # The custom firmware content is appended afterward.
    # --------------------------------------------------------

    merged_text = (
        original_text.rstrip()
        + "\n\n"
        + "// ============================================================\n"
        + "// MERGED CUSTOM FUNCTIONS FROM myOpenCatEsp32.ino\n"
        + "// ============================================================\n\n"
        + filtered_custom_text.strip()
        + "\n"
    )

    # --------------------------------------------------------
    # Ensure WiFi include exists in final merged file.
    #
    # Even though we preserve the custom include, this additional
    # safety check guarantees the final merged file has WiFi.h if
    # the custom code uses WiFi classes.
    # --------------------------------------------------------

    if (
        "#include <WiFi.h>" not in merged_text
        and (
            "WiFiServer" in merged_text
            or "WiFiClient" in merged_text
            or "WiFi.begin" in merged_text
        )
    ):
        if '#include "src/OpenCat.h"' in merged_text:
            merged_text = merged_text.replace(
                '#include "src/OpenCat.h"',
                '#include <WiFi.h>\n#include "src/OpenCat.h"',
                1
            )
        else:
            merged_text = '#include <WiFi.h>\n' + merged_text

        print("\nAdded missing #include <WiFi.h> to merged output.")

    # --------------------------------------------------------
    # Write ONE merged firmware file.
    # --------------------------------------------------------

    print("\nWriting merged firmware...")
    print("Output path:", os.path.abspath(merged_ino))

    with open(merged_ino, "w", encoding="utf-8", newline="\n") as f:
        f.write(merged_text)

    # --------------------------------------------------------
    # Confirm output file exists.
    # --------------------------------------------------------

    print("\n============================================================")
    print(" MERGE COMPLETE")
    print("============================================================")

    if os.path.isfile(merged_ino):
        print("SUCCESS:")
        print(os.path.abspath(merged_ino))
        print("File size:", os.path.getsize(merged_ino), "bytes")
    else:
        raise RuntimeError(
            "Merge appeared to finish, but output file was not found:\n"
            f"{os.path.abspath(merged_ino)}"
        )

    print("============================================================")


# ============================================================
# REMOVE setup() AND loop() WRAPPERS
# ============================================================

def remove_setup_and_loop_wrappers(text):
    """
    Removes custom setup() and loop() blocks from myOpenCatEsp32.ino
    before appending the custom code to the original OpenCat firmware.

    This prevents duplicate Arduino setup()/loop() definitions while
    preserving all helper functions and comments.
    """

    text = remove_named_void_function(text, "setup")
    text = remove_named_void_function(text, "loop")

    return text


# ============================================================
# REMOVE A NAMED void FUNCTION BLOCK
# ============================================================

def remove_named_void_function(text, function_name):
    """
    Removes one top-level C/C++ function block such as:

        void setup() { ... }

    or:

        void loop() { ... }

    The rest of the file, including comments, remains unchanged.
    """

    signature = f"void {function_name}()"

    while True:
        idx = text.find(signature)

        if idx == -1:
            return text

        brace_start = text.find("{", idx)

        if brace_start == -1:
            return text

        depth = 0
        i = brace_start

        while i < len(text):
            char = text[i]

            if char == "{":
                depth += 1

            elif char == "}":
                depth -= 1

                if depth == 0:
                    block_end = i + 1

                    while (
                        block_end < len(text)
                        and text[block_end] in " \t\r\n"
                    ):
                        block_end += 1

                    removed_block = text[idx:block_end]

                    print(
                        f"\nRemoved duplicate custom {function_name}() block "
                        f"from appended custom code."
                    )
                    print(
                        f"Removed {function_name}() block size:",
                        len(removed_block),
                        "characters"
                    )

                    return text[:idx] + text[block_end:]

            i += 1

        return text


# ============================================================
# OPTION 5
# COPY MERGED FILE INTO bittle_env OpenCatEsp32.ino
# ============================================================

def install_merged_firmware_into_bittle_env():
    """
    PURPOSE:
    ------------------------------------------------------------
    Copy the generated merged firmware file:

        mergeOpenCatEsp32.ino

    from the current script directory into the OpenCat firmware
    folder inside bittle_env as:

        OpenCatEsp32.ino

    BACKUP RULE:
    ------------------------------------------------------------
    Before replacing the current bittle_env OpenCatEsp32.ino,
    the script creates a backup named:

        OpenCatEsp32.ino.org

    IMPORTANT:
    ------------------------------------------------------------
    If OpenCatEsp32.ino.org already exists, the script leaves it
    unchanged. This protects the first/original backup from being
    overwritten by later runs.
    """

    # --------------------------------------------------------
    # Source merged firmware file
    # --------------------------------------------------------

    merged_ino = os.path.join(
        SCRIPT_DIR,
        "mergeOpenCatEsp32.ino"
    )

    # --------------------------------------------------------
    # Target OpenCat firmware file inside bittle_env
    # --------------------------------------------------------

    target_ino = os.path.join(
        PROJECT_DIR,
        "OpenCatEsp32.ino"
    )

    # --------------------------------------------------------
    # Backup file for original OpenCat firmware
    # --------------------------------------------------------

    backup_ino = os.path.join(
        PROJECT_DIR,
        "OpenCatEsp32.ino.org"
    )

    print("\n============================================================")
    print(" OPTION 5 - INSTALL MERGED FIRMWARE INTO bittle_env")
    print("============================================================")
    print("Merged source :", os.path.abspath(merged_ino))
    print("Target file   :", os.path.abspath(target_ino))
    print("Backup file   :", os.path.abspath(backup_ino))
    print("============================================================")

    # --------------------------------------------------------
    # Verify merged file exists
    # --------------------------------------------------------

    if not os.path.isfile(merged_ino):
        raise RuntimeError(
            "Merged firmware file not found:\n"
            f"{os.path.abspath(merged_ino)}\n\n"
            "Run option 4 first to create mergeOpenCatEsp32.ino."
        )

    # --------------------------------------------------------
    # Verify target OpenCat firmware folder exists
    # --------------------------------------------------------

    if not os.path.isdir(PROJECT_DIR):
        raise RuntimeError(
            "OpenCat firmware folder not found:\n"
            f"{os.path.abspath(PROJECT_DIR)}\n\n"
            "Run option 1 first, or make sure bittle_env\\OpenCatEsp32 exists."
        )

    # --------------------------------------------------------
    # Verify current target OpenCatEsp32.ino exists
    # --------------------------------------------------------

    if not os.path.isfile(target_ino):
        raise RuntimeError(
            "Target OpenCatEsp32.ino not found:\n"
            f"{os.path.abspath(target_ino)}"
        )

    # --------------------------------------------------------
    # Create backup only if it does not already exist
    # --------------------------------------------------------

    if os.path.isfile(backup_ino):
        print("\nBackup already exists.")
        print("Keeping existing backup unchanged:")
        print(os.path.abspath(backup_ino))

    else:
        print("\nCreating backup of current OpenCatEsp32.ino...")
        shutil.copy2(target_ino, backup_ino)

        print("Backup created:")
        print(os.path.abspath(backup_ino))

    # --------------------------------------------------------
    # Replace current OpenCatEsp32.ino with mergeOpenCatEsp32.ino
    # --------------------------------------------------------

    print("\nReplacing bittle_env OpenCatEsp32.ino with merged firmware...")

    shutil.copy2(merged_ino, target_ino)

    # --------------------------------------------------------
    # Verify replacement
    # --------------------------------------------------------

    print("\n============================================================")
    print(" INSTALL MERGED FIRMWARE COMPLETE")
    print("============================================================")

    if os.path.isfile(target_ino):
        print("SUCCESS:")
        print(os.path.abspath(target_ino))
        print("New file size:", os.path.getsize(target_ino), "bytes")

    else:
        raise RuntimeError(
            "Replacement failed. Target file not found:\n"
            f"{os.path.abspath(target_ino)}"
        )

    print("============================================================")





# ============================================================
# OPTION 6
# DIRECTLY UPLOAD COMPILED EXECUTABLE/BINARY TO ESP32 OVER USB
#
# OPTION 6 ONLY FLASHES:
# ------------------------------------------------------------
# This option does NOT compile. It assumes option 2 already
# generated the .bin files, then it uploads those files to ESP32
# over USB using esptool.
# ============================================================

def upload_compiled_binary_to_esp32():
    """
    PURPOSE:
    ------------------------------------------------------------
    Upload the already-compiled ESP32 firmware binary files from:

        bittle_env\\OpenCatEsp32

    directly into the ESP32 board over USB.

    WHEN TO USE THIS OPTION:
    ------------------------------------------------------------
    Use option 6 after option 2 has successfully compiled the
    firmware and generated/copied these files into bittle_env:

        OpenCatEsp32.ino.bootloader.bin
        OpenCatEsp32.ino.partitions.bin
        OpenCatEsp32.ino.bin

    DIFFERENCE BETWEEN OPTION 3 AND OPTION 6:
    ------------------------------------------------------------
    Option 3:
        Uses arduino-cli upload. It depends on the Arduino build
        workflow.

    Option 6:
        Uses esptool directly to flash the compiled binary files
        over USB.

    ESP32 FLASH OFFSETS:
    ------------------------------------------------------------
    Typical Arduino ESP32 flash layout:

        0x1000   bootloader
        0x8000   partition table
        0x10000  application firmware

    IMPORTANT:
    ------------------------------------------------------------
    If your ESP32 partition layout changes, these offsets may need
    adjustment. For the standard Arduino ESP32 workflow, these
    offsets are normally correct.
    """

    # --------------------------------------------------------
    # Make sure esptool and pyserial are available.
    # --------------------------------------------------------

    ensure_python_package("serial", "pyserial")
    ensure_python_package("esptool", "esptool")

    # --------------------------------------------------------
    # Detect ESP32 USB COM port.
    # --------------------------------------------------------

    port = detect_com_port()

    # --------------------------------------------------------
    # Compiled firmware artifacts expected in bittle_env project.
    # --------------------------------------------------------

    bootloader_bin = os.path.join(
        PROJECT_DIR,
        "OpenCatEsp32.ino.bootloader.bin"
    )

    partitions_bin = os.path.join(
        PROJECT_DIR,
        "OpenCatEsp32.ino.partitions.bin"
    )

    app_bin = os.path.join(
        PROJECT_DIR,
        "OpenCatEsp32.ino.bin"
    )

    print("\n============================================================")
    print(" OPTION 6 - DIRECT USB FLASH TO ESP32")
    print("============================================================")
    print("Selected port :", port)
    print("Bootloader    :", os.path.abspath(bootloader_bin))
    print("Partitions    :", os.path.abspath(partitions_bin))
    print("Application   :", os.path.abspath(app_bin))
    print("============================================================")

    # --------------------------------------------------------
    # Verify compiled files exist before flashing.
    # --------------------------------------------------------

    missing_files = []

    for file_path in [bootloader_bin, partitions_bin, app_bin]:

        if not os.path.isfile(file_path):
            missing_files.append(file_path)

    if missing_files:

        print("\nMissing compiled firmware files:")

        for file_path in missing_files:
            print("  -", os.path.abspath(file_path))

        raise RuntimeError(
            "Compiled firmware files are missing.\n\n"
            "Run option 2 first to compile the firmware and generate "
            "the binary files inside bittle_env\\OpenCatEsp32."
        )

    # --------------------------------------------------------
    # Flash command using esptool.
    #
    # This uses python -m esptool so it uses the same Python
    # environment that is running this script.
    # --------------------------------------------------------

    flash_cmd = [
        sys.executable,
        "-m",
        "esptool",
        "--chip",
        "esp32",
        "--port",
        port,
        "--baud",
        "921600",
        "--before",
        "default_reset",
        "--after",
        "hard_reset",
        "write_flash",
        "-z",
        "--flash_mode",
        "dio",
        "--flash_freq",
        "40m",
        "--flash_size",
        "detect",
        "0x1000",
        bootloader_bin,
        "0x8000",
        partitions_bin,
        "0x10000",
        app_bin
    ]

    print("\nRunning ESP32 flash command:")
    print(" ".join(flash_cmd))

    run_cmd(flash_cmd)

    print("\n============================================================")
    print(" ESP32 USB FLASH COMPLETE")
    print("============================================================")
    print("Firmware uploaded successfully to ESP32 over USB.")
    print("============================================================")



# ============================================================
# MAIN MENU
# ============================================================

def main():
    while True:
        print("\n======================================")
        print(" Bittle X ESP32 Firmware Tool")
        print("======================================")
        print("1 - Install all necessary packages, firmware, and libraries")
        print("2 - Compile firmware only and generate .bin/.elf/.map files")
        print("4 - Merge original bittle_env OpenCatEsp32.ino + current myOpenCatEsp32.ino into mergeOpenCatEsp32.ino")
        print("5 - Copy mergeOpenCatEsp32.ino into bittle_env OpenCatEsp32.ino")
        print("6 - Flash/upload compiled .bin files to ESP32 over USB")
        print("0 - Exit")

        choice = get_menu_choice()

        try:
            if choice == "1":
                install_all()

            elif choice == "2":
                compile_firmware()

            elif choice == "4":
                merge_firmware_files()

            elif choice == "5":
                install_merged_firmware_into_bittle_env()

            elif choice == "6":
                upload_compiled_binary_to_esp32()

            elif choice == "0":
                print("\nExiting.")
                break

            else:
                print("\nInvalid option.")

        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")

        except Exception as e:
            print("\nERROR:")
            print(str(e))
if __name__ == "__main__":
    main()