from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

# ============================================================
# ESP32 Generic Firmware Tool + Python Serial Status Communication
# NOTE: Python library install and standalone-example creation menu options were removed by request.
# ============================================================
# This script helps you:
#   1) Show paths.
#   2) Create a fresh merge_esp32_generic.ino every time.
#      The merge is always built from the ORIGINAL esp32_generic.ino
#      plus my_esp32_generic.ino, never from an older merged file.
#   3) Replace the generated/base sketch with a fresh merged sketch.
#   4) Compile firmware using Arduino CLI.
#   5) Upload firmware to ESP32.
#   6) Install Python libraries needed for ESP32 <-> PC communication.
#   7) List available COM ports.
#   8) Monitor ESP32 status messages over USB serial.
#   9) Send commands from Python/PC to ESP32 over USB serial.
#  10) Generate a standalone Python serial-control example program.
#
# Important:
#   This version intentionally removes the menu option that restores the
#   original esp32_generic.ino. The .org file is still kept internally as
#   the clean original source used for every fresh merge.
#
# ESP32 firmware requirement:
#   In your ESP32 .ino file, include Serial.begin(115200) in setup().
#
# Example ESP32 code:
#
#   void setup()
#   {
#     Serial.begin(115200);
#     delay(1000);
#     Serial.println("ESP32_READY");
#   }
#
#   void loop()
#   {
#     Serial.println("STATUS: ESP32 running");
#     delay(1000);
#
#     if (Serial.available())
#     {
#       String cmd = Serial.readStringUntil('\n');
#       cmd.trim();
#       Serial.print("RECEIVED_CMD: ");
#       Serial.println(cmd);
#     }
#   }
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
ENV_DIR = ROOT_DIR / "esp32_env" / "esp32_generic"
BASE_INO = ENV_DIR / "esp32_generic.ino"
CUSTOM_INO = ROOT_DIR / "my_esp32_generic.ino"
MERGED_INO = ROOT_DIR / "merge_esp32_generic.ino"
BACKUP_INO = ENV_DIR / "esp32_generic.ino.org"

# IMPORTANT MERGE RULE
# --------------------
# BACKUP_INO is treated as the clean original esp32_generic.ino.
# If BACKUP_INO does not exist yet, the current BASE_INO is copied to BACKUP_INO
# before the first merge.
#
# This prevents a common problem:
#   esp32_generic.ino + my_esp32_generic.ino -> merge_esp32_generic.ino
#   then the merged file replaces esp32_generic.ino
#   then a later merge accidentally merges the already-merged file again.
#
# With this script, every merge uses:
#   original esp32_generic.ino.org + my_esp32_generic.ino
#
# so your original comments and original base logic are kept clean.
SERIAL_CONTROL_EXAMPLE = ROOT_DIR / "esp32_python_serial_control.py"
DEFAULT_FQBN = "esp32:esp32:esp32"
DEFAULT_BAUD = 115200

# Python-side libraries:
# pyserial          -> USB serial communication with ESP32 and COM-port listing.
# requests          -> Optional HTTP communication with ESP32 web server over Wi-Fi.
# websocket-client  -> Optional WebSocket communication if ESP32 exposes WebSocket.
# colorama          -> Optional colored terminal output on Windows CMD / VS Terminal.
REQUIRED_PYTHON_PACKAGES = [
    "pyserial",
    "requests",
    "websocket-client",
    "colorama",
]


def pause():
    input("\nPress Enter...")


def arduino_cli_path():
    p = Path(r"C:\Program Files\Arduino CLI\arduino-cli.exe")
    return str(p) if p.exists() else "arduino-cli"


def run_cmd(cmd):
    print("\nRUNNING:")
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
    try:
        r = subprocess.run(cmd)
    except FileNotFoundError as e:
        print("\nERROR: command not found.")
        print("Details:", e)
        print("Recommendation:")
        print(" - Install Arduino CLI, or")
        print(" - Add arduino-cli.exe to your Windows PATH, or")
        print(r" - Put it in C:\Program Files\Arduino CLI\arduino-cli.exe")
        return False
    if r.returncode != 0:
        print("\nERROR: command failed.")
        print("Return code:", r.returncode)
        return False
    return True


def install_python_libraries():
    print("\nInstalling/updating Python libraries needed for ESP32 communication...")
    print("Python executable:", sys.executable)
    ok = True
    for pkg in REQUIRED_PYTHON_PACKAGES:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", pkg]
        if not run_cmd(cmd):
            ok = False
    if ok:
        print("\nPython communication libraries installed successfully.")
        print("Installed/updated:")
        for pkg in REQUIRED_PYTHON_PACKAGES:
            print(" -", pkg)
    else:
        print("\nSome packages failed to install. Check the error messages above.")


def require_pyserial():
    try:
        import serial  # noqa: F401
        import serial.tools.list_ports  # noqa: F401
        return True
    except Exception as e:
        print("\nERROR: pyserial is not installed or cannot be imported.")
        print("Details:", e)
        print("Run menu option 7 first: Install/update Python communication libraries.")
        return False


def list_serial_ports(return_ports=False):
    if not require_pyserial():
        return [] if return_ports else None

    import serial.tools.list_ports

    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("\nNo serial COM ports detected.")
        print("Check that:")
        print(" - ESP32 is connected by USB.")
        print(" - USB cable supports data, not charge-only.")
        print(" - CP210x / CH340 / FTDI driver is installed if needed.")
        return [] if return_ports else None

    print("\nDetected serial ports:")
    for i, p in enumerate(ports, start=1):
        print(f" {i}) {p.device}  |  {p.description}")
        if p.hwid:
            print(f"     HWID: {p.hwid}")

    return ports if return_ports else None


def ask_serial_port():
    ports = list_serial_ports(return_ports=True)
    if not ports:
        return input("\nEnter ESP32 COM port manually, for example COM3: ").strip()

    print("\nSelect port number or type COM port manually.")
    choice = input("Port selection: ").strip()
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(ports):
            return ports[idx - 1].device
        print("Invalid port number.")
        return ""
    return choice


def monitor_esp32_serial_status():
    if not require_pyserial():
        return

    import serial

    port = ask_serial_port()
    if not port:
        print("\nERROR: Port is required.")
        return

    baud_text = input(f"Enter baud rate [{DEFAULT_BAUD}]: ").strip()
    baud = int(baud_text) if baud_text else DEFAULT_BAUD

    print("\nOpening ESP32 serial monitor...")
    print("Port:", port)
    print("Baud:", baud)
    print("Press Ctrl+C to stop monitoring.")

    try:
        with serial.Serial(port, baudrate=baud, timeout=1) as ser:
            time.sleep(2.0)  # ESP32 may reset when serial opens.
            ser.reset_input_buffer()
            print("\nListening for ESP32 status messages...")
            while True:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    print("[ESP32]", line)
    except KeyboardInterrupt:
        print("\nSerial monitor stopped by user.")
    except Exception as e:
        print("\nERROR: Could not monitor ESP32 serial output.")
        print("Details:", e)
        print("Common fixes:")
        print(" - Close Arduino Serial Monitor if it is open.")
        print(" - Close any other Python program using the same COM port.")
        print(" - Confirm the selected COM port is the ESP32.")


def send_command_to_esp32():
    if not require_pyserial():
        return

    import serial

    port = ask_serial_port()
    if not port:
        print("\nERROR: Port is required.")
        return

    baud_text = input(f"Enter baud rate [{DEFAULT_BAUD}]: ").strip()
    baud = int(baud_text) if baud_text else DEFAULT_BAUD

    print("\nCommand examples:")
    print(" - STATUS")
    print(" - SERVO 1 90")
    print(" - LED ON")
    print(" - LED OFF")
    cmd_text = input("Enter command to send to ESP32: ").strip()
    if not cmd_text:
        print("\nERROR: Command is required.")
        return

    try:
        with serial.Serial(port, baudrate=baud, timeout=1) as ser:
            time.sleep(2.0)  # ESP32 may reset when serial opens.
            ser.reset_input_buffer()
            ser.write((cmd_text + "\n").encode("utf-8"))
            ser.flush()
            print("\nSent to ESP32:", cmd_text)
            print("Waiting briefly for ESP32 response...")
            end_time = time.time() + 3.0
            got_response = False
            while time.time() < end_time:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    got_response = True
                    print("[ESP32]", line)
            if not got_response:
                print("No response received. Make sure ESP32 code reads Serial commands.")
    except Exception as e:
        print("\nERROR: Could not send command to ESP32.")
        print("Details:", e)


def create_serial_control_example():
    example = '''"""
ESP32 Python Serial Control Example
===================================
Run this file from CMD or Visual Studio Terminal:

    python esp32_python_serial_control.py

Purpose:
    - List COM ports.
    - Connect to ESP32 over USB serial.
    - Receive ESP32 status messages.
    - Send commands to ESP32.

Required ESP32 Arduino code pattern:

    void setup()
    {
      Serial.begin(115200);
      delay(1000);
      Serial.println("ESP32_READY");
    }

    void loop()
    {
      Serial.println("STATUS: ESP32 running");
      delay(1000);

      if (Serial.available())
      {
        String cmd = Serial.readStringUntil('\\n');
        cmd.trim();
        Serial.print("RECEIVED_CMD: ");
        Serial.println(cmd);
      }
    }
"""

import sys
import time
import subprocess


def install_if_missing():
    try:
        import serial  # noqa: F401
        import serial.tools.list_ports  # noqa: F401
    except Exception:
        print("pyserial is missing. Installing pyserial...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pyserial"], check=False)


def list_ports():
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    print("\\nAvailable COM ports:")
    for i, p in enumerate(ports, start=1):
        print(f" {i}) {p.device} - {p.description}")
    return ports


def select_port():
    ports = list_ports()
    choice = input("\\nSelect port number or type COM port, for example COM3: ").strip()
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(ports):
            return ports[idx - 1].device
    return choice


def main():
    install_if_missing()
    import serial

    port = select_port()
    baud = 115200

    print(f"\\nOpening {port} at {baud} baud...")
    with serial.Serial(port, baudrate=baud, timeout=0.2) as ser:
        time.sleep(2.0)
        ser.reset_input_buffer()
        print("Connected. Type commands and press Enter.")
        print("Examples: STATUS, LED ON, LED OFF, SERVO 1 90")
        print("Type q to quit.\\n")

        while True:
            while ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="replace").strip()
                if line:
                    print("[ESP32]", line)

            cmd = input("PC> ").strip()
            if cmd.lower() in ("q", "quit", "exit"):
                print("Exiting.")
                break

            ser.write((cmd + "\\n").encode("utf-8"))
            ser.flush()

            end_time = time.time() + 1.5
            while time.time() < end_time:
                if ser.in_waiting:
                    line = ser.readline().decode("utf-8", errors="replace").strip()
                    if line:
                        print("[ESP32]", line)
                else:
                    time.sleep(0.05)


if __name__ == "__main__":
    main()
'''
    SERIAL_CONTROL_EXAMPLE.write_text(example, encoding="utf-8")
    print("\nCreated standalone Python serial-control example:")
    print(" ", SERIAL_CONTROL_EXAMPLE)


def ensure_original_backup():
    """
    Guarantee that BACKUP_INO contains the clean original esp32_generic.ino.

    The first time this tool is run, BACKUP_INO may not exist yet. In that case
    the current BASE_INO is copied to BACKUP_INO before any merge happens.

    After BACKUP_INO exists, all future merges read from BACKUP_INO instead of
    BASE_INO. This makes the merge repeatable and prevents duplicated code.
    """
    if BACKUP_INO.exists():
        return True

    if not BASE_INO.exists():
        print("\nERROR: Cannot create original backup because base sketch is missing:")
        print(" -", BASE_INO)
        return False

    BACKUP_INO.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASE_INO, BACKUP_INO)
    print("\nOriginal esp32_generic.ino backed up to:")
    print(" ", BACKUP_INO)
    print("\nFuture merges will always use this original backup as the base.")
    print("The menu no longer includes a restore-original option; this .org file is only used as merge input.")
    return True


def get_original_base_ino():
    """
    Return the path that should be used as the original base sketch.

    Always prefer BACKUP_INO, because BASE_INO may already have been replaced by
    a previously merged sketch.
    """
    return BACKUP_INO if BACKUP_INO.exists() else BASE_INO


def check_required_files():
    missing = []

    # We need the current base at least once so we can create the .org backup.
    if not BASE_INO.exists() and not BACKUP_INO.exists():
        missing.append(str(BASE_INO))

    if not CUSTOM_INO.exists():
        missing.append(str(CUSTOM_INO))

    if missing:
        print("\nERROR: Missing required file(s):")
        for x in missing:
            print(" -", x)
        return False

    if not ensure_original_backup():
        return False

    return True


def find_function_block(text, function_name):
    pattern = re.compile(r"\bvoid\s+" + re.escape(function_name) + r"\s*\([^)]*\)\s*\{", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return None
    start = m.start()
    brace_start = text.find("{", m.start())
    if brace_start == -1:
        return None
    depth = 0
    i = brace_start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
        i += 1
    return None


def has_function(text, function_name):
    return find_function_block(text, function_name) is not None


def remove_function_block(text, function_name):
    block = find_function_block(text, function_name)
    if not block:
        return text, False
    start, end = block
    return text[:start].rstrip() + "\n\n" + text[end:].lstrip(), True


def ensure_wifi_include(text):
    uses_wifi = "WiFiClient" in text or "WiFiServer" in text or "WiFi." in text
    has_include = "#include <WiFi.h>" in text or '#include "WiFi.h"' in text
    if uses_wifi and not has_include:
        text = "#include <WiFi.h>\n" + text
    return text


def add_ledc_compat(text):
    if "ledcSetup(" not in text and "ledcAttachPin(" not in text:
        return text
    if "ESP32_LEDC_COMPAT_HELPERS" in text:
        return text

    compat = r'''
// ============================================================
// ESP32 Arduino Core 3.x LEDC compatibility helpers
// ============================================================
#ifndef ESP32_LEDC_COMPAT_HELPERS
#define ESP32_LEDC_COMPAT_HELPERS

#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
static int __compat_ledc_freq[16];
static int __compat_ledc_resolution[16];

static inline void compat_ledcSetup(int channel, int freq, int resolution)
{
  if (channel >= 0 && channel < 16)
  {
    __compat_ledc_freq[channel] = freq;
    __compat_ledc_resolution[channel] = resolution;
  }
}

static inline void compat_ledcAttachPin(int pin, int channel)
{
  int freq = 5000;
  int resolution = 8;

  if (channel >= 0 && channel < 16)
  {
    if (__compat_ledc_freq[channel] > 0) freq = __compat_ledc_freq[channel];
    if (__compat_ledc_resolution[channel] > 0) resolution = __compat_ledc_resolution[channel];
  }

  ledcAttachChannel(pin, freq, resolution, channel);
}

#define ledcSetup(channel, freq, resolution) compat_ledcSetup(channel, freq, resolution)
#define ledcAttachPin(pin, channel) compat_ledcAttachPin(pin, channel)
#endif

#endif
'''
    include_matches = list(re.finditer(r'^\s*#include\s+[<"].+[>"]\s*$', text, re.MULTILINE))
    if include_matches:
        insert_at = include_matches[-1].end()
        return text[:insert_at] + "\n" + compat + text[insert_at:]
    return compat + "\n" + text


def ensure_serial_helper_comment(text):
    if "ESP32_SERIAL_STATUS_HELPER_COMMENT" in text:
        return text
    comment = r'''
/*
ESP32_SERIAL_STATUS_HELPER_COMMENT
============================================================
Python serial status communication reminder
============================================================
To allow the PC Python program to receive ESP32 status messages,
your ESP32 sketch should initialize USB serial in setup():

  Serial.begin(115200);
  delay(1000);
  Serial.println("ESP32_READY");

Then print status messages anywhere in loop() or functions:

  Serial.println("STATUS: WiFi connected");
  Serial.println("SERVO:1:90");
  Serial.println("BATTERY:7.4");

To receive commands from Python:

  if (Serial.available())
  {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    Serial.print("RECEIVED_CMD: ");
    Serial.println(cmd);
  }
============================================================
*/
'''
    return comment + "\n" + text


def smart_merge(base_text, custom_text):
    cleaned_base = base_text
    removed_setup = False
    removed_loop = False

    if has_function(cleaned_base, "setup") and has_function(custom_text, "setup"):
        cleaned_base, removed_setup = remove_function_block(cleaned_base, "setup")

    if has_function(cleaned_base, "loop") and has_function(custom_text, "loop"):
        cleaned_base, removed_loop = remove_function_block(cleaned_base, "loop")

    merged = f'''
/*
============================================================
MERGED ESP32 SKETCH
Base file: {BASE_INO}
Custom file: {CUSTOM_INO}
Removed duplicate setup() from base: {removed_setup}
Removed duplicate loop() from base: {removed_loop}
============================================================
*/

{cleaned_base.rstrip()}

/*
============================================================
CUSTOM FILE CONTENT START
============================================================
*/

{custom_text.rstrip()}

/*
============================================================
CUSTOM FILE CONTENT END
============================================================
*/
'''
    merged = ensure_wifi_include(merged)
    merged = add_ledc_compat(merged)
    merged = ensure_serial_helper_comment(merged)
    return merged


def show_paths():
    print("\nCurrent paths:")
    print("Script directory:", ROOT_DIR)
    print("Base sketch:", BASE_INO)
    print("Original .org source used internally for fresh merges:", BACKUP_INO)
    print("Custom sketch:", CUSTOM_INO)
    print("Merge output:", MERGED_INO)


def create_merge_file():
    """
    Create merge_esp32_generic.ino from:

        ORIGINAL esp32_generic.ino.org  +  my_esp32_generic.ino

    This function intentionally does NOT use BASE_INO after the .org backup
    exists, because BASE_INO may already be a previously merged sketch.
    """
    if not check_required_files():
        return False

    original_base = get_original_base_ino()
    print("\nCreating fresh merge from:")
    print(" Original base:", original_base)
    print(" Custom file:  ", CUSTOM_INO)
    print(" Output file:  ", MERGED_INO)

    base_text = original_base.read_text(encoding="utf-8", errors="replace")
    custom_text = CUSTOM_INO.read_text(encoding="utf-8", errors="replace")
    MERGED_INO.write_text(smart_merge(base_text, custom_text), encoding="utf-8")

    print("\nFresh merge file created:")
    print(" ", MERGED_INO)
    print("\nThis merge used the original esp32_generic.ino backup, not an older merged file.")
    print("Serial communication reminder comments were added to the merged sketch.")
    return True


def replace_base_with_merge():
    """
    Always rebuild the merge first, then replace BASE_INO.

    This guarantees esp32_env/esp32_generic/esp32_generic.ino always becomes:

        original esp32_generic.ino.org + my_esp32_generic.ino

    and never an old merge stacked on top of another merge.
    """
    if not create_merge_file():
        return False

    if not MERGED_INO.exists():
        print("\nERROR: Merge file was not created:", MERGED_INO)
        return False

    shutil.copy2(MERGED_INO, BASE_INO)
    print("\nBase file replaced with fresh merged file:")
    print(" ", BASE_INO)
    return True



def compile_firmware():
    """
    Before compiling, always rebuild and install a fresh merged sketch.
    """
    if not replace_base_with_merge():
        return

    if not BASE_INO.exists():
        print("\nERROR: Base sketch not found:", BASE_INO)
        return

    cmd = [
        arduino_cli_path(),
        "compile",
        "--fqbn",
        DEFAULT_FQBN,
        str(ENV_DIR),
        "--output-dir",
        str(ENV_DIR),
    ]
    print("\nBuild output directory:")
    print(" ", ENV_DIR)
    if run_cmd(cmd):
        print("\nCompile completed successfully.")
        print("Generated build files are inside:", ENV_DIR)


def upload_firmware():
    """
    Before uploading, always rebuild and install a fresh merged sketch.
    """
    if not replace_base_with_merge():
        return

    if not BASE_INO.exists():
        print("\nERROR: Base sketch not found:", BASE_INO)
        return

    port = input("Enter ESP32 port, for example COM3: ").strip()
    if not port:
        print("\nERROR: Port is required.")
        return

    cmd = [
        arduino_cli_path(),
        "upload",
        "-p",
        port,
        "--fqbn",
        DEFAULT_FQBN,
        str(ENV_DIR),
    ]
    run_cmd(cmd)


def get_key():
    try:
        import msvcrt
        key = msvcrt.getwch()
        print(key)
        return key.lower()
    except Exception:
        return input("Select option: ").strip().lower()


def main():
    while True:
        print("=" * 72)
        print(" ESP32 Generic Firmware Tool + Python Serial Status Communication")
        print("=" * 72)
        print("1 - Show paths")
        print("2 - Create fresh merge_esp32_generic.ino from original + my_esp32_generic")
        print("3 - Rebuild fresh merge and replace esp32_env esp32_generic.ino")
        print("4 - Compile firmware")
        print("5 - Load/upload firmware to ESP32")
        print("6 - List available ESP32 COM ports")
        print("7 - Monitor ESP32 status messages over USB serial")
        print("8 - Send command from Python to ESP32 over USB serial")
        print("q - Exit")
        print("=" * 72)
        print("Press one key. You do not need to press Enter.")
        print("Select option: ", end="", flush=True)

        choice = get_key()

        if choice == "1":
            show_paths()
            pause()
        elif choice == "2":
            create_merge_file()
            pause()
        elif choice == "3":
            replace_base_with_merge()
            pause()
        elif choice == "4":
            compile_firmware()
            pause()
        elif choice == "5":
            upload_firmware()
            pause()
        elif choice == "6":
            list_serial_ports()
            pause()
        elif choice == "7":
            monitor_esp32_serial_status()
            pause()
        elif choice == "8":
            send_command_to_esp32()
            pause()
        elif choice == "q":
            print("\nExiting.")
            break
        else:
            print("\nInvalid option.")
            pause()


if __name__ == "__main__":
    main()
