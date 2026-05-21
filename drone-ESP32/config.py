# ============================================================
# GENERIC ESP32 DRONE APPLICATION LOCAL WINDOWS TOOL
# ============================================================
#
# PURPOSE
# ------------------------------------------------------------
# This script creates and manages a GENERIC ESP32 drone project.
# It removes the dependency on OpenCat / Petoi / Bittle firmware.
#
# It automates:
#   1. Installing Python dependencies
#   2. Installing Arduino CLI locally
#   3. Installing ESP32 board support
#   4. Creating a clean DroneESP32 Arduino sketch
#   5. Installing optional drone libraries
#   6. Compiling and uploading firmware to ESP32
#
# SAFETY
# ------------------------------------------------------------
# This is a starter framework, not a certified flight controller.
# Always test ESC/motor output with propellers removed.
# ============================================================

import os
import sys
import zipfile
import subprocess
import urllib.request


# ============================================================
# SELF-INSTALL REQUIRED PYTHON PACKAGES
# ============================================================

def ensure_python_package(import_name, pip_name):
    try:
        __import__(import_name)
        print(f"{pip_name} already installed.")
    except ImportError:
        print(f"{pip_name} is missing. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        print(f"{pip_name} installed successfully.")


def install_required_python_packages():
    ensure_python_package("serial", "pyserial")
    ensure_python_package("esptool", "esptool")


install_required_python_packages()


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR
DRONE_ENV_DIR = os.path.join(BASE_DIR, "drone_env")
DATA_DIR = os.path.join(DRONE_ENV_DIR, "data")
LOCAL_LIBRARIES_DIR = os.path.join(DRONE_ENV_DIR, "libraries")

CLI_EXE = os.path.join(DRONE_ENV_DIR, "arduino-cli.exe")
CLI_CONFIG = os.path.join(DATA_DIR, "cli.yaml")
CLI_ZIP = os.path.join(DRONE_ENV_DIR, "arduino-cli.zip")

CLI_URL = "https://downloads.arduino.cc/arduino-cli/arduino-cli_latest_Windows_64bit.zip"

DEFAULT_SKETCH_FILE_NAME = "DroneESP32.ino"


def normalize_sketch_file_name(file_name):
    value = (file_name or "").strip().strip('"').strip("'")
    if not value:
        value = DEFAULT_SKETCH_FILE_NAME
    value = os.path.basename(value)
    if not value.lower().endswith(".ino"):
        value += ".ino"
    invalid_chars = '<>:"/\\|?*'
    if any(ch in value for ch in invalid_chars):
        print("Invalid sketch filename. Using default.")
        value = DEFAULT_SKETCH_FILE_NAME
    return value


def select_main_sketch_file_name():
    env_value = os.environ.get("DRONE_SKETCH_FILE", "").strip()
    if env_value:
        selected = normalize_sketch_file_name(env_value)
        print("Selected sketch file from DRONE_SKETCH_FILE:", selected)
        return selected

    print("\n============================================================")
    print(" SELECT MAIN ARDUINO SKETCH FILE")
    print("============================================================")
    print("Default:", DEFAULT_SKETCH_FILE_NAME)
    print("Press Enter to use default.")
    value = input("Enter main .ino filename: ").strip()
    selected = normalize_sketch_file_name(value)
    print("Selected main sketch file:", selected)
    return selected


MAIN_SKETCH_FILE_NAME = select_main_sketch_file_name()
MAIN_SKETCH_BASE_NAME = os.path.splitext(MAIN_SKETCH_FILE_NAME)[0]
PROJECT_DIR = os.path.join(DRONE_ENV_DIR, MAIN_SKETCH_BASE_NAME)
INO_FILE = os.path.join(PROJECT_DIR, MAIN_SKETCH_FILE_NAME)
CONFIG_HEADER_FILE = os.path.join(PROJECT_DIR, "drone_config.h")


# ============================================================
# SINGLE KEY INPUT
# ============================================================

def get_single_key(prompt="Select option: "):
    print(prompt, end="", flush=True)
    if os.name == "nt":
        import msvcrt
        key = msvcrt.getch()
        try:
            value = key.decode("utf-8")
        except UnicodeDecodeError:
            value = ""
        print(value)
        return value.strip()
    return input().strip()


# ============================================================
# ESP32 BOARD PROFILES
# ============================================================

ESP32_CORE = "esp32:esp32"

BOARD_PROFILES = {
    "classic_esp32_4m": {
        "label": "Classic ESP32 / 4 MB flash",
        "fqbn_candidates": [
            "esp32:esp32:esp32:PartitionScheme=huge_app",
            "esp32:esp32:esp32",
        ],
    },
    "classic_esp32_16m": {
        "label": "Classic ESP32 / 16 MB flash",
        "fqbn_candidates": [
            "esp32:esp32:esp32:FlashSize=16M,PartitionScheme=app3M_fat9M_16MB",
            "esp32:esp32:esp32:FlashSize=16M,PartitionScheme=huge_app",
            "esp32:esp32:esp32",
        ],
    },
    "esp32s3_4m": {
        "label": "ESP32-S3 / 4 MB flash",
        "fqbn_candidates": [
            "esp32:esp32:esp32s3:PartitionScheme=huge_app",
            "esp32:esp32:esp32s3",
        ],
    },
    "esp32s3_16m": {
        "label": "ESP32-S3 / 16 MB flash",
        "fqbn_candidates": [
            "esp32:esp32:esp32s3:FlashSize=16M,PartitionScheme=app3M_fat9M_16MB",
            "esp32:esp32:esp32s3:FlashSize=16M,PartitionScheme=huge_app",
            "esp32:esp32:esp32s3",
        ],
    },
    "esp32c3_4m": {
        "label": "ESP32-C3 / 4 MB flash",
        "fqbn_candidates": [
            "esp32:esp32:esp32c3:PartitionScheme=huge_app",
            "esp32:esp32:esp32c3",
        ],
    },
    "esp32c6_4m": {
        "label": "ESP32-C6 / 4 MB flash",
        "fqbn_candidates": ["esp32:esp32:esp32c6"],
    },
}

DEFAULT_BOARD_PROFILE = "classic_esp32_4m"
ACTIVE_BOARD_PROFILE = os.environ.get("DRONE_ESP32_MODEL", DEFAULT_BOARD_PROFILE).strip()


def get_active_board_profile_name():
    global ACTIVE_BOARD_PROFILE
    if ACTIVE_BOARD_PROFILE not in BOARD_PROFILES:
        print("Unknown board profile. Falling back to:", DEFAULT_BOARD_PROFILE)
        ACTIVE_BOARD_PROFILE = DEFAULT_BOARD_PROFILE
    return ACTIVE_BOARD_PROFILE


def get_active_board_profile():
    return BOARD_PROFILES[get_active_board_profile_name()]


def get_active_fqbn_candidates():
    return get_active_board_profile()["fqbn_candidates"]


def print_active_board_profile():
    name = get_active_board_profile_name()
    profile = get_active_board_profile()
    print("\n============================================================")
    print(" ACTIVE ESP32 TARGET")
    print("============================================================")
    print("Profile:", name)
    print("Label  :", profile["label"])
    print("FQBN candidates:")
    for fqbn in profile["fqbn_candidates"]:
        print("  -", fqbn)
    print("============================================================")


def select_board_profile():
    global ACTIVE_BOARD_PROFILE
    names = list(BOARD_PROFILES.keys())
    keys = list("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    key_to_profile = {}
    print("\n============================================================")
    print(" SELECT ESP32 TARGET MODEL")
    print("============================================================")
    for i, name in enumerate(names):
        key = keys[i]
        key_to_profile[key.lower()] = name
        current = "  <== current" if name == get_active_board_profile_name() else ""
        print(f"{key} - {name}: {BOARD_PROFILES[name]['label']}{current}")
    print("0 - Cancel")
    choice = get_single_key("Select target model: ").lower()
    if choice == "0":
        return
    if choice in key_to_profile:
        ACTIVE_BOARD_PROFILE = key_to_profile[choice]
    else:
        print("Invalid selection. Keeping current profile.")
    print_active_board_profile()


select_board_profile()


# ============================================================
# DRONE APPLICATION PROFILES
# ============================================================

DRONE_APP_PROFILES = {
    "minimal": {
        "label": "Minimal framework, Wi-Fi telemetry only",
        "libraries": [],
        "use_esc": 0,
        "use_imu": 0,
        "use_gps": 0,
        "use_wifi": 1,
    },
    "esc_only": {
        "label": "ESC motor control + Wi-Fi telemetry",
        "libraries": ["ESP32Servo"],
        "use_esc": 1,
        "use_imu": 0,
        "use_gps": 0,
        "use_wifi": 1,
    },
    "drone_core": {
        "label": "ESC + MPU6050 IMU + Wi-Fi telemetry",
        "libraries": ["ESP32Servo", "Adafruit MPU6050", "Adafruit Unified Sensor", "Adafruit BusIO"],
        "use_esc": 1,
        "use_imu": 1,
        "use_gps": 0,
        "use_wifi": 1,
    },
    "drone_full": {
        "label": "ESC + MPU6050 IMU + GPS + Wi-Fi telemetry",
        "libraries": ["ESP32Servo", "TinyGPSPlus", "Adafruit MPU6050", "Adafruit Unified Sensor", "Adafruit BusIO"],
        "use_esc": 1,
        "use_imu": 1,
        "use_gps": 1,
        "use_wifi": 1,
    },
}

DEFAULT_DRONE_APP_PROFILE = "drone_core"
ACTIVE_DRONE_APP_PROFILE = os.environ.get("DRONE_APP_PROFILE", DEFAULT_DRONE_APP_PROFILE).strip()


def get_active_drone_app_profile_name():
    global ACTIVE_DRONE_APP_PROFILE
    if ACTIVE_DRONE_APP_PROFILE not in DRONE_APP_PROFILES:
        print("Unknown drone profile. Falling back to:", DEFAULT_DRONE_APP_PROFILE)
        ACTIVE_DRONE_APP_PROFILE = DEFAULT_DRONE_APP_PROFILE
    return ACTIVE_DRONE_APP_PROFILE


def get_active_drone_app_profile():
    return DRONE_APP_PROFILES[get_active_drone_app_profile_name()]


def print_active_drone_app_profile():
    name = get_active_drone_app_profile_name()
    profile = get_active_drone_app_profile()
    print("\n============================================================")
    print(" ACTIVE DRONE APPLICATION PROFILE")
    print("============================================================")
    print("Profile:", name)
    print("Label  :", profile["label"])
    print("Libraries:")
    if profile["libraries"]:
        for lib in profile["libraries"]:
            print("  -", lib)
    else:
        print("  - none")
    print("============================================================")


def select_drone_app_profile():
    global ACTIVE_DRONE_APP_PROFILE
    names = list(DRONE_APP_PROFILES.keys())
    keys = list("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    key_to_profile = {}
    print("\n============================================================")
    print(" SELECT DRONE APPLICATION PROFILE")
    print("============================================================")
    for i, name in enumerate(names):
        key = keys[i]
        key_to_profile[key.lower()] = name
        current = "  <== current" if name == get_active_drone_app_profile_name() else ""
        print(f"{key} - {name}: {DRONE_APP_PROFILES[name]['label']}{current}")
    print("0 - Cancel")
    choice = get_single_key("Select drone profile: ").lower()
    if choice == "0":
        return
    if choice in key_to_profile:
        ACTIVE_DRONE_APP_PROFILE = key_to_profile[choice]
    else:
        print("Invalid selection. Keeping current profile.")
    print_active_drone_app_profile()


select_drone_app_profile()


# ============================================================
# COMMAND HELPERS
# ============================================================

def run_cmd(cmd):
    print("\nRUNNING:")
    print(" ".join(cmd))
    print("\nOUTPUT:\n--------------------------------------------------")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    print("--------------------------------------------------")
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}")


def run_cmd_return_code(cmd):
    print("\nRUNNING:")
    print(" ".join(cmd))
    print("\nOUTPUT:\n--------------------------------------------------")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    print("--------------------------------------------------")
    return process.returncode


def download_file(url, output_path):
    print("\nDownloading:", url)
    urllib.request.urlretrieve(url, output_path)
    print("Saved:", output_path)


def extract_zip(zip_path, output_dir):
    print("\nExtracting:", zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)


# ============================================================
# ARDUINO ENVIRONMENT
# ============================================================

def install_arduino_cli():
    os.makedirs(DRONE_ENV_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOCAL_LIBRARIES_DIR, exist_ok=True)
    if not os.path.exists(CLI_EXE):
        download_file(CLI_URL, CLI_ZIP)
        extract_zip(CLI_ZIP, DRONE_ENV_DIR)
    else:
        print("Arduino CLI already installed:", CLI_EXE)


def install_arduino_environment():
    install_arduino_cli()
    run_cmd([CLI_EXE, "config", "init", "--config-file", CLI_CONFIG])
    run_cmd([CLI_EXE, "core", "update-index", "--config-file", CLI_CONFIG])
    run_cmd([CLI_EXE, "core", "install", ESP32_CORE, "--config-file", CLI_CONFIG])

    profile = get_active_drone_app_profile()
    for lib in profile["libraries"]:
        run_cmd([CLI_EXE, "lib", "install", lib, "--config-file", CLI_CONFIG])


# ============================================================
# DRONE PROJECT GENERATION
# ============================================================

def generate_config_header():
    p = get_active_drone_app_profile()
    return f"""#pragma once

// ============================================================
// DRONE FEATURE SWITCHES
// ============================================================
#define USE_ESC {p['use_esc']}
#define USE_IMU {p['use_imu']}
#define USE_GPS {p['use_gps']}
#define USE_WIFI_TELEMETRY {p['use_wifi']}

// ============================================================
// WIFI TELEMETRY
// ============================================================
#define WIFI_AP_SSID "ESP32_DRONE"
#define WIFI_AP_PASSWORD "12345678"
#define TELEMETRY_PORT 4210

// ============================================================
// MOTOR / ESC PIN MAPPING
// ============================================================
// Change these pins to match your actual ESP32 wiring.
#define MOTOR_FL_PIN 25
#define MOTOR_FR_PIN 26
#define MOTOR_RL_PIN 27
#define MOTOR_RR_PIN 14

#define ESC_MIN_US 1000
#define ESC_IDLE_US 1100
#define ESC_MAX_US 2000

// ============================================================
// GPS SERIAL PINS
// ============================================================
#define GPS_RX_PIN 16
#define GPS_TX_PIN 17
#define GPS_BAUD 9600

// ============================================================
// CONTROL LOOP
// ============================================================
#define CONTROL_LOOP_HZ 100
#define CONTROL_LOOP_PERIOD_MS (1000 / CONTROL_LOOP_HZ)
#define START_DISARMED 1
"""


def generate_drone_ino():
    return f"""// ============================================================
// GENERIC ESP32 DRONE APPLICATION
// ============================================================
// Main sketch: {MAIN_SKETCH_FILE_NAME}
// This firmware is generic and has no OpenCat/Petoi dependency.
// ============================================================

#include <Arduino.h>
#include "drone_config.h"

#if USE_WIFI_TELEMETRY
#include <WiFi.h>
WiFiServer telemetryServer(TELEMETRY_PORT);
WiFiClient telemetryClient;
#endif

#if USE_ESC
#include <ESP32Servo.h>
Servo motorFL;
Servo motorFR;
Servo motorRL;
Servo motorRR;
#endif

#if USE_IMU
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
Adafruit_MPU6050 imu;
#endif

#if USE_GPS
#include <TinyGPSPlus.h>
TinyGPSPlus gps;
HardwareSerial gpsSerial(2);
#endif

bool droneArmed = false;
unsigned long lastControlLoopMs = 0;
float rollDeg = 0.0;
float pitchDeg = 0.0;
float yawDeg = 0.0;
double gpsLatitude = 0.0;
double gpsLongitude = 0.0;
double gpsAltitudeM = 0.0;

void disarmDrone() {{
  droneArmed = false;
#if USE_ESC
  motorFL.writeMicroseconds(ESC_MIN_US);
  motorFR.writeMicroseconds(ESC_MIN_US);
  motorRL.writeMicroseconds(ESC_MIN_US);
  motorRR.writeMicroseconds(ESC_MIN_US);
#endif
  Serial.println("[SAFETY] Drone disarmed.");
}}

void armDrone() {{
  droneArmed = true;
  Serial.println("[SAFETY] Drone armed.");
}}

void setupMotors() {{
#if USE_ESC
  Serial.println("[MOTOR] Attaching ESC outputs...");
  motorFL.attach(MOTOR_FL_PIN, ESC_MIN_US, ESC_MAX_US);
  motorFR.attach(MOTOR_FR_PIN, ESC_MIN_US, ESC_MAX_US);
  motorRL.attach(MOTOR_RL_PIN, ESC_MIN_US, ESC_MAX_US);
  motorRR.attach(MOTOR_RR_PIN, ESC_MIN_US, ESC_MAX_US);
  disarmDrone();
#else
  Serial.println("[MOTOR] ESC support disabled.");
#endif
}}

void writeAllMotorsMicroseconds(int valueUs) {{
#if USE_ESC
  valueUs = constrain(valueUs, ESC_MIN_US, ESC_MAX_US);
  if (!droneArmed) valueUs = ESC_MIN_US;
  motorFL.writeMicroseconds(valueUs);
  motorFR.writeMicroseconds(valueUs);
  motorRL.writeMicroseconds(valueUs);
  motorRR.writeMicroseconds(valueUs);
#else
  (void)valueUs;
#endif
}}

void setupIMU() {{
#if USE_IMU
  Serial.println("[IMU] Initializing MPU6050...");
  if (!imu.begin()) {{
    Serial.println("[IMU] MPU6050 not detected. Check wiring.");
    return;
  }}
  imu.setAccelerometerRange(MPU6050_RANGE_8_G);
  imu.setGyroRange(MPU6050_RANGE_500_DEG);
  imu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("[IMU] MPU6050 ready.");
#else
  Serial.println("[IMU] IMU support disabled.");
#endif
}}

void updateIMU() {{
#if USE_IMU
  sensors_event_t accel, gyro, temp;
  imu.getEvent(&accel, &gyro, &temp);
  rollDeg = atan2(accel.acceleration.y, accel.acceleration.z) * 57.2958;
  pitchDeg = atan2(-accel.acceleration.x,
                   sqrt(accel.acceleration.y * accel.acceleration.y +
                        accel.acceleration.z * accel.acceleration.z)) * 57.2958;
  yawDeg += gyro.gyro.z * 57.2958 * (CONTROL_LOOP_PERIOD_MS / 1000.0);
#endif
}}

void setupGPS() {{
#if USE_GPS
  gpsSerial.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  Serial.println("[GPS] GPS serial ready.");
#else
  Serial.println("[GPS] GPS support disabled.");
#endif
}}

void updateGPS() {{
#if USE_GPS
  while (gpsSerial.available() > 0) gps.encode(gpsSerial.read());
  if (gps.location.isValid()) {{
    gpsLatitude = gps.location.lat();
    gpsLongitude = gps.location.lng();
  }}
  if (gps.altitude.isValid()) gpsAltitudeM = gps.altitude.meters();
#endif
}}

void setupTelemetry() {{
#if USE_WIFI_TELEMETRY
  WiFi.mode(WIFI_AP);
  WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASSWORD);
  telemetryServer.begin();
  Serial.print("[WIFI] SSID: "); Serial.println(WIFI_AP_SSID);
  Serial.print("[WIFI] IP: "); Serial.println(WiFi.softAPIP());
  Serial.print("[WIFI] Port: "); Serial.println(TELEMETRY_PORT);
#else
  Serial.println("[WIFI] Telemetry disabled.");
#endif
}}

void sendTelemetryLine(String line) {{
#if USE_WIFI_TELEMETRY
  if (telemetryClient && telemetryClient.connected()) telemetryClient.println(line);
#else
  (void)line;
#endif
}}

void handleTelemetryCommand(String cmd) {{
  cmd.trim();
  cmd.toLowerCase();
  if (cmd == "arm") {{
    armDrone();
    sendTelemetryLine("OK armed");
  }} else if (cmd == "disarm") {{
    disarmDrone();
    sendTelemetryLine("OK disarmed");
  }} else if (cmd.startsWith("throttle ")) {{
    int valueUs = cmd.substring(9).toInt();
    writeAllMotorsMicroseconds(valueUs);
    sendTelemetryLine("OK throttle " + String(valueUs));
  }} else if (cmd == "status") {{
    sendTelemetryLine(
      "armed=" + String(droneArmed ? "1" : "0") +
      ", roll=" + String(rollDeg, 2) +
      ", pitch=" + String(pitchDeg, 2) +
      ", yaw=" + String(yawDeg, 2) +
      ", lat=" + String(gpsLatitude, 6) +
      ", lon=" + String(gpsLongitude, 6) +
      ", alt=" + String(gpsAltitudeM, 2)
    );
  }} else {{
    sendTelemetryLine("ERROR unknown command");
  }}
}}

void updateTelemetry() {{
#if USE_WIFI_TELEMETRY
  if (!telemetryClient || !telemetryClient.connected()) telemetryClient = telemetryServer.available();
  if (telemetryClient && telemetryClient.connected()) {{
    while (telemetryClient.available()) {{
      String cmd = telemetryClient.readStringUntil('\\n');
      handleTelemetryCommand(cmd);
    }}
  }}
#endif
}}

void droneControlLoop() {{
  updateIMU();
  updateGPS();
  updateTelemetry();

  // Add future functionality here:
  //   - PID control
  //   - motor mixing
  //   - altitude hold
  //   - GPS waypoint navigation
  //   - failsafe checks
  if (!droneArmed) writeAllMotorsMicroseconds(ESC_MIN_US);
}}

void setup() {{
  Serial.begin(115200);
  delay(1000);
  Serial.println("GENERIC ESP32 DRONE APPLICATION");
#if START_DISARMED
  droneArmed = false;
#else
  droneArmed = true;
#endif
  setupMotors();
  setupIMU();
  setupGPS();
  setupTelemetry();
  Serial.println("[SYSTEM] Setup complete.");
}}

void loop() {{
  unsigned long nowMs = millis();
  if ((nowMs - lastControlLoopMs) >= CONTROL_LOOP_PERIOD_MS) {{
    lastControlLoopMs = nowMs;
    droneControlLoop();
  }}
}}
"""


def create_drone_project(force=False):
    os.makedirs(PROJECT_DIR, exist_ok=True)
    if os.path.isfile(INO_FILE) and not force:
        print("Drone sketch already exists:", INO_FILE)
    else:
        with open(INO_FILE, "w", encoding="utf-8", newline="\n") as f:
            f.write(generate_drone_ino())
        print("Drone sketch written:", INO_FILE)

    if os.path.isfile(CONFIG_HEADER_FILE) and not force:
        print("Drone config header already exists:", CONFIG_HEADER_FILE)
    else:
        with open(CONFIG_HEADER_FILE, "w", encoding="utf-8", newline="\n") as f:
            f.write(generate_config_header())
        print("Drone config header written:", CONFIG_HEADER_FILE)


def verify_drone_project():
    if not os.path.isdir(PROJECT_DIR):
        raise RuntimeError("Drone project folder not found. Run option 1 first.")
    if not os.path.isfile(INO_FILE):
        raise RuntimeError("Drone .ino file not found. Run option 8 to regenerate.")
    if not os.path.isfile(CONFIG_HEADER_FILE):
        raise RuntimeError("drone_config.h not found. Run option 8 to regenerate.")


# ============================================================
# BUILD AND UPLOAD
# ============================================================

def install_all():
    install_arduino_environment()
    create_drone_project(force=False)
    print("\nINSTALLATION COMPLETE.")


def compile_firmware():
    if not os.path.exists(CLI_EXE):
        print("Arduino CLI missing. Installing first...")
        install_all()
    verify_drone_project()
    print_active_board_profile()
    print_active_drone_app_profile()
    last_failed = None
    for fqbn in get_active_fqbn_candidates():
        rc = run_cmd_return_code([
            CLI_EXE, "compile", "--verbose", "--fqbn", fqbn,
            PROJECT_DIR, "--output-dir", PROJECT_DIR,
            "--libraries", LOCAL_LIBRARIES_DIR,
            "--config-file", CLI_CONFIG,
        ])
        if rc == 0:
            print("\nCOMPILE COMPLETE. Successful FQBN:", fqbn)
            return
        last_failed = fqbn
    raise RuntimeError("All FQBN candidates failed. Last failed: " + str(last_failed))


def detect_com_port():
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No COM ports detected. Check USB cable and driver.")
    print("\nDetected COM ports:")
    for p in ports:
        print(f"  {p.device} - {p.description}")
    for p in ports:
        desc = p.description.lower()
        if "cp210" in desc or "ch340" in desc or "usb" in desc or "uart" in desc or "silicon labs" in desc:
            print("Selected ESP32 port:", p.device)
            return p.device
    print("Using first port:", ports[0].device)
    return ports[0].device


def upload_firmware():
    if not os.path.exists(CLI_EXE):
        print("Arduino CLI missing. Installing first...")
        install_all()
    verify_drone_project()
    port = detect_com_port()
    fqbn = get_active_fqbn_candidates()[0]
    run_cmd([
        CLI_EXE, "upload", "-p", port, "--fqbn", fqbn,
        PROJECT_DIR, "--libraries", LOCAL_LIBRARIES_DIR,
        "--config-file", CLI_CONFIG,
    ])
    print("\nUPLOAD COMPLETE.")


# ============================================================
# INFORMATION
# ============================================================

def print_debug_paths():
    print("\n============================================================")
    print(" DEBUG PATHS")
    print("============================================================")
    print("SCRIPT_DIR       :", SCRIPT_DIR)
    print("DRONE_ENV_DIR    :", DRONE_ENV_DIR)
    print("PROJECT_DIR      :", PROJECT_DIR)
    print("INO_FILE         :", INO_FILE)
    print("CONFIG_HEADER    :", CONFIG_HEADER_FILE)
    print("CLI_EXE          :", CLI_EXE)
    print("CLI_CONFIG       :", CLI_CONFIG)
    print("LOCAL_LIBRARIES  :", LOCAL_LIBRARIES_DIR)
    print("============================================================")


def show_extension_guide():
    print("\n============================================================")
    print(" HOW TO ADD MORE DRONE FUNCTIONALITY")
    print("============================================================")
    print("1. Edit pin settings in:")
    print("   ", CONFIG_HEADER_FILE)
    print("2. Add setup/update functions in:")
    print("   ", INO_FILE)
    print("3. Add libraries in DRONE_APP_PROFILES inside this Python file.")
    print("4. Add your logic inside droneControlLoop().")
    print("5. Compile with option 2 and upload with option 3.")
    print("============================================================")


# ============================================================
# MAIN MENU
# ============================================================

def print_menu():
    print("\n============================================================")
    print(" GENERIC ESP32 DRONE FIRMWARE TOOL")
    print("============================================================")
    print("1 - Install tools, ESP32 core, libraries, and create drone project")
    print("2 - Compile drone firmware")
    print("3 - Upload drone firmware to ESP32")
    print("4 - Show active board and drone profile")
    print("5 - Select ESP32 board profile")
    print("6 - Select drone application profile")
    print("7 - Show debug paths")
    print("8 - Regenerate drone project files")
    print("9 - Show extension guide")
    print("q - Exit")
    print("============================================================")


def main():
    while True:
        print_menu()
        choice = get_single_key("Select option: ").lower()
        try:
            if choice == "1":
                install_all()
            elif choice == "2":
                compile_firmware()
            elif choice == "3":
                upload_firmware()
            elif choice == "4":
                print_active_board_profile()
                print_active_drone_app_profile()
            elif choice == "5":
                select_board_profile()
            elif choice == "6":
                select_drone_app_profile()
            elif choice == "7":
                print_debug_paths()
            elif choice == "8":
                create_drone_project(force=True)
            elif choice == "9":
                show_extension_guide()
            elif choice == "q":
                print("Exiting.")
                break
            else:
                print("Invalid option.")
        except Exception as e:
            print("\nERROR:")
            print(str(e))


if __name__ == "__main__":
    main()
