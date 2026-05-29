"""
ESP32 Motor Controller - USB First, WiFi Fallback
=================================================

Purpose
-------
This Python program controls your ESP32 motor firmware and receives motor
status messages.

Connection order:
  1. Try USB Serial first.
  2. Ask ESP32 for WiFi IP using @IP and cache it.
  3. If USB Serial is not detected or connection fails, use cached WiFi TCP.
  4. If no cached IP exists, try esp32motor.local.
  5. The program never asks the user to type the ESP32 IP address.

Supported commands from this program:
  - Request all motor status
  - Request one motor status
  - Increase speed
  - Decrease speed
  - Stop one motor
  - Stop all motors
  - Move CW / CCW
  - Set angle/speed/torque
  - Enable/disable periodic status
  - Receive live MOTOR_STATUS messages from ESP32
  - Request and cache ESP32 WiFi IP address with @IP
  - Reconnect to the ESP32 WiFi server when USB is down

Required Python packages:
  python -m pip install pyserial

WiFi requirement on ESP32:
  USB Serial works with the merged firmware directly.
  WiFi fallback requires the ESP32 firmware to run a TCP command/status server
  that accepts the same text commands, for example on port 3333:

      @STATUS_ALL\n
  and sends status lines like:

      MOTOR_STATUS,index=0,type=0,gpio=4,id=1,angle=90.00,speed=50.00,torque=0.00,direction=0,enabled=1

Run examples:
  python esp32_motor_usb_first_wifi_fallback_controller.py
  python esp32_motor_usb_first_wifi_fallback_controller.py --port COM5
  python esp32_motor_no_manual_ip.py
  python esp32_motor_no_manual_ip.py --port COM5
  python esp32_motor_no_manual_ip.py --wifi-only
"""

from __future__ import annotations

# ============================================================
# Automatic Python package installer
# ============================================================
# This block runs before the ESP32 controller imports optional packages.
# It makes the script easier to run on a new Windows/VS Code/CMD setup.
#
# Required packages:
#   pyserial  -> USB Serial / COM port communication with ESP32
#   colorama  -> optional colored console output support
#   requests  -> optional future HTTP/REST support
#   zeroconf  -> optional mDNS discovery support, e.g. esp32motor.local
# ============================================================
import importlib.util
import subprocess
import sys

REQUIRED_PACKAGES = {
    "pyserial": "serial",
    "colorama": "colorama",
    "requests": "requests",
    "zeroconf": "zeroconf",
}


def ensure_required_packages() -> None:
    """Install missing Python packages automatically before the controller starts."""
    print("=" * 60)
    print("Checking required Python packages...")
    print("=" * 60)

    for package_name, import_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(import_name) is not None:
            print(f"[OK] {package_name} already installed")
            continue

        print(f"[INSTALL] {package_name} is missing. Installing now...")
        try:
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                package_name,
            ])
            print(f"[OK] Installed {package_name}")
        except Exception as exc:
            print(f"[WARN] Could not install {package_name}: {exc}")
            print("       You can install it manually with:")
            print(f"       python -m pip install --upgrade {package_name}")

    print("=" * 60)


ensure_required_packages()


import argparse
import json
import os
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Protocol

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None  # type: ignore

BAUD_RATE = 115200
ESP32_BOOT_DELAY_SECONDS = 2.0
DEFAULT_WIFI_PORT = 3333
SOCKET_TIMEOUT_SECONDS = 0.5
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esp32_wifi_cache.json")
IP_RESPONSE_TIMEOUT_SECONDS = 4.0

ESP32_USB_KEYWORDS = [
    "esp32",
    "espressif",
    "cp210",
    "cp210x",
    "silicon labs",
    "ch340",
    "wch",
    "usb serial",
    "ftdi",
    "uart",
]


@dataclass
class MotorStatus:
    """Latest status known for one motor."""

    index: int
    motor_type: Optional[int] = None
    gpio: Optional[int] = None
    motor_id: Optional[int] = None
    angle: Optional[float] = None
    speed: Optional[float] = None
    torque: Optional[float] = None
    direction: Optional[int] = None
    enabled: Optional[int] = None
    last_update: str = field(default_factory=lambda: "")


class Transport(Protocol):
    """Common interface for USB and WiFi connections."""

    name: str

    def connect(self) -> None: ...
    def close(self) -> None: ...
    def write_bytes(self, data: bytes) -> None: ...
    def readline(self) -> bytes: ...


class USBSerialTransport:
    """USB Serial transport for ESP32."""

    def __init__(self, port: str, baud_rate: int = BAUD_RATE) -> None:
        if serial is None:
            raise RuntimeError("pyserial is not installed. Run: python -m pip install pyserial")
        self.port = port
        self.baud_rate = baud_rate
        self.name = f"USB Serial {port}"
        self.ser: Optional[serial.Serial] = None

    def connect(self) -> None:
        print(f"Opening USB Serial {self.port} at {self.baud_rate} baud...")
        self.ser = serial.Serial(self.port, self.baud_rate, timeout=0.5)
        print("Waiting for ESP32 boot/reset...")
        time.sleep(ESP32_BOOT_DELAY_SECONDS)

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def write_bytes(self, data: bytes) -> None:
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("USB Serial is not open.")
        self.ser.write(data)
        self.ser.flush()

    def readline(self) -> bytes:
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("USB Serial is not open.")
        return self.ser.readline()


class WiFiTCPTransport:
    """WiFi TCP transport for ESP32."""

    def __init__(self, ip: str, port: int = DEFAULT_WIFI_PORT) -> None:
        self.ip = ip
        self.port = port
        self.name = f"WiFi TCP {ip}:{port}"
        self.sock: Optional[socket.socket] = None
        self.rx_buffer = b""

    def connect(self) -> None:
        print(f"Opening WiFi TCP connection to {self.ip}:{self.port}...")
        self.sock = socket.create_connection((self.ip, self.port), timeout=5.0)
        self.sock.settimeout(SOCKET_TIMEOUT_SECONDS)

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def write_bytes(self, data: bytes) -> None:
        if not self.sock:
            raise RuntimeError("WiFi socket is not open.")
        self.sock.sendall(data)

    def readline(self) -> bytes:
        if not self.sock:
            raise RuntimeError("WiFi socket is not open.")

        while b"\n" not in self.rx_buffer:
            try:
                chunk = self.sock.recv(1024)
                if not chunk:
                    raise ConnectionError("WiFi socket closed by ESP32.")
                self.rx_buffer += chunk
            except socket.timeout:
                return b""

        line, self.rx_buffer = self.rx_buffer.split(b"\n", 1)
        return line + b"\n"


class ESP32MotorController:
    """Controller for ESP32 motor firmware over USB Serial or WiFi TCP."""

    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        self.stop_reader = threading.Event()
        self.reader: Optional[threading.Thread] = None
        self.motor_status: Dict[int, MotorStatus] = {}
        self.connected = False
        self.last_wifi_ip: Optional[str] = None
        self.last_wifi_port: int = DEFAULT_WIFI_PORT
        self.wifi_response_active = False

    def connect(self) -> None:
        self.transport.connect()
        self.connected = True
        self.stop_reader.clear()
        self.reader = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader.start()
        print(f"Connected using {self.transport.name}.")
        self.status_all()

    def close(self) -> None:
        self.stop_reader.set()
        time.sleep(0.2)
        self.transport.close()
        self.connected = False
        print("Connection closed.")

    def send_line_command(self, command: str) -> None:
        command = command.strip()
        if not command:
            return
        if not command.startswith("@"):
            command = "@" + command
        self.transport.write_bytes((command + "\n").encode("utf-8"))
        print(f"[PY -> ESP32] {command}")

    def send_single_key(self, key: str) -> None:
        if len(key) != 1:
            print("ERROR: single-key command must be exactly one character.")
            return
        self.transport.write_bytes(key.encode("utf-8"))
        print(f"[PY -> ESP32] single-key: {key}")

    def select_motor(self, motor_index: int) -> None:
        if motor_index < 0 or motor_index > 15:
            print("ERROR: Motor index must be between 0 and 15.")
            return
        if motor_index <= 9:
            self.send_single_key(str(motor_index))
        else:
            self.send_single_key(chr(ord("A") + (motor_index - 10)))
        time.sleep(0.05)

    def increase_speed(self, motor_index: int) -> None:
        self.select_motor(motor_index)
        self.send_single_key("q")
        time.sleep(0.05)
        self.status(motor_index)

    def decrease_speed(self, motor_index: int) -> None:
        self.select_motor(motor_index)
        self.send_single_key("a")
        time.sleep(0.05)
        self.status(motor_index)

    def stop_motor(self, motor_index: int) -> None:
        self.send_line_command(f"STOP {motor_index}")

    def stop_all(self) -> None:
        self.send_line_command("STOP_ALL")

    def status(self, motor_index: int) -> None:
        self.send_line_command(f"STATUS {motor_index}")

    def status_all(self) -> None:
        self.send_line_command("STATUS_ALL")

    def periodic_on(self) -> None:
        self.send_line_command("PERIODIC_ON")

    def periodic_off(self) -> None:
        self.send_line_command("PERIODIC_OFF")

    def move_cw(self, motor_index: int, speed: float, torque: float = 0.0) -> None:
        self.send_line_command(f"CW {motor_index} {speed} {torque}")
        time.sleep(0.05)
        self.status(motor_index)

    def move_ccw(self, motor_index: int, speed: float, torque: float = 0.0) -> None:
        self.send_line_command(f"CCW {motor_index} {speed} {torque}")
        time.sleep(0.05)
        self.status(motor_index)

    def set_motor(self, motor_index: int, angle: float, speed: float, torque: float = 0.0) -> None:
        self.send_line_command(f"SET {motor_index} {angle} {speed} {torque}")
        time.sleep(0.05)
        self.status(motor_index)


    def request_wifi_ip(self, timeout_seconds: float = IP_RESPONSE_TIMEOUT_SECONDS) -> Optional[str]:
        """
        Ask the ESP32 firmware for its WiFi IP address.

        This sends @IP. The ESP32 firmware should reply with lines like:
            WIFI_STATUS_BEGIN
            WIFI_CONNECTED=YES
            IP_ADDRESS=192.168.1.55
            PORT=3333
            WIFI_STATUS_END

        The reader thread parses IP_ADDRESS and PORT automatically.
        """
        self.last_wifi_ip = None
        self.wifi_response_active = True
        self.send_line_command("IP")

        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if self.last_wifi_ip:
                save_wifi_cache(self.last_wifi_ip, self.last_wifi_port)
                print(f"Cached ESP32 WiFi server: {self.last_wifi_ip}:{self.last_wifi_port}")
                self.wifi_response_active = False
                return self.last_wifi_ip
            time.sleep(0.05)

        self.wifi_response_active = False
        print("No IP_ADDRESS response received from ESP32.")
        return None

    def reconnect_using_cached_wifi(self) -> bool:
        """Close the current transport and reconnect to the ESP32 WiFi server using cached IP."""
        cached = load_wifi_cache()
        if not cached:
            print("No cached WiFi IP found. Connect by USB first and use option 14 to request IP.")
            return False

        ip = cached.get("ip")
        port = int(cached.get("port", DEFAULT_WIFI_PORT))
        if not ip:
            print("Cached WiFi file exists but does not contain an IP address.")
            return False

        print(f"Switching to ESP32 WiFi server {ip}:{port}...")
        try:
            self.stop_reader.set()
            time.sleep(0.2)
            self.transport.close()

            self.transport = WiFiTCPTransport(ip=ip, port=port)
            self.transport.connect()
            self.stop_reader.clear()
            self.connected = True
            self.reader = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader.start()
            print(f"Connected using {self.transport.name}.")
            self.status_all()
            return True
        except Exception as exc:
            self.connected = False
            print(f"WiFi reconnect failed: {exc}")
            return False


    def _reader_loop(self) -> None:
        while not self.stop_reader.is_set():
            try:
                raw = self.transport.readline()
                if not raw:
                    continue
                line = raw.decode(errors="replace").strip()
                if line:
                    self._handle_esp32_line(line)
            except Exception as exc:
                print(f"\nCONNECTION READ ERROR: {exc}")
                self.stop_reader.set()
                self.connected = False
                break

    def _handle_esp32_line(self, line: str) -> None:
        # Parse WiFi/IP response from ESP32.
        # The firmware should answer @IP with lines such as:
        #   IP_ADDRESS=192.168.1.55
        #   PORT=3333
        if line.startswith("IP_ADDRESS="):
            ip = line.split("=", 1)[1].strip()
            if ip and ip != "0.0.0.0":
                self.last_wifi_ip = ip
                save_wifi_cache(self.last_wifi_ip, self.last_wifi_port)
                print(f"[ESP32 WIFI] IP_ADDRESS={ip}")
            else:
                print(f"[ESP32 WIFI] IP address not available: {ip}")
            return

        if line.startswith("PORT="):
            port_text = line.split("=", 1)[1].strip()
            try:
                self.last_wifi_port = int(port_text)
                if self.last_wifi_ip:
                    save_wifi_cache(self.last_wifi_ip, self.last_wifi_port)
                print(f"[ESP32 WIFI] PORT={self.last_wifi_port}")
            except ValueError:
                print(f"[ESP32 WIFI] Invalid PORT response: {line}")
            return

        if line.startswith("MOTOR_STATUS,"):
            status = self._parse_motor_status(line)
            if status:
                self.motor_status[status.index] = status
                self._print_motor_status(status)
            return
        print(f"[ESP32] {line}")

    def _parse_motor_status(self, line: str) -> Optional[MotorStatus]:
        values: Dict[str, str] = {}
        for item in line.split(",")[1:]:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            values[key.strip()] = value.strip()
        try:
            index = int(values.get("index", "-1"))
            status = MotorStatus(index=index)
            status.motor_type = _to_int(values.get("type"))
            status.gpio = _to_int(values.get("gpio"))
            status.motor_id = _to_int(values.get("id"))
            status.angle = _to_float(values.get("angle"))
            status.speed = _to_float(values.get("speed"))
            status.torque = _to_float(values.get("torque"))
            status.direction = _to_int(values.get("direction"))
            status.enabled = _to_int(values.get("enabled"))
            status.last_update = datetime.now().strftime("%H:%M:%S")
            return status
        except Exception:
            print(f"[WARN] Could not parse MOTOR_STATUS line: {line}")
            return None

    def _print_motor_status(self, status: MotorStatus) -> None:
        direction_name = {0: "STOP", 1: "CW", 2: "CCW"}.get(status.direction, str(status.direction))
        print(
            f"[MOTOR {status.index}] "
            f"speed={status.speed} angle={status.angle} torque={status.torque} "
            f"direction={direction_name} gpio={status.gpio} enabled={status.enabled} "
            f"time={status.last_update}"
        )

    def print_cached_status(self) -> None:
        if not self.motor_status:
            print("No cached status yet. Send STATUS_ALL first.")
            return
        print("\nLatest cached motor status:")
        print("index | speed | angle | torque | direction | gpio | enabled | time")
        print("------|-------|-------|--------|-----------|------|---------|------")
        for index in sorted(self.motor_status):
            s = self.motor_status[index]
            direction_name = {0: "STOP", 1: "CW", 2: "CCW"}.get(s.direction, str(s.direction))
            print(
                f"{s.index:>5} | {str(s.speed):>5} | {str(s.angle):>5} | "
                f"{str(s.torque):>6} | {direction_name:>9} | {str(s.gpio):>4} | "
                f"{str(s.enabled):>7} | {s.last_update}"
            )
        print()


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None




def load_wifi_cache() -> Optional[Dict[str, object]]:
    """Load the last known ESP32 WiFi IP/port from disk."""
    try:
        if not os.path.exists(CACHE_FILE):
            return None
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if not data.get("ip"):
            return None
        return data
    except Exception:
        return None


def save_wifi_cache(ip: str, port: int = DEFAULT_WIFI_PORT) -> None:
    """Save the last known ESP32 WiFi IP/port to disk."""
    data = {
        "ip": ip,
        "port": int(port),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        print(f"[WARN] Could not save WiFi cache: {exc}")


def discover_esp32_by_cached_wifi(port: int = DEFAULT_WIFI_PORT) -> Optional[WiFiTCPTransport]:
    """
    Try the cached WiFi IP first.

    This is used when USB is down. It lets the Python program connect to the ESP32
    WiFi server without asking the user for the IP every time.
    """
    cached = load_wifi_cache()
    if not cached:
        return None

    ip = str(cached.get("ip", "")).strip()
    cached_port = int(cached.get("port", port))
    if not ip:
        return None

    print(f"Trying cached ESP32 WiFi server {ip}:{cached_port}...")
    transport = WiFiTCPTransport(ip=ip, port=cached_port)
    try:
        transport.connect()
        transport._already_connected = True  # type: ignore[attr-defined]
        return transport
    except Exception as exc:
        print(f"Cached WiFi connection failed: {exc}")
        return None


def discover_esp32_by_hostname(port: int = DEFAULT_WIFI_PORT, hostname: str = "esp32motor.local") -> Optional[WiFiTCPTransport]:
    """
    Try the ESP32 mDNS hostname without asking the user for an IP address.

    This requires the ESP32 firmware to start mDNS with:
        MDNS.begin("esp32motor");

    If mDNS works on Windows, the hostname esp32motor.local resolves to the
    current ESP32 IP address. If it fails, the program prints a clear message
    and asks the user to connect once by USB and request/cache the IP.
    """
    print(f"Trying ESP32 hostname {hostname}:{port}...")
    transport = WiFiTCPTransport(ip=hostname, port=port)
    try:
        transport.connect()
        transport._already_connected = True  # type: ignore[attr-defined]
        return transport
    except Exception as exc:
        print(f"Hostname WiFi connection failed: {exc}")
        return None


def get_serial_ports() -> List[object]:
    if serial is None:
        return []
    return list(serial.tools.list_ports.comports())


def find_esp32_usb_port() -> Optional[str]:
    ports = get_serial_ports()
    if not ports:
        return None

    scored = []
    for p in ports:
        text = f"{p.device} {p.description} {getattr(p, 'manufacturer', '')} {getattr(p, 'hwid', '')}".lower()
        score = sum(1 for keyword in ESP32_USB_KEYWORDS if keyword in text)
        scored.append((score, p))

    scored.sort(key=lambda item: item[0], reverse=True)

    if scored and scored[0][0] > 0:
        best = scored[0][1]
        print(f"Auto-detected likely ESP32 USB port: {best.device} ({best.description})")
        return best.device

    if len(ports) == 1:
        only = ports[0]
        print(f"Only one serial port found. Using: {only.device} ({only.description})")
        return only.device

    print("Serial ports found, but no clear ESP32 match:")
    for i, p in enumerate(ports, start=1):
        print(f"  [{i}] {p.device:8} {p.description}")
    choice = input("Select USB port number, or press Enter to skip USB and use WiFi: ").strip()
    if not choice:
        return None
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(ports):
            return ports[idx].device
    except ValueError:
        pass
    print("Invalid USB selection. Skipping USB.")
    return None


def choose_transport(args: argparse.Namespace) -> Optional[Transport]:
    """
    Try USB first. If USB is not available, use WiFi fallback automatically.

    IMPORTANT:
    This version never asks the user to manually type the ESP32 IP address.
    The IP must come from one of these automatic methods:
      1. USB connection where Python sends @IP and caches IP_ADDRESS.
      2. Existing esp32_wifi_cache.json from a previous USB @IP request.
      3. mDNS hostname esp32motor.local.
    """

    if not args.wifi_only:
        port = args.port or find_esp32_usb_port()
        if port:
            try:
                usb = USBSerialTransport(port=port, baud_rate=args.baud)
                usb.connect()
                # We already connected successfully; return an already-open transport.
                # The controller will start its reader thread without reconnecting.
                usb._already_connected = True  # type: ignore[attr-defined]
                return usb
            except Exception as exc:
                print(f"USB connection failed: {exc}")
                print("Falling back to WiFi automatically...")

    # USB is not available. Try cached WiFi IP automatically.
    cached_transport = discover_esp32_by_cached_wifi(args.wifi_port)
    if cached_transport:
        return cached_transport

    # Try mDNS hostname automatically. No manual IP entry.
    hostname_transport = discover_esp32_by_hostname(args.wifi_port, args.hostname)
    if hostname_transport:
        return hostname_transport

    print("No USB connection and no automatic WiFi address was found.")
    print("The program will not ask for an IP address manually.")
    print("To cache the IP automatically:")
    print("  1. Connect the ESP32 by USB.")
    print("  2. Run this program.")
    print("  3. Select option 14: Request ESP32 WiFi IP address and cache it.")
    print("After that, WiFi fallback will work without typing the IP.")
    return None

def connect_controller(transport: Transport) -> ESP32MotorController:
    controller = ESP32MotorController(transport)
    if getattr(transport, "_already_connected", False):
        # Start controller reader without reconnecting the already-open USB port.
        controller.connected = True
        controller.stop_reader.clear()
        controller.reader = threading.Thread(target=controller._reader_loop, daemon=True)
        controller.reader.start()
        print(f"Connected using {transport.name}.")
        controller.status_all()
    else:
        controller.connect()
    return controller


def ask_motor_index() -> Optional[int]:
    text = input("Motor index 0-15: ").strip()
    if not text:
        return None
    try:
        motor_index = int(text)
    except ValueError:
        print("Invalid motor index.")
        return None
    if motor_index < 0 or motor_index > 15:
        print("Motor index must be between 0 and 15.")
        return None
    return motor_index


def ask_float(label: str, default: float) -> float:
    text = input(f"{label} [{default}]: ").strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        print(f"Invalid value. Using default {default}.")
        return default


def print_menu(controller: ESP32MotorController) -> None:
    print("\n" + "=" * 74)
    print(f" ESP32 MOTOR CONTROLLER - connection: {controller.transport.name}")
    print("=" * 74)
    print("1  - Request all motor status")
    print("2  - Request one motor status")
    print("3  - Increase speed of one motor")
    print("4  - Decrease speed of one motor")
    print("5  - Stop one motor")
    print("6  - Stop all motors")
    print("7  - Move motor clockwise with speed")
    print("8  - Move motor counter-clockwise with speed")
    print("9  - Set motor angle/speed/torque")
    print("10 - Enable periodic ESP32 status messages")
    print("11 - Disable periodic ESP32 status messages")
    print("12 - Show Python cached latest status table")
    print("13 - Send raw command to ESP32")
    print("14 - Request ESP32 WiFi IP address and cache it")
    print("15 - Switch/reconnect using cached WiFi IP")
    print("0  - Exit")
    print("=" * 74)


def interactive_menu(controller: ESP32MotorController) -> None:
    while True:
        print_menu(controller)
        choice = input("Select option: ").strip().lower()

        if choice in {"0", "q", "quit", "exit"}:
            break
        if choice == "1":
            controller.status_all()
        elif choice == "2":
            motor = ask_motor_index()
            if motor is not None:
                controller.status(motor)
        elif choice == "3":
            motor = ask_motor_index()
            if motor is not None:
                controller.increase_speed(motor)
        elif choice == "4":
            motor = ask_motor_index()
            if motor is not None:
                controller.decrease_speed(motor)
        elif choice == "5":
            motor = ask_motor_index()
            if motor is not None:
                controller.stop_motor(motor)
        elif choice == "6":
            controller.stop_all()
        elif choice == "7":
            motor = ask_motor_index()
            if motor is not None:
                speed = ask_float("Speed", 50.0)
                torque = ask_float("Torque", 0.0)
                controller.move_cw(motor, speed, torque)
        elif choice == "8":
            motor = ask_motor_index()
            if motor is not None:
                speed = ask_float("Speed", 50.0)
                torque = ask_float("Torque", 0.0)
                controller.move_ccw(motor, speed, torque)
        elif choice == "9":
            motor = ask_motor_index()
            if motor is not None:
                angle = ask_float("Angle", 90.0)
                speed = ask_float("Speed", 50.0)
                torque = ask_float("Torque", 0.0)
                controller.set_motor(motor, angle, speed, torque)
        elif choice == "10":
            controller.periodic_on()
        elif choice == "11":
            controller.periodic_off()
        elif choice == "12":
            controller.print_cached_status()
        elif choice == "13":
            print("Examples: STATUS_ALL, STATUS 0, STOP 0, STOP_ALL, CW 0 50 0, IP")
            print("You can also send original one-key commands by typing: key:q or key:a")
            raw = input("Raw command: ").strip()
            if raw.startswith("key:") and len(raw) == 5:
                controller.send_single_key(raw[-1])
            else:
                controller.send_line_command(raw)
        elif choice == "14":
            controller.request_wifi_ip()
        elif choice == "15":
            controller.reconnect_using_cached_wifi()
        else:
            print("Invalid option.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ESP32 motor controller with USB-first and WiFi fallback.")
    parser.add_argument("--port", help="USB COM port, example COM5. If omitted, auto-detect is used.")
    parser.add_argument("--baud", type=int, default=BAUD_RATE, help=f"USB baud rate. Default: {BAUD_RATE}")
    parser.add_argument("--wifi-port", type=int, default=DEFAULT_WIFI_PORT, help=f"ESP32 TCP port. Default: {DEFAULT_WIFI_PORT}")
    parser.add_argument("--hostname", default="esp32motor.local", help="ESP32 mDNS hostname. Default: esp32motor.local")
    parser.add_argument("--wifi-only", action="store_true", help="Skip USB and connect using WiFi only.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if serial is None and not args.wifi_only:
        print("pyserial is not installed, so USB cannot be used.")
        print("Install it with: python -m pip install pyserial")
        print("Trying WiFi fallback...")

    transport = choose_transport(args)
    if transport is None:
        return

    controller: Optional[ESP32MotorController] = None
    try:
        controller = connect_controller(transport)
        interactive_menu(controller)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt.")
    except Exception as exc:
        print(f"ERROR: {exc}")
    finally:
        if controller:
            controller.close()
        else:
            transport.close()


if __name__ == "__main__":
    main()
