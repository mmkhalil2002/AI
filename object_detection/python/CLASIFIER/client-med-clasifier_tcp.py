# ============================================================
# tcp_classifier_client_raw_rgb.py
# ============================================================
# TCP CLIENT FOR IMAGE CLASSIFIER SERVER
#
# PURPOSE
# ------------------------------------------------------------
# This program sends images to the TCP classifier server.
#
# The client:
#   1) Reads .env global variables if available
#   2) Gets TEST_IMAGE_DIR from .env
#   3) If TEST_IMAGE_DIR is not defined, uses:
#
#          os.path.join(MODEL_PATH, "data", "CLASIFIER_TEST")
#
#   4) Prompts for TCP server IP address
#   5) If IP is not provided, uses the current machine IP address
#   6) Sends each image using this protocol:
#
#          1 byte  width
#          1 byte  height
#          1 byte  channels
#          raw RGB image bytes
#
#   7) Receives the classifier result from the server
#   8) Displays the classification result on the screen
#
# IMPORTANT PROTOCOL NOTE
# ------------------------------------------------------------
# Width and height are sent as 1 byte each.
# Therefore maximum image width/height is 255.
#
# Examples:
#   32x32 RGB  -> 1 byte width=32, 1 byte height=32, channels=3
#   64x64 RGB  -> 1 byte width=64, 1 byte height=64, channels=3
#
# The server can calculate image size as:
#
#   image_size = width * height * channels
#
# ============================================================


# ============================================================
# AUTO-INSTALL DEPENDENCIES
# ============================================================

import sys
import subprocess
import importlib


def _pip_install(pkgs):
    """
    Install packages into the SAME Python interpreter running this script.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *pkgs])


def _ensure_import(import_name, pip_name=None):
    """
    Try import. If missing, install, then import again.
    """
    try:
        importlib.import_module(import_name)
    except Exception:
        _pip_install([pip_name or import_name])
        importlib.import_module(import_name)


def ensure_deps_for_this_script():
    """
    Ensure all required packages are available.
    """
    _ensure_import("dotenv", "python-dotenv")
    _ensure_import("PIL", "pillow")
    _ensure_import("dotenv", "python-dotenv")


ensure_deps_for_this_script()


# ============================================================
# NORMAL IMPORTS
# ============================================================

import os
import socket
import struct
import json
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv, find_dotenv
from PIL import Image


# ============================================================
# LOAD .env
# ============================================================

load_dotenv(find_dotenv())


# ============================================================
# ENVIRONMENT HELPERS
# ============================================================

def get_str(name, default=""):
    """
    Read a string variable from .env / environment.
    """
    return os.getenv(name, default)


def get_int(name, default=0):
    """
    Read an integer variable from .env / environment.
    """
    try:
        return int(os.getenv(name, default))
    except Exception:
        return int(default)


def get_float(name, default=0.0):
    """
    Read a float variable from .env / environment.
    """
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def get_bool(name, default="False"):
    """
    Read a boolean variable from .env / environment.
    """
    return str(os.getenv(name, default)).lower() in ("true", "1", "yes", "on")


# ============================================================
# GLOBAL CONFIG
# ============================================================

MODEL_PATH = get_str("MODEL_BASE", "../../../../")

TEST_IMAGE_DIR = os.path.expandvars(
    get_str(
        "TEST_IMAGE_DIR",
        os.path.join(MODEL_PATH, "data", "CLASIFIER_TEST")
    )
)

TCP_PORT = get_int("TCP_PORT", 5055)
TCP_CONNECT_TIMEOUT_SEC = get_float("TCP_CONNECT_TIMEOUT_SEC", 10.0)
TCP_RECV_TIMEOUT_SEC = get_float("TCP_RECV_TIMEOUT_SEC", 60.0)

# If RESIZE_IMAGE_BEFORE_SEND is True, the client resizes every image before
# sending it to the server. This is useful when the server expects 32x32 or 64x64.
RESIZE_IMAGE_BEFORE_SEND = get_bool("RESIZE_IMAGE_BEFORE_SEND", "False")
SEND_IMAGE_WIDTH = get_int("SEND_IMAGE_WIDTH", 32)
SEND_IMAGE_HEIGHT = get_int("SEND_IMAGE_HEIGHT", 32)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# IP ADDRESS HELPER
# ============================================================

def get_current_machine_ip() -> str:
    """
    Try to get the current machine LAN IP address.
    If this fails, return 127.0.0.1.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ============================================================
# IMAGE DIRECTORY HELPER
# ============================================================

def list_images_in_dir(root_dir: str) -> List[str]:
    """
    Return all valid image file paths sorted alphabetically.
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Input directory not found: {root_dir}")

    paths = []

    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)

        if not os.path.isfile(p):
            continue

        ext = os.path.splitext(name)[1].lower()
        if ext in ALLOWED_EXTS:
            paths.append(p)

    paths.sort()
    return paths


# ============================================================
# IMAGE CONVERSION HELPER
# ============================================================

def image_to_raw_rgb(image_path: str) -> Tuple[int, int, int, bytes]:
    """
    Open image file and convert it to raw RGB bytes.

    Return:
        width, height, channels, raw_rgb_bytes

    Protocol restriction:
        width  must be 1..255
        height must be 1..255
        channels is 3 for RGB
    """
    img = Image.open(image_path).convert("RGB")

    if RESIZE_IMAGE_BEFORE_SEND:
        img = img.resize((SEND_IMAGE_WIDTH, SEND_IMAGE_HEIGHT), Image.BILINEAR)

    width, height = img.size
    channels = 3

    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size.")

    if width > 255 or height > 255:
        raise ValueError(
            f"Image too large for 1-byte width/height protocol: {width}x{height}. "
            f"Set RESIZE_IMAGE_BEFORE_SEND=True in .env or use smaller images."
        )

    raw_bytes = img.tobytes()

    expected_size = width * height * channels
    if len(raw_bytes) != expected_size:
        raise ValueError(
            f"Raw image size mismatch. Expected {expected_size}, got {len(raw_bytes)}"
        )

    return width, height, channels, raw_bytes


# ============================================================
# TCP RECEIVE HELPER
# ============================================================

def recv_exact(sock_obj: socket.socket, nbytes: int) -> bytes:
    """
    Receive exactly nbytes from the TCP socket.

    Raises ConnectionError if the server disconnects early.
    """
    chunks = []
    remaining = nbytes

    while remaining > 0:
        chunk = sock_obj.recv(min(65536, remaining))

        if not chunk:
            raise ConnectionError("Server disconnected before all bytes were received.")

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


# ============================================================
# TCP CLIENT ROUTINE
# ============================================================

def send_raw_rgb_image_and_receive_result(
    server_ip: str,
    server_port: int,
    image_path: str
) -> Dict[str, Any]:
    """
    Send one raw RGB image to the classifier server.

    Client -> Server:
        1 byte width
        1 byte height
        1 byte channels
        raw RGB bytes

    Server -> Client:
        The recommended response format is:
            4 bytes unsigned big-endian JSON size
            JSON bytes

    This client also supports an older 8-byte JSON-size response if needed
    by changing SERVER_RESPONSE_LENGTH_BYTES in .env.
    """
    width, height, channels, raw_bytes = image_to_raw_rgb(image_path)

    response_length_bytes = get_int("SERVER_RESPONSE_LENGTH_BYTES", 4)
    if response_length_bytes not in (2, 4, 8):
        raise ValueError("SERVER_RESPONSE_LENGTH_BYTES must be 2, 4, or 8")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(float(TCP_CONNECT_TIMEOUT_SEC))
        s.connect((server_ip, int(server_port)))

        # After connecting, allow more time for classifier inference.
        s.settimeout(float(TCP_RECV_TIMEOUT_SEC))

        # Send compact raw RGB header.
        s.sendall(bytes([width]))
        s.sendall(bytes([height]))
        s.sendall(bytes([channels]))

        # Send raw RGB payload.
        s.sendall(raw_bytes)

        # Receive JSON response length.
        header = recv_exact(s, response_length_bytes)

        if response_length_bytes == 2:
            response_size = struct.unpack("!H", header)[0]
        elif response_length_bytes == 4:
            response_size = struct.unpack("!I", header)[0]
        else:
            response_size = struct.unpack("!Q", header)[0]

        if response_size <= 0:
            raise ValueError("Server returned an empty response.")

        response_bytes = recv_exact(s, response_size)

    response_text = response_bytes.decode("utf-8", errors="replace")

    try:
        return json.loads(response_text)
    except Exception:
        # If the server returns plain text instead of JSON, still show it.
        return {
            "ok": True,
            "raw_response": response_text
        }


# ============================================================
# RESULT DISPLAY
# ============================================================

def print_result_to_screen(image_path: str, result: Dict[str, Any]):
    """
    Display classification result on the command screen.
    """
    print("------------------------------------------------------------")
    print(f"IMAGE FILE       : {os.path.basename(image_path)}")

    if not result.get("ok", False):
        print("SERVER RESULT    : ERROR")
        print(f"ERROR MESSAGE    : {result.get('error', 'Unknown error')}")
        return

    if "raw_response" in result:
        print("SERVER RESPONSE  :")
        print(result["raw_response"])
        return

    print("SERVER RESULT    : OK")
    print(f"FINAL DETECTION  : {result.get('final_detection', '')}")
    print(f"CONFIDENCE       : {result.get('final_confidence_percent', result.get('confidence', ''))}")
    print(f"WINNING MODEL    : {result.get('winning_model', '')}")

    model_results = result.get("model_results", [])
    if model_results:
        print("MODEL-BY-MODEL RESULTS:")
        for r in model_results:
            print(
                f"  - {str(r.get('model_name', '')):<15} -> "
                f"{str(r.get('detected_class', '')):<30} "
                f"{r.get('confidence_percent', r.get('confidence', ''))}"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main program:
      1) Read .env settings
      2) Ask for server IP
      3) Read images from TEST_IMAGE_DIR
      4) Send each image as raw RGB using 1-byte width/height/channels
      5) Print server classification result
    """
    print("============================================================")
    print("RAW RGB TCP CLASSIFIER CLIENT")
    print("============================================================")
    print(f"MODEL_PATH                : {MODEL_PATH}")
    print(f"TEST_IMAGE_DIR            : {TEST_IMAGE_DIR}")
    print(f"TCP_PORT                  : {TCP_PORT}")
    print(f"RESIZE_IMAGE_BEFORE_SEND  : {RESIZE_IMAGE_BEFORE_SEND}")
    print(f"SEND_IMAGE_WIDTH          : {SEND_IMAGE_WIDTH}")
    print(f"SEND_IMAGE_HEIGHT         : {SEND_IMAGE_HEIGHT}")
    print("============================================================")

    local_ip = get_current_machine_ip()

    print()
    print("Enter TCP server IP address.")
    print(f"Press ENTER to use current machine IP: {local_ip}")
    user_ip = input("Server IP: ").strip()

    server_ip = user_ip if user_ip else local_ip

    print()
    print(f"[CLIENT] Using server: {server_ip}:{TCP_PORT}")
    print()

    image_paths = list_images_in_dir(TEST_IMAGE_DIR)

    if not image_paths:
        print(f"No images found in: {TEST_IMAGE_DIR}")
        return

    print(f"[CLIENT] Images found: {len(image_paths)}")

    ok_count = 0
    error_count = 0

    try:
        for image_path in image_paths:
            try:
                print()
                print(f"[CLIENT] Sending image: {image_path}")

                result = send_raw_rgb_image_and_receive_result(
                    server_ip=server_ip,
                    server_port=TCP_PORT,
                    image_path=image_path
                )

                print_result_to_screen(image_path, result)

                if result.get("ok", False):
                    ok_count += 1
                else:
                    error_count += 1

                cmd = input("Press ENTER for next image or E to exit: ").strip().lower()
                if cmd == "e":
                    break

            except KeyboardInterrupt:
                raise

            except Exception as e:
                error_count += 1
                print("------------------------------------------------------------")
                print(f"IMAGE FILE       : {os.path.basename(image_path)}")
                print("CLIENT ERROR     :", e)

                cmd = input("Press ENTER for next image or E to exit: ").strip().lower()
                if cmd == "e":
                    break

    except KeyboardInterrupt:
        print("\n[CLIENT] Ctrl+C detected. Stopping client...")

    print()
    print("============================================================")
    print("CLIENT SUMMARY")
    print("============================================================")
    print(f"Images processed OK : {ok_count}")
    print(f"Images with errors  : {error_count}")
    print("============================================================")


if __name__ == "__main__":
    main()
