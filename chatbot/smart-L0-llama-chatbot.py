# ==========================================================
# smart_json_llama_chatbot_fixed.py
# ==========================================================
# PURPOSE
# -------
# This script creates a chatbot using:
#
#   1) LLaMA models running through Ollama
#
# The chatbot works in this order:
#
#   User Question
#        ↓
#   Ask LLaMA
#        ↓
#   Print response
#
# ----------------------------------------------------------
# BEFORE RUNNING
# ----------------------------------------------------------
# Install required Python package:
#
#     pip install requests
#
# ----------------------------------------------------------
# RUN
# ----------------------------------------------------------
#
#     python smart_json_llama_chatbot_fixed.py
#
# ----------------------------------------------------------

import os
import sys
import time
import json
import shutil
import subprocess
import requests


# ----------------------------------------------------------
# CONFIGURATION SECTION
# ----------------------------------------------------------

# Fast model used first
SMALL_MODEL = "llama3.2"

# Larger model fallback if the small model fails
LARGE_MODEL = "llama3.1"

# ----------------------------------------------------------
# OLLAMA SERVER ADDRESS
# ----------------------------------------------------------
#
# 127.0.0.1 is the "loopback" address which means:
#
#     THIS COMPUTER
#
# When Ollama starts using:
#
#     ollama serve
#
# it launches a local HTTP API server listening on:
#
#     127.0.0.1:11434
#
# Our Python chatbot communicates with Ollama through this API.
#
# Example request sent by the script:
#
#     http://127.0.0.1:11434/api/tags
#
# which returns the list of installed models.
#
# Using a local address has advantages:
#
#   • very fast communication
#   • no internet required
#   • no API cost
#   • data stays on this machine
#
# "localhost" and "127.0.0.1" are equivalent.
#
# If Ollama runs on another computer, replace the host with
# that machine's IP address.
# ----------------------------------------------------------

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# Timeouts and behavior
REQUEST_TIMEOUT = 300
SERVER_READY_WAIT_SECONDS = 30
KEEP_ALIVE = "10m"

AUTO_PULL_MODELS = True
USE_LARGE_MODEL_FALLBACK = True


# ----------------------------------------------------------
# SYSTEM PROMPTS
# ----------------------------------------------------------

SYSTEM_PROMPT_SMALL = (
    "You are a helpful assistant. "
    "Answer clearly and briefly unless more detail is requested."
)

SYSTEM_PROMPT_LARGE = (
    "You are a knowledgeable assistant. "
    "Provide accurate and clear explanations."
)


# ----------------------------------------------------------
# GENERATION OPTIONS
# ----------------------------------------------------------

SMALL_MODEL_OPTIONS = {
    "temperature": 0.2,
    "num_predict": 200,
    "top_p": 0.9,
}

LARGE_MODEL_OPTIONS = {
    "temperature": 0.2,
    "num_predict": 300,
    "top_p": 0.9,
}


# ----------------------------------------------------------
# HELPER: Print separator line
# ----------------------------------------------------------

def print_line():
    print("=" * 70)


# ----------------------------------------------------------
# FIND OLLAMA EXECUTABLE
# ----------------------------------------------------------

def find_ollama_exe():
    # ----------------------------------------------------------
    # STEP 1 — Try to locate the Ollama executable using PATH
    # ----------------------------------------------------------
    #
    # shutil.which("ollama") searches for the command "ollama"
    # in the system PATH environment variable.
    #
    # PATH contains directories where executable programs live.
    #
    # Example PATH entries on Windows might include:
    #
    #   C:\Windows\System32
    #   C:\Program Files\Ollama
    #   C:\Users\John\AppData\Local\Programs\Python
    #
    # If "ollama.exe" exists in one of these directories,
    # shutil.which() returns the full path to it.
    #
    # Example return value:
    #
    #   "C:\Program Files\Ollama\ollama.exe"
    #
    # If it cannot find the command, it returns None.

    found = shutil.which("ollama")

    # ----------------------------------------------------------
    # STEP 2 — If Ollama was found in PATH, return it immediately
    # ----------------------------------------------------------
    #
    # Example:
    #
    # found = "C:\Program Files\Ollama\ollama.exe"
    #
    # In that case we return this path and stop searching.

    if found:
        return found

    # ----------------------------------------------------------
    # STEP 3 — Get the current Windows username
    # ----------------------------------------------------------
    #
    # Some Ollama installations are placed inside the user's
    # home directory instead of Program Files.
    #
    # os.environ.get("USERNAME") reads the Windows environment
    # variable USERNAME.
    #
    # Example result:
    #
    #   username = "Mohamed"
    #
    # The second parameter "" is a default value returned
    # if USERNAME does not exist.

    username = os.environ.get("USERNAME", "")

    # ----------------------------------------------------------
    # STEP 4 — Define common installation locations
    # ----------------------------------------------------------
    #
    # These are typical folders where Ollama might be installed.
    #
    # Example paths if username = "Mohamed":
    #
    #   C:\Users\Mohamed\AppData\Local\Programs\Ollama\ollama.exe
    #   C:\Users\Mohamed\AppData\Local\Ollama\ollama.exe
    #   C:\Program Files\Ollama\ollama.exe
    #
    # The rf"" string means:
    #
    #   r → raw string (backslashes are not escaped)
    #   f → formatted string (we can insert variables like {username})

    possible_paths = [
        rf"C:\Users\{username}\AppData\Local\Programs\Ollama\ollama.exe",
        rf"C:\Users\{username}\AppData\Local\Ollama\ollama.exe",
        r"C:\Program Files\Ollama\ollama.exe",
    ]

    # ----------------------------------------------------------
    # STEP 5 — Check each possible path
    # ----------------------------------------------------------
    #
    # We loop through each location and verify whether the file exists.
    #
    # os.path.exists(path) returns:
    #
    #   True  → file exists
    #   False → file does not exist
    #
    # Example:
    #
    #   path = "C:\Program Files\Ollama\ollama.exe"
    #
    # If the file exists, we immediately return that path.

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # ----------------------------------------------------------
    # STEP 6 — If Ollama was not found anywhere
    # ----------------------------------------------------------
    #
    # If we reach this point it means:
    #
    #   • not found in PATH
    #   • not found in typical install locations
    #
    # So we return None to indicate failure.
    #
    # The calling function can then decide to install Ollama.

    return None


# ----------------------------------------------------------
# INSTALL OLLAMA IF NECESSARY
# ----------------------------------------------------------

def install_ollama():
    # ----------------------------------------------------------
    # STEP 1 — Print section header
    # ----------------------------------------------------------
    print_line()
    print("Checking whether Ollama is installed...")

    # ----------------------------------------------------------
    # STEP 2 — First try to find Ollama normally
    # ----------------------------------------------------------
    # If Ollama is already installed, we return its path and stop.
    #
    # Example:
    #   C:\Program Files\Ollama\ollama.exe
    #   /usr/bin/ollama
    #   /usr/local/bin/ollama
    #
    ollama_exe = find_ollama_exe()

    if ollama_exe:
        print(f"[OK] Ollama found: {ollama_exe}")
        return ollama_exe

    print("[INFO] Ollama was not found.")
    print("[INFO] Attempting OS-specific installation...")

    # ----------------------------------------------------------
    # STEP 3 — Detect operating system
    # ----------------------------------------------------------
    # sys.platform examples:
    #   "win32"   -> Windows
    #   "linux"   -> Linux
    #   "darwin"  -> macOS
    #
    platform_name = sys.platform
    print(f"[INFO] Detected platform: {platform_name}")

    try:
        # ------------------------------------------------------
        # STEP 4A — Windows installation
        # ------------------------------------------------------
        # The official Windows download page shows a PowerShell
        # install command:
        #
        #   irm https://ollama.com/install.ps1 | iex
        #
        # In this script we first try winget, since many Windows
        # systems already have it installed.
        #
        # If winget is missing or fails, we fall back to the
        # official PowerShell installer command.
        #
        if platform_name.startswith("win"):
            print("[INFO] Windows detected.")

            # Try winget first
            print("[INFO] Trying winget install...")
            result = subprocess.run(
                ["winget", "install", "-e", "--id", "Ollama.Ollama"],
                check=False
            )

            if result.returncode != 0:
                print("[WARNING] winget install failed.")
                print("[INFO] Trying official PowerShell installer...")

                # Official installer command from Ollama download page
                ps_cmd = "irm https://ollama.com/install.ps1 | iex"
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
                    check=False
                )

                if result.returncode != 0:
                    print("[ERROR] Windows installation failed.")
                    print("[ACTION] Please install Ollama manually from https://ollama.com/download/windows")
                    return None

        # ------------------------------------------------------
        # STEP 4B — Linux installation
        # ------------------------------------------------------
        # The official Linux docs show the installer command:
        #
        #   curl -fsSL https://ollama.com/install.sh | sh
        #
        # That is the standard method used here.
        #
        elif platform_name.startswith("linux"):
            print("[INFO] Linux detected.")
            print("[INFO] Running official shell installer...")

            shell_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
            result = subprocess.run(
                ["sh", "-c", shell_cmd],
                check=False
            )

            if result.returncode != 0:
                print("[ERROR] Linux installation failed.")
                print("[ACTION] Please check the Ollama Linux docs or install manually.")
                return None

        # ------------------------------------------------------
        # STEP 4C — macOS installation
        # ------------------------------------------------------
        # The official docs say the preferred macOS installation
        # method is mounting the DMG and dragging Ollama into
        # Applications.
        #
        # Because GUI installation is the preferred method,
        # this script gives guidance instead of forcing an
        # unsupported silent install.
        #
        elif platform_name == "darwin":
            print("[INFO] macOS detected.")
            print("[ACTION] Please install Ollama using the official macOS app installer.")
            print("[ACTION] Preferred method: download the DMG and move Ollama to Applications.")
            print("[ACTION] Download: https://ollama.com/download/mac")
            return None

        # ------------------------------------------------------
        # STEP 4D — Unsupported OS
        # ------------------------------------------------------
        else:
            print(f"[ERROR] Unsupported platform: {platform_name}")
            return None

    except Exception as e:
        print(f"[ERROR] Automatic installation failed: {e}")
        print("[ACTION] Please install Ollama manually from the official download page.")
        return None

    # ----------------------------------------------------------
    # STEP 5 — Wait a few seconds after install
    # ----------------------------------------------------------
    # Some installers need a short delay before the executable
    # becomes visible in PATH or on disk.
    #
    time.sleep(6)

    # ----------------------------------------------------------
    # STEP 6 — Search for Ollama again
    # ----------------------------------------------------------
    # If installation succeeded, find_ollama_exe() should now
    # return the executable path.
    #
    ollama_exe = find_ollama_exe()

    if ollama_exe:
        print(f"[OK] Ollama installed successfully: {ollama_exe}")
        return ollama_exe

    # ----------------------------------------------------------
    # STEP 7 — Final fallback message
    # ----------------------------------------------------------
    print("[ERROR] Ollama may be installed, but Python cannot find it yet.")
    print("[ACTION] Close the terminal, open a new one, and rerun the script.")
    return None


def is_ollama_server_running():
    """
    PURPOSE
    -------
    Check whether the Ollama server is currently running and reachable.

    The Ollama server exposes an HTTP API on a specific port
    (usually 11434). If the server is running, we should be able
    to make a request to one of its endpoints.

    This function attempts to contact the server using a simple
    API call and verifies whether the server responds correctly.

    If the server responds successfully → return True
    If the server is unreachable or errors occur → return False


    ------------------------------------------------------------
    WHAT ENDPOINT IS USED
    ------------------------------------------------------------

    We call:

        /api/tags

    Example full URL:

        http://127.0.0.1:11434/api/tags

    This endpoint returns the list of models installed in Ollama.

    Example response from Ollama:

    {
        "models": [
            {"name": "llama3.2"},
            {"name": "llama3.1"}
        ]
    }

    If we receive a valid HTTP response, we know the server is running.
    """

    try:
        # ---------------------------------------------------------
        # STEP 1 — Send HTTP GET request to Ollama server
        # ---------------------------------------------------------
        #
        # requests.get() performs an HTTP GET request.
        #
        # Example request:
        #
        #   GET http://127.0.0.1:11434/api/tags
        #
        # This asks the Ollama server:
        #
        #   "Please send me the list of available models."
        #
        # If the server is running, it will respond immediately.
        #
        # ---------------------------------------------------------
        # PARAMETERS
        # ---------------------------------------------------------
        #
        # f"{OLLAMA_BASE_URL}/api/tags"
        #
        # Example if:
        #
        #   OLLAMA_BASE_URL = "http://127.0.0.1:11434"
        #
        # Then the final URL becomes:
        #
        #   http://127.0.0.1:11434/api/tags
        #
        # ---------------------------------------------------------
        #
        # timeout=2
        #
        # This means:
        #
        #   "Wait at most 2 seconds for the server to respond."
        #
        # If the server does not respond within 2 seconds,
        # the request will raise a timeout exception.
        #
        # This prevents the program from hanging forever.
        #

        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)

        # ---------------------------------------------------------
        # STEP 2 — Check HTTP response status code
        # ---------------------------------------------------------
        #
        # When a web server responds to a request, it returns
        # a status code indicating the result.
        #
        # Common HTTP status codes:
        #
        #   200 → Success
        #   404 → Page not found
        #   500 → Internal server error
        #   503 → Service unavailable
        #
        # In this case:
        #
        #   r.status_code
        #
        # holds the numeric code returned by the server.
        #
        # Example successful response:
        #
        #   r.status_code = 200
        #
        # This means the Ollama API responded correctly.
        #

        return r.status_code == 200

        # ---------------------------------------------------------
        # WHAT THIS RETURN STATEMENT DOES
        # ---------------------------------------------------------
        #
        # r.status_code == 200
        #
        # is a Boolean expression that evaluates to:
        #
        #   True  → if status_code is 200
        #   False → otherwise
        #
        # So the function returns:
        #
        #   True  → server is reachable and working
        #   False → server responded with an error code
        #

    except Exception:
        # ---------------------------------------------------------
        # STEP 3 — Handle connection errors
        # ---------------------------------------------------------
        #
        # If the server is NOT running, requests.get() may raise
        # an exception such as:
        #
        #   ConnectionError
        #   Timeout
        #   DNS error
        #
        # Example error:
        #
        #   requests.exceptions.ConnectionError:
        #   Failed to establish a new connection
        #
        # The try/except block prevents the program from crashing.
        #
        # Instead, we catch ANY exception and simply return False.
        #
        # This means:
        #
        #   "The server is not running or cannot be reached."
        #

        return False


# ----------------------------------------------------------
# START OLLAMA SERVER
# ----------------------------------------------------------
def start_ollama_server(ollama_exe):
    # ----------------------------------------------------------
    # PURPOSE
    # ----------------------------------------------------------
    # Start the Ollama server if it is not already running.
    #
    # When Ollama starts using:
    #
    #     ollama serve
    #
    # it launches a local HTTP API server that binds to:
    #
    #     127.0.0.1:11434
    #
    # Meaning:
    #
    #     IP address : 127.0.0.1  (this computer only)
    #     Port       : 11434
    #
    # Once bound, the server listens for requests such as:
    #
    #     http://127.0.0.1:11434/api/tags
    #     http://127.0.0.1:11434/api/generate
    #
    # Your Python chatbot communicates with the Ollama server
    # through this local API.
    #
    # ----------------------------------------------------------

    # Print a visual separator in the console
    print_line()

    # ----------------------------------------------------------
    # STEP 1 — Check whether the server is already running
    # ----------------------------------------------------------
    # is_ollama_server_running() sends a request to:
    #
    #     http://127.0.0.1:11434/api/tags
    #
    # If the request succeeds (HTTP 200), the server is already
    # running and bound to the address 127.0.0.1:11434.
    #
    # In that case we do NOT start another server.
    #
    if is_ollama_server_running():
        print("[OK] Ollama server already running.")
        return True

    # ----------------------------------------------------------
    # STEP 2 — Start the Ollama server
    # ----------------------------------------------------------
    print("[INFO] Starting Ollama server...")

    try:
        # ------------------------------------------------------
        # STEP 3 — Configure process creation flags
        # ------------------------------------------------------
        # On Windows we create a new process group so that
        # the Ollama process runs independently of the Python
        # script.
        #
        # This helps avoid termination issues if the script exits.
        #
        creation_flags = 0

        if os.name == "nt":
            # Windows-specific process flag
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

        # ------------------------------------------------------
        # STEP 4 — Launch the Ollama server process
        # ------------------------------------------------------
        # This runs the command:
        #
        #     ollama serve
        #
        # Internally Ollama performs:
        #
        #     socket()
        #     bind(127.0.0.1, 11434)
        #     listen()
        #
        # Which means the server begins listening for requests
        # on the local address:
        #
        #     127.0.0.1:11434
        #
        subprocess.Popen(
            [ollama_exe, "serve"],

            # Suppress standard output so the console stays clean
            stdout=subprocess.DEVNULL,

            # Suppress error output
            stderr=subprocess.DEVNULL,

            # Apply Windows process flags if needed
            creationflags=creation_flags
        )

    except Exception as e:
        # If the server fails to start, report the error
        print(f"[ERROR] Failed to start Ollama server: {e}")
        return False

    # ----------------------------------------------------------
    # STEP 5 — Wait for the server to become ready
    # ----------------------------------------------------------
    # Starting the server may take a few seconds.
    #
    # We repeatedly check if the server responds at:
    #
    #     http://127.0.0.1:11434/api/tags
    #
    # If the server responds successfully, it means the
    # binding to 127.0.0.1:11434 completed and the server
    # is ready to accept requests.
    #
    for _ in range(SERVER_READY_WAIT_SECONDS):

        if is_ollama_server_running():
            print("[OK] Ollama server is ready.")
            return True

        # Wait 1 second before checking again
        time.sleep(1)

    # ----------------------------------------------------------
    # STEP 6 — Server did not start within the expected time
    # ----------------------------------------------------------
    print("[ERROR] Ollama server failed to start.")
    return False


# ----------------------------------------------------------
# CHECK IF MODEL EXISTS
# ----------------------------------------------------------

def model_exists(model_name):
    # ----------------------------------------------------------
    # PURPOSE
    # ----------------------------------------------------------
    # Check whether a given Ollama model is already installed
    # and available on the local Ollama server.
    #
    # Example:
    #   model_name = "llama3.2"
    #
    # If Ollama already has a model whose name begins with
    # "llama3.2", this function returns:
    #
    #   True
    #
    # Otherwise it returns:
    #
    #   False
    #
    # Example use:
    #
    #   if model_exists("llama3.2"):
    #       print("Model is installed")
    #   else:
    #       print("Model is missing")
    # ----------------------------------------------------------

    try:
        # ------------------------------------------------------
        # STEP 1 — Ask Ollama for the list of installed models
        # ------------------------------------------------------
        #
        # requests.get(...) sends an HTTP GET request to:
        #
        #   http://127.0.0.1:11434/api/tags
        #
        # This Ollama API endpoint returns metadata about the
        # models currently installed on the system.
        #
        # Example final URL:
        #
        #   f"{OLLAMA_BASE_URL}/api/tags"
        #   -> "http://127.0.0.1:11434/api/tags"
        #
        # timeout=5 means:
        #
        #   "Wait at most 5 seconds for a response."
        #
        # If the server does not respond in 5 seconds,
        # an exception is raised.
        #
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)

        # ------------------------------------------------------
        # STEP 2 — Verify the HTTP request succeeded
        # ------------------------------------------------------
        #
        # raise_for_status() checks the HTTP status code.
        #
        # If the server returned:
        #   200 -> success, continue
        #
        # If the server returned an error like:
        #   404 -> not found
        #   500 -> server error
        #
        # then raise_for_status() throws an exception and
        # execution jumps to the except block.
        #
        r.raise_for_status()

        # ------------------------------------------------------
        # STEP 3 — Convert JSON response into a Python dictionary
        # ------------------------------------------------------
        #
        # r.json() reads the response body and converts it from
        # JSON text into Python data structures.
        #
        # Example Ollama response:
        #
        # {
        #   "models": [
        #       {"name": "llama3.2:latest"},
        #       {"name": "llama3.1:latest"},
        #       {"name": "mistral:latest"}
        #   ]
        # }
        #
        # After r.json(), data becomes a Python dictionary:
        #
        # data = {
        #   "models": [
        #       {"name": "llama3.2:latest"},
        #       {"name": "llama3.1:latest"},
        #       {"name": "mistral:latest"}
        #   ]
        # }
        #
        data = r.json()

        # ------------------------------------------------------
        # STEP 4 — Loop through all installed models
        # ------------------------------------------------------
        #
        # data.get("models", [])
        #
        # means:
        #   - get the value stored under the key "models"
        #   - if "models" does not exist, use [] instead
        #
        # This prevents a crash if the response is missing
        # the "models" field.
        #
        # Example:
        #
        # data.get("models", []) ->
        #
        # [
        #   {"name": "llama3.2:latest"},
        #   {"name": "llama3.1:latest"}
        # ]
        #
        for model in data.get("models", []):

            # --------------------------------------------------
            # STEP 5 — Extract the model name safely
            # --------------------------------------------------
            #
            # model.get("name", "")
            #
            # means:
            #   - get the "name" field from this model entry
            #   - if "name" does not exist, use "" instead
            #
            # Example:
            #
            # model = {"name": "llama3.2:latest"}
            #
            # name becomes:
            #
            #   "llama3.2:latest"
            #
            name = model.get("name", "")

            # --------------------------------------------------
            # STEP 6 — Check whether the installed model name
            #          begins with the requested model_name
            # --------------------------------------------------
            #
            # startswith(model_name) is used instead of == because
            # Ollama model names often include a tag suffix.
            #
            # Example installed name:
            #   "llama3.2:latest"
            #
            # Example requested name:
            #   "llama3.2"
            #
            # Then:
            #
            #   "llama3.2:latest".startswith("llama3.2")
            #
            # returns:
            #
            #   True
            #
            # Another example:
            #
            # installed name = "mistral:latest"
            # model_name     = "llama3.2"
            #
            # Then:
            #
            #   "mistral:latest".startswith("llama3.2")
            #
            # returns:
            #
            #   False
            #
            if name.startswith(model_name):
                # ------------------------------------------------
                # STEP 7 — Matching model found
                # ------------------------------------------------
                #
                # As soon as we find one installed model that
                # matches the requested prefix, we return True.
                #
                # Example:
                #   requested: "llama3.2"
                #   found:     "llama3.2:latest"
                #
                return True

        # ------------------------------------------------------
        # STEP 8 — No matching model was found
        # ------------------------------------------------------
        #
        # If the loop finishes and no model name matched,
        # return False.
        #
        # Example:
        #   requested model: "llama3.2"
        #   installed models:
        #       "mistral:latest"
        #       "phi3:latest"
        #
        # Result:
        #   False
        #
        return False

    except Exception:
        # ------------------------------------------------------
        # STEP 9 — Handle any error safely
        # ------------------------------------------------------
        #
        # Possible reasons for failure:
        #
        #   • Ollama server is not running
        #   • connection refused
        #   • timeout occurred
        #   • invalid JSON response
        #   • HTTP error code returned
        #
        # In all such cases, return False.
        #
        # Meaning:
        #   "We could not confirm that the model exists."
        #
        return False


# ----------------------------------------------------------
# PULL MODEL IF NEEDED
# ----------------------------------------------------------
def pull_model(ollama_exe, model_name):

    # ----------------------------------------------------------
    # PURPOSE
    # ----------------------------------------------------------
    # Ensure that a specific Ollama model is available locally.
    #
    # If the model already exists → do nothing.
    # If the model does not exist → download it using:
    #
    #     ollama pull <model_name>
    #
    # Example command executed:
    #
    #     ollama pull llama3.2
    #
    # This downloads the model files the first time.
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # STEP 1 — Check whether the model already exists
    # ----------------------------------------------------------
    # model_exists() calls the Ollama API (/api/tags)
    # and verifies if the model is already installed.
    #
    # Example:
    #   model_name = "llama3.2"
    #
    # If installed:
    #   return True immediately.
    #
    if model_exists(model_name):
        print(f"[OK] Model already available: {model_name}")
        return True

    # ----------------------------------------------------------
    # STEP 2 — Check whether automatic downloading is allowed
    # ----------------------------------------------------------
    # AUTO_PULL_MODELS is a configuration flag.
    #
    # If False → do not download automatically.
    # If True  → allow downloading missing models.
    #
    if not AUTO_PULL_MODELS:
        print(f"[WARNING] Model not found: {model_name}")
        return False

    # Inform the user that the model download is starting
    print(f"[INFO] Pulling model: {model_name}")

    # First-time model downloads may take time because
    # model weights can be several GB.
    print("[INFO] This may take time the first time...")

    try:

        # ------------------------------------------------------
        # STEP 3 — Execute the Ollama pull command
        # ------------------------------------------------------
        #
        # subprocess.run() executes external programs.
        #
        # The command executed is equivalent to typing:
        #
        #     ollama pull llama3.2
        #
        # Example list passed to subprocess:
        #
        #     [ollama_exe, "pull", model_name]
        #
        result = subprocess.run(
            [ollama_exe, "pull", model_name],
            check=False
        )

        # ------------------------------------------------------
        # STEP 4 — Check if the command succeeded
        # ------------------------------------------------------
        #
        # returncode values:
        #
        #   0  → success
        #   >0 → failure
        #
        if result.returncode != 0:
            print(f"[ERROR] Failed to pull model: {model_name}")
            return False

        # If the pull command succeeded
        print(f"[OK] Model pulled successfully: {model_name}")
        return True

    except Exception as e:

        # ------------------------------------------------------
        # STEP 5 — Handle errors
        # ------------------------------------------------------
        #
        # Possible errors:
        #
        # • Ollama executable not found
        # • network failure during download
        # • invalid model name
        #
        print(f"[ERROR] Failed while pulling model '{model_name}': {e}")
        return False


# ----------------------------------------------------------
# SHOW PROCESSOR INFO
# ----------------------------------------------------------

def show_processor_info(ollama_exe):
    # ----------------------------------------------------------
    # PURPOSE
    # ----------------------------------------------------------
    # Display information about currently running Ollama models.
    # It executes the command:
    #
    #     ollama ps
    #
    # which shows active models and whether they are using CPU/GPU.
    #
    # Example output:
    #
    # NAME        ID            SIZE    PROCESSOR
    # llama3.2    a1b2c3d4      4.7 GB  GPU
    # ----------------------------------------------------------

    print_line()
    print("[INFO] Current Ollama process info:")

    try:
        # Run the external command:  ollama ps
        # subprocess.run executes system commands from Python.
        result = subprocess.run(
            [ollama_exe, "ps"],   # command: ollama ps

            # Capture the command output instead of printing it directly
            capture_output=True,

            # Return output as text (string) instead of raw bytes
            text=True,

            # Do not raise exception automatically if command fails
            check=False
        )

        # returncode = 0 means the command executed successfully
        # result.stdout contains the command output
        if result.returncode == 0 and result.stdout.strip():

            # Print the process information returned by Ollama
            print(result.stdout)

        else:
            # Happens when no model is currently loaded
            print("[INFO] No process information available yet.")

    except Exception as e:
        # Handle errors such as:
        # - Ollama not installed
        # - wrong executable path
        # - command execution failure
        print(f"[INFO] Could not run 'ollama ps': {e}")


# ----------------------------------------------------------
# ASK MODEL (STREAM OUTPUT)
# ----------------------------------------------------------

def ask_model_stream(model_name, prompt, system_prompt=None, options=None):
    # ------------------------------------------------------------
    # Build the request payload sent to the Ollama API.
    #
    # model      : name of the LLM model to use (e.g., llama3)
    # prompt     : the user prompt or question
    # stream     : True means the response will arrive in chunks
    # keep_alive : keeps the model loaded in memory for reuse
    # ------------------------------------------------------------
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "keep_alive": KEEP_ALIVE
    }

    # ------------------------------------------------------------
    # Optional system prompt.
    # This allows defining model behavior, personality,
    # or instructions (e.g., "You are a helpful assistant").
    # ------------------------------------------------------------
    if system_prompt:
        payload["system"] = system_prompt

    # ------------------------------------------------------------
    # Optional generation parameters such as:
    # temperature, top_p, top_k, num_predict, etc.
    # ------------------------------------------------------------
    if options:
        payload["options"] = options

    # ------------------------------------------------------------
    # Variable to accumulate the full model response as text
    # while it is being streamed token-by-token.
    # ------------------------------------------------------------
    full_text = ""

    try:
        # ------------------------------------------------------------
        # Send HTTP POST request to Ollama generation endpoint.
        #
        # stream=True allows receiving partial responses as they are
        # generated instead of waiting for the full completion.
        # ------------------------------------------------------------
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT
        ) as r:

            # ------------------------------------------------------------
            # Raise an exception if HTTP status code is not successful
            # (e.g., 400, 404, 500).
            # ------------------------------------------------------------
            r.raise_for_status()

            # ------------------------------------------------------------
            # Print a label showing which model produced the output.
            # flush=True ensures immediate display in terminal.
            # ------------------------------------------------------------
            print("\nSOURCE: LLAMA MODEL")
            print(f"{model_name}: ", end="", flush=True)

            # ------------------------------------------------------------
            # Iterate over the streaming response lines returned by
            # the Ollama server.
            #
            # Each line is a JSON object containing partial text
            # generated by the model.
            # ------------------------------------------------------------
            for line in r.iter_lines():

                # Skip empty lines
                if line:
                    try:
                        # Decode bytes to string and parse JSON
                        data = json.loads(line.decode("utf-8"))
                    except Exception:
                        # Skip malformed lines if parsing fails
                        continue

                    # ------------------------------------------------------------
                    # Extract generated text chunk from JSON response.
                    # Ollama sends partial tokens under the key "response".
                    # ------------------------------------------------------------
                    chunk = data.get("response", "")

                    if chunk:
                        # Print chunk immediately to console
                        print(chunk, end="", flush=True)

                        # Append chunk to complete response buffer
                        full_text += chunk

            # Print newline after streaming finishes
            print("\n")

    except Exception as e:
        # ------------------------------------------------------------
        # Handle request failures such as:
        # - network issues
        # - server timeout
        # - API errors
        # ------------------------------------------------------------
        print(f"\n[ERROR] Model request failed: {e}\n")
        return ""

    # ------------------------------------------------------------
    # Return the complete response text once streaming is finished.
    # ------------------------------------------------------------
    return full_text


# ----------------------------------------------------------
# MAIN QUESTION ROUTER
# ----------------------------------------------------------

def answer_user_question(user_text):
    # ------------------------------------------------------------
    # This function sends the user's question to an LLM.
    #
    # Strategy:
    # 1. First try a SMALL LLaMA model (faster and cheaper).
    # 2. If the small model fails or returns an empty response,
    #    optionally fall back to a LARGE model.
    # ------------------------------------------------------------

    # Inform the user that the small model is being queried
    print("[INFO] Asking small LLaMA model...")

    # ------------------------------------------------------------
    # Send the question to the small model using the streaming
    # request function (ask_model_stream).
    #
    # Parameters:
    #   model_name    → which model to use
    #   prompt        → the user question
    #   system_prompt → instructions defining model behavior
    #   options       → model generation settings
    # ------------------------------------------------------------
    response = ask_model_stream(
        model_name=SMALL_MODEL,
        prompt=user_text,
        system_prompt=SYSTEM_PROMPT_SMALL,
        options=SMALL_MODEL_OPTIONS
    )

    # ------------------------------------------------------------
    # Check if the small model returned a valid answer.
    #
    # response.strip() removes whitespace and ensures the response
    # is not empty.
    # ------------------------------------------------------------
    if response and response.strip():
        print("[INFO] Small model returned an answer.")
        return

    # ------------------------------------------------------------
    # If the small model failed to produce a usable answer,
    # optionally try a larger model as a fallback.
    # This is controlled by the USE_LARGE_MODEL_FALLBACK flag.
    # ------------------------------------------------------------
    if USE_LARGE_MODEL_FALLBACK:

        # Inform the user that the small model failed
        print("[INFO] Small model returned no usable answer.")

        # Notify that the system will now query the larger model
        print("[INFO] Asking larger LLaMA model...")

        # --------------------------------------------------------
        # Send the same user question to the larger model.
        # Larger models are typically slower but more capable.
        # --------------------------------------------------------
        response = ask_model_stream(
            model_name=LARGE_MODEL,
            prompt=user_text,
            system_prompt=SYSTEM_PROMPT_LARGE,
            options=LARGE_MODEL_OPTIONS
        )

        # --------------------------------------------------------
        # Check whether the large model produced a valid response.
        # --------------------------------------------------------
        if response and response.strip():
            print("[INFO] Large model returned an answer.")
            return

    # ------------------------------------------------------------
    # If both models fail to return a usable response,
    # print a warning message.
    # ------------------------------------------------------------
    print("[WARNING] No usable answer was returned.")

# ----------------------------------------------------------
# HELP COMMANDS
# ----------------------------------------------------------

def print_help():
    print_line()
    print("Commands:")
    print("  exit        -> quit the program")
    print("  /ps         -> show Ollama processor / GPU / CPU info")
    print_line()


# ----------------------------------------------------------
# MAIN PROGRAM
# ----------------------------------------------------------

def main():
    print_line()
    print("LLaMA Chatbot")
    print_line()

    ollama_exe = install_ollama()

    if not ollama_exe:
        sys.exit(1)

    if not start_ollama_server(ollama_exe):
        sys.exit(1)

    print_line()

    if not pull_model(ollama_exe, SMALL_MODEL):
        print("[ERROR] Could not prepare small model.")
        sys.exit(1)

    if USE_LARGE_MODEL_FALLBACK:
        pull_model(ollama_exe, LARGE_MODEL)

    show_processor_info(ollama_exe)

    print_help()
    print("Chatbot ready.")
    print("Flow: LLaMA direct response")
    print(f"Ollama server: {OLLAMA_BASE_URL}")
    print("Type 'exit' to quit.")
    print_line()

    while True:
        try:
            user_text = input("You: ").strip()
        except KeyboardInterrupt:
            print("\n[INFO] Exiting...")
            break

        if user_text.lower() == "exit":
            break

        if user_text.lower() == "/ps":
            show_processor_info(ollama_exe)
            continue

        if not user_text:
            continue

        answer_user_question(user_text)


# ----------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()