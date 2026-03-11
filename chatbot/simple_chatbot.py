# ==========================================================
# install_and_run_llama_fast.py
# ==========================================================
# PURPOSE:
#   Improved all-in-one Windows Python script to:
#     1) Check whether Ollama is installed
#     2) Install Ollama with winget if missing
#     3) Find ollama.exe even if PATH is not refreshed yet
#     4) Start Ollama server only if it is not already running
#     5) Wait until the server is ready
#     6) Pull a Llama model only if missing
#     7) Run a faster chatbot loop with STREAMING output
#     8) Keep prompts short for better speed
#     9) Optionally show whether the model is using GPU/CPU
#
# SPEED IMPROVEMENTS INCLUDED:
#   - Does NOT restart server if already running
#   - Streams output instead of waiting for full response
#   - Keeps model loaded in memory between prompts
#   - Lets you use a smaller/faster model easily
#   - Uses shorter generation settings for better responsiveness
#   - Optional warm-up request after model pull
#
# BEFORE RUNNING:
#   pip install requests
#
# RUN:
#   python install_and_run_llama_fast.py
# ==========================================================

import os
import sys
import time
import json
import shutil
import subprocess
import requests

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
MODEL_NAME = "llama3.2"      # Change to a smaller/faster model if needed
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# Speed / behavior tuning
REQUEST_TIMEOUT = 300
SERVER_READY_WAIT_SECONDS = 30
KEEP_ALIVE = "10m"           # Keep model loaded in memory for 10 minutes
WARMUP_MODEL = True          # Send tiny prompt once to reduce first real delay
SHOW_PROCESSOR_INFO = True   # Try to show GPU / CPU info using "ollama ps"

# Generation tuning for speed
# Lower num_predict = shorter answers = faster
GEN_OPTIONS = {
    "temperature": 0.3,
    "num_predict": 200,
    "top_p": 0.9,
}

# Optional system-style instruction to keep answers concise
SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer clearly and briefly unless the user asks for more detail."
)

# ----------------------------------------------------------
# HELPER: print separator
# ----------------------------------------------------------
def print_line():
    print("=" * 70)

# ----------------------------------------------------------
# HELPER: find ollama.exe
# ----------------------------------------------------------
def find_ollama_exe():
    """
    Try several ways to find ollama.exe.

    Returns:
        Full path to ollama.exe if found, otherwise None
    """
    found = shutil.which("ollama")
    if found:
        return found

    username = os.environ.get("USERNAME", "")

    possible_paths = [
        rf"C:\Users\{username}\AppData\Local\Programs\Ollama\ollama.exe",
        rf"C:\Users\{username}\AppData\Local\Ollama\ollama.exe",
        r"C:\Program Files\Ollama\ollama.exe",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None

# ----------------------------------------------------------
# HELPER: install Ollama if missing
# ----------------------------------------------------------
def install_ollama():
    """
    Install Ollama using winget if it is not already installed.

    Returns:
        Full path to ollama.exe if successful, otherwise None
    """
    print_line()
    print("Checking whether Ollama is already installed...")

    ollama_exe = find_ollama_exe()
    if ollama_exe:
        print(f"[OK] Ollama found: {ollama_exe}")
        return ollama_exe

    print("[INFO] Ollama was not found.")
    print("[INFO] Trying to install Ollama using winget...")

    try:
        result = subprocess.run(
            ["winget", "install", "-e", "--id", "Ollama.Ollama"],
            capture_output=False,
            text=True,
            check=False
        )
    except FileNotFoundError:
        print("[ERROR] winget is not available on this machine.")
        print("[ACTION] Install Ollama manually from: https://ollama.com/download/windows")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error while running winget: {e}")
        return None

    if result.returncode != 0:
        print("[ERROR] winget install did not complete successfully.")
        print("[ACTION] Please install Ollama manually from: https://ollama.com/download/windows")
        return None

    print("[INFO] Ollama installation command completed.")
    print("[INFO] Waiting a few seconds for Windows to finish registration...")
    time.sleep(6)

    ollama_exe = find_ollama_exe()
    if ollama_exe:
        print(f"[OK] Ollama installed successfully: {ollama_exe}")
        return ollama_exe

    print("[ERROR] Ollama may be installed, but Python cannot locate ollama.exe yet.")
    print("[ACTION] Close and reopen terminal, or verify installation path manually.")
    return None

# ----------------------------------------------------------
# HELPER: check whether server is running
# ----------------------------------------------------------
def is_ollama_server_running():
    """
    Check if Ollama server responds on localhost.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

# ----------------------------------------------------------
# HELPER: start server only if needed
# ----------------------------------------------------------
def start_ollama_server(ollama_exe):
    """
    Start Ollama server if it is not already running.
    """
    print_line()

    if is_ollama_server_running():
        print("[OK] Ollama server is already running.")
        return True

    print("[INFO] Starting Ollama server...")

    try:
        creation_flags = 0
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

        subprocess.Popen(
            [ollama_exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags
        )
    except Exception as e:
        print(f"[ERROR] Failed to start Ollama server: {e}")
        return False

    print("[INFO] Waiting for Ollama server to become ready...")

    for _ in range(SERVER_READY_WAIT_SECONDS):
        if is_ollama_server_running():
            print("[OK] Ollama server is ready.")
            return True
        time.sleep(1)

    print("[ERROR] Ollama server did not become ready in time.")
    return False

# ----------------------------------------------------------
# HELPER: check whether model exists
# ----------------------------------------------------------
def model_exists():
    """
    Check if the desired model already exists in Ollama.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()

        for model in data.get("models", []):
            name = model.get("name", "")
            if name.startswith(MODEL_NAME):
                return True

        return False
    except Exception:
        return False

# ----------------------------------------------------------
# HELPER: pull model only if missing
# ----------------------------------------------------------
def pull_model(ollama_exe):
    """
    Pull the configured model if it does not already exist.
    """
    print_line()

    if model_exists():
        print(f"[OK] Model already available: {MODEL_NAME}")
        return True

    print(f"[INFO] Pulling model: {MODEL_NAME}")
    print("[INFO] This may take time the first time, depending on internet speed.")

    try:
        result = subprocess.run([ollama_exe, "pull", MODEL_NAME], check=False)
    except Exception as e:
        print(f"[ERROR] Exception while pulling model: {e}")
        return False

    if result.returncode != 0:
        print(f"[ERROR] Failed to pull model: {MODEL_NAME}")
        return False

    print(f"[OK] Model pulled successfully: {MODEL_NAME}")
    return True

# ----------------------------------------------------------
# HELPER: optional processor / GPU info
# ----------------------------------------------------------
def show_processor_info(ollama_exe):
    """
    Show 'ollama ps' output so user can inspect whether model is on GPU/CPU.
    """
    if not SHOW_PROCESSOR_INFO:
        return

    print_line()
    print("[INFO] Trying to display current Ollama process info...")
    try:
        result = subprocess.run(
            [ollama_exe, "ps"],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0 and result.stdout.strip():
            print(result.stdout)
        else:
            print("[INFO] Could not read 'ollama ps' output yet.")
    except Exception as e:
        print(f"[INFO] Could not run 'ollama ps': {e}")

# ----------------------------------------------------------
# HELPER: warm up model
# ----------------------------------------------------------
def warmup_model():
    """
    Send a very small prompt once so the first real question is faster.
    """
    if not WARMUP_MODEL:
        return

    print_line()
    print("[INFO] Warming up model for faster first response...")

    payload = {
        "model": MODEL_NAME,
        "prompt": "Hi",
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0.0,
            "num_predict": 5
        },
        "system": SYSTEM_PROMPT
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        print("[OK] Warm-up completed.")
    except Exception as e:
        print(f"[INFO] Warm-up skipped or failed: {e}")

# ----------------------------------------------------------
# HELPER: build prompt
# ----------------------------------------------------------
def build_prompt(user_text):
    """
    Keep prompt simple and concise for better speed.
    """
    return user_text.strip()

# ----------------------------------------------------------
# HELPER: ask model with streaming
# ----------------------------------------------------------
def ask_model_stream(prompt):
    """
    Send a prompt to Ollama and stream the response live.
    This improves perceived speed a lot.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": build_prompt(prompt),
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": GEN_OPTIONS,
        "system": SYSTEM_PROMPT
    }

    full_text = ""

    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT
        ) as response:
            response.raise_for_status()

            print("\nLlama: ", end="", flush=True)

            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))
                except Exception:
                    continue

                chunk = data.get("response", "")
                if chunk:
                    full_text += chunk
                    print(chunk, end="", flush=True)

            print("\n")
        return full_text

    except Exception as e:
        print(f"\n[ERROR] Failed to query model: {e}\n")
        return ""

# ----------------------------------------------------------
# OPTIONAL: quick test prompt
# ----------------------------------------------------------
def run_quick_test():
    """
    Run a simple test prompt to confirm everything works.
    """
    print_line()
    print("[TEST] Sending a simple streaming test prompt to the model...")

    test_prompt = "Say hello in one short sentence."
    ask_model_stream(test_prompt)

# ----------------------------------------------------------
# HELPER: print speed tips
# ----------------------------------------------------------
def print_speed_tips():
    print_line()
    print("Speed tips:")
    print("1) Use a smaller model if still slow.")
    print("2) Keep prompts short.")
    print("3) Avoid sending large documents directly.")
    print("4) Check 'ollama ps' to see whether model uses GPU or CPU.")
    print("5) Keep this program open so the model stays loaded in memory.")
    print_line()

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    print_line()
    print("Improved Fast Python Llama Installer + Chatbot")
    print_line()

    # Step 1: ensure Ollama exists
    ollama_exe = install_ollama()
    if not ollama_exe:
        sys.exit(1)

    # Step 2: start server only if needed
    if not start_ollama_server(ollama_exe):
        sys.exit(1)

    # Step 3: ensure model exists
    if not pull_model(ollama_exe):
        sys.exit(1)

    # Step 4: optional warm-up
    warmup_model()

    # Step 5: optional processor info
    show_processor_info(ollama_exe)

    # Step 6: quick test
    run_quick_test()

    # Step 7: user loop
    print_speed_tips()
    print("Llama is ready.")
    print("Type your question and press Enter.")
    print("Type 'exit' to quit.")
    print_line()

    while True:
        try:
            user_text = input("You: ").strip()
        except KeyboardInterrupt:
            print("\n[INFO] Exiting...")
            break

        if not user_text:
            continue

        if user_text.lower() in ["exit", "quit"]:
            print("[INFO] Goodbye.")
            break

        # Special local command to inspect processor usage again
        if user_text.lower() == "/ps":
            show_processor_info(ollama_exe)
            continue

        ask_model_stream(user_text)

# ----------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------
if __name__ == "__main__":
    main()