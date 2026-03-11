# ==========================================================
# json_llama_chatbot.py
# ==========================================================
# PURPOSE:
#   1) Check/install Ollama
#   2) Start Ollama if needed
#   3) Load local JSON knowledge
#   4) Search JSON first
#   5) If no JSON answer, ask a small model first
#   6) Optionally fall back to a larger model
#   7) Stream output for better responsiveness
#
# BEFORE RUNNING:
#   pip install requests
#
# OPTIONAL JSON FORMAT:
# [
#   {"question": "what is ai", "answer": "AI is ..."},
#   {"question": "what is 5g", "answer": "5G is ..."}
# ]
#
# RUN:
#   python json_llama_chatbot.py
# ==========================================================

import os
import sys
import time
import json
import shutil
import subprocess
import requests
import re

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
JSON_FILE = "knowledge.json"

# Fast model used first when JSON has no answer
SMALL_MODEL = "llama3.2"

# Larger model fallback
LARGE_MODEL = "llama3.1"

# Ollama API info
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# Behavior tuning
REQUEST_TIMEOUT = 300
SERVER_READY_WAIT_SECONDS = 30
KEEP_ALIVE = "10m"
AUTO_PULL_MODELS = True
USE_LARGE_MODEL_FALLBACK = True

# JSON search tuning
MIN_MATCH_SCORE = 0.60

# Short system prompts
SYSTEM_PROMPT_SMALL = (
    "You are a helpful assistant. "
    "Answer clearly and briefly unless the user asks for more detail."
)

SYSTEM_PROMPT_LARGE = (
    "You are a helpful assistant. "
    "Provide a clear, accurate answer. "
    "Be concise but complete."
)

# Generation options
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

# Global memory for loaded JSON
knowledge_data = []

# ----------------------------------------------------------
# HELPER: print separator
# ----------------------------------------------------------
def print_line():
    print("=" * 70)

# ----------------------------------------------------------
# HELPER: normalize text
# ----------------------------------------------------------
def normalize_text(text):
    """
    Normalize text for easier matching:
    - lowercase
    - remove punctuation
    - collapse spaces
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------------------------------------
# HELPER: token overlap score
# ----------------------------------------------------------
def token_overlap_score(user_text, candidate_text):
    """
    A simple score based on overlap between user question tokens
    and stored JSON question tokens.
    """
    user_tokens = set(normalize_text(user_text).split())
    candidate_tokens = set(normalize_text(candidate_text).split())

    if not user_tokens or not candidate_tokens:
        return 0.0

    common = user_tokens.intersection(candidate_tokens)
    score = len(common) / max(len(candidate_tokens), 1)
    return score

# ----------------------------------------------------------
# FIND OLLAMA EXE
# ----------------------------------------------------------
def find_ollama_exe():
    """
    Try to locate ollama.exe from PATH or common Windows paths.
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
# INSTALL OLLAMA
# ----------------------------------------------------------
def install_ollama():
    """
    Install Ollama using winget if needed.
    """
    print_line()
    print("Checking whether Ollama is installed...")

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
        print("[ERROR] winget is not available.")
        print("[ACTION] Install Ollama manually from:")
        print("         https://ollama.com/download/windows")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected installation error: {e}")
        return None

    if result.returncode != 0:
        print("[ERROR] winget install failed.")
        print("[ACTION] Install Ollama manually from:")
        print("         https://ollama.com/download/windows")
        return None

    print("[INFO] Waiting a few seconds for installation to complete...")
    time.sleep(6)

    ollama_exe = find_ollama_exe()
    if ollama_exe:
        print(f"[OK] Ollama installed successfully: {ollama_exe}")
        return ollama_exe

    print("[ERROR] Ollama may be installed, but Python cannot locate it yet.")
    print("[ACTION] Close and reopen terminal, then rerun the script.")
    return None

# ----------------------------------------------------------
# CHECK SERVER
# ----------------------------------------------------------
def is_ollama_server_running():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

# ----------------------------------------------------------
# START SERVER
# ----------------------------------------------------------
def start_ollama_server(ollama_exe):
    """
    Start Ollama server only if needed.
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
        print(f"[ERROR] Could not start Ollama server: {e}")
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
# MODEL EXISTS
# ----------------------------------------------------------
def model_exists(model_name):
    """
    Check if a model is already available in Ollama.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()

        for model in data.get("models", []):
            name = model.get("name", "")
            if name.startswith(model_name):
                return True

        return False
    except Exception:
        return False

# ----------------------------------------------------------
# PULL MODEL
# ----------------------------------------------------------
def pull_model(ollama_exe, model_name):
    """
    Pull model if not present.
    """
    if model_exists(model_name):
        print(f"[OK] Model already available: {model_name}")
        return True

    if not AUTO_PULL_MODELS:
        print(f"[WARNING] Model not found and AUTO_PULL_MODELS is off: {model_name}")
        return False

    print(f"[INFO] Pulling model: {model_name}")
    print("[INFO] First download may take some time...")

    try:
        result = subprocess.run([ollama_exe, "pull", model_name], check=False)
        if result.returncode != 0:
            print(f"[ERROR] Failed to pull model: {model_name}")
            return False
        print(f"[OK] Model pulled successfully: {model_name}")
        return True
    except Exception as e:
        print(f"[ERROR] Exception while pulling model '{model_name}': {e}")
        return False

# ----------------------------------------------------------
# SHOW PROCESSOR INFO
# ----------------------------------------------------------
def show_processor_info(ollama_exe):
    """
    Show ollama ps info so user can see CPU/GPU use.
    """
    print_line()
    print("[INFO] Current Ollama process info:")
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
            print("[INFO] No process information available yet.")
    except Exception as e:
        print(f"[INFO] Could not run 'ollama ps': {e}")

# ----------------------------------------------------------
# LOAD JSON KNOWLEDGE
# ----------------------------------------------------------
def load_knowledge():
    """
    Load knowledge from JSON file.
    Expected format:
    [
      {"question": "...", "answer": "..."},
      ...
    ]
    """
    global knowledge_data
    knowledge_data = []

    print_line()
    print(f"Loading JSON knowledge from: {JSON_FILE}")

    if not os.path.exists(JSON_FILE):
        print(f"[WARNING] JSON file not found: {JSON_FILE}")
        print("[WARNING] The chatbot will still work, but only with local models.")
        return

    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            print("[WARNING] JSON file must contain a list of entries.")
            return

        for item in raw:
            if not isinstance(item, dict):
                continue

            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()

            if question and answer:
                knowledge_data.append({
                    "question": question,
                    "answer": answer
                })

        print(f"[OK] Loaded {len(knowledge_data)} valid JSON knowledge entries.")

    except Exception as e:
        print(f"[ERROR] Failed to load JSON file: {e}")

# ----------------------------------------------------------
# SEARCH JSON KNOWLEDGE
# ----------------------------------------------------------
def search_knowledge(user_text):
    """
    PURPOSE
    -------
    Search the JSON knowledge base to find the best matching answer
    for the user's question.

    The JSON file typically looks like:

    knowledge.json
    [
        {"question": "who is mohamed",
         "answer": "Mohamed Khalil is an engineer working on telecom and AI systems."},

        {"question": "what is ai",
         "answer": "Artificial Intelligence is the ability of machines to perform tasks that normally require human intelligence."}
    ]

    Matching Strategy (in order of priority):

    1) Exact normalized match
    2) Contains match (substring)
    3) Token overlap scoring

    If a match is found, the corresponding answer is returned.
    Otherwise, the function returns None and the script will
    ask the Llama model instead.
    """

    # -----------------------------------------------------
    # Step 0: Check if JSON knowledge is empty
    # -----------------------------------------------------
    # If the JSON file was not loaded or has no entries,
    # we cannot search anything.
    #
    # Example:
    # knowledge_data = []
    #
    # Result:
    # return None so the system will ask the LLM instead.
    # -----------------------------------------------------
    if not knowledge_data:
        return None


    # -----------------------------------------------------
    # Step 1: Normalize the user question
    # -----------------------------------------------------
    # The function normalize_text() typically does:
    #
    # 1) lowercase
    # 2) remove punctuation
    # 3) remove extra spaces
    #
    # Example:
    #
    # User input:
    #   "Who is Mohamed?"
    #
    # After normalization:
    #   "who is mohamed"
    #
    # Another example:
    #   "  MOHAMED   "
    #
    # becomes:
    #   "mohamed"
    #
    # This helps match different forms of the same question.
    # -----------------------------------------------------
    user_norm = normalize_text(user_text)


    # -----------------------------------------------------
    # STEP 1: Exact normalized match
    # -----------------------------------------------------
    # Here we check if the user's normalized question
    # exactly matches a stored JSON question.
    #
    # Example JSON question:
    #   "who is mohamed"
    #
    # User input:
    #   "Who is Mohamed?"
    #
    # After normalization both become:
    #   "who is mohamed"
    #
    # Therefore they match exactly.
    #
    # If match found -> return the stored answer immediately.
    # -----------------------------------------------------
    for item in knowledge_data:

        q_norm = normalize_text(item["question"])

        if user_norm == q_norm:
            return item["answer"]


    # -----------------------------------------------------
    # STEP 2: Contains match (substring)
    # -----------------------------------------------------
    # If exact match failed, we check if one string
    # appears inside the other.
    #
    # Example JSON:
    #   "who is mohamed"
    #
    # User input:
    #   "mohamed"
    #
    # Check:
    #   "mohamed" in "who is mohamed"  → TRUE
    #
    # Therefore we assume it refers to the same concept.
    #
    # Another example:
    #
    # JSON question:
    #   "what is ai"
    #
    # User input:
    #   "please explain what is ai"
    #
    # Check:
    #   "what is ai" in "please explain what is ai" → TRUE
    #
    # So we return the stored answer.
    # -----------------------------------------------------
    for item in knowledge_data:

        q_norm = normalize_text(item["question"])

        if q_norm and (q_norm in user_norm or user_norm in q_norm):
            return item["answer"]


    # -----------------------------------------------------
    # STEP 3: Token overlap scoring
    # -----------------------------------------------------
    # If substring matching also fails, we compute a similarity
    # score based on overlapping words (tokens).
    #
    # Example:
    #
    # JSON question:
    #   "who is mohamed"
    #
    # User question:
    #   "tell me about mohamed"
    #
    # Token sets:
    #
    # user_tokens     = {tell, me, about, mohamed}
    # candidate_tokens= {who, is, mohamed}
    #
    # common tokens   = {mohamed}
    #
    # Score example:
    #   overlap = 1 / 3 = 0.33
    #
    # If the score is greater than MIN_MATCH_SCORE,
    # we accept it as a match.
    #
    # Example threshold:
    #
    #   MIN_MATCH_SCORE = 0.60
    #
    # If best score >= 0.60 → return answer
    #
    # Otherwise → treat as unknown question.
    # -----------------------------------------------------
    best_score = 0.0
    best_answer = None

    for item in knowledge_data:

        score = token_overlap_score(user_text, item["question"])

        if score > best_score:
            best_score = score
            best_answer = item["answer"]


    # -----------------------------------------------------
    # STEP 4: Accept match if score is high enough
    # -----------------------------------------------------
    # If similarity exceeds the configured threshold,
    # we return the answer.
    #
    # Example:
    #
    # user question:
    #   "information about mohamed"
    #
    # JSON question:
    #   "who is mohamed"
    #
    # score = 0.67
    #
    # If threshold = 0.60 → ACCEPT
    #
    # Otherwise → reject.
    # -----------------------------------------------------
    if best_score >= MIN_MATCH_SCORE:
        return best_answer


    # -----------------------------------------------------
    # STEP 5: No match found
    # -----------------------------------------------------
    # If none of the rules matched, we return None.
    #
    # The main chatbot program will then do this:
    #
    #   JSON search failed
    #           ↓
    #   ask the Llama model
    #
    # Example question not in JSON:
    #
    #   "Explain quantum computing"
    #
    # JSON does not contain it,
    # so the system calls the LLM.
    # -----------------------------------------------------
    return None
# ----------------------------------------------------------
# WARM UP MODEL
# ----------------------------------------------------------
def warmup_model(model_name, system_prompt):
    """
    Tiny prompt to warm up model for faster first response.
    """
    print(f"[INFO] Warming up model: {model_name}")

    payload = {
        "model": model_name,
        "prompt": "Hi",
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0.0,
            "num_predict": 5
        },
        "system": system_prompt
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        print(f"[OK] Warm-up completed for model: {model_name}")
    except Exception as e:
        print(f"[WARNING] Warm-up failed for model '{model_name}': {e}")

# ----------------------------------------------------------
# STREAM MODEL RESPONSE
# ----------------------------------------------------------
def ask_model_stream(model_name, prompt, system_prompt, options):
    """
    Ask a model and stream the answer live.
    Returns the full generated text.
    """
    payload = {
        "model": model_name,
        "prompt": prompt.strip(),
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": options,
        "system": system_prompt
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

            print(f"\n{model_name}: ", end="", flush=True)

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
        print(f"\n[ERROR] Failed to query model '{model_name}': {e}\n")
        return ""

# ----------------------------------------------------------
# ROUTER: JSON -> SMALL MODEL -> LARGE MODEL
# ----------------------------------------------------------
def answer_user_question(user_text):
    """
    Main answering flow:
    1) Search JSON knowledge
    2) If found, return immediately
    3) Else ask small model
    4) If large fallback enabled, optionally allow user to request it

    Current policy:
    - JSON answer is used first if found
    - If not found, small model answers
    - Then large model can be used automatically or when small fails
    """
    # Step 1: JSON search
    local_answer = search_knowledge(user_text)
    if local_answer:
        print("\nJSON Knowledge:", local_answer, "\n")
        return

    # Step 2: small model
    print("\n[INFO] No answer found in JSON. Asking smaller model first...")
    small_response = ask_model_stream(
        model_name=SMALL_MODEL,
        prompt=user_text,
        system_prompt=SYSTEM_PROMPT_SMALL,
        options=SMALL_MODEL_OPTIONS
    )

    # Step 3: larger model fallback
    # Here we use a simple fallback rule:
    # If small response is empty, automatically use larger model.
    if USE_LARGE_MODEL_FALLBACK and not small_response.strip():
        print("[INFO] Small model did not return a usable answer. Trying larger model...")
        ask_model_stream(
            model_name=LARGE_MODEL,
            prompt=user_text,
            system_prompt=SYSTEM_PROMPT_LARGE,
            options=LARGE_MODEL_OPTIONS
        )

# ----------------------------------------------------------
# HELP MESSAGE
# ----------------------------------------------------------
def print_help():
    print_line()
    print("Commands:")
    print("  exit        -> quit the program")
    print("  /ps         -> show Ollama processor / GPU / CPU info")
    print("  /reload     -> reload knowledge.json")
    print("  /small      -> show current small model")
    print("  /large      -> show current large model")
    print("  /asklarge   -> ask the larger model directly")
    print_line()

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    print_line()
    print("JSON + Llama Chatbot")
    print_line()

    # 1) Install / find Ollama
    ollama_exe = install_ollama()
    if not ollama_exe:
        sys.exit(1)

    # 2) Start Ollama
    if not start_ollama_server(ollama_exe):
        sys.exit(1)

    # 3) Pull small model
    print_line()
    if not pull_model(ollama_exe, SMALL_MODEL):
        sys.exit(1)

    # 4) Pull large model
    if USE_LARGE_MODEL_FALLBACK:
        if not pull_model(ollama_exe, LARGE_MODEL):
            print(f"[WARNING] Could not prepare larger model: {LARGE_MODEL}")
            print("[WARNING] Script will continue with JSON + small model only.")

    # 5) Load JSON
    load_knowledge()

    # 6) Warm up models
    print_line()
    warmup_model(SMALL_MODEL, SYSTEM_PROMPT_SMALL)

    if USE_LARGE_MODEL_FALLBACK and model_exists(LARGE_MODEL):
        warmup_model(LARGE_MODEL, SYSTEM_PROMPT_LARGE)

    # 7) Show process info
    show_processor_info(ollama_exe)

    # 8) Start chat loop
    print_help()
    print("Chatbot is ready.")
    print("Flow: JSON first -> small model -> larger model if needed")
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

        if user_text.lower() == "/ps":
            show_processor_info(ollama_exe)
            continue

        if user_text.lower() == "/reload":
            load_knowledge()
            continue

        if user_text.lower() == "/small":
            print(f"Small model: {SMALL_MODEL}")
            continue

        if user_text.lower() == "/large":
            print(f"Large model: {LARGE_MODEL}")
            continue

        if user_text.lower() == "/asklarge":
            question = input("Enter question for larger model: ").strip()
            if question:
                ask_model_stream(
                    model_name=LARGE_MODEL,
                    prompt=question,
                    system_prompt=SYSTEM_PROMPT_LARGE,
                    options=LARGE_MODEL_OPTIONS
                )
            continue

        answer_user_question(user_text)

# ----------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------
if __name__ == "__main__":
    main()