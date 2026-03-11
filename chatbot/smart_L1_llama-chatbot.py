# ==========================================================
# smart_json_llama_chatbot_fixed.py
# ==========================================================
# PURPOSE
# -------
# This script creates a self-learning chatbot using:
#
#   1) Local JSON knowledge storage
#   2) LLaMA models running through Ollama
#
# The chatbot works in this order:
#
#   User Question
#        ↓
#   Search smart_knowledge.json
#        ↓
#   If found → answer immediately
#        ↓
#   If not found → ask LLaMA
#        ↓
#   Save LLaMA answer into smart_knowledge.json
#        ↓
#   Next time → answer instantly from JSON
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
import re


# ----------------------------------------------------------
# CONFIGURATION SECTION
# ----------------------------------------------------------

# File used as the ONLY knowledge base
SMART_JSON_FILE = "smart_knowledge.json"

# Fast model used first
SMALL_MODEL = "llama3.2"

# Larger model fallback if the small model fails
LARGE_MODEL = "llama3.1"

# Ollama API location
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# Timeouts and behavior
REQUEST_TIMEOUT = 300
SERVER_READY_WAIT_SECONDS = 30
KEEP_ALIVE = "10m"

AUTO_PULL_MODELS = True
USE_LARGE_MODEL_FALLBACK = True

# Threshold used when comparing questions
MIN_MATCH_SCORE = 0.50

# Debug mode for search details
DEBUG_SEARCH = True


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
# GLOBAL MEMORY
# ----------------------------------------------------------

# Knowledge loaded from smart_knowledge.json
knowledge_data = []


# ----------------------------------------------------------
# HELPER: Print separator line
# ----------------------------------------------------------

def print_line():
    print("=" * 70)


# ----------------------------------------------------------
# TEXT NORMALIZATION
# ----------------------------------------------------------
# PURPOSE
# -------
# Convert text to a simplified form so different
# writing styles still match.
#
# Example:
#
#    "Who is Mohamed?"
#        ↓
#    "who is mohamed"
#
# Steps performed:
#
#   1) convert to lowercase
#   2) remove punctuation
#   3) collapse extra spaces
# ----------------------------------------------------------

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------------------
# REDUCE QUESTION TO IMPORTANT WORDS
# ----------------------------------------------------------
# PURPOSE
# -------
# Remove common filler words to improve matching.
#
# Example:
#   "how i can terminate the program"
#       ↓
#   "terminate program"
#
#   "how can i terminate the program"
#       ↓
#   "terminate program"
# ----------------------------------------------------------
def reduce_question(text):
    """
    PURPOSE
    -------
    Simplify a question by removing common words that do not
    carry important meaning (called "stopwords").

    This helps the chatbot compare questions more intelligently.

    Example problem:
        "how can I terminate the program"
        "how I can terminate program"
        "terminate the program"

    All of these should match the same stored question.

    After reduction they become:

        "terminate program"

    which makes matching easier.


    ----------------------------------------------------------
    STEP 1 — Define stopwords
    ----------------------------------------------------------
    Stopwords are very common words that usually do not change
    the meaning of a question.

    Examples:
        how
        what
        the
        is
        can
        please
        about

    These words are removed during reduction.
    """

    stopwords = {
        "how", "what", "who", "where", "when", "why", "which",
        "is", "are", "am", "was", "were", "be", "been", "being",
        "i", "me", "my", "you", "your", "he", "she", "it", "we", "they",
        "can", "could", "would", "should", "do", "does", "did",
        "please", "tell", "about", "the", "a", "an", "to", "of", "in",
        "on", "for", "and", "or"
    }

    """
    ----------------------------------------------------------
    STEP 2 — Normalize the text
    ----------------------------------------------------------
    The function normalize_text() converts the text into a
    simplified format so comparisons are easier.

    Example input:
        "How Can I Terminate the Program?"

    After normalize_text():
        "how can i terminate the program"

    Normalization typically:
        • converts to lowercase
        • removes punctuation
        • removes extra spaces
    """

    norm = normalize_text(text)

    """
    ----------------------------------------------------------
    STEP 3 — Split the sentence into words
    ----------------------------------------------------------

    Example:

        norm = "how can i terminate the program"

    After split():

        words = ["how","can","i","terminate","the","program"]
    """

    words = norm.split()

    """
    ----------------------------------------------------------
    STEP 4 — Remove stopwords
    ----------------------------------------------------------

    We keep only words that are NOT in the stopword list.

    Example:

        words = ["how","can","i","terminate","the","program"]

    stopwords removed:

        "how"
        "can"
        "i"
        "the"

    remaining words:

        ["terminate","program"]
    """

    reduced_words = [w for w in words if w not in stopwords]

    """
    ----------------------------------------------------------
    STEP 5 — Join remaining words back into a sentence
    ----------------------------------------------------------

    reduced_words = ["terminate","program"]

    After join():

        "terminate program"
    """

    return " ".join(reduced_words)


# ----------------------------------------------------------
# TOKEN OVERLAP SCORING
# ----------------------------------------------------------
# PURPOSE
# -------
# Compute similarity between two questions.
#
# Example:
#
#   user question
#       "tell me about mohamed"
#
#   stored question
#       "who is mohamed"
#
# tokens(user) = {tell, me, about, mohamed}
# tokens(json) = {who, is, mohamed}
#
# common tokens = {mohamed}
#
# score = 1 / 3 = 0.33
# ----------------------------------------------------------

def token_overlap_score(user_text, candidate_text):
    user_tokens = set(normalize_text(user_text).split())
    candidate_tokens = set(normalize_text(candidate_text).split())

    if not user_tokens or not candidate_tokens:
        return 0.0

    common = user_tokens.intersection(candidate_tokens)
    score = len(common) / max(len(candidate_tokens), 1)
    return score


# ----------------------------------------------------------
# FIND OLLAMA EXECUTABLE
# ----------------------------------------------------------

def find_ollama_exe():
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
# INSTALL OLLAMA IF NECESSARY
# ----------------------------------------------------------

def install_ollama():
    print_line()
    print("Checking whether Ollama is installed...")

    ollama_exe = find_ollama_exe()

    if ollama_exe:
        print(f"[OK] Ollama found: {ollama_exe}")
        return ollama_exe

    print("[INFO] Ollama was not found.")
    print("[INFO] Installing Ollama using winget...")

    try:
        result = subprocess.run(
            ["winget", "install", "-e", "--id", "Ollama.Ollama"],
            check=False
        )

        if result.returncode != 0:
            print("[ERROR] winget install failed.")
            print("[ACTION] Please install Ollama manually from https://ollama.com")
            return None

    except Exception as e:
        print(f"[ERROR] Could not install Ollama automatically: {e}")
        print("[ACTION] Please install Ollama manually from https://ollama.com")
        return None

    time.sleep(6)

    ollama_exe = find_ollama_exe()

    if ollama_exe:
        print(f"[OK] Ollama installed successfully: {ollama_exe}")
        return ollama_exe

    print("[ERROR] Ollama may be installed, but Python cannot find it yet.")
    print("[ACTION] Close terminal, open it again, then rerun the script.")
    return None


# ----------------------------------------------------------
# CHECK IF OLLAMA SERVER IS RUNNING
# ----------------------------------------------------------

def is_ollama_server_running():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ----------------------------------------------------------
# START OLLAMA SERVER
# ----------------------------------------------------------

def start_ollama_server(ollama_exe):
    print_line()

    if is_ollama_server_running():
        print("[OK] Ollama server already running.")
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

    for _ in range(SERVER_READY_WAIT_SECONDS):
        if is_ollama_server_running():
            print("[OK] Ollama server is ready.")
            return True
        time.sleep(1)

    print("[ERROR] Ollama server failed to start.")
    return False


# ----------------------------------------------------------
# CHECK IF MODEL EXISTS
# ----------------------------------------------------------

def model_exists(model_name):
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json()

        for model in data.get("models", []):
            name = model.get("name", "")
            if name.startswith(model_name):
                return True

        return False

    except Exception:
        return False


# ----------------------------------------------------------
# PULL MODEL IF NEEDED
# ----------------------------------------------------------

def pull_model(ollama_exe, model_name):
    if model_exists(model_name):
        print(f"[OK] Model already available: {model_name}")
        return True

    if not AUTO_PULL_MODELS:
        print(f"[WARNING] Model not found: {model_name}")
        return False

    print(f"[INFO] Pulling model: {model_name}")
    print("[INFO] This may take time the first time...")

    try:
        result = subprocess.run([ollama_exe, "pull", model_name], check=False)

        if result.returncode != 0:
            print(f"[ERROR] Failed to pull model: {model_name}")
            return False

        print(f"[OK] Model pulled successfully: {model_name}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed while pulling model '{model_name}': {e}")
        return False


# ----------------------------------------------------------
# SHOW PROCESSOR INFO
# ----------------------------------------------------------

def show_processor_info(ollama_exe):
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
# CREATE SMART KNOWLEDGE FILE IF MISSING
# ----------------------------------------------------------

def ensure_smart_json_exists():
    if not os.path.exists(SMART_JSON_FILE):
        try:
            with open(SMART_JSON_FILE, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2, ensure_ascii=False)

            print(f"[OK] Created {SMART_JSON_FILE}")
            print(f"[INFO] File location: {os.path.abspath(SMART_JSON_FILE)}")

        except Exception as e:
            print(f"[ERROR] Failed to create {SMART_JSON_FILE}: {e}")


# ----------------------------------------------------------
# LOAD KNOWLEDGE FROM JSON
# ----------------------------------------------------------

def load_knowledge():
    global knowledge_data

    knowledge_data = []

    print_line()
    print(f"Loading knowledge from {SMART_JSON_FILE}")

    ensure_smart_json_exists()

    try:
        with open(SMART_JSON_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            print("[WARNING] JSON root must be a list.")
            return

        for item in raw:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                knowledge_data.append(item)

        print(f"[OK] Loaded {len(knowledge_data)} knowledge entries.")
        print(f"[INFO] File location: {os.path.abspath(SMART_JSON_FILE)}")

    except Exception as e:
        print("Failed to load JSON:", e)


# ----------------------------------------------------------
# SAVE NEW KNOWLEDGE
# ----------------------------------------------------------

def save_smart_knowledge(question, answer):
    print_line()
    print("[SAVE] Starting save process...")

    if not question or not question.strip():
        print("[SAVE] Question is empty. Save skipped.")
        return False

    if not answer or not answer.strip():
        print("[SAVE] Answer is empty. Save skipped.")
        return False

    ensure_smart_json_exists()

    abs_path = os.path.abspath(SMART_JSON_FILE)
    print(f"[SAVE] Target file: {abs_path}")

    data = []

    print("[SAVE] Loading existing JSON data...")

    try:
        with open(SMART_JSON_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)

            if isinstance(loaded, list):
                data = loaded
                print(f"[SAVE] Existing entries loaded: {len(data)}")
            else:
                print("[SAVE] Existing file is not a JSON list. Resetting to empty list.")
                data = []

    except Exception as e:
        print(f"[SAVE] Could not load existing data. Starting fresh. Reason: {e}")
        data = []

    print("[SAVE] Checking for duplicate question...")

    normalized_new_question = normalize_text(question)

    for item in data:
        existing_question = str(item.get("question", "")).strip()

        if normalize_text(existing_question) == normalized_new_question:
            print("[SAVE] Duplicate question found. Save skipped.")
            return False

    print("[SAVE] No duplicate found. Preparing new entry...")

    new_entry = {
        "question": question.strip(),
        "answer": answer.strip()
    }

    data.append(new_entry)
    print(f"[SAVE] New entry appended. New total entries: {len(data)}")

    print("[SAVE] Writing updated JSON file...")

    try:
        with open(SMART_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print("[SAVE] JSON write completed successfully.")

    except Exception as e:
        print(f"[SAVE] Failed while writing JSON file: {e}")
        return False

    print("[SAVE] Updating in-memory knowledge...")

    knowledge_data.append({
        "question": question.strip(),
        "answer": answer.strip()
    })

    print("[SMART LEARN] Added new knowledge successfully.")
    print("[SAVE] Save process finished.")
    return True


def search_knowledge(user_text):
    """
    Search smart_knowledge.json for the best matching answer.

    FIXED LOGIC:
    1) exact normalized match
    2) exact reduced-question match
    3) strong contains match
    4) high token overlap only if at least 2 important words match

    This prevents bad matches like:
        "population of egypt"
    matching:
        "where is egypt"
    """

    if DEBUG_SEARCH:
        print_line()
        print(f"[DEBUG] Searching JSON for: {user_text}")

    if not knowledge_data:
        if DEBUG_SEARCH:
            print("[DEBUG] knowledge_data is empty")
        return None

    user_norm = normalize_text(user_text)
    user_reduced = reduce_question(user_text)
    user_tokens = set(user_reduced.split())

    if DEBUG_SEARCH:
        print(f"[DEBUG] normalized user: {user_norm}")
        print(f"[DEBUG] reduced user   : {user_reduced}")
        print(f"[DEBUG] reduced tokens : {user_tokens}")

    # --------------------------------------------------
    # 1) Exact normalized match
    # Example:
    #   user  = "what is ai"
    #   json  = "what is ai"
    # --------------------------------------------------
    for item in knowledge_data:
        q_norm = normalize_text(item["question"])

        if DEBUG_SEARCH:
            print(f"[DEBUG] exact compare with: {q_norm}")

        if user_norm == q_norm:
            if DEBUG_SEARCH:
                print("[DEBUG] EXACT NORMALIZED MATCH FOUND")
            return item["answer"]

    # --------------------------------------------------
    # 2) Exact reduced-question match
    # Example:
    #   "how can i terminate the program"
    #   "how i can terminate the program"
    # both reduce to something similar like:
    #   "terminate program"
    # --------------------------------------------------
    for item in knowledge_data:
        q_reduced = reduce_question(item["question"])

        if DEBUG_SEARCH:
            print(f"[DEBUG] reduced compare with: {q_reduced}")

        if user_reduced and q_reduced and user_reduced == q_reduced:
            if DEBUG_SEARCH:
                print("[DEBUG] EXACT REDUCED MATCH FOUND")
            return item["answer"]

    # --------------------------------------------------
    # 3) Strong contains match
    # Only allow contains if the shorter side has at least 2 words.
    # This avoids bad matches like:
    #   user = "population egypt"
    #   json = "where egypt"
    # where only one useful word overlaps.
    # --------------------------------------------------
    for item in knowledge_data:
        q_reduced = reduce_question(item["question"])
        q_tokens = set(q_reduced.split())

        if not user_reduced or not q_reduced:
            continue

        shorter_token_count = min(len(user_tokens), len(q_tokens))

        if shorter_token_count >= 2:
            if q_reduced in user_reduced or user_reduced in q_reduced:
                if DEBUG_SEARCH:
                    print("[DEBUG] STRONG CONTAINS MATCH FOUND")
                return item["answer"]

    # --------------------------------------------------
    # 4) High token overlap with minimum important-word match
    #
    # We only accept token overlap if:
    #   - score >= MIN_MATCH_SCORE
    #   - AND at least 2 important tokens overlap
    #
    # This prevents:
    #   "population of egypt"
    # from matching
    #   "where is egypt"
    #
    # because overlap would only be:
    #   {"egypt"}  -> only 1 important token
    # --------------------------------------------------
    best_score = 0.0
    best_answer = None
    best_question = None

    for item in knowledge_data:
        q_reduced = reduce_question(item["question"])
        q_tokens = set(q_reduced.split())

        common_tokens = user_tokens.intersection(q_tokens)

        if not q_tokens:
            continue

        score = len(common_tokens) / max(len(q_tokens), 1)

        if DEBUG_SEARCH:
            print(f"[DEBUG] overlap vs '{item['question']}': score={score:.2f}, common={common_tokens}")

        # Require at least 2 meaningful common words
        if len(common_tokens) >= 2 and score > best_score:
            best_score = score
            best_answer = item["answer"]
            best_question = item["question"]

    if DEBUG_SEARCH:
        print(f"[DEBUG] best score: {best_score:.2f}")
        print(f"[DEBUG] best question match: {best_question}")

    if best_score >= MIN_MATCH_SCORE:
        if DEBUG_SEARCH:
            print("[DEBUG] TOKEN OVERLAP MATCH ACCEPTED")
        return best_answer

    if DEBUG_SEARCH:
        print("[DEBUG] NO JSON MATCH FOUND")

    return None


# ----------------------------------------------------------
# ASK MODEL (STREAM OUTPUT)
# ----------------------------------------------------------

def ask_model_stream(model_name, prompt, system_prompt=None, options=None):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "keep_alive": KEEP_ALIVE
    }

    if system_prompt:
        payload["system"] = system_prompt

    if options:
        payload["options"] = options

    full_text = ""

    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT
        ) as r:

            r.raise_for_status()

            print("\nSOURCE: LLAMA MODEL")
            print(f"{model_name}: ", end="", flush=True)

            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue

                    chunk = data.get("response", "")

                    if chunk:
                        print(chunk, end="", flush=True)
                        full_text += chunk

            print("\n")

    except Exception as e:
        print(f"\n[ERROR] Model request failed: {e}\n")
        return ""

    return full_text


# ----------------------------------------------------------
# MAIN QUESTION ROUTER
# ----------------------------------------------------------

def answer_user_question(user_text):
    local_answer = search_knowledge(user_text)

    if local_answer:
        print("\nSOURCE: SMART JSON")
        print("SMART JSON:", local_answer, "\n")
        return

    print("[INFO] Question not found in smart_knowledge.json")
    print("[INFO] Asking small LLaMA model...")

    response = ask_model_stream(
        model_name=SMALL_MODEL,
        prompt=user_text,
        system_prompt=SYSTEM_PROMPT_SMALL,
        options=SMALL_MODEL_OPTIONS
    )

    if response and response.strip():
        print("[INFO] Small model returned an answer.")
        save_smart_knowledge(user_text, response)
        return

    if USE_LARGE_MODEL_FALLBACK:
        print("[INFO] Small model returned no usable answer.")
        print("[INFO] Asking larger LLaMA model...")

        response = ask_model_stream(
            model_name=LARGE_MODEL,
            prompt=user_text,
            system_prompt=SYSTEM_PROMPT_LARGE,
            options=LARGE_MODEL_OPTIONS
        )

        if response and response.strip():
            print("[INFO] Large model returned an answer.")
            save_smart_knowledge(user_text, response)
            return

    print("[WARNING] No usable answer was returned.")


# ----------------------------------------------------------
# HELP COMMANDS
# ----------------------------------------------------------

def print_help():
    print_line()
    print("Commands:")
    print("  exit        -> quit the program")
    print("  /ps         -> show Ollama processor / GPU / CPU info")
    print("  /reload     -> reload smart_knowledge.json")
    print("  /path       -> show full path of smart_knowledge.json")
    print_line()


# ----------------------------------------------------------
# MAIN PROGRAM
# ----------------------------------------------------------

def main():
    print_line()
    print("Self-Learning JSON + LLaMA Chatbot")
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

    load_knowledge()
    show_processor_info(ollama_exe)

    print_help()
    print("Chatbot ready.")
    print("Flow: JSON → LLaMA → JSON learning")
    print(f"Knowledge file: {os.path.abspath(SMART_JSON_FILE)}")
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

        if user_text.lower() == "/reload":
            load_knowledge()
            continue

        if user_text.lower() == "/path":
            print(os.path.abspath(SMART_JSON_FILE))
            continue

        if not user_text:
            continue

        answer_user_question(user_text)


# ----------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()