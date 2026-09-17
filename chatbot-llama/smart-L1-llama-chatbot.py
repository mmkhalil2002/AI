# ==========================================================
# MOST GENERAL CROSS-PLATFORM AUTO-INSTALL ROUTINE
# ==========================================================
# PURPOSE
# -------
# This routine provides a reusable dependency installer for
# Python scripts.
#
# It supports:
#
#   1) Ensuring pip exists
#   2) Installing one or more packages
#   3) Mapping import names to pip package names
#   4) Optional version constraints
#   5) Optional extra pip arguments
#   6) Verifying imports after install
#   7) Clean failure handling
#
# Works on:
#   • Windows
#   • Linux / Ubuntu
#   • macOS
#
# IMPORTANT
# ---------
# Put this block at the TOP of your script BEFORE importing
# third-party modules such as:
#
#   import requests
#   import torch
#   import numpy
#   import PIL
# ==========================================================

import sys
import os
import subprocess
import importlib
import tempfile
import urllib.request


# ==========================================================
# HELPER: print status line
# ==========================================================
def install_print_line():
    print("=" * 70)


# ==========================================================
# HELPER: run command
# ==========================================================
def run_command(cmd, quiet=False):
    """
    Run a system command and return the subprocess result.

    PARAMETERS
    ----------
    cmd : list[str]
        Command arguments.

    quiet : bool
        If True, suppress stdout/stderr.

    RETURNS
    -------
    subprocess.CompletedProcess
    """
    if quiet:
        return subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )

    return subprocess.run(cmd, check=False)


# ==========================================================
# STEP 1 — Ensure pip exists
# ==========================================================
def ensure_pip_available(verbose=True):
    """
    Ensure pip is available for the CURRENT Python interpreter.

    CHECK ORDER
    -----------
    1) python -m pip --version
    2) python -m ensurepip --upgrade
    3) download and run get-pip.py

    RETURNS
    -------
    True  -> pip is available
    False -> pip could not be installed
    """

    if verbose:
        install_print_line()
        print("[INFO] Checking whether pip is available...")

    # ------------------------------------------------------
    # Try existing pip first
    # ------------------------------------------------------
    try:
        result = run_command(
            [sys.executable, "-m", "pip", "--version"],
            quiet=True
        )
        if result.returncode == 0:
            if verbose:
                print("[OK] pip is already available.")
            return True
    except Exception:
        pass

    if verbose:
        print("[INFO] pip was not found.")
        print("[INFO] Trying ensurepip...")

    # ------------------------------------------------------
    # Try ensurepip
    # ------------------------------------------------------
    try:
        result = run_command(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            quiet=not verbose
        )

        if result.returncode == 0:
            verify = run_command(
                [sys.executable, "-m", "pip", "--version"],
                quiet=True
            )

            if verify.returncode == 0:
                if verbose:
                    print("[OK] pip installed successfully using ensurepip.")
                return True

    except Exception as e:
        if verbose:
            print(f"[WARNING] ensurepip failed: {e}")

    if verbose:
        print("[INFO] ensurepip did not work.")
        print("[INFO] Trying fallback installation using get-pip.py ...")

    # ------------------------------------------------------
    # Fallback: get-pip.py
    # ------------------------------------------------------
    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp_file:
            temp_path = tmp_file.name

        urllib.request.urlretrieve(get_pip_url, temp_path)

        result = run_command(
            [sys.executable, temp_path],
            quiet=not verbose
        )

        if result.returncode != 0:
            if verbose:
                print("[ERROR] get-pip.py failed.")
            return False

        verify = run_command(
            [sys.executable, "-m", "pip", "--version"],
            quiet=True
        )

        if verify.returncode == 0:
            if verbose:
                print("[OK] pip installed successfully using get-pip.py.")
            return True

        if verbose:
            print("[ERROR] pip still not available after get-pip.py.")
        return False

    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to install pip automatically: {e}")
        return False

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ==========================================================
# STEP 2 — Check import
# ==========================================================
def can_import_module(import_name):
    """
    Check whether a Python module can be imported.

    RETURNS
    -------
    True or False
    """
    try:
        importlib.import_module(import_name)
        return True
    except Exception:
        return False


# ==========================================================
# STEP 3 — Build pip package specifier
# ==========================================================
def build_pip_spec(pip_name, version=None):
    """
    Build the package specifier passed to pip.

    EXAMPLES
    --------
    build_pip_spec("requests")              -> "requests"
    build_pip_spec("numpy", "==1.26.4")     -> "numpy==1.26.4"
    build_pip_spec("torch", ">=2.2")        -> "torch>=2.2"
    """
    if version:
        return f"{pip_name}{version}"
    return pip_name


# ==========================================================
# STEP 4 — Install one package
# ==========================================================
def ensure_python_package(
    import_name,
    pip_name=None,
    version=None,
    upgrade=True,
    user=False,
    extra_pip_args=None,
    verbose=True
):
    """
    Ensure one Python package is installed and importable.

    PARAMETERS
    ----------
    import_name : str
        Name used in Python import statement.

        Example:
            "requests"
            "PIL"
            "cv2"

    pip_name : str or None
        Name used with pip.

        Example:
            import_name="PIL", pip_name="pillow"
            import_name="cv2", pip_name="opencv-python"

    version : str or None
        Optional version constraint.

        Examples:
            "==2.31.0"
            ">=2.0"
            "<3"

    upgrade : bool
        If True, pass --upgrade to pip.

    user : bool
        If True, pass --user to pip.

    extra_pip_args : list[str] or None
        Any extra pip arguments.

        Example:
            ["--index-url", "https://download.pytorch.org/whl/cu121"]

    verbose : bool
        Print status messages.

    RETURNS
    -------
    True  -> package available
    False -> installation failed
    """

    if pip_name is None:
        pip_name = import_name

    if extra_pip_args is None:
        extra_pip_args = []

    # ------------------------------------------------------
    # Try importing first
    # ------------------------------------------------------
    if can_import_module(import_name):
        if verbose:
            print(f"[OK] Python package already installed: {pip_name}")
        return True

    if verbose:
        print(f"[INFO] Missing Python package: {pip_name}")

    # ------------------------------------------------------
    # Ensure pip exists
    # ------------------------------------------------------
    if not ensure_pip_available(verbose=verbose):
        if verbose:
            print("[ERROR] pip is not available.")
        return False

    # ------------------------------------------------------
    # Build pip install command
    # ------------------------------------------------------
    package_spec = build_pip_spec(pip_name, version)

    cmd = [sys.executable, "-m", "pip", "install"]

    if upgrade:
        cmd.append("--upgrade")

    if user:
        cmd.append("--user")

    cmd.extend(extra_pip_args)
    cmd.append(package_spec)

    if verbose:
        print(f"[INFO] Installing Python package: {package_spec}")

    try:
        result = run_command(cmd, quiet=not verbose)

        if result.returncode != 0:
            if verbose:
                print(f"[ERROR] Failed to install package: {package_spec}")
            return False

    except Exception as e:
        if verbose:
            print(f"[ERROR] Exception while installing '{package_spec}': {e}")
        return False

    # ------------------------------------------------------
    # Verify after installation
    # ------------------------------------------------------
    if can_import_module(import_name):
        if verbose:
            print(f"[OK] Installed Python package successfully: {package_spec}")
        return True

    if verbose:
        print(f"[ERROR] Package installed but still cannot be imported: {import_name}")
    return False


# ==========================================================
# STEP 5 — Install many packages
# ==========================================================
def ensure_python_packages(package_specs, verbose=True, stop_on_failure=True):
    """
    Ensure multiple packages are installed.

    PARAMETERS
    ----------
    package_specs : list[dict]
        Each dictionary may contain:

            {
                "import_name": "requests",
                "pip_name": "requests",
                "version": None,
                "upgrade": True,
                "user": False,
                "extra_pip_args": []
            }

    verbose : bool
        Whether to print progress.

    stop_on_failure : bool
        If True, stop immediately when one package fails.

    RETURNS
    -------
    dict with:
        {
            "success": bool,
            "installed": list[str],
            "failed": list[str]
        }
    """

    installed = []
    failed = []

    if verbose:
        install_print_line()
        print("[INFO] Ensuring required Python packages...")

    for spec in package_specs:
        import_name = spec["import_name"]
        pip_name = spec.get("pip_name")
        version = spec.get("version")
        upgrade = spec.get("upgrade", True)
        user = spec.get("user", False)
        extra_pip_args = spec.get("extra_pip_args", [])

        ok = ensure_python_package(
            import_name=import_name,
            pip_name=pip_name,
            version=version,
            upgrade=upgrade,
            user=user,
            extra_pip_args=extra_pip_args,
            verbose=verbose
        )

        display_name = pip_name or import_name

        if ok:
            installed.append(display_name)
        else:
            failed.append(display_name)
            if stop_on_failure:
                break

    success = len(failed) == 0

    if verbose:
        install_print_line()
        print(f"[INFO] Installed/verified packages: {installed}")
        if failed:
            print(f"[ERROR] Failed packages: {failed}")
        else:
            print("[OK] All required Python packages are available.")

    return {
        "success": success,
        "installed": installed,
        "failed": failed
    }


# ==========================================================
# STEP 6 — Optional helper to exit on failure
# ==========================================================
def ensure_python_packages_or_exit(package_specs, verbose=True):
    """
    Ensure packages and exit the program if any fail.
    """
    result = ensure_python_packages(
        package_specs=package_specs,
        verbose=verbose,
        stop_on_failure=True
    )

    if not result["success"]:
        print("[ERROR] Cannot continue because required packages are missing.")
        sys.exit(1)


# ==========================================================
# EXAMPLE USAGE
# ==========================================================
# Define third-party dependencies for THIS script here.
#
# IMPORTANT:
# Do NOT include standard library modules such as:
#   os, sys, json, time, re, math, shutil, subprocess
#
# Only include packages that normally require pip.
# ==========================================================

REQUIRED_PACKAGES = [
    {
        "import_name": "requests",
        "pip_name": "requests",
    },

    # Example mappings:
    # {"import_name": "PIL", "pip_name": "pillow"},
    # {"import_name": "cv2", "pip_name": "opencv-python"},
    # {"import_name": "yaml", "pip_name": "pyyaml"},
    # {"import_name": "numpy", "pip_name": "numpy", "version": ">=1.26"},
    # {
    #     "import_name": "torch",
    #     "pip_name": "torch",
    #     "extra_pip_args": ["--index-url", "https://download.pytorch.org/whl/cpu"]
    # },
]

# ----------------------------------------------------------
# RUN INSTALLER NOW
# ----------------------------------------------------------
ensure_python_packages_or_exit(REQUIRED_PACKAGES, verbose=True)
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
SMART_JSON_FILE = "smart_knowledge_L1.json"

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
    """
    PURPOSE
    -------
    Convert user input into a standardized format so that
    text comparisons become easier and more reliable.

    When users ask questions they may use:
        • different letter cases (AI vs ai)
        • punctuation (?, ., !)
        • extra spaces
        • commas or symbols

    This function removes those differences so two
    sentences with the same meaning become easier to match.

    Example:

        Original input:
            "  How Can I Terminate the Program??? "

        After normalization:
            "how can i terminate the program"
    """

    # ---------------------------------------------------------
    # STEP 1: Convert text to lowercase and remove outer spaces
    # ---------------------------------------------------------

    # text.lower()
    # converts all uppercase letters to lowercase.
    #
    # Example:
    #   "HELLO World" → "hello world"
    #
    # Why?
    # Because comparisons should not fail due to case differences.
    #
    # Example:
    #   "AI"
    #   "ai"
    #   "Ai"
    #
    # should all be treated the same.

    # text.strip()
    # removes spaces at the beginning and end of the string.
    #
    # Example:
    #   "   hello world   " → "hello world"

    text = text.lower().strip()

    # ---------------------------------------------------------
    # STEP 2: Remove punctuation and special characters
    # ---------------------------------------------------------

    # re.sub(pattern, replacement, text)
    #
    # re.sub() is a Regular Expression substitution function.
    # It replaces text that matches a pattern.

    # Pattern used:
    #
    #   r"[^a-z0-9\s]"
    #
    # Explanation:
    #
    #   ^ inside [] means "NOT"
    #
    #   a-z     → lowercase letters
    #   0-9     → digits
    #   \s      → whitespace (spaces, tabs)
    #
    # So the pattern means:
    #
    #   "anything that is NOT a letter, number, or space"

    # All such characters are replaced with a space " ".

    # Example:

    # Input text:
    #   "how can I terminate the program???"

    # After step 1:
    #   "how can i terminate the program???"

    # After this regex:
    #   "how can i terminate the program   "

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # ---------------------------------------------------------
    # STEP 3: Remove extra spaces inside the sentence
    # ---------------------------------------------------------

    # After step 2 we may have multiple spaces.
    #
    # Example:
    #   "how can i terminate the program   "
    #
    # or
    #   "hello     world"

    # Pattern:
    #
    #   r"\s+"
    #
    # Meaning:
    #
    #   \s  → whitespace
    #   +   → one or more occurrences
    #
    # So this pattern matches:
    #
    #   "   "
    #   "     "
    #   "\t\t"

    # Replace them with a single space.

    # Example:
    #
    #   "hello     world" → "hello world"

    text = re.sub(r"\s+", " ", text).strip()

    # strip() again removes spaces that might remain
    # at the beginning or end.

    # Example:
    #
    #   " hello world " → "hello world"

    # ---------------------------------------------------------
    # STEP 4: Return the cleaned text
    # ---------------------------------------------------------

    # Now the text is normalized and ready to be used
    # in comparisons or matching algorithms.

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






def token_overlap_score(user_text, candidate_text):
    """
    PURPOSE
    -------
    Compute similarity between two questions using keyword comparison.

    The function:
        1. Removes stopwords from both questions
        2. Converts questions to sets of important keywords
        3. Computes INTERSECTION and UNION
        4. Calculates Jaccard similarity score

    Similarity score range:
        0.0 → no similarity
        1.0 → identical keywords

    Example questions:

        User question:
            "what is the population of egypt"

        Stored question:
            "population of egypt"

        After reduction both contain:
            {"population","egypt"}

        Result:
            similarity = 2 / 2 = 1.0
    """

    # ---------------------------------------------------------
    # STEP 1 — Reduce the user question
    # ---------------------------------------------------------
    # reduce_question() removes stopwords such as:
    #   what, is, the, of, where, how, etc.
    #
    # Example:
    #
    # user_text:
    #     "what is the population of egypt"
    #
    # reduce_question(user_text) →
    #
    #     "population egypt"

    user_reduced = reduce_question(user_text)

    # Convert the reduced text into tokens using split()
    #
    # Example:
    #
    # "population egypt"
    #
    # becomes:
    #
    # ["population","egypt"]

    # Convert list to set
    #
    # Why use a set?
    #   - removes duplicates
    #   - enables fast intersection/union operations
    #
    # Result:
    #
    # {"population","egypt"}

    user_tokens = set(user_reduced.split())

    # ---------------------------------------------------------
    # STEP 2 — Reduce the candidate question
    # ---------------------------------------------------------
    #
    # candidate_text example:
    #
    # "what is the population of egypt"
    #
    # reduce_question(candidate_text) →
    #
    # "population egypt"

    candidate_reduced = reduce_question(candidate_text)

    candidate_tokens = set(candidate_reduced.split())

    # Example result:
    #
    # {"population","egypt"}

    # ---------------------------------------------------------
    # STEP 3 — Prevent empty token comparison
    # ---------------------------------------------------------
    #
    # If either sentence produced no keywords,
    # similarity cannot be computed.

    if not user_tokens or not candidate_tokens:
        return 0.0

    # ---------------------------------------------------------
    # STEP 4 — Compute INTERSECTION
    # ---------------------------------------------------------
    #
    # Intersection = words appearing in BOTH questions.
    #
    # Mathematical notation:
    #
    #   A ∩ B
    #
    # Example:
    #
    # user_tokens =
    #     {"population","egypt"}
    #
    # candidate_tokens =
    #     {"population","egypt"}
    #
    # intersection =
    #
    #     {"population","egypt"}

    common = user_tokens.intersection(candidate_tokens)

    # ---------------------------------------------------------
    # STEP 5 — Compute UNION
    # ---------------------------------------------------------
    #
    # Union = all unique words appearing in either question.
    #
    # Mathematical notation:
    #
    #   A ∪ B
    #
    # Example:
    #
    # user_tokens =
    #     {"population","egypt"}
    #
    # candidate_tokens =
    #     {"population","egypt"}
    #
    # union =
    #
    #     {"population","egypt"}

    union = user_tokens.union(candidate_tokens)

    # ---------------------------------------------------------
    # DEBUG OUTPUT (for understanding the process)
    # ---------------------------------------------------------

    print("User tokens:", user_tokens)
    print("Candidate tokens:", candidate_tokens)
    print("Common tokens:", common)
    print("Union tokens:", union)

    # ---------------------------------------------------------
    # STEP 6 — Require at least two matching keywords
    # ---------------------------------------------------------
    #
    # Example bad match we want to avoid:
    #
    # user question:
    #     "population of egypt"
    #
    # stored question:
    #     "where is egypt"
    #
    # Reduced tokens:
    #
    # user_tokens = {"population","egypt"}
    # candidate_tokens = {"egypt"}
    #
    # intersection = {"egypt"}
    #
    # Only one word overlaps → weak similarity.
    #
    # So we reject it.

    if len(common) < 2:
        print("Too few matching keywords → score = 0")
        return 0.0

    # ---------------------------------------------------------
    # STEP 7 — Compute Jaccard similarity
    # ---------------------------------------------------------
    #
    # Formula:
    #
    # similarity = |intersection| / |union|
    #
    # Example:
    #
    # intersection size = 2
    # union size = 2
    #
    # similarity = 2 / 2 = 1.0

    score = len(common) / len(union)

    print("Similarity score:", score)

    return score

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

    # ------------------------------------------------------------
    # FIRST: search local JSON knowledge before calling the model.
    #
    # If a matching answer is found in smart_knowledge.json,
    # return it immediately and do not call LLaMA.
    # ------------------------------------------------------------
    local_answer = search_knowledge(user_text)

    if local_answer:
        print("\nSOURCE: SMART JSON")
        print("SMART JSON:", local_answer, "\n")
        return

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

        # ------------------------------------------------------------
        # Build EXACT text that was printed to the console
        #
        # This ensures the stored answer matches what the user saw
        # on the screen.
        # ------------------------------------------------------------
        saved_output = (
            f"SOURCE: LLAMA MODEL\n"
            f"{SMALL_MODEL}: {response.strip()}"
        )

        # Save the newly learned answer into smart_knowledge.json
        save_smart_knowledge(user_text, saved_output)

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

            # --------------------------------------------------------
            # Store exactly what was printed to the console
            # --------------------------------------------------------
            saved_output = (
                f"SOURCE: LLAMA MODEL\n"
                f"{LARGE_MODEL}: {response.strip()}"
            )

            # Save the newly learned answer
            save_smart_knowledge(user_text, saved_output)

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