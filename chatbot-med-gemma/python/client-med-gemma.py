"""Windows desktop GUI for MedGemma via local or remote Ollama.

Run: py "client-med-gemma(1).py"

The client connects to an Ollama server on Windows or Linux. Enter the
server's network address in the GUI. This client does not install models or
start server processes. All imports belong to Python's standard library.
"""

import json
import os
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def load_settings():
    """Read simple KEY=VALUE lines from .env next to this script.

    Existing Windows environment variables take precedence over .env values.
    Empty lines and lines starting with # are ignored. No pip package needed.
    """
    values = {}
    path = Path(__file__).resolve().parent / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key in ("OLLAMA_URL", "MEDGEMMA_MODEL"):
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                values[key] = value
    return values


settings_from_file = load_settings()
DEFAULT_SERVER_URL = os.environ.get(
    "OLLAMA_URL", settings_from_file.get("OLLAMA_URL", "http://127.0.0.1:11434")
)
DEFAULT_MODEL = os.environ.get(
    "MEDGEMMA_MODEL", settings_from_file.get("MEDGEMMA_MODEL", "medgemma1.5:4b")
)
DEFAULT_MODELS = (DEFAULT_MODEL,)


def base_url(value):
    """Accept a base URL or an /api/generate URL from the GUI."""
    value = value.strip().rstrip("/")
    if value.endswith("/api/generate"):
        value = value.removesuffix("/api/generate")
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Enter a full server URL, such as http://192.168.1.50:11434")
    if parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("Enter only the server address and port, without a path or key.")
    return value


def list_models(url):
    """GET /api/tags: verifies connectivity and lists installed models."""
    with urlopen(f"{url}/api/tags", timeout=5) as response:
        data = json.load(response)
    return [item["name"] for item in data.get("models", []) if item.get("name")]


def generate(url, model, question):
    """POST /api/generate and extract the complete answer from Ollama."""
    prefix = (
        "This is for educational demonstration only. Provide a clear educational "
        "explanation. Do not provide a personal diagnosis or treatment plan.\n\n"
    )
    payload = {
        "model": model,
        "prompt": prefix + question,
        "stream": False,
        "options": {"num_predict": 300, "temperature": 0.1},
    }
    request = Request(
        f"{url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            data = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Cannot reach Ollama: {error.reason}") from error
    answer = data.get("response", "")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The server returned no answer.")
    return answer.strip()


def show_status(message):
    """Only the Tkinter main thread calls this function."""
    status_label.config(text=message)


def finish(answer=None, error=None):
    """Display a finished answer or error and restore the Send button."""
    send_button.config(state=tk.NORMAL)
    show_status("Completed" if error is None else "Request failed")
    answer_box.delete("1.0", tk.END)
    answer_box.insert(tk.END, answer if error is None else f"Error: {error}")


def run_request(url, model, question):
    """The HTTP call runs in the background so the window stays responsive."""
    try:
        # Give a useful message before trying generation with a missing model.
        models = list_models(url)
        if model not in models and f"{model}:latest" not in models:
            raise RuntimeError(
                f"Model {model} is not installed on the selected server. "
                "Check the model name and server address."
            )
        result = generate(url, model, question)
        window.after(0, lambda: finish(answer=result))
    except Exception as error:
        message = str(error)
        window.after(0, lambda: finish(error=message))


def send_question():
    """Validate GUI values and send one question to the selected model."""
    question = prompt_box.get("1.0", tk.END).strip()
    model = model_name.get().strip()
    try:
        url = base_url(server_url.get())
        if not question or not model:
            raise ValueError("Enter a question and a model name.")
    except ValueError as error:
        show_status(str(error))
        return
    send_button.config(state=tk.DISABLED)
    answer_box.delete("1.0", tk.END)
    show_status(f"Waiting for {model}...")
    threading.Thread(target=run_request, args=(url, model, question), daemon=True).start()


def refresh_models():
    """Read the model list from the selected local or remote Ollama server."""
    try:
        url = base_url(server_url.get())
    except ValueError as error:
        show_status(str(error))
        return
    refresh_button.config(state=tk.DISABLED)
    show_status("Checking Ollama...")

    def worker():
        try:
            models = list_models(url)
            def complete():
                refresh_button.config(state=tk.NORMAL)
                model_picker.configure(values=models or DEFAULT_MODELS)
                show_status(f"Connected. {len(models)} installed model(s): {', '.join(models)}")
            window.after(0, complete)
        except Exception as error:
            message = str(error)
            def failed():
                refresh_button.config(state=tk.NORMAL)
                show_status(f"Cannot reach the selected server: {message}")
            window.after(0, failed)

    threading.Thread(target=worker, daemon=True).start()


# Build the Tkinter desktop interface on the main thread.
window = tk.Tk()
window.title("MedGemma Client")
window.geometry("760x690")

settings = ttk.LabelFrame(window, text="Ollama server and model")
settings.pack(fill=tk.X, padx=12, pady=12)
settings.columnconfigure(1, weight=1)
server_url = tk.StringVar(value=DEFAULT_SERVER_URL)
model_name = tk.StringVar(value=DEFAULT_MODEL)
ttk.Label(settings, text="Server URL:").grid(row=0, column=0, padx=8, pady=7, sticky="w")
ttk.Entry(settings, textvariable=server_url).grid(row=0, column=1, padx=8, pady=7, sticky="ew")
ttk.Label(settings, text="Model:").grid(row=1, column=0, padx=8, pady=7, sticky="w")
model_picker = ttk.Combobox(settings, textvariable=model_name, values=DEFAULT_MODELS)
model_picker.grid(row=1, column=1, padx=8, pady=7, sticky="ew")

controls = ttk.Frame(window)
controls.pack(fill=tk.X, padx=12)
refresh_button = ttk.Button(controls, text="Check server / models", command=refresh_models)
refresh_button.pack(side=tk.LEFT)

# Check the configured server after the window appears; this does not launch
# or install software on another computer.
window.after(300, refresh_models)

ttk.Label(window, text="Question:").pack(anchor="w", padx=12, pady=(14, 4))
prompt_box = scrolledtext.ScrolledText(window, height=7, wrap=tk.WORD)
prompt_box.pack(fill=tk.X, padx=12)
send_button = ttk.Button(window, text="Send question", command=send_question)
send_button.pack(pady=10)
status_label = ttk.Label(window, text="Ready. Select a server and model.", wraplength=730)
status_label.pack(anchor="w", padx=12)
ttk.Label(window, text="Answer:").pack(anchor="w", padx=12, pady=(12, 4))
answer_box = scrolledtext.ScrolledText(window, wrap=tk.WORD)
answer_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

window.mainloop()
