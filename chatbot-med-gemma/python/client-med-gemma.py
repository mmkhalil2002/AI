"""Windows MedGemma client: how text and image data reach Ollama.

Run: py "client-med-gemma(1).py" (install Pillow: py -m pip install pillow).

INPUT AND SELECTION
-------------------
Type a question in the left text box. The right panel displays images found
in the 'data' directory next to this script. Click ONE thumbnail to select
the corresponding image; click it again to remove the selection. The gallery
thumbnails are 64 x 64 pixels for display ONLY. They are not uploaded.

HOW THE REQUEST IS FORMED
-------------------------
1. The text box contains a Python string. An educational instruction is
   prepended to this question. It becomes the JSON field 'prompt'.
2. If an image is selected, the worker reads that image from disk. JPEG and
   PNG bytes are used as-is; other supported formats are converted to PNG in
   memory. This preserves image resolution (the thumbnail is never sent).
3. base64.b64encode converts those binary bytes to printable ASCII text.
   The resulting string becomes one item in the JSON 'images' array. An
   image path is NOT sent; Ollama receives the actual image contents.
4. A Python dictionary is serialized with json.dumps, encoded as UTF-8, and
   sent as the body of HTTP POST <server URL>/api/generate with the header
   Content-Type: application/json. 'stream': false asks for one JSON reply.

Example request shape (the base64 string is much longer in reality):
    {
      "model": "medgemma1.5:4b",
      "prompt": "Educational instruction...\\n\\nWhat is in this image?",
      "images": ["iVBORw0KGgo..."],
      "stream": false,
      "options": {"num_predict": 300, "temperature": 0.1}
    }

If no image is selected, the 'images' field is omitted and only text is sent.
The response JSON contains a 'response' string, displayed in the left panel.
The client runs on Windows; Ollama can run on a Windows or Linux server.
"""

import base64
import io
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
IMAGE_DIR = Path(__file__).resolve().parent / "data"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
THUMBNAIL_SIZE = (64, 64)


def select_image(path, button):
    """Record one clicked image, with a visible pressed-button selection."""
    previous = selected_image["button"]
    if previous is not None:
        previous.config(relief=tk.RAISED)
    if selected_image["path"] == path:
        selected_image.update(path=None, button=None)
        selected_image_label.config(text="No image selected (text only)")
    else:
        selected_image.update(path=path, button=button)
        button.config(relief=tk.SUNKEN)
        selected_image_label.config(text=f"Selected: {path.name}")


def encode_image(path):
    """Return the original full-resolution image as a base64 ASCII string."""
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        image_bytes = path.read_bytes()
    else:
        from PIL import Image
        with Image.open(path) as source:
            buffer = io.BytesIO()
            source.convert("RGB").save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode("ascii")


def load_images():
    """Display data-folder images in three columns at exactly 64 x 64 pixels."""
    # Destroy old widgets before reloading; keep references to PhotoImage
    # objects so Tkinter does not remove the image data from the screen.
    for widget in image_grid.winfo_children():
        widget.destroy()
    thumbnails.clear()
    selected_image.update(path=None, button=None)
    selected_image_label.config(text="No image selected (text only)")
    IMAGE_DIR.mkdir(exist_ok=True)
    paths = sorted(
        (p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )
    if not paths:
        ttk.Label(image_grid, text=f"No images found in:\n{IMAGE_DIR}", wraplength=230).grid(
            row=0, column=0, columnspan=3, padx=6, pady=12
        )
        image_count.config(text="0 images")
        return

    try:
        from PIL import Image, ImageOps, ImageTk
    except ImportError:
        ttk.Label(
            image_grid,
            text="To display JPG, PNG, and other images, run:\npy -m pip install pillow",
            wraplength=230,
        ).grid(row=0, column=0, columnspan=3, padx=6, pady=12)
        image_count.config(text=f"{len(paths)} images; Pillow required")
        return

    errors = 0
    for path in paths:
        try:
            with Image.open(path) as source:
                # Preserve the whole picture within the requested fixed size.
                picture = ImageOps.pad(
                    source.convert("RGB"), THUMBNAIL_SIZE,
                    method=Image.Resampling.LANCZOS, color="white",
                )
            photo = ImageTk.PhotoImage(picture)
        except (OSError, ValueError):
            errors += 1
            continue

        thumbnails.append(photo)
        index = len(thumbnails) - 1
        tile = ttk.Frame(image_grid)
        tile.grid(row=index // 3, column=index % 3, padx=5, pady=6, sticky="n")
        button = tk.Button(tile, image=photo, relief=tk.RAISED, borderwidth=2)
        button.config(command=lambda p=path, b=button: select_image(p, b))
        button.pack()
        ttk.Label(tile, text=path.name[:12], width=12, anchor="center").pack()

    image_count.config(text=f"{len(thumbnails)} images" + (f"; {errors} unreadable" if errors else ""))
    image_grid.update_idletasks()
    image_canvas.configure(scrollregion=image_canvas.bbox("all"))


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


def generate(url, model, question, image_path=None):
    """POST prompt and optionally full-resolution image to Ollama."""
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
    if image_path is not None:
        # Ollama /api/generate expects base64 strings inside "images".
        # The gallery's resized PhotoImage object is NOT used here.
        payload["images"] = [encode_image(image_path)]
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


def run_request(url, model, question, image_path):
    """The HTTP call runs in the background so the window stays responsive."""
    try:
        # Give a useful message before trying generation with a missing model.
        models = list_models(url)
        if model not in models and f"{model}:latest" not in models:
            raise RuntimeError(
                f"Model {model} is not installed on the selected server. "
                "Check the model name and server address."
            )
        result = generate(url, model, question, image_path)
        window.after(0, lambda: finish(answer=result))
    except Exception as error:
        message = str(error)
        window.after(0, lambda: finish(error=message))


def send_question():
    """Validate GUI values and send one question to the selected model."""
    question = prompt_box.get("1.0", tk.END).strip()
    image_path = selected_image["path"]
    if not question and image_path is not None:
        question = "Describe this image for educational purposes."
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
    show_status(f"Waiting for {model}" + (f" with {image_path.name}" if image_path else " (text only)") + "...")
    threading.Thread(
        target=run_request, args=(url, model, question, image_path), daemon=True
    ).start()


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
window.geometry("1060x690")

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

# Place text on the left and the image gallery beside it on the right.
body = ttk.Frame(window)
body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 12))
body.columnconfigure(0, weight=2)
body.columnconfigure(1, weight=1)
body.rowconfigure(0, weight=1)

text_panel = ttk.Frame(body)
text_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
ttk.Label(text_panel, text="Question:").pack(anchor="w", pady=(4, 4))
prompt_box = scrolledtext.ScrolledText(text_panel, height=7, wrap=tk.WORD)
prompt_box.pack(fill=tk.X)
send_button = ttk.Button(text_panel, text="Send question", command=send_question)
send_button.pack(pady=10)
status_label = ttk.Label(text_panel, text="Ready. Select a server and model.", wraplength=690)
status_label.pack(anchor="w")
ttk.Label(text_panel, text="Answer:").pack(anchor="w", pady=(12, 4))
answer_box = scrolledtext.ScrolledText(text_panel, wrap=tk.WORD)
answer_box.pack(fill=tk.BOTH, expand=True)

image_panel = ttk.LabelFrame(body, text="Images from data folder")
image_panel.grid(row=0, column=1, sticky="nsew")
image_panel.rowconfigure(2, weight=1)
image_panel.columnconfigure(0, weight=1)
gallery_controls = ttk.Frame(image_panel)
gallery_controls.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
image_count = ttk.Label(gallery_controls, text="Loading images...")
image_count.pack(side=tk.LEFT)
ttk.Button(gallery_controls, text="Refresh images", command=load_images).pack(side=tk.RIGHT)
selected_image = {"path": None, "button": None}
selected_image_label = ttk.Label(
    image_panel, text="No image selected (text only)", wraplength=260
)
selected_image_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 5))

image_canvas = tk.Canvas(image_panel, highlightthickness=0)
image_canvas.grid(row=2, column=0, sticky="nsew")
image_scroll = ttk.Scrollbar(image_panel, orient=tk.VERTICAL, command=image_canvas.yview)
image_scroll.grid(row=2, column=1, sticky="ns")
image_canvas.configure(yscrollcommand=image_scroll.set)
image_grid = ttk.Frame(image_canvas)
gallery_window = image_canvas.create_window((0, 0), window=image_grid, anchor="nw")
image_grid.bind(
    "<Configure>", lambda event: image_canvas.configure(scrollregion=image_canvas.bbox("all"))
)
image_canvas.bind(
    "<Configure>", lambda event: image_canvas.itemconfigure(gallery_window, width=event.width)
)
thumbnails = []

# Load images and check the server after the complete window is assembled.
window.after(100, load_images)
window.after(300, refresh_models)

window.mainloop()