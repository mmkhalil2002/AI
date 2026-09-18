"""Windows MedGemma client: how text and image data reach Ollama.

Run: py "med-gemma.py" (install Pillow: py -m pip install pillow).

INPUT AND SELECTION
-------------------
Type a question in the left text box. The right panel displays images found
in the 'data' directory next to this script. Click thumbnails to select
multiple images; click a selected thumbnail again to deselect it. Selected
images have a blue background and border. The gallery
thumbnails use IMAGE_WIDTH x IMAGE_HEIGHT from .env for display ONLY
(256 x 256 pixels if missing or invalid). They are not uploaded.

HOW THE REQUEST IS FORMED
-------------------------
1. The text box contains your question as a Python string. Its exact text
   becomes the JSON field 'prompt'; no fixed instruction is added.
2. For each selected image, the worker reads that image from disk. JPEG and
   PNG bytes are used as-is; other supported formats are converted to PNG in
   memory. This preserves image resolution (the thumbnail is never sent).
3. base64.b64encode converts those binary bytes to printable ASCII text.
   Each resulting string becomes one item in the JSON 'images' array, in
   selection order. Image paths are NOT sent; Ollama receives image contents.
4. A Python dictionary is serialized with json.dumps, encoded as UTF-8, and
   sent as the body of HTTP POST <server URL>/api/generate with the header
   Content-Type: application/json. 'stream': false asks for one JSON reply.

Example request shape (the base64 string is much longer in reality):
    {
      "model": "medgemma1.5:4b",
      "prompt": "What are these images representing?",
      "images": ["iVBORw0KGgo...", "anotherBase64Image..."],
      "stream": false,
      "options": {"num_predict": 300, "temperature": 0.1}
    }

If no images are selected, the 'images' field is omitted and only text is sent.
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
            if key in ("OLLAMA_URL", "MEDGEMMA_MODEL", "IMAGE_WIDTH", "IMAGE_HEIGHT", "WINDOW_WIDTH", "WINDOW_HEIGHT"):
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


def image_dimension(name):
    """Read a positive thumbnail size; use 256 when absent or invalid."""
    value = os.environ.get(name, settings_from_file.get(name, "256"))
    try:
        number = int(value)
        if 1 <= number <= 2048:
            return number
    except (TypeError, ValueError):
        pass
    print(f"Invalid {name}={value!r}; using 256 pixels.")
    return 256


THUMBNAIL_SIZE = (image_dimension("IMAGE_WIDTH"), image_dimension("IMAGE_HEIGHT"))


def window_dimension(name, default):
    """Use a positive window dimension from .env or the calculated default."""
    value = os.environ.get(name, settings_from_file.get(name, str(default)))
    try:
        dimension = int(value)
        if dimension > 0:
            return dimension
    except (TypeError, ValueError):
        pass
    print(f"Invalid {name}={value!r}; using {default} pixels.")
    return default


def select_image(path, button, tile):
    """Toggle one image; a blue tile makes each selected image easy to see."""
    if path in selected_images:
        del selected_images[path]
        button.config(relief=tk.RAISED, background=default_button_color)
        tile.config(background="#eeeeee")
    else:
        selected_images[path] = (button, tile)
        button.config(relief=tk.SUNKEN, background="#90c6ff")
        tile.config(background="#1976d2")
    count = len(selected_images)
    selected_image_label.config(
        text=f"{count} image{'s' if count != 1 else ''} selected" if count else "No images selected (text only)"
    )


def encode_image(path):
    """Return the original full-resolution image as a base64 ASCII string."""
    # This path identifies a file on the Windows client. Only the encoded
    # contents below are sent to Ollama; the path is never added to the JSON.
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        # PNG/JPEG already have formats Ollama accepts. Read every byte from
        # the original file; the resized gallery thumbnail is not involved.
        image_bytes = path.read_bytes()
    else:
        from PIL import Image
        with Image.open(path) as source:
            # Convert formats such as BMP, GIF, or WebP to PNG in memory.
            # The source keeps its original dimensions; this does not resize it.
            # The current script handles them differently:

            #    PNG, JPG, JPEG: sends the original file bytes, encoded as Base64.
            #    BMP, GIF, WebP, TIFF: converts the image to PNG in memory, then sends those PNG bytes encoded as Base64.
            buffer = io.BytesIO()
            source.convert("RGB").save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
    # JSON contains text, not raw bytes. Base64 represents the image bytes as
    # ASCII text; Ollama decodes this string when it receives the request.
    return base64.b64encode(image_bytes).decode("ascii")


def load_images():
    """Display data-folder images in three columns at the configured size."""
    # Destroy old widgets before reloading; keep references to PhotoImage
    # objects so Tkinter does not remove the image data from the screen.
    for widget in image_grid.winfo_children():
        widget.destroy()
    thumbnails.clear()
    selected_images.clear()
    selected_image_label.config(text="No images selected (text only)")
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
        tile = tk.Frame(image_grid, background="#eeeeee", padx=3, pady=3)
        tile.grid(row=index // 3, column=index % 3, padx=5, pady=6, sticky="n")
        button = tk.Button(tile, image=photo, relief=tk.RAISED, borderwidth=2)
        button.config(command=lambda p=path, b=button, t=tile: select_image(p, b, t))
        button.pack()
        ttk.Label(tile, text=path.name[:12], width=12, anchor="center").pack()

    image_count.config(text=f"{len(thumbnails)} images ({THUMBNAIL_SIZE[0]} x {THUMBNAIL_SIZE[1]} px)" + (f"; {errors} unreadable" if errors else ""))
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


def generate(url, model, question, image_paths=()):
    """POST prompt and all selected full-resolution images to Ollama."""
    # Form a Python dictionary that will become the HTTP request's JSON body.
    # "question" is exactly what the user entered in the Question box (apart
    # from leading/trailing whitespace removed by send_question). No fixed
    # instruction or filename is added to the prompt.
    payload = {
        # The installed Ollama model selected in the GUI.
        "model": model,
        # The user's question, for example: "What do these images show?"
        "prompt": question,
        # Ask Ollama for one complete JSON reply instead of streamed chunks.
        "stream": False,
        # Maximum generated tokens and sampling temperature for the answer.
        "options": {"num_predict": 300, "temperature": 0.1},
    }
    if image_paths:
        # encode_image returns one Base64 string for each selected source file.
        # Dictionary selection order is preserved in image_paths, so the JSON
        # array follows the order in which images were clicked. The gallery's
        # resized PhotoImage objects are never sent. With no selected images,
        # omit the "images" key entirely and submit a text-only request.
        payload["images"] = [encode_image(path) for path in image_paths]
    # json.dumps turns the dictionary into JSON text. UTF-8 then converts that
    # text into bytes for HTTP. Example: {"prompt": "My question", "images":
    # ["<base64 image 1>", "<base64 image 2>"], ...}.
    request = Request(
        # The Ollama server address comes from the GUI/.env; /api/generate is
        # the endpoint that receives a model, a prompt, and optional images.
        f"{url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        # Tell the server that the HTTP body is JSON rather than plain text.
        headers={"Content-Type": "application/json"},
        # POST sends the JSON body to Ollama.
        method="POST",
    )
    try:
        # Network requests can take time. run_request calls this function in a
        # worker thread so the GUI does not freeze while Ollama responds.
        with urlopen(request, timeout=300) as response:
            # Parse Ollama's reply JSON into a Python dictionary.
            data = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Cannot reach Ollama: {error.reason}") from error
    # /api/generate places the generated answer in its "response" field.
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


def run_request(url, model, question, image_paths):
    """The HTTP call runs in the background so the window stays responsive."""
    try:
        # Give a useful message before trying generation with a missing model.
        models = list_models(url)
        if model not in models and f"{model}:latest" not in models:
            raise RuntimeError(
                f"Model {model} is not installed on the selected server. "
                "Check the model name and server address."
            )
        result = generate(url, model, question, image_paths)
        window.after(0, lambda: finish(answer=result))
    except Exception as error:
        message = str(error)
        window.after(0, lambda: finish(error=message))


def send_question():
    """Validate GUI values and send one question to the selected model."""
    # Read all lines in the Question box, then trim surrounding whitespace.
    # This text is passed unchanged as the JSON "prompt" value.
    question = prompt_box.get("1.0", tk.END).strip()
    # Capture the paths of every selected image in click order. This snapshot
    # means that later clicks cannot change a request already in progress.
    image_paths = tuple(selected_images)
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
    show_status(f"Waiting for {model}" + (f" with {len(image_paths)} image(s)" if image_paths else " (text only)") + "...")
    # Pass the question and image snapshot together to the background worker.
    # run_request calls generate(), which builds and sends one HTTP POST.
    threading.Thread(
        target=run_request, args=(url, model, question, image_paths), daemon=True
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
window.title("MedGemma Client - your question only v6")
# Reserve the full width of three thumbnails, including button borders and
# spacing, so the third image is not hidden by the answer pane. On a narrow
# screen, use the horizontal scrollbar underneath the images.
gallery_width = 3 * (THUMBNAIL_SIZE[0] + 24) + 24
# Leave enough space for the question on monitors too narrow for all three
# thumbnails; users can scroll horizontally to reach the third image.
gallery_viewport_width = min(gallery_width, max(280, window.winfo_screenwidth() - 370))
# WINDOW_WIDTH and WINDOW_HEIGHT in .env can override the initial size.
preferred_width = gallery_width + 540
window_width = window_dimension("WINDOW_WIDTH", max(2400, round(preferred_width * 1.5)))
window_height = window_dimension("WINDOW_HEIGHT", 850)
# A requested width wider than the physical monitor cannot be fully visible.
# Fit the window on screen and allow the image panel to scroll if necessary.
visible_width = min(window_width, window.winfo_screenwidth() - 30)
visible_height = min(window_height, window.winfo_screenheight() - 80)
window.geometry(f"{visible_width}x{visible_height}")

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
# Give the image panel enough room before distributing extra window width.
body.columnconfigure(0, weight=1, minsize=310)
body.columnconfigure(1, weight=0, minsize=gallery_viewport_width)
body.rowconfigure(0, weight=1)

text_panel = ttk.Frame(body)
text_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
ttk.Label(text_panel, text="Question:").pack(anchor="w", pady=(4, 4))
prompt_box = scrolledtext.ScrolledText(text_panel, width=35, height=7, wrap=tk.WORD)
prompt_box.pack(fill=tk.X)
send_button = ttk.Button(text_panel, text="Send question", command=send_question)
send_button.pack(pady=10)
status_label = ttk.Label(text_panel, text="Ready. Select a server and model.", wraplength=690)
status_label.pack(anchor="w")
ttk.Label(text_panel, text="Answer:").pack(anchor="w", pady=(12, 4))
answer_box = scrolledtext.ScrolledText(text_panel, width=35, wrap=tk.WORD)
answer_box.pack(fill=tk.BOTH, expand=True)

image_panel = ttk.LabelFrame(body, text="Images from data folder")
image_panel.grid(row=0, column=1, sticky="nsew")
image_panel.rowconfigure(2, weight=1)
image_panel.columnconfigure(0, weight=1)
gallery_controls = ttk.Frame(image_panel)
gallery_controls.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
image_count = ttk.Label(gallery_controls, text=f"Loading images... ({THUMBNAIL_SIZE[0]} x {THUMBNAIL_SIZE[1]} px)")
image_count.pack(side=tk.LEFT)
ttk.Button(gallery_controls, text="Refresh images", command=load_images).pack(side=tk.RIGHT)
selected_images = {}  # Insertion order is the order sent to Ollama.
sample_button = tk.Button(image_panel)
default_button_color = sample_button.cget("background")
sample_button.destroy()
selected_image_label = ttk.Label(
    image_panel, text="No images selected (text only)", wraplength=260
)
selected_image_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 5))

image_canvas = tk.Canvas(image_panel, width=gallery_viewport_width, highlightthickness=0)
image_canvas.grid(row=2, column=0, sticky="nsew")
image_scroll = ttk.Scrollbar(image_panel, orient=tk.VERTICAL, command=image_canvas.yview)
image_scroll.grid(row=2, column=1, sticky="ns")
image_canvas.configure(yscrollcommand=image_scroll.set)
image_hscroll = ttk.Scrollbar(image_panel, orient=tk.HORIZONTAL, command=image_canvas.xview)
image_hscroll.grid(row=3, column=0, sticky="ew")
image_canvas.configure(xscrollcommand=image_hscroll.set)
image_grid = ttk.Frame(image_canvas)
gallery_window = image_canvas.create_window((0, 0), window=image_grid, anchor="nw")
image_grid.bind(
    "<Configure>", lambda event: image_canvas.configure(scrollregion=image_canvas.bbox("all"))
)
image_canvas.bind(
    "<Configure>", lambda event: image_canvas.itemconfigure(
        gallery_window, width=max(event.width, gallery_width - 24)
    )
)
thumbnails = []

# Load images and check the server after the complete window is assembled.
window.after(100, load_images)
window.after(300, refresh_models)

window.mainloop()
