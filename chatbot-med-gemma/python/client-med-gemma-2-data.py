"""Windows MedGemma client: how text and image data reach Ollama.

Run: py "med-gemma.py" (install Pillow: py -m pip install pillow).

INPUT AND SELECTION
-------------------
Type a question in the left text box. Make two folders such as
date-091826 and date-092526 beside this script (or inside its data folder). Their images appear
in two separate date-labeled columns. You can also use Images > Choose dataset
folder to select an image folder elsewhere on your computer. Click thumbnails in either column to
select multiple images; click a selected thumbnail again to deselect it.
Selections in the first column are blue and in the second column are orange.
Images are sent in column order, then in click order within each column. The
folder names are visible in the GUI but are not added to your prompt or sent.
Gallery thumbnails use IMAGE_WIDTH x IMAGE_HEIGHT from .env as their maximum
display size (256 x 256 pixels by default). If the window is narrow, they
shrink to fit three across in each dataset column. They are not uploaded.

HOW THE REQUEST IS FORMED
-------------------------
1. The text box contains your question as a Python string. Its exact text
   becomes the JSON field 'prompt'; no fixed instruction is added.
2. For each selected image, the worker reads that image from disk. JPEG and
   PNG bytes are used as-is; other supported formats are converted to PNG in
   memory. This preserves image resolution (the thumbnail is never sent).
3. base64.b64encode converts those binary bytes to printable ASCII text.
   Each resulting string becomes one item in the JSON 'images' array. Images
   from the earlier date come first, then the later date; within each column
   they follow click order. Paths and date labels are NOT sent to Ollama.
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
from datetime import datetime
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
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
            if key in ("OLLAMA_URL", "MEDGEMMA_MODEL", "IMAGE_DIR", "IMAGE_WIDTH", "IMAGE_HEIGHT", "WINDOW_WIDTH", "WINDOW_HEIGHT"):
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
# IMAGE_DIR can be an absolute path or relative to the directory containing
# this script. Each automatic dataset is a date-MMDDYY folder within it.
image_dir_setting = os.environ.get("IMAGE_DIR", settings_from_file.get("IMAGE_DIR", "data"))
IMAGE_DIR = Path(image_dir_setting).expanduser()
if not IMAGE_DIR.is_absolute():
    IMAGE_DIR = Path(__file__).resolve().parent / IMAGE_DIR
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
DATASET_COLORS = (("#1976d2", "#90c6ff"), ("#c05713", "#ffd197"))


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


def select_image(path, button, tile, column):
    """Toggle one image with the selection color of its dataset column."""
    selection = selected_images[column]
    if path in selection:
        del selection[path]
        button.config(relief=tk.RAISED, background=default_button_color)
        tile.config(background="#eeeeee")
    else:
        selection[path] = (button, tile)
        border, fill = DATASET_COLORS[column]
        button.config(relief=tk.SUNKEN, background=fill)
        tile.config(background=border)
    # The colored image border is the visible selection indicator.


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
            buffer = io.BytesIO()
            source.convert("RGB").save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
    # JSON contains text, not raw bytes. Base64 represents the image bytes as
    # ASCII text; Ollama decodes this string when it receives the request.
    return base64.b64encode(image_bytes).decode("ascii")


def dated_folders():
    """Discover datasets beside this script and inside IMAGE_DIR."""
    script_dir = Path(__file__).resolve().parent
    candidates = []
    seen = set()
    # Look in both places. A configured IMAGE_DIR takes priority if a folder
    # of the same name also exists beside the script.
    for parent in (IMAGE_DIR, script_dir):
        if not parent.is_dir():
            continue
        for folder in parent.iterdir():
            if not folder.is_dir() or folder.name.startswith(".") or folder.name == "__pycache__":
                continue
            # IMAGE_DIR is a container for datasets, not itself a second
            # dataset when we also scan its parent (the script directory).
            if parent == script_dir and folder.resolve() == IMAGE_DIR.resolve():
                continue
            if folder.resolve() in seen:
                continue
            seen.add(folder.resolve())
            # Accept date-MMDDYY first, but also allow two ordinary image
            # folders placed next to the script, regardless of their names.
            try:
                date = datetime.strptime(folder.name[5:], "%m%d%y") if folder.name.lower().startswith("date-") else None
                if date and date.strftime("%m%d%y") != folder.name[5:]:
                    date = None
            except ValueError:
                date = None
            has_images = any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.rglob("*"))
            if has_images:
                candidates.append((date, folder))
    # Also handle loose images placed directly beside the script or directly
    # in the configured data directory when there are no two image folders.
    for parent in (script_dir, IMAGE_DIR):
        if parent.is_dir() and parent.resolve() not in seen:
            if any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in parent.iterdir()):
                candidates.append((None, parent))
                seen.add(parent.resolve())
    dated = sorted((item for item in candidates if item[0] is not None), key=lambda item: item[0])
    ordinary = sorted((item for item in candidates if item[0] is None), key=lambda item: item[1].name.lower())
    chosen = dated[-2:] if len(dated) >= 2 else dated + ordinary[:2 - len(dated)]
    return [folder for _, folder in chosen]


def choose_dataset(column):
    """Allow each image column to display an explicitly selected folder."""
    # Windows' folder picker returns an empty string when the user cancels.
    chosen = filedialog.askdirectory(
        title=f"Choose image folder for dataset {column + 1}",
        initialdir=str(dataset_overrides[column] or IMAGE_DIR),
    )
    if chosen:
        dataset_overrides[column] = Path(chosen)
        load_images()


def load_images():
    """Show each dated dataset in its own independently scrolling column."""
    # Destroy old widgets before reloading; keep PhotoImage references so
    # Tkinter does not remove the image pixels from the screen.
    for gallery in galleries:
        for widget in gallery["grid"].winfo_children():
            widget.destroy()
    thumbnails.clear()
    for selection in selected_images:
        selection.clear()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    folders = dated_folders()

    try:
        from PIL import Image, ImageOps, ImageTk
    except ImportError:
        Image = ImageOps = ImageTk = None

    for column, gallery in enumerate(galleries):
        grid = gallery["grid"]
        # Explicit folder choices take priority; discovered image folders
        # beside the script or inside IMAGE_DIR fill the remaining columns.
        folder = dataset_overrides[column] or (folders[column] if column < len(folders) else None)
        if folder is None:
            gallery["title"].config(text=f"Dataset {column + 1} (missing)")
            ttk.Label(grid, text="No dataset found. Put an image folder beside this script, or select one from the Images menu.", wraplength=220).grid(row=0, column=0, padx=5, pady=12)
            continue
        gallery["title"].config(text=f"{folder.name} ({'blue' if column == 0 else 'orange'} selection)")
        paths = sorted((p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: str(p.relative_to(folder)).lower())
        if not paths:
            ttk.Label(grid, text="No supported images in this folder. Select another folder from the Images menu.", wraplength=220).grid(row=0, column=0, padx=5, pady=12)
        if Image is None and paths:
            ttk.Label(grid, text="Install Pillow: py -m pip install pillow", wraplength=220).grid(row=0, column=0, padx=5, pady=12)
            continue
        errors = 0
        displayed = 0
        for path in paths:
            try:
                with Image.open(path) as source:
                    # Only the displayed thumbnail is resized; encode_image
                    # later reads the original source when sending to Ollama.
                    picture = ImageOps.pad(source.convert("RGB"), gallery_thumbnail_size, method=Image.Resampling.LANCZOS, color="white")
                photo = ImageTk.PhotoImage(picture)
            except (OSError, ValueError):
                errors += 1
                continue
            thumbnails.append(photo)
            tile = tk.Frame(grid, background="#eeeeee", padx=3, pady=3)
            # Three pictures per row within each of the two dataset columns.
            # gallery_thumbnail_size is calculated from the space available
            # on this monitor, so the third picture stays inside the viewport.
            tile.grid(row=displayed // 3, column=displayed % 3, padx=2, pady=5, sticky="n")
            button = tk.Button(tile, image=photo, relief=tk.RAISED, borderwidth=2)
            button.config(command=lambda p=path, b=button, t=tile, c=column: select_image(p, b, t, c))
            button.pack()
            ttk.Label(tile, text=path.name[:14], anchor="center", wraplength=thumbnail_width).pack()
            displayed += 1
        grid.update_idletasks()
        gallery["canvas"].configure(scrollregion=gallery["canvas"].bbox("all"))


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
        # image_paths contains the first date's selection followed by the
        # second date's selection, preserving click order within each. The gallery's
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
    # Capture blue-column images followed by orange-column images. Click order
    # within each dataset is retained. Later clicks cannot change this request.
    image_paths = tuple(path for group in selected_images for path in group)
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
window.title("MedGemma Client - two image datasets v12")
# Reserve two image columns beside the question/answer column.
gallery_width = 2 * (3 * (THUMBNAIL_SIZE[0] + 14) + 30) + 30
# WINDOW_WIDTH and WINDOW_HEIGHT in .env can override the initial size.
preferred_width = gallery_width + 540
window_width = window_dimension("WINDOW_WIDTH", max(2400, round(preferred_width * 1.5)))
window_height = window_dimension("WINDOW_HEIGHT", 850)
# A requested width wider than the physical monitor cannot be fully visible.
# Fit the window on screen and allow the image panel to scroll if necessary.
visible_width = min(window_width, window.winfo_screenwidth() - 30)
visible_height = min(window_height, window.winfo_screenheight() - 80)
window.geometry(f"{visible_width}x{visible_height}")
# Each image column gets 4/15 (26.7%) of the body instead of 1/3 (33.3%):
# that is a 20% width reduction. The text area receives the freed space and
# takes 7/15 (46.7%). Allow for pane borders, padding, and scrollbars.
gallery_viewport_width = max(105, round((visible_width - 45) * 4 / 15) - 24)
thumbnail_width = min(THUMBNAIL_SIZE[0], max(24, (gallery_viewport_width - 45) // 3))
thumbnail_height = max(1, round(THUMBNAIL_SIZE[1] * thumbnail_width / THUMBNAIL_SIZE[0]))
gallery_thumbnail_size = (thumbnail_width, thumbnail_height)

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
# The text pane gets 7 shares; the two galleries together get 8 shares.
# Both gallery columns below get equal width (4 shares each).
body.columnconfigure(0, weight=7, minsize=190, uniform="body_parts")
body.columnconfigure(1, weight=8, minsize=gallery_viewport_width * 2 + 32, uniform="body_parts")
body.rowconfigure(0, weight=1)

text_panel = ttk.Frame(body)
text_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
ttk.Label(text_panel, text="Question:").pack(anchor="w", pady=(4, 4))
prompt_box = scrolledtext.ScrolledText(text_panel, width=18, height=4, wrap=tk.WORD)
prompt_box.pack(fill=tk.X)
send_button = ttk.Button(text_panel, text="Send question", command=send_question)
send_button.pack(pady=10)
status_label = ttk.Label(text_panel, text="Ready. Select a server and model.", wraplength=240)
status_label.pack(anchor="w")
ttk.Label(text_panel, text="Answer:").pack(anchor="w", pady=(12, 4))
answer_box = scrolledtext.ScrolledText(text_panel, width=18, wrap=tk.WORD)
answer_box.pack(fill=tk.BOTH, expand=True)

image_panel = ttk.LabelFrame(body, text="Dated image datasets")
image_panel.grid(row=0, column=1, sticky="nsew")
image_panel.rowconfigure(0, weight=1)
image_panel.columnconfigure(0, weight=1)
image_panel.columnconfigure(1, weight=1)
selected_images = [{}, {}]  # Blue first, then orange; each keeps click order.
dataset_overrides = [None, None]  # Optional folders selected through Images menu.
sample_button = tk.Button(image_panel)
default_button_color = sample_button.cget("background")
sample_button.destroy()

# Keep folder selection and refresh available without taking gallery space.
menu_bar = tk.Menu(window)
images_menu = tk.Menu(menu_bar, tearoff=False)
images_menu.add_command(label="Choose dataset 1 folder...", command=lambda: choose_dataset(0))
images_menu.add_command(label="Choose dataset 2 folder...", command=lambda: choose_dataset(1))
images_menu.add_command(label="Refresh images", command=load_images)
menu_bar.add_cascade(label="Images", menu=images_menu)
window.config(menu=menu_bar)

galleries = []
for column in range(2):
    pane = ttk.LabelFrame(image_panel, text=f"Dataset {column + 1}")
    pane.grid(row=0, column=column, sticky="nsew", padx=(0, 5) if column == 0 else (5, 0))
    pane.columnconfigure(0, weight=1)
    pane.rowconfigure(0, weight=1)
    canvas = tk.Canvas(pane, width=gallery_viewport_width, highlightthickness=0)
    canvas.grid(row=0, column=0, sticky="nsew")
    vscroll = ttk.Scrollbar(pane, orient=tk.VERTICAL, command=canvas.yview)
    vscroll.grid(row=0, column=1, sticky="ns")
    hscroll = ttk.Scrollbar(pane, orient=tk.HORIZONTAL, command=canvas.xview)
    hscroll.grid(row=1, column=0, sticky="ew")
    canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
    grid = ttk.Frame(canvas)
    canvas_window = canvas.create_window((0, 0), window=grid, anchor="nw")
    grid.bind("<Configure>", lambda event, c=canvas: c.configure(scrollregion=c.bbox("all")))
    canvas.bind("<Configure>", lambda event, c=canvas, w=canvas_window: c.itemconfigure(
        w, width=max(event.width, 3 * (gallery_thumbnail_size[0] + 14))))
    galleries.append({"title": pane, "canvas": canvas, "grid": grid})
thumbnails = []

# Load images and check the server after the complete window is assembled.
window.after(100, load_images)
window.after(300, refresh_models)

window.mainloop()
