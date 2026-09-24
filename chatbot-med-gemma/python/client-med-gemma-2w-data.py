"""Windows MedGemma client: how text and image data reach Ollama.

Required packages are checked and installed automatically at startup:
    Pillow   - image loading and display
    pyttsx3  - offline Windows text-to-speech

Manual installation, if automatic installation is blocked:
    py -m pip install pillow pyttsx3

Run:
    py "med-gemma.py"

INPUT AND SELECTION
-------------------
Type a question in the left text box. Make folders such as xray-data-091826 and
ekg-data-091826 beside this script (or inside IMAGE_DIR). X-Ray images appear in
the top gallery and EKG images in the bottom gallery. Click a thumbnail to
select it; click it again to deselect it. X-Ray selections are blue and EKG
selections are orange. Selected X-Ray originals are sent first, followed by
selected EKG originals, in click order. Folder names are not sent.
Gallery display sizes come from XRAY_WIDTH, XRAY_HEIGHT, EKG_WIDTH, and
EKG_HEIGHT in .env (256 by default). A zero dimension keeps that original
image dimension. Display thumbnails are never uploaded.
NUM_PREDICT in .env controls the maximum number of tokens generated for each
answer (4096 by default).
TEMPERATURE controls how predictable the wording is. Lower values produce
more focused, repeatable reports; 0.2 is the default.
REPEAT_PENALTY discourages reuse of recently generated words and phrases.
1.0 applies no extra penalty; 1.2 is the default moderate penalty.
REPEAT_LAST_N controls how many recent tokens Ollama checks when applying the
repetition penalty. The default is 256 tokens.
THINK in .env controls whether Ollama enables model reasoning (false by
default). Accepted values include true/false, yes/no, on/off, and 1/0.
GENERATION_TIMEOUT in .env controls how many seconds the client waits for
Ollama to finish one response (900 seconds by default).

HOW THE REQUEST IS FORMED
-------------------------
1. The text box contains your question as a Python string. Its exact text
   becomes the JSON field 'prompt'; no fixed instruction is added.
2. For each selected image, the worker reads that image from disk. JPEG and
   PNG bytes are used as-is; other supported formats are converted to PNG in
   memory. This preserves image resolution (the thumbnail is never sent).
3. base64.b64encode converts those binary bytes to printable ASCII text.
   Each resulting string becomes one item in the JSON 'images' array. Images
   from X-Ray come first, then EKG; within each gallery they follow click
   order. Paths and folder labels are NOT sent to Ollama.
4. A Python dictionary is serialized with json.dumps, encoded as UTF-8, and
   sent as the body of HTTP POST <server URL>/api/generate with the header
   Content-Type: application/json. 'stream': false asks for one JSON reply.

Example request shape (the base64 string is much longer in reality):
    {
      "model": "medgemma1.5:4b",
      "prompt": "What are these images representing?",
      "images": ["iVBORw0KGgo...", "anotherBase64Image..."],
      "stream": false,
      "think": false,
      "options": {
        "num_predict": 4096,
        "temperature": 0.2,
        "repeat_penalty": 1.2,
        "repeat_last_n": 256
      }
    }

If no images are selected, the 'images' field is omitted and only text is sent.
The response JSON contains a 'response' string, displayed in the left panel.
When Voice is ON, the completed response is read aloud with the default
Windows text-to-speech voice. Voice processing runs in a background thread.
The client runs on Windows; Ollama can run on a Windows or Linux server.
"""

import base64
import importlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
from datetime import datetime
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def install_missing_package(import_name, package_name):
    """Install one required package only when its import is unavailable."""
    if importlib.util.find_spec(import_name) is not None:
        return

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                package_name,
            ],
            check=True,
        )
        importlib.invalidate_caches()
    except (OSError, subprocess.CalledProcessError) as error:
        startup_window = tk.Tk()
        startup_window.withdraw()
        messagebox.showerror(
            "Required package installation failed",
            f"Could not install {package_name}.\n\n"
            f"Run this command manually:\n"
            f'"{sys.executable}" -m pip install {package_name}\n\n'
            f"Details: {error}",
            parent=startup_window,
        )
        startup_window.destroy()
        raise SystemExit(1) from error

    if importlib.util.find_spec(import_name) is None:
        raise SystemExit(
            f"{package_name} was installed but {import_name} is still unavailable. "
            "Restart Python and run the script again."
        )


# These fixed, trusted package names are installed only when missing. Using
# sys.executable ensures pip installs them into the Python running this script.
install_missing_package("PIL", "Pillow")
install_missing_package("pyttsx3", "pyttsx3")

import pyttsx3


voice_enabled = False
speech_engine = None
speech_lock = threading.Lock()


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
            if key in (
                "OLLAMA_URL", "MEDGEMMA_MODEL", "IMAGE_DIR", "BLOOD_TEST_DIR",
                "XRAY_WIDTH", "XRAY_HEIGHT", "EKG_WIDTH", "EKG_HEIGHT",
                "WINDOW_WIDTH", "WINDOW_HEIGHT", "NUM_PREDICT", "THINK",
                "GENERATION_TIMEOUT", "TEMPERATURE", "REPEAT_PENALTY",
                "REPEAT_LAST_N",
            ):
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
# this script. Automatic folders are xray-data-MMDDYY and
# ekg-data-MMDDYY within it.
image_dir_setting = os.environ.get("IMAGE_DIR", settings_from_file.get("IMAGE_DIR", "data"))
IMAGE_DIR = Path(image_dir_setting).expanduser()
if not IMAGE_DIR.is_absolute():
    IMAGE_DIR = Path(__file__).resolve().parent / IMAGE_DIR
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
# Cheerful clinical-dashboard palette.
APP_BACKGROUND = "#eef6fb"
CARD_BACKGROUND = "#ffffff"
TEXT_COLOR = "#17324d"
MUTED_TEXT = "#5f7182"
PRIMARY_BLUE = "#1976d2"
PRIMARY_BLUE_DARK = "#125ca6"
TEAL = "#00a896"
CORAL = "#f26b5e"
SOFT_BLUE = "#d9efff"
SOFT_TEAL = "#d8f5ef"
SOFT_CORAL = "#ffe2dd"
TILE_BACKGROUND = "#f5f9fc"
DATASET_COLORS = ((PRIMARY_BLUE, SOFT_BLUE), (TEAL, SOFT_TEAL))
GALLERY_NAMES = ("XRAY", "EKG/ECG")
FOLDER_PREFIXES = ("xray-data", "ekg-data")
blood_test_dir_setting = os.environ.get(
    "BLOOD_TEST_DIR", settings_from_file.get("BLOOD_TEST_DIR", "")
).strip()
BLOOD_TEST_DIR = Path(blood_test_dir_setting).expanduser() if blood_test_dir_setting else None
if BLOOD_TEST_DIR is not None and not BLOOD_TEST_DIR.is_absolute():
    BLOOD_TEST_DIR = Path(__file__).resolve().parent / BLOOD_TEST_DIR

# Remote MedGemma can take several minutes when full-resolution images are sent.
# Keep the quick server/model check short, but allow inference much longer.
SERVER_CHECK_TIMEOUT = 15


def positive_integer_setting(name, default):
    """Read a positive integer from the environment or .env."""
    value = os.environ.get(name, settings_from_file.get(name, str(default)))
    try:
        number = int(value)
        if number > 0:
            return number
    except (TypeError, ValueError):
        pass
    print(f"Invalid {name}={value!r}; using {default}.")
    return default


NUM_PREDICT = positive_integer_setting("NUM_PREDICT", 4096)
GENERATION_TIMEOUT = positive_integer_setting("GENERATION_TIMEOUT", 900)
REPEAT_LAST_N = positive_integer_setting("REPEAT_LAST_N", 256)


def float_setting(name, default, minimum, maximum):
    """Read a floating-point .env value and keep it within a safe range."""
    value = os.environ.get(name, settings_from_file.get(name, str(default)))
    try:
        number = float(value)
        if minimum <= number <= maximum:
            return number
    except (TypeError, ValueError):
        pass
    print(
        f"Invalid {name}={value!r}; expected {minimum} through {maximum}. "
        f"Using {default}."
    )
    return default


# Low temperature keeps medical-report wording focused and consistent.
TEMPERATURE = float_setting("TEMPERATURE", 0.2, 0.0, 2.0)
# A moderate value above 1.0 discourages loops without making wording unnatural.
REPEAT_PENALTY = float_setting("REPEAT_PENALTY", 1.2, 0.1, 2.0)


def boolean_setting(name, default):
    """Read a true/false setting from the environment or .env."""
    value = os.environ.get(name, settings_from_file.get(name, str(default)))
    normalized = str(value).strip().lower()
    if normalized in ("true", "yes", "on", "1"):
        return True
    if normalized in ("false", "no", "off", "0"):
        return False
    print(f"Invalid {name}={value!r}; using {default}.")
    return default


THINK = boolean_setting("THINK", False)


def image_dimension(name):
    """Read a thumbnail dimension; zero preserves that original dimension."""
    value = os.environ.get(name, settings_from_file.get(name, "256"))
    try:
        number = int(value)
        if 0 <= number <= 4096:
            return number
    except (TypeError, ValueError):
        pass
    print(f"Invalid {name}={value!r}; using 256 pixels.")
    return 256


GALLERY_SIZES = (
    (image_dimension("XRAY_WIDTH"), image_dimension("XRAY_HEIGHT")),
    (image_dimension("EKG_WIDTH"), image_dimension("EKG_HEIGHT")),
)


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
    """Toggle one image with the selection color of its gallery."""
    selection = selected_images[column]
    if path in selection:
        del selection[path]
        button.config(relief=tk.RAISED, background=default_button_color)
        tile.config(background=TILE_BACKGROUND)
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


def image_folders():
    """Find the newest xray-data-MMDDYY and ekg-data-MMDDYY folders."""
    script_dir = Path(__file__).resolve().parent
    candidates = [[], []]
    seen = set()
    for parent in (IMAGE_DIR, script_dir):
        if not parent.is_dir():
            continue
        for folder in parent.iterdir():
            if not folder.is_dir() or folder.name.startswith(".") or folder.name == "__pycache__":
                continue
            if parent == script_dir and folder.resolve() == IMAGE_DIR.resolve():
                continue
            if folder.resolve() in seen:
                continue
            seen.add(folder.resolve())
            if not any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.rglob("*")):
                continue
            for index, prefix in enumerate(FOLDER_PREFIXES):
                marker = f"{prefix}-"
                if not folder.name.lower().startswith(marker):
                    continue
                suffix = folder.name[len(marker):]
                try:
                    folder_date = datetime.strptime(suffix, "%m%d%y")
                except ValueError:
                    continue
                if folder_date.strftime("%m%d%y") == suffix:
                    candidates[index].append((folder_date, folder))
    return [max(items, key=lambda item: item[0])[1] if items else None for items in candidates]


def friendly_folder_date(folder):
    """Format the final MMDDYY folder digits as a friendly calendar date."""
    date_text = folder.name.rsplit("-", 1)[-1]
    try:
        folder_date = datetime.strptime(date_text, "%m%d%y")
    except ValueError:
        return folder.name
    return folder_date.strftime("%b %d, %Y")


def latest_blood_test_folder():
    """Find the newest valid bloodtest-data-MMDDYY directory containing JSON."""
    if BLOOD_TEST_DIR is not None:
        parents = (BLOOD_TEST_DIR,)
    else:
        parents = (IMAGE_DIR, Path(__file__).resolve().parent)
    candidates = []
    seen = set()
    for parent in parents:
        if not parent.is_dir():
            continue
        for folder in parent.iterdir():
            if not folder.is_dir() or folder.resolve() in seen:
                continue
            seen.add(folder.resolve())
            marker = "bloodtest-data-"
            if not folder.name.lower().startswith(marker):
                continue
            suffix = folder.name[len(marker):]
            try:
                folder_date = datetime.strptime(suffix, "%m%d%y")
            except ValueError:
                continue
            if folder_date.strftime("%m%d%y") != suffix:
                continue
            if any(path.is_file() and path.suffix.lower() == ".json" for path in folder.rglob("*.json")):
                candidates.append((folder_date, folder))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def load_blood_tests():
    """Load one selectable list row per JSON blood-test record."""
    blood_test_view.config(state=tk.NORMAL)
    blood_test_view.delete("1.0", tk.END)
    blood_test_records.clear()
    selected_blood_test_indices.clear()
    folder = latest_blood_test_folder()
    if folder is None:
        blood_test_panel.config(text="BLOOD TEST (missing)")
        blood_test_view.insert(tk.END, "No bloodtest-data-MMDDYY directory found")
        blood_test_view.config(state=tk.DISABLED)
        return

    blood_test_panel.config(text=f"BLOOD TEST — {friendly_folder_date(folder)}")
    errors = 0
    for path in sorted(folder.rglob("*.json"), key=lambda item: str(item).lower()):
        if path.name.lower() == "index.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            errors += 1
            continue
        records = data if isinstance(data, list) else [data]
        for record in records:
            if not isinstance(record, dict):
                errors += 1
                continue
            title = record.get("title")
            if not isinstance(title, str) or not title.strip():
                errors += 1
                continue
            record_index = len(blood_test_records)
            blood_test_records.append(record)
            line_tag = f"blood_record_{record_index}"
            remaining_data = {key: value for key, value in record.items() if key != "title"}
            blood_test_view.insert(tk.END, title.strip(), (line_tag, "blood_title"))
            blood_test_view.insert(
                tk.END,
                " : "
                + json.dumps(remaining_data, ensure_ascii=False, separators=(", ", ": "))
                + "\n",
                (line_tag,),
            )

    if not blood_test_records:
        blood_test_view.insert(tk.END, "No valid titled JSON records found")
    blood_test_view.config(state=tk.DISABLED)
    if errors:
        print(f"Skipped {errors} invalid blood-test record(s).")


def toggle_blood_test_record(event):
    """Select or deselect the complete blood-test record on the clicked line."""
    line_number = int(blood_test_view.index(f"@{event.x},{event.y}").split(".")[0])
    record_index = line_number - 1
    if not 0 <= record_index < len(blood_test_records):
        return "break"
    line_tag = f"blood_record_{record_index}"
    if record_index in selected_blood_test_indices:
        selected_blood_test_indices.remove(record_index)
        blood_test_view.tag_configure(line_tag, background="white")
    else:
        selected_blood_test_indices.add(record_index)
        blood_test_view.tag_configure(line_tag, background="#bbdefb")
    blood_test_view.tag_raise("blood_title")
    return "break"


def choose_dataset(column):
    """Allow each image column to display an explicitly selected folder."""
    # Windows' folder picker returns an empty string when the user cancels.
    chosen = filedialog.askdirectory(
        title=f"Choose {GALLERY_NAMES[column]} image folder",
        initialdir=str(dataset_overrides[column] or IMAGE_DIR),
    )
    if chosen:
        dataset_overrides[column] = Path(chosen)
        load_images()


def update_gallery_scrollbars(canvas, canvas_window, grid):
    """Keep the canvas viewport scrollable when image content is oversized."""
    grid.update_idletasks()
    # Fill the visible canvas when content is small, but retain the complete
    # requested content width when an image is wider than the physical window.
    content_width = grid.winfo_reqwidth()
    viewport_width = max(canvas.winfo_width(), 1)
    canvas.itemconfigure(canvas_window, width=max(viewport_width, content_width))
    canvas.configure(scrollregion=canvas.bbox("all"))


def load_images():
    """Show X-Ray and EKG images in independently scrolling stacked galleries."""
    # Destroy old widgets before reloading; keep PhotoImage references so
    # Tkinter does not remove the image pixels from the screen.
    for gallery in galleries:
        for widget in gallery["grid"].winfo_children():
            widget.destroy()
    thumbnails.clear()
    for selection in selected_images:
        selection.clear()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    folders = image_folders()

    try:
        from PIL import Image, ImageOps, ImageTk
    except ImportError:
        Image = ImageOps = ImageTk = None

    for column, gallery in enumerate(galleries):
        grid = gallery["grid"]
        # Give each active image column equal width. The EKG gallery has only
        # one active column, so each EKG image is centered in its own row.
        for grid_column in range(3):
            grid.columnconfigure(
                grid_column,
                weight=1 if (column == 0 or grid_column == 0) else 0,
            )
        # Explicit folder choices take priority; discovered image folders
        # beside the script or inside IMAGE_DIR fill the remaining columns.
        folder = dataset_overrides[column] or folders[column]
        if folder is None:
            expected = f"{FOLDER_PREFIXES[column]}-MMDDYY"
            gallery["title"].config(text=f"{GALLERY_NAMES[column]} (missing)")
            ttk.Label(grid, text=f"No {expected} folder found. Create one beside this script or inside IMAGE_DIR, or choose a folder from the Images menu.", wraplength=500).grid(row=0, column=0, padx=5, pady=12)
            continue
        friendly_date = friendly_folder_date(folder)
        gallery["title"].config(
            text=f"{GALLERY_NAMES[column]} — {friendly_date}"
        )
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
                    original_width, original_height = source.size
                    configured_width, configured_height = GALLERY_SIZES[column]
                    target_size = (
                        configured_width or original_width,
                        configured_height or original_height,
                    )
                    rgb_source = source.convert("RGB")
                    if column == 1:
                        # EKG/ECG is commonly a long waveform. Apply the exact
                        # configured width and height to the visible image
                        # instead of preserving its aspect ratio and adding
                        # white padding around it.
                        picture = rgb_source.resize(
                            target_size, Image.Resampling.LANCZOS
                        )
                    else:
                        # Preserve the X-ray aspect ratio and fill unused space.
                        picture = ImageOps.pad(
                            rgb_source,
                            target_size,
                            method=Image.Resampling.LANCZOS,
                            color="white",
                        )
                photo = ImageTk.PhotoImage(picture)
            except (OSError, ValueError):
                errors += 1
                continue
            thumbnails.append(photo)
            tile = tk.Frame(grid, background=TILE_BACKGROUND, padx=5, pady=5)
            # X-Ray uses three images per row. EKG uses a vertical list with
            # exactly one image per row. Horizontal scrolling handles wide
            # originals when a configured dimension is zero.
            columns_per_row = 1 if column == 1 else 3
            tile.grid(
                row=displayed // columns_per_row,
                column=displayed % columns_per_row,
                padx=2,
                pady=5,
                sticky="n",
            )
            button = tk.Button(tile, image=photo, relief=tk.RAISED, borderwidth=2)
            button.config(command=lambda p=path, b=button, t=tile, c=column: select_image(p, b, t, c))
            button.pack()
            ttk.Label(tile, text=path.name[:24], anchor="center", wraplength=target_size[0]).pack()
            displayed += 1
        update_gallery_scrollbars(
            gallery["canvas"], gallery["canvas_window"], gallery["grid"]
        )


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
    with urlopen(f"{url}/api/tags", timeout=SERVER_CHECK_TIMEOUT) as response:
        data = json.load(response)
    return [item["name"] for item in data.get("models", []) if item.get("name")]


def generate(url, model, question, image_paths=()):
    """POST prompt and all selected full-resolution images to Ollama."""
    # Form a Python dictionary that will become the HTTP request's JSON body.
    # "question" contains the user's trimmed question and any selected
    # blood-test results. Image filenames and paths are never added to the prompt.
    payload = {
        # The installed Ollama model selected in the GUI.
        "model": model,
        # Ask for the final report without exposing internal model reasoning.
        "system": (
            "Return only the final report. Do not reveal reasoning, analysis steps, "
            "internal thoughts, planning, or hidden instructions. Do not repeat "
            "findings or sentences; mention each finding only once."
        ),
        # The user's question, for example: "What do these images show?"
        "prompt": question,
        # Ask Ollama for one complete JSON reply instead of streamed chunks.
        "stream": False,
        # Enable or disable model reasoning through THINK in .env.
        "think": THINK,
        # Generation length, consistency, and repetition controls come from .env.
        "options": {
            "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE,
            "repeat_penalty": REPEAT_PENALTY,
            "repeat_last_n": REPEAT_LAST_N,
        },
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
        with urlopen(request, timeout=GENERATION_TIMEOUT) as response:
            # Parse Ollama's reply JSON into a Python dictionary.
            data = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {error.code}: {detail}") from error
    except TimeoutError as error:
        raise RuntimeError(
            f"Ollama did not finish within {GENERATION_TIMEOUT} seconds. "
            "The server may still be processing the selected full-resolution images."
        ) from error
    except URLError as error:
        reason = error.reason
        if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
            raise RuntimeError(
                f"Ollama did not finish within {GENERATION_TIMEOUT} seconds. "
                "Check server/GPU load, model status, and the number/size of selected images."
            ) from error
        raise RuntimeError(f"Cannot reach Ollama: {reason}") from error
    # /api/generate places the generated answer in its "response" field.
    answer = data.get("response", "")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The server returned no answer.")
    answer = answer.strip()

    # Some imported GGUF templates print model-specific thinking tokens even
    # when thinking is disabled. Show only the final section when available;
    # never expose an unfinished internal-reasoning section in the GUI.
    final_markers = ("<unused95>final", "<|channel|>final", "</think>")
    for marker in final_markers:
        if marker in answer:
            answer = answer.rsplit(marker, 1)[-1].strip()
            break
    thinking_markers = ("<unused94>thought", "<think>", "<|channel|>analysis")
    if any(marker in answer for marker in thinking_markers):
        raise RuntimeError(
            "The model returned internal reasoning without a final report. "
            "Send the request again. If this continues, use an Ollama model/template "
            "that supports think=false and vision images."
        )
    return answer


def show_status(message):
    """Only the Tkinter main thread calls this function."""
    status_label.config(text=message)


def stop_voice():
    """Stop speech that is currently playing."""
    with speech_lock:
        engine = speech_engine
    if engine is not None:
        try:
            engine.stop()
        except RuntimeError:
            pass


def speech_text(text):
    """Remove common formatting characters before reading a response aloud."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[`*_#]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def speak_response(text):
    """Read one completed response without blocking the Tkinter GUI."""
    if not voice_enabled or not text or pyttsx3 is None:
        return

    stop_voice()

    def worker():
        global speech_engine
        engine = None
        try:
            engine = pyttsx3.init("sapi5")
            with speech_lock:
                if not voice_enabled:
                    return
                speech_engine = engine
            engine.setProperty("rate", 165)
            engine.setProperty("volume", 1.0)
            engine.say(speech_text(text))
            engine.runAndWait()
        except Exception as error:
            message = str(error)
            window.after(0, lambda: show_status(f"Voice error: {message}"))
        finally:
            with speech_lock:
                if speech_engine is engine:
                    speech_engine = None

    threading.Thread(target=worker, daemon=True).start()


def toggle_voice():
    """Toggle automatic reading and visually show whether voice is enabled."""
    global voice_enabled

    voice_enabled = not voice_enabled
    if voice_enabled:
        voice_toggle_button.configure(text="Voice: ON", style="VoiceOn.TButton")
        show_status("Voice is ON. Completed responses will be read aloud.")
        current_answer = answer_box.get("1.0", tk.END).strip()
        if current_answer and not current_answer.startswith("Error:"):
            speak_response(current_answer)
    else:
        voice_toggle_button.configure(text="Voice: OFF", style="VoiceOff.TButton")
        stop_voice()
        show_status("Voice is OFF.")


def finish(answer=None, error=None):
    """Display a finished answer or error and restore the Send button."""
    send_button.config(state=tk.NORMAL)
    show_status("Completed" if error is None else "Request failed")
    answer_box.delete("1.0", tk.END)
    answer_box.insert(tk.END, answer if error is None else f"Error: {error}")
    if error is None and voice_enabled:
        speak_response(answer)


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
    user_question = prompt_box.get("1.0", tk.END).strip()
    question = user_question
    selected_blood_tests = [
        blood_test_records[index].get("results", [])
        for index in sorted(selected_blood_test_indices)
        if index < len(blood_test_records)
    ]
    if selected_blood_tests:
        question += (
            "\n\nSelected blood test data (JSON):\n"
            + json.dumps(selected_blood_tests, indent=2, ensure_ascii=False)
        )
    # Capture blue-column images followed by orange-column images. Click order
    # within each dataset is retained. Later clicks cannot change this request.
    image_paths = tuple(path for group in selected_images for path in group)
    model = model_name.get().strip()
    try:
        url = base_url(server_url.get())
        if not user_question or not model:
            raise ValueError("Enter a question and a model name.")
    except ValueError as error:
        show_status(str(error))
        return
    send_button.config(state=tk.DISABLED)
    answer_box.delete("1.0", tk.END)
    attachments = []
    if image_paths:
        attachments.append(f"{len(image_paths)} image(s)")
    if selected_blood_tests:
        attachments.append(f"{len(selected_blood_tests)} blood-test record(s)")
    show_status(
        f"Waiting for {model}"
        + (f" with {', '.join(attachments)}" if attachments else " (text only)")
        + "..."
    )
    # Pass the question and image snapshot together to the background worker.
    # run_request calls generate(), which builds and sends one HTTP POST.
    threading.Thread(
        target=run_request,
        args=(url, model, question, image_paths),
        daemon=True,
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
window.title("AI Clinical Assistant Center")
window.configure(background=APP_BACKGROUND)

# Use a clean, consistent theme instead of platform-dependent default colors.
ui_style = ttk.Style(window)
try:
    ui_style.theme_use("clam")
except tk.TclError:
    pass
ui_style.configure("App.TFrame", background=APP_BACKGROUND)
ui_style.configure(
    "Settings.TLabelframe",
    background=CARD_BACKGROUND,
    bordercolor=SOFT_BLUE,
    relief="solid",
    borderwidth=1,
)
ui_style.configure(
    "Settings.TLabelframe.Label",
    background=APP_BACKGROUND,
    foreground=PRIMARY_BLUE_DARK,
    font=("Segoe UI", 11, "bold"),
)
ui_style.configure(
    "Assistant.TLabelframe",
    background=CARD_BACKGROUND,
    bordercolor=SOFT_BLUE,
    relief="solid",
    borderwidth=1,
)
ui_style.configure(
    "Assistant.TLabelframe.Label",
    background=APP_BACKGROUND,
    foreground=CORAL,
    font=("Segoe UI", 16, "bold"),
)
ui_style.configure(
    "Gallery.TLabelframe",
    background=CARD_BACKGROUND,
    bordercolor=SOFT_TEAL,
    relief="solid",
    borderwidth=1,
)
ui_style.configure(
    "Gallery.TLabelframe.Label",
    background=APP_BACKGROUND,
    foreground=TEAL,
    font=("Segoe UI", 12, "bold"),
)
ui_style.configure(
    "Field.TLabel",
    background=CARD_BACKGROUND,
    foreground=TEXT_COLOR,
    font=("Segoe UI", 10, "bold"),
)
ui_style.configure(
    "Status.TLabel",
    background=CARD_BACKGROUND,
    foreground=MUTED_TEXT,
    font=("Segoe UI", 9),
)
ui_style.configure(
    "Accent.TButton",
    background=PRIMARY_BLUE,
    foreground="white",
    borderwidth=0,
    padding=(14, 7),
    font=("Segoe UI", 10, "bold"),
)
ui_style.map(
    "Accent.TButton",
    background=[("active", PRIMARY_BLUE_DARK), ("disabled", "#9eb6c9")],
    foreground=[("disabled", "#edf3f7")],
)
ui_style.configure(
    "Secondary.TButton",
    background=TEAL,
    foreground="white",
    borderwidth=0,
    padding=(10, 6),
    font=("Segoe UI", 9, "bold"),
)
ui_style.map("Secondary.TButton", background=[("active", "#008577")])
ui_style.configure(
    "VoiceOn.TButton",
    background=TEAL,
    foreground="white",
    borderwidth=0,
    padding=(12, 7),
    font=("Segoe UI", 10, "bold"),
)
ui_style.map("VoiceOn.TButton", background=[("active", "#008577")])
ui_style.configure(
    "VoiceOff.TButton",
    background="#d9e2e8",
    foreground=TEXT_COLOR,
    borderwidth=0,
    padding=(12, 7),
    font=("Segoe UI", 10, "bold"),
)
ui_style.map("VoiceOff.TButton", background=[("active", "#c7d4dc")])
ui_style.configure("TEntry", fieldbackground="white", foreground=TEXT_COLOR, padding=5)
ui_style.configure("TCombobox", fieldbackground="white", foreground=TEXT_COLOR, padding=4)
ui_style.configure("TScrollbar", background=SOFT_BLUE, troughcolor="#e8f1f7")
# Reserve one full-width image area containing two stacked galleries.
gallery_width = 3 * (max(width or 256 for width, _ in GALLERY_SIZES) + 14) + 60
# WINDOW_WIDTH and WINDOW_HEIGHT in .env can override the initial size.
preferred_width = gallery_width + 540
window_width = window_dimension("WINDOW_WIDTH", max(2400, round(preferred_width * 1.5)))
window_height = window_dimension("WINDOW_HEIGHT", 850)
# A requested width wider than the physical monitor cannot be fully visible.
# Fit the window on screen and allow the image panel to scroll if necessary.
visible_width = min(window_width, window.winfo_screenwidth() - 30)
visible_height = min(window_height, window.winfo_screenheight() - 80)
window.geometry(f"{visible_width}x{visible_height}")
# The medical-data panels use 65% of the body width.
gallery_viewport_width = max(105, round((visible_width - 45) * 65 / 100) - 24)

settings = ttk.LabelFrame(
    window, text="Connection & Model", style="Settings.TLabelframe"
)
settings.pack(fill=tk.X, padx=14, pady=(10, 5))
settings.columnconfigure(1, weight=0)
settings.columnconfigure(3, weight=0)
server_url = tk.StringVar(value=DEFAULT_SERVER_URL)
model_name = tk.StringVar(value=DEFAULT_MODEL)
ttk.Label(settings, text="Server URL", style="Field.TLabel").grid(row=0, column=0, padx=(10, 5), pady=7, sticky="w")
server_entry = ttk.Entry(
    settings,
    textvariable=server_url,
    width=max(24, min(42, len(DEFAULT_SERVER_URL) + 2)),
)
server_entry.grid(row=0, column=1, padx=4, pady=3, sticky="w")
ttk.Label(settings, text="Model", style="Field.TLabel").grid(row=0, column=2, padx=(14, 5), pady=7, sticky="w")
model_picker = ttk.Combobox(
    settings,
    textvariable=model_name,
    values=DEFAULT_MODELS,
    width=max(16, min(28, len(DEFAULT_MODEL) + 2)),
)
model_picker.grid(row=0, column=3, padx=4, pady=3, sticky="w")
refresh_button = ttk.Button(
    settings,
    text="Check Connection",
    command=refresh_models,
    style="Secondary.TButton",
)
refresh_button.grid(row=0, column=4, padx=(10, 8), pady=3)

# Place text on the left and the image gallery beside it on the right.
body = ttk.Frame(window, style="App.TFrame")
body.pack(fill=tk.BOTH, expand=True, padx=14, pady=(5, 14))
# Question and AI Response receive 35% of the width; the medical image and
# blood-test panels receive the remaining 65%.
body.columnconfigure(0, weight=35, minsize=190, uniform="body_parts")
body.columnconfigure(1, weight=65, minsize=gallery_viewport_width + 16, uniform="body_parts")
# Blood Test Result receives 30% of the height. The upper 70% is split
# equally between XRAY and EKG, giving each image window 35% overall.
body.rowconfigure(0, weight=7, uniform="body_rows")
body.rowconfigure(1, weight=3, uniform="body_rows")

# One unified question-and-answer window.
qa_panel = ttk.LabelFrame(
    body,
    text="AI Clinical Assistant Center",
    labelanchor="n",
    style="Assistant.TLabelframe",
)
qa_panel.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 12))
ttk.Label(qa_panel, text="Your Question", style="Field.TLabel").pack(anchor="w", padx=10, pady=(10, 5))
prompt_box = scrolledtext.ScrolledText(
    qa_panel,
    width=18,
    height=4,
    wrap=tk.WORD,
    font=("Segoe UI", 11),
    background="#fbfdff",
    foreground=TEXT_COLOR,
    insertbackground=PRIMARY_BLUE,
    selectbackground=SOFT_BLUE,
    relief=tk.FLAT,
    borderwidth=6,
)
prompt_box.pack(fill=tk.X, padx=10)
action_bar = ttk.Frame(qa_panel, style="App.TFrame")
action_bar.pack(pady=10)
send_button = ttk.Button(
    action_bar, text="Send to MedGemma", command=send_question, style="Accent.TButton"
)
send_button.pack(side=tk.LEFT, padx=(0, 8))
voice_toggle_button = ttk.Button(
    action_bar,
    text="Voice: OFF",
    command=toggle_voice,
    style="VoiceOff.TButton",
)
voice_toggle_button.pack(side=tk.LEFT)
status_label = ttk.Label(
    qa_panel,
    text="Ready. Select a server and model.",
    wraplength=240,
    style="Status.TLabel",
)
status_label.pack(anchor="w")
ttk.Label(qa_panel, text="AI Response", style="Field.TLabel").pack(anchor="w", padx=10, pady=(12, 5))
answer_box = scrolledtext.ScrolledText(
    qa_panel,
    width=18,
    wrap=tk.WORD,
    font=("Segoe UI", 11),
    background="#f8fcff",
    foreground=TEXT_COLOR,
    insertbackground=PRIMARY_BLUE,
    selectbackground=SOFT_BLUE,
    relief=tk.FLAT,
    borderwidth=6,
)
answer_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

image_panel = ttk.LabelFrame(
    body,
    text="Medical Image Datasets",
    labelanchor="n",
    style="Gallery.TLabelframe",
)
image_panel.grid(row=0, column=1, sticky="nsew")
image_panel.rowconfigure(0, weight=1, uniform="gallery_rows")
image_panel.rowconfigure(1, weight=1, uniform="gallery_rows")
image_panel.columnconfigure(0, weight=1)
selected_images = [{}, {}]  # Blue first, then orange; each keeps click order.
dataset_overrides = [None, None]  # Optional folders selected through Images menu.
sample_button = tk.Button(image_panel)
default_button_color = sample_button.cget("background")
sample_button.destroy()

# Keep folder selection and refresh available without taking gallery space.
menu_bar = tk.Menu(window, background=CARD_BACKGROUND, foreground=TEXT_COLOR)
images_menu = tk.Menu(
    menu_bar,
    tearoff=False,
    background=CARD_BACKGROUND,
    foreground=TEXT_COLOR,
    activebackground=SOFT_BLUE,
    activeforeground=PRIMARY_BLUE_DARK,
)
images_menu.add_command(label="Choose X-Ray folder...", command=lambda: choose_dataset(0))
images_menu.add_command(label="Choose EKG folder...", command=lambda: choose_dataset(1))
images_menu.add_command(
    label="Refresh images and blood tests",
    command=lambda: (load_images(), load_blood_tests()),
)
menu_bar.add_cascade(label="Images", menu=images_menu)
window.config(menu=menu_bar)

galleries = []
for gallery_index in range(2):
    pane = ttk.LabelFrame(
        image_panel,
        text=GALLERY_NAMES[gallery_index],
        style="Gallery.TLabelframe",
    )
    pane.grid(
        row=gallery_index,
        column=0,
        sticky="nsew",
        pady=(0, 4) if gallery_index == 0 else (4, 0),
    )
    pane.columnconfigure(0, weight=1)
    pane.rowconfigure(0, weight=1)
    canvas = tk.Canvas(
        pane,
        width=gallery_viewport_width,
        background=TILE_BACKGROUND,
        highlightthickness=0,
    )
    canvas.grid(row=0, column=0, sticky="nsew")
    vscroll = ttk.Scrollbar(pane, orient=tk.VERTICAL, command=canvas.yview)
    vscroll.grid(row=0, column=1, sticky="ns")
    hscroll = ttk.Scrollbar(pane, orient=tk.HORIZONTAL, command=canvas.xview)
    hscroll.grid(row=1, column=0, sticky="ew")
    canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
    grid = ttk.Frame(canvas, style="App.TFrame")
    canvas_window = canvas.create_window((0, 0), window=grid, anchor="nw")
    grid.bind(
        "<Configure>",
        lambda event, c=canvas, w=canvas_window, g=grid: update_gallery_scrollbars(c, w, g),
    )
    canvas.bind(
        "<Configure>",
        lambda event, c=canvas, w=canvas_window, g=grid: update_gallery_scrollbars(c, w, g),
    )
    galleries.append(
        {
            "title": pane,
            "canvas": canvas,
            "canvas_window": canvas_window,
            "grid": grid,
        }
    )

# Keep the selectable Blood Test Data list below the image galleries.
blood_test_panel = ttk.LabelFrame(
    body,
    text="BLOOD TEST",
    labelanchor="nw",
    style="Gallery.TLabelframe",
)
blood_test_panel.grid(row=1, column=1, sticky="nsew", pady=(8, 0))
blood_test_panel.rowconfigure(0, weight=1)
blood_test_panel.columnconfigure(0, weight=1)
blood_test_records = []
selected_blood_test_indices = set()
blood_test_view = tk.Text(
    blood_test_panel,
    wrap=tk.NONE,
    font=("Segoe UI", 10),
    cursor="hand2",
    state=tk.DISABLED,
    background="#fffdfb",
    foreground=TEXT_COLOR,
    selectbackground=SOFT_CORAL,
    selectforeground=TEXT_COLOR,
    relief=tk.FLAT,
    borderwidth=5,
)
blood_test_view.tag_configure("blood_title", foreground=CORAL, font=("Segoe UI", 10, "bold"))
blood_test_view.bind("<Button-1>", toggle_blood_test_record)
blood_test_view.grid(row=0, column=0, sticky="nsew", padx=(5, 0), pady=(5, 0))
blood_test_scroll = ttk.Scrollbar(
    blood_test_panel,
    orient=tk.VERTICAL,
    command=blood_test_view.yview,
)
blood_test_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 5), pady=(5, 0))
blood_test_hscroll = ttk.Scrollbar(
    blood_test_panel,
    orient=tk.HORIZONTAL,
    command=blood_test_view.xview,
)
blood_test_hscroll.grid(row=1, column=0, sticky="ew", padx=(5, 0), pady=(0, 5))
blood_test_view.configure(
    yscrollcommand=blood_test_scroll.set,
    xscrollcommand=blood_test_hscroll.set,
)

thumbnails = []

# Load images and check the server after the complete window is assembled.
window.after(100, load_images)
window.after(150, load_blood_tests)
window.after(300, refresh_models)

window.mainloop()
