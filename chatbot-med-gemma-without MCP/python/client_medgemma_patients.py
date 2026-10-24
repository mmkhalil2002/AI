"""Single-file patient GUI, adapted from the original MedGemma Tkinter client.

Configuration is read from .env beside this file. Optional setting:
PATIENT_RECORDS_DIR=patient_records (relative to this script, or absolute).
OLLAMA_URL and MEDGEMMA_MODEL retain the existing .env values.
Select a patient, select records, type a question, and press Send.
Original image bytes are uploaded; thumbnails remain local.
AI-generated output requires clinician review.
"""

import base64
import importlib
import importlib.util
import io
import json
import os
import re
import sys
import threading
import queue
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen




# Voice is optional; no automatic package or model installation.
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

PYTHON_DIR = Path(__file__).resolve().parent

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
                "REPEAT_LAST_N", "DISK", "PATIENT_RECORDS_DIR",
            ):
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                values[key] = value
    return values


settings_from_file = load_settings()

def patient_records_root():
    """Resolve and validate the configured root before any Tk window exists.

    Iterating the directory verifies real access rather than relying solely on
    permission bits. Missing roots are never created or replaced.
    """
    value = os.environ.get("PATIENT_RECORDS_DIR", settings_from_file.get(
        "PATIENT_RECORDS_DIR", "patient_records"))
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PYTHON_DIR / path
    path = path.resolve()
    try:
        if not path.is_dir() or not os.access(path, os.R_OK | os.X_OK):
            raise OSError("Not a readable directory")
        with os.scandir(path) as entries:
            next(entries, None)
    except OSError as error:
        raise RuntimeError(
            f"[ERROR] Patient records directory not found or not accessible: {path}\n"
            "[ACTION] Restore the patient_records directory alongside the GUI, "
            "or correct PATIENT_RECORDS_DIR in python/.env.\n"
            "[STOP] Application terminated."
        ) from error
    return path


def safe_path(path, patient):
    """Reject symlinks/junctions and files resolving outside the patient folder."""
    path, patient = Path(path), Path(patient).resolve()
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(patient):
        raise ValueError("Record path escapes the active patient directory")
    current = path
    while current != patient:
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise ValueError("Linked record paths are not allowed")
        if current == current.parent:
            raise ValueError("Record path is outside the active patient")
        current = current.parent
    return resolved


def discover_patients(root):
    """List immediate ordinary patient directories without name-pattern rules."""
    patients = {}
    for path in sorted(root.iterdir(), key=lambda p: p.name.casefold()):
        try:
            attributes = getattr(path.stat(), "st_file_attributes", 0)
            if path.name.startswith('.') or path.name == '__pycache__' or attributes & 6:
                continue
            if path.is_dir() and not path.is_symlink() and not getattr(path, 'is_junction', lambda: False)():
                safe_path(path, root)
                patients[path.name] = path.resolve()
        except (OSError, ValueError):
            continue
    return patients


def patient_files(modality, extensions):
    """Return safe immediate modality files and a nonfatal loading message."""
    if active_patient is None:
        return [], "Select a patient to load records."
    folder = active_patient / modality
    try:
        safe_path(folder, active_patient)
        if not folder.is_dir():
            return [], f"No {modality} folder for this patient."
        paths = []
        for path in sorted(folder.iterdir(), key=lambda p: p.name.casefold()):
            if path.suffix.lower() in extensions and path.is_file():
                try:
                    safe_path(path, active_patient)
                    paths.append(path)
                except (OSError, ValueError):
                    print(f"Skipped unsafe record: {path}")
        return paths, f"No supported records in {modality}." if not paths else ""
    except (OSError, ValueError):
        return [], f"Missing or inaccessible {modality} folder."


active_patient = None
request_in_flight = False
ui_callbacks = queue.Queue()


def post_ui(callback):
    """Queue work for Tk; background workers never invoke Tcl directly."""
    ui_callbacks.put(callback)


def drain_ui():
    """Process queued network completions on the main GUI thread."""
    try:
        while True:
            ui_callbacks.get_nowait()()
    except queue.Empty:
        pass
    window.after(50, drain_ui)


def select_patient(event=None):
    """Clear previous context and selections before loading the chosen patient."""
    global active_patient
    if request_in_flight:
        return
    stop_voice()
    active_patient = patients.get(patient_name.get())
    answer_box.delete("1.0", tk.END)
    prompt_box.delete("1.0", tk.END)
    load_images()
    load_blood_tests()
    show_status(f"Patient: {patient_name.get()}" if active_patient else "No patient directories found")


DEFAULT_SERVER_URL = os.environ.get(
    "OLLAMA_URL", settings_from_file.get("OLLAMA_URL", "http://127.0.0.1:11434")
)
DEFAULT_MODEL = os.environ.get(
    "MEDGEMMA_MODEL", settings_from_file.get("MEDGEMMA_MODEL", "medgemma1.5:4b")
)
DEFAULT_MODELS = (DEFAULT_MODEL,)
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
    """Encode original source bytes; gallery resizing never changes uploads."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")



def load_blood_tests():
    """Load selectable JSON records, preserving units and reference ranges.

    A title is optional: panel or filename supplies the visible label. Each
    displayed record retains its source path for validation at request time.
    """
    blood_test_view.config(state=tk.NORMAL)
    blood_test_view.delete("1.0", tk.END)
    blood_test_records.clear()
    blood_test_paths.clear()
    selected_blood_test_indices.clear()
    paths, message = patient_files("blood_tests", {".json"})
    errors = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            records = data if isinstance(data, list) else [data]
            for record in records:
                if not isinstance(record, dict) or not record:
                    raise ValueError("Expected a nonempty JSON object")
                index = len(blood_test_records)
                blood_test_records.append(record)
                blood_test_paths.append(path)
                tag = f"blood_record_{index}"
                title = str(record.get("title") or record.get("panel") or path.stem)
                blood_test_view.insert(tk.END, title.replace("\n", " "), (tag, "blood_title"))
                blood_test_view.insert(tk.END, " : " + json.dumps(record, ensure_ascii=False) + "\n", (tag,))
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(f"{path.name}: {error}")
    if not blood_test_records:
        blood_test_view.insert(tk.END, (message or "No valid blood-test JSON records.") + "\n")
    if errors:
        blood_test_view.insert(tk.END, "Invalid JSON: " + "; ".join(errors))
        print("Invalid blood-test JSON: " + "; ".join(errors))
    blood_test_view.config(state=tk.DISABLED)



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
        modality = ("xrays", "ekgs")[column]
        paths, message = patient_files(modality, IMAGE_EXTENSIONS)
        gallery["title"].config(text=GALLERY_NAMES[column])
        if not paths:
            ttk.Label(grid, text=message, wraplength=400).grid(row=0, column=0, padx=5, pady=12)
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
        if errors:
            ttk.Label(grid, text=f"Could not load {errors} invalid image(s).").grid(
                row=displayed + 1, column=0, padx=5, pady=5)
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
        raise ValueError("Enter a full HTTP or HTTPS server URL including its port.")
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
            "findings or sentences; mention each finding only once. Organize the "
            "report into three separate sections with these exact plain-text headings: "
            "X-RAY, ECG/EKG, and BLOOD TEST. Never combine information from different "
            "sections. If a section has no supplied data, write 'No data provided' "
            "under that section."
        ),
        # The user's question, for example: "What do these images show?"
        "prompt": question,
        # Ask Ollama for one complete JSON reply instead of streamed chunks.
        "stream": False,
        # Generation length, consistency, and repetition controls come from .env.
        "options": {
            "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE,
            "repeat_penalty": REPEAT_PENALTY,
            "repeat_last_n": REPEAT_LAST_N,
        },
    }
    payload["think"] = THINK
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

    # Some MedGemma/Ollama templates emit private planning between <unused94>
    # and <unused95>. The text after <unused95> is the final report. Templates
    # vary by inserting spaces or the word "final", so accept each variation.
    final_patterns = (
        r"<unused95>\s*(?:final\s*)?",
        r"<\|channel\|>\s*final\s*",
        r"</think>\s*",
    )
    for pattern in final_patterns:
        matches = list(re.finditer(pattern, answer, flags=re.IGNORECASE))
        if matches:
            answer = answer[matches[-1].end():].strip()
            break

    # Never display an unfinished reasoning-only response.
    thinking_pattern = r"(?:<unused94>\s*thought|<think>|<\|channel\|>\s*analysis)"
    if re.search(thinking_pattern, answer, flags=re.IGNORECASE):
        raise RuntimeError(
            "The model returned internal reasoning without a final report. "
            "Send the request again. If this continues, use an Ollama model/template "
            "that produces a final response after its reasoning."
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
            engine = pyttsx3.init("sapi5" if sys.platform == "win32" else None)
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
            post_ui(lambda: show_status(f"Voice error: {message}"))
        finally:
            with speech_lock:
                if speech_engine is engine:
                    speech_engine = None

    threading.Thread(target=worker, daemon=True).start()


def toggle_voice():
    """Toggle automatic reading and visually show whether voice is enabled."""
    global voice_enabled

    if pyttsx3 is None:
        show_status("Voice unavailable: install pyttsx3 to enable it.")
        return
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
    global request_in_flight
    request_in_flight = False
    patient_picker.config(state="readonly")
    send_button.config(state=tk.NORMAL)
    show_status("Completed" if error is None else "Request failed")
    answer_box.delete("1.0", tk.END)
    if error is None:
        answer_box.insert(tk.END, "AI-generated — requires clinician review.\n\n")
        insert_highlighted_report(answer)
    else:
        answer_box.insert(tk.END, f"Error: {error}")
    if error is None and voice_enabled:
        speak_response(answer)


def insert_highlighted_report(report):
    """Display the report and color medical-section names red."""
    # Match common spellings regardless of capitalization:
    # X-ray/Xray/X ray, EKG, ECG, and Blood Test/Blood-Test.
    pattern = re.compile(
        r"\b(?:x[\s-]?ray|ecg\s*/\s*ekg|ekg\s*/\s*ecg|ekg|ecg|blood[\s-]?test)\b",
        flags=re.IGNORECASE,
    )
    position = 0
    for match in pattern.finditer(report):
        answer_box.insert(tk.END, report[position:match.start()])
        answer_box.insert(tk.END, match.group(0), ("report_medical_heading",))
        position = match.end()
    answer_box.insert(tk.END, report[position:])


def run_request(url, model, question, image_paths, patient):
    """The HTTP call runs in the background so the window stays responsive."""
    try:
        for path in image_paths:
            safe_path(path, patient)
        # Give a useful message before trying generation with a missing model.
        models = list_models(url)
        if model not in models and f"{model}:latest" not in models:
            raise RuntimeError(
                f"Model {model} is not installed on the selected server. "
                "Check the model name and server address."
            )
        result = generate(url, model, question, image_paths)
        post_ui(lambda: finish(answer=result) if active_patient == patient else None)
    except Exception as error:
        message = str(error)
        post_ui(lambda: finish(error=message) if active_patient == patient else None)


def send_question():
    """Validate GUI values and send one question to the selected model."""
    global request_in_flight
    if request_in_flight:
        return
    patient = active_patient
    if patient is None:
        show_status("Select a patient first.")
        return
    try:
        for path in tuple(p for group in selected_images for p in group):
            safe_path(path, patient)
        for index in selected_blood_test_indices:
            safe_path(blood_test_paths[index], patient)
    except (OSError, ValueError) as error:
        show_status(f"Invalid selected record: {error}")
        return
    # Read all lines in the Question box, then trim surrounding whitespace.
    user_question = prompt_box.get("1.0", tk.END).strip()
    question = user_question
    selected_blood_tests = [
        blood_test_records[index]
        for index in sorted(selected_blood_test_indices)
        if index < len(blood_test_records)
    ]
    if selected_blood_tests:
        question += (
            "\n\nSelected blood test data (JSON):\n"
            + json.dumps(selected_blood_tests, indent=2, ensure_ascii=False)
        )
    xray_count = len(selected_images[0])
    ekg_count = len(selected_images[1])
    question += (
        "\n\nImage input identification:\n"
        f"- The first {xray_count} image(s) are X-ray images.\n"
        f"- The next {ekg_count} image(s) are ECG/EKG images.\n"
        "\nRequired final report format:\n"
        "X-RAY\n"
        "[X-ray findings, or No data provided]\n\n"
        "ECG/EKG\n"
        "[ECG/EKG findings, or No data provided]\n\n"
        "BLOOD TEST\n"
        "[Blood-test findings, or No data provided]\n"
        "Keep these as three separate sections and do not combine them."
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
    request_in_flight = True
    patient_picker.config(state=tk.DISABLED)
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
        args=(url, model, question, image_paths, patient),
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
            post_ui(complete)
        except Exception as error:
            message = str(error)
            def failed():
                refresh_button.config(state=tk.NORMAL)
                show_status(f"Cannot reach the selected server: {message}")
            post_ui(failed)

    threading.Thread(target=worker, daemon=True).start()


def main():
    """Validate startup data, then construct the preserved Tkinter interface."""
    global PATIENT_RECORDS_DIR, patients, window, ui_style, server_url, model_name, server_entry, model_picker, refresh_button, patient_name, patient_picker, prompt_box, send_button, voice_toggle_button, status_label, answer_box, selected_images, default_button_color, galleries, blood_test_panel, blood_test_records, blood_test_paths, selected_blood_test_indices, blood_test_view, thumbnails, gallery_viewport_width, actions_placeholder
    # Fatal data-root validation precedes construction of any Tk widget.
    try:
        PATIENT_RECORDS_DIR = patient_records_root()
    except (RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
    patients = discover_patients(PATIENT_RECORDS_DIR)

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

    patient_bar = ttk.LabelFrame(window, text="Patient", style="Settings.TLabelframe")
    patient_bar.pack(fill=tk.X, padx=14, pady=5)
    patient_name = tk.StringVar()
    patient_picker = ttk.Combobox(patient_bar, textvariable=patient_name,
                                  values=tuple(patients), state="readonly", width=40)
    patient_picker.pack(side=tk.LEFT, padx=10, pady=5)
    patient_picker.bind("<<ComboboxSelected>>", select_patient)
    ttk.Label(patient_bar, text="Choose the patient whose selected records will be sent.",
              style="Status.TLabel").pack(side=tk.LEFT, padx=8)

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
    actions_placeholder = ttk.LabelFrame(qa_panel, text="Actions", height=38,
                                         style="Settings.TLabelframe")
    actions_placeholder.pack(fill=tk.X, padx=10, pady=5)
    actions_placeholder.pack_propagate(False)
    ttk.Label(qa_panel, text="AI-generated output requires clinician review.",
              style="Status.TLabel", wraplength=300).pack(anchor="w", padx=10)
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
    answer_box.tag_configure(
        "report_medical_heading",
        foreground="#d32f2f",
        font=("Segoe UI", 11, "bold"),
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
    sample_button = tk.Button(image_panel)
    default_button_color = sample_button.cget("background")
    sample_button.destroy()

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
    blood_test_paths = []
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
    window.after(50, drain_ui)
    window.after(100, select_patient)
    # Connection checks are explicit; startup never transmits clinical data.

    window.mainloop()


if __name__ == "__main__":
    main()
