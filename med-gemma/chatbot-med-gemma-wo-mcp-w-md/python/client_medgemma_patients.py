"""Single-file patient GUI, adapted from the original MedGemma Tkinter client.

Configuration is read from .env beside this file. Optional setting:
PATIENT_RECORDS_DIR=patient_records (relative to this script, or absolute).
OLLAMA_URL and MEDGEMMA_MODEL retain the existing .env values.
Select a patient, select records, type a question, and press Send.
Original image bytes are uploaded; thumbnails remain local.
AI-generated output requires clinician review.
"""

import base64
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.client import IncompleteRead, RemoteDisconnected
import importlib
import importlib.util
import html
import io
import json
import os
import re
import tempfile
import sys
import threading
import queue
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

# Path.is_relative_to is used by the record confinement checks.
if sys.version_info < (3, 9):
    print(
        "[ERROR] Python 3.9 or newer is required.\n"
        "[ACTION] Install a supported Python version and retry.\n"
        "[STOP] Application terminated.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(2)

# Tkinter is required for the GUI. Convert a broken Python install into a short,
# readable startup message instead of leaving a long traceback in the console.
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except Exception as error:
    print(
        "[ERROR] Required dependency unavailable: Tkinter could not be loaded.\n"
        "[ACTION] Install Python with Tcl/Tk support (on Ubuntu, install python3-tk).\n"
        "[STOP] Application terminated.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(2) from error

# Voice is optional; no automatic package or model installation.
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import PIL
except Exception:
    PIL = None

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
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError):
            print(
                "[ERROR] Application settings could not be read.\n"
                "[ACTION] Check access to python/.env and retry.\n"
                "[STOP] Application terminated.",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(2)
        for line in lines:
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


def runtime_diagnostics():
    """Return sanitized runtime versions without exposing record or config paths."""
    try:
        tk_version = str(tk.Tcl().eval("info patchlevel"))
    except Exception:
        tk_version = "unavailable"
    pillow_version = getattr(PIL, "__version__", "unavailable")
    voice_version = getattr(pyttsx3, "__version__", "available") if pyttsx3 else "optional/unavailable"
    return (
        f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        f" | Tk {tk_version} | Pillow {pillow_version} | Voice {voice_version}"
    )


def server_error_category(error):
    """Map transport failures to safe categories without raw bodies or paths."""
    if isinstance(error, HTTPError):
        return f"HTTP error {error.code}"
    if isinstance(error, (TimeoutError,)):
        return "request timed out"
    if isinstance(error, URLError):
        if isinstance(error.reason, TimeoutError):
            return "request timed out"
        if isinstance(error.reason, (IncompleteRead, RemoteDisconnected, ConnectionResetError, BrokenPipeError)):
            return "connection disconnected before completion"
        return "connection failed"
    if isinstance(error, (json.JSONDecodeError, UnicodeDecodeError)):
        return "malformed response event"
    if isinstance(error, (IncompleteRead, RemoteDisconnected, ConnectionResetError, BrokenPipeError)):
        return "connection disconnected before completion"
    if isinstance(error, OSError):
        return "connection failed"
    return "server check failed"

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


def parse_record_dates(value):
    """Extract MMDDYY dates from names or common ISO/date representations."""
    text = str(value)
    found = []
    patterns = (
        (r"(?<!\d)(\d{2})[._/-]?(\d{2})[._/-]?(\d{2})(?!\d)", "parts"),
        (r"(?<!\d)(\d{4})[-_/](\d{2})[-_/](\d{2})(?!\d)", "%Y-%m-%d"),
    )
    for pattern, format_string in patterns:
        for match in re.finditer(pattern, text):
            try:
                if format_string == "parts":
                    parsed = datetime.strptime("".join(match.groups()), "%m%d%y").date()
                else:
                    parsed = datetime.strptime("-".join(match.groups()), format_string).date()
                found.append(parsed)
            except ValueError:
                continue
    return found


def path_record_dates(path):
    """Read dates from the most specific file and folder names first."""
    try:
        parts = Path(path).relative_to(active_patient).parts
    except (ValueError, TypeError):
        parts = Path(path).parts
    for part in reversed(parts):
        dates = parse_record_dates(part)
        if dates:
            return dates
    return []


def json_record_dates(value):
    """Find dates in common clinical JSON fields, including nested records."""
    dates = []
    date_keys = {"date", "test_date", "collection_date", "collected_date", "sample_date",
                 "draw_date", "date_of_test", "performed_at", "collected_at", "timestamp", "datetime", "date_time"}
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in date_keys or normalized.endswith("_date") or normalized.endswith("_datetime"):
                dates.extend(parse_record_dates(item))
            dates.extend(json_record_dates(item))
    elif isinstance(value, list):
        for item in value:
            dates.extend(json_record_dates(item))
    return dates


def date_is_in_active_range(record_dates):
    """Return whether a date falls inside the currently applied inclusive range."""
    if active_date_start is None and active_date_end is None:
        return True
    return any(
        (active_date_start is None or value >= active_date_start)
        and (active_date_end is None or value <= active_date_end)
        for value in record_dates
    )


def active_date_range_label(modality=None):
    """Return the selected snapshot's date/time label, never a range."""
    if selected_snapshot_dir is None:
        return "No date/time selected"
    return f"{selected_snapshot_dir.parent.name} {selected_snapshot_dir.name}"


def set_snapshot_heading(name_label, date_label, time_label, title):
    """Render a panel heading with green date and contrasting purple time."""
    if selected_snapshot_dir is None:
        name_label.config(text=f"{title} (No date/time selected)")
        date_label.config(text="")
        time_label.config(text="")
        return
    name_label.config(text=f"{title} (")
    date_label.config(text=selected_snapshot_dir.parent.name)
    time_label.config(text=f" {selected_snapshot_dir.name})")


def record_matches_date_range(record, path):
    """Match clinical JSON dates, falling back to its dated file/folder name."""
    if active_date_start is None and active_date_end is None:
        return True
    dates = json_record_dates(record)
    if not dates:
        dates = path_record_dates(path)
    return date_is_in_active_range(dates)


def file_matches_date_range(path, modality):
    """Filter images by path dates and blood tests by JSON or path dates."""
    if active_date_start is None and active_date_end is None:
        return True
    if modality != "blood_tests":
        return date_is_in_active_range(path_record_dates(path))
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        # Keep invalid JSON visible so the existing loader can report it.
        return date_is_in_active_range(path_record_dates(path)) or not path_record_dates(path)
    records = data if isinstance(data, list) else [data]
    return any(record_matches_date_range(record, path) for record in records)


def patient_files(modality, extensions):
    """Return safe patient-modality files filtered by the active date range.

    New record sets use ``<patient>/data/<modality>``.  Some of the bundled
    legacy record sets predate that wrapper and use ``<patient>/<modality>``.
    Prefer the documented layout, but safely fall back to the legacy location
    when the documented folder is absent.  This keeps discovery patient-scoped
    and never broadens it to another patient's files.
    """
    if active_patient is None:
        return [], "Select a patient to load records."
    if selected_snapshot_dir is None and patient_snapshots(active_patient):
        return [], "Choose a date and time to load this patient's data."
    if selected_snapshot_dir is not None:
        try:
            safe_path(selected_snapshot_dir, active_patient)
            folder = selected_snapshot_dir / modality
            safe_path(folder, active_patient)
        except (OSError, ValueError):
            return [], "Selected date/time data is no longer available."
    else:
        folder = None
    if folder is not None:
        # A selected snapshot is deliberately confined to one date/time folder.
        pass
    else:
        preferred_folder = active_patient / "data" / modality
        legacy_folder = active_patient / modality
        folder = preferred_folder if preferred_folder.is_dir() else legacy_folder
    try:
        # Do not merge layouts: if a patient has the documented folder it is
        # authoritative.  Falling back only when it is missing avoids showing
        # duplicate or stale legacy records alongside current ones.
        safe_path(folder, active_patient)
        if not folder.is_dir():
            return [], f"No {modality} folder for this patient."
        paths = []
        for path in sorted(folder.rglob("*"), key=lambda p: str(p).casefold()):
            try:
                if (not path.is_file() or path.suffix.lower() not in extensions
                        or any(part.startswith(".") for part in path.relative_to(folder).parts)):
                    continue
                safe_path(path, active_patient)
                attributes = getattr(path.stat(), "st_file_attributes", 0)
                if attributes & 6 or not file_matches_date_range(path, modality):
                    continue
                paths.append(path)
            except (OSError, ValueError):
                # Do not follow linked paths outside the selected patient.
                continue
        if paths:
            return paths, ""
        if active_date_start is not None or active_date_end is not None:
            return [], f"No dated {modality} records match the selected range; undated files are excluded."
        return [], f"No supported records in {modality}."
    except (OSError, ValueError):
        return [], f"Missing or inaccessible {modality} folder."


active_patient = None
active_date_start = None
active_date_end = None
selected_snapshot_dir = None
image_viewer = None
ui_callbacks = queue.Queue()
# Thumbnail work is streamed through bounded queues. The decoder keeps only one
# source image open; Tk PhotoImage instances are created and cached on the UI thread.
THUMBNAIL_CACHE_LIMIT = 32
THUMBNAIL_REQUEST_LIMIT = 12
THUMBNAIL_RESULT_LIMIT = 4
THUMBNAIL_MAX_DIMENSION = 640
# RGB cache ceiling: 32 × 640 × 640 × 3 ≈ 39.3 MB; four queued
# decoded results add at most ≈4.9 MB, plus the single active decode.
thumbnail_requests = queue.Queue(maxsize=THUMBNAIL_REQUEST_LIMIT)
thumbnail_results = queue.Queue(maxsize=THUMBNAIL_RESULT_LIMIT)
thumbnail_cache = OrderedDict()
thumbnail_pending = set()
thumbnail_worker_thread = None
thumbnail_generation = 0
thumbnail_items = [[], []]
thumbnail_started_at = 0.0
thumbnail_first_paint = None
thumbnail_decoded_count = 0
thumbnail_worker_shutdown = threading.Event()
current_request = None
save_quick_button = None
report_original_draft = None
report_patient = None
report_snapshot = None
report_history_bundle = None
report_history_directory = None
report_history_paths = []
report_completed = False
report_review_acknowledged = False
report_rendering = False
report_review_var = None
REPORT_SCHEMA_VERSION = "1.0.0"
REPORT_TEMPLATE_VERSION = "medgemma-report-v1"


@dataclass(frozen=True)
class RequestSnapshot:
    """Immutable inputs captured before a worker begins network activity."""

    request_id: str
    patient: Path
    url: str
    model: str
    question: str
    image_paths: tuple
    blood_paths: tuple = ()
    xray_count: int = 0
    ekg_count: int = 0


@dataclass
class RequestState:
    """Lifecycle and cancellation handles for one generation request."""

    snapshot: RequestSnapshot
    started_at: float = field(default_factory=time.monotonic)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    connection_lock: threading.Lock = field(default_factory=threading.Lock)
    response: object = None
    terminal: bool = False
    stage: str = "Checking server"
    completion: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    """Final user-facing text and non-content completion metadata."""

    answer: str
    metadata: dict


def request_is_current(request_id):
    """Return whether callbacks still belong to the active nonterminal request."""
    return (
        current_request is not None
        and current_request.snapshot.request_id == request_id
        and not current_request.terminal
        and not current_request.cancel_event.is_set()
    )


def post_ui(callback):
    """Queue work for Tk; background workers never invoke Tcl directly."""
    ui_callbacks.put(callback)


def drain_ui():
    """Process network and bounded thumbnail results on the Tk main thread."""
    try:
        while True:
            ui_callbacks.get_nowait()()
    except queue.Empty:
        pass
    drain_thumbnail_results(limit=3)
    for column in range(2):
        request_visible_thumbnails(column)
    window.after(35, drain_ui)


def parse_mmddyy_input(value):
    """Validate one optional date bound in the requested six-digit MMDDYY format."""
    value = value.strip()
    if not value:
        return None
    if not re.fullmatch(r"\d{6}", value):
        raise ValueError("Enter dates as six digits in MMDDYY format.")
    try:
        return datetime.strptime(value, "%m%d%y").date()
    except ValueError as error:
        raise ValueError(f"{value} is not a valid MMDDYY calendar date.") from error


def apply_date_range():
    """Apply inclusive date bounds and reload all three original record panels."""
    global active_date_start, active_date_end
    if current_request is not None:
        show_status("Wait for the current request to finish before changing the date range.")
        return
    try:
        start = parse_mmddyy_input(date_from_var.get())
        end = parse_mmddyy_input(date_to_var.get())
        if start and end and start > end:
            raise ValueError("The start date must be on or before the end date.")
    except ValueError as error:
        messagebox.showerror("Invalid date range", str(error), parent=window)
        return
    active_date_start, active_date_end = start, end
    if active_date_start is None and active_date_end is None:
        date_filter_status.config(text="Blank range shows all records.")
    else:
        start_text = active_date_start.strftime("%m%d%y") if active_date_start else "any"
        end_text = active_date_end.strftime("%m%d%y") if active_date_end else "any"
        date_filter_status.config(text=f"Showing records {start_text} through {end_text}; undated records are excluded.")
    if active_patient is not None:
        # The normal patient reload clears any selections and stale response context.
        select_patient()
    else:
        update_selection_workspace()


def clear_date_range():
    """Clear both date bounds and restore all records for the active patient."""
    if current_request is not None:
        show_status("Wait for the current request to finish before changing the date range.")
        return
    date_from_var.set("")
    date_to_var.set("")
    apply_date_range()


def patient_snapshots(patient):
    """Return safe ``date/time`` record folders for one patient.

    The new layout is ``patient_records/<id>/<date>/<time>/<modality>``.
    Only time folders containing at least one expected modality directory are
    offered; legacy folders are never shown in this date/time picker.
    """
    snapshots = []
    if patient is None:
        return snapshots
    try:
        for date_dir in sorted(patient.iterdir(), key=lambda item: item.name, reverse=True):
            if not date_dir.is_dir() or date_dir.name.startswith(".") or not parse_record_dates(date_dir.name):
                continue
            safe_path(date_dir, patient)
            for time_dir in sorted(date_dir.iterdir(), key=lambda item: item.name, reverse=True):
                if not time_dir.is_dir() or time_dir.name.startswith("."):
                    continue
                safe_path(time_dir, patient)
                if any((time_dir / modality).is_dir() for modality in ("xrays", "ekgs", "blood_tests")):
                    snapshots.append((date_dir.name, time_dir.name, time_dir.resolve()))
    except (OSError, ValueError):
        return []
    return snapshots


def choose_snapshot():
    """Show one date/time picker and load a snapshot when a time is clicked."""
    if active_patient is None:
        show_status("Select a patient first.")
        return
    snapshots = patient_snapshots(active_patient)
    if not snapshots:
        messagebox.showinfo(
            "No date/time data",
            "No date/time record folders were found for this patient.",
            parent=window,
        )
        return
    dialog = tk.Toplevel(window)
    dialog.title("Choose patient date and time")
    dialog.transient(window)
    fit_dialog_to_display(dialog, 0.30, 0.50, 440, 360)
    dialog.columnconfigure(0, weight=1)
    dialog.rowconfigure(1, weight=1)
    patient_id_var = tk.StringVar(value=active_patient.name)
    picker_header = ttk.Frame(dialog, padding=(10, 6))
    picker_header.grid(row=0, column=0, columnspan=2, sticky="w")
    ttk.Label(picker_header, text="Patient ID", style="Field.TLabel").pack(side=tk.LEFT, padx=(0, 4))
    dialog_patient_picker = ttk.Combobox(picker_header, textvariable=patient_id_var, values=tuple(patients),
                                         state="readonly", width=18)
    dialog_patient_picker.pack(side=tk.LEFT)
    tree = ttk.Treeview(dialog, show="tree", selectmode="browse")
    tree.grid(row=1, column=0, sticky="nsew", padx=(10, 0), pady=(0, 10))
    scrollbar = ttk.Scrollbar(dialog, orient=tk.VERTICAL, command=tree.yview)
    scrollbar.grid(row=1, column=1, sticky="ns", padx=(0, 10), pady=(0, 10))
    tree.configure(yscrollcommand=scrollbar.set)
    # Dates keep the default theme color; selectable times use a contrasting
    # purple so the two hierarchy levels are visually distinct.
    tree.tag_configure("snapshot_time", foreground="#6a1b9a")
    date_nodes = {}
    snapshot_paths = {}
    for index, (date_name, time_name, path) in enumerate(snapshots):
        parent = date_nodes.get(date_name)
        if parent is None:
            parent = tree.insert("", tk.END, text=date_name, open=True)
            date_nodes[date_name] = parent
        item_id = f"snapshot_{index}"
        snapshot_paths[item_id] = path
        tree.insert(parent, tk.END, iid=item_id, text=time_name, tags=("snapshot_time",))

    def load_selected(_event=None):
        item = tree.focus()
        path = snapshot_paths.get(item)
        if path is None:
            return
        select_snapshot(path)
        dialog.destroy()

    def change_dialog_patient(_event=None):
        if patient_id_var.get() == patient_name.get():
            return
        dialog.destroy()
        patient_name.set(patient_id_var.get())
        select_patient()
        window.after_idle(choose_snapshot)

    tree.bind("<Double-Button-1>", load_selected)
    dialog_patient_picker.bind("<<ComboboxSelected>>", change_dialog_patient)
    footer = ttk.Frame(dialog, padding=(10, 0, 10, 10))
    footer.grid(row=2, column=0, columnspan=2, sticky="e")
    ttk.Button(footer, text="Load selected date & time", command=load_selected).pack(side=tk.LEFT)
    ttk.Button(footer, text="Close", command=dialog.destroy).pack(side=tk.LEFT, padx=(8, 0))


def select_snapshot(path):
    """Load only one selected date/time folder for the active patient."""
    global selected_snapshot_dir, active_date_start, active_date_end
    if active_patient is None or current_request is not None:
        return
    try:
        path = safe_path(path, active_patient)
        date_value = parse_record_dates(path.parent.name)
        if not date_value:
            raise ValueError("Snapshot date is invalid")
    except (OSError, ValueError):
        show_status("The selected date/time folder is unavailable.")
        return
    selected_snapshot_dir = path
    active_date_start = active_date_end = date_value[0]
    snapshot_selection_label.config(text=f"Loaded: {path.parent.name} {path.name}")
    clear_report_review()
    prompt_box.delete("1.0", tk.END)
    load_images()
    load_blood_tests()
    update_selection_workspace()
    show_status(f"Loaded {active_patient.name}: {path.parent.name} at {path.name}")


def select_patient(event=None):
    """Clear previous context and selections before loading the chosen patient."""
    global active_patient, selected_snapshot_dir, active_date_start, active_date_end
    if current_request is not None:
        return
    next_patient = patients.get(patient_name.get())
    if next_patient != active_patient and report_has_unsaved_edits():
        if not offer_report_recovery():
            patient_name.set(active_patient.name if active_patient else "")
            return
    close_image_viewer()  # Never leave one patient's image visible after rebinding.
    stop_voice()
    active_patient = next_patient
    selected_snapshot_dir = None
    active_date_start = active_date_end = None
    clear_report_review()
    # A patient change must not insert a canned request into the doctor's
    # question field.  Every request begins with explicit clinician input.
    prompt_box.delete("1.0", tk.END)
    snapshots = patient_snapshots(active_patient)
    if snapshots:
        # The newest folder is listed first; load it immediately so each
        # patient opens with a usable, isolated date/time snapshot.
        select_snapshot(snapshots[0][2])
    else:
        load_images()
        load_blood_tests()
        if active_patient is not None:
            snapshot_selection_label.config(text="Choose a date and time to load data.")
    show_status(f"Patient: {patient_name.get()}" if active_patient else "No patient directories found")
    refresh_saved_report_history()
    update_selection_workspace()


def report_has_unsaved_edits():
    """Return whether the completed report currently differs from its AI draft."""
    if not report_completed or report_original_draft is None:
        return False
    if report_history_bundle is not None:
        saved_drafts = report_history_bundle["drafts"]
        saved_reviewed = report_history_bundle["review"]["status"] == "acknowledged_locally_not_a_signature"
        return (
            answer_box.get("1.0", "end-1c") != saved_drafts["user_edited"]
            or bool(report_review_acknowledged) != saved_reviewed
        )
    return answer_box.get("1.0", "end-1c") != report_original_draft


def report_input_references(snapshot):
    """Format only record paths captured in the immutable request snapshot."""
    if snapshot is None:
        return "No completed request snapshot."
    patient = snapshot.patient.resolve()
    # Image paths are snapshotted in upload order; classify them by the original
    # selection boundary recorded when the request was submitted.
    xray_count = snapshot.xray_count
    ekg_count = snapshot.ekg_count
    grouped = (
        ("X-ray", snapshot.image_paths[:xray_count]),
        ("EKG", snapshot.image_paths[xray_count:xray_count + ekg_count]),
        ("Blood test", snapshot.blood_paths),
    )
    lines = []
    for label, paths in grouped:
        if paths:
            refs = []
            for path in paths:
                try:
                    refs.append(Path(path).resolve().relative_to(patient).as_posix())
                except (OSError, ValueError):
                    refs.append("[record reference unavailable]")
            lines.append(f"{label}: " + ", ".join(refs))
    return "\n".join(lines) if lines else "No records were supplied (text-only request)."


def report_input_reference_data(snapshot):
    """Return actual request files as patient-relative references for export."""
    if snapshot is None:
        return {"xray": [], "ekg": [], "blood_test": []}
    patient = Path(snapshot.patient).resolve()

    def relative_refs(paths):
        refs = []
        for path in paths:
            try:
                refs.append(Path(path).resolve(strict=True).relative_to(patient).as_posix())
            except (OSError, ValueError):
                # Do not turn an unavailable or escaped source into an absolute path.
                refs.append("[record reference unavailable]")
        return refs

    images = tuple(snapshot.image_paths)
    xray_count = snapshot.xray_count
    ekg_count = snapshot.ekg_count
    return {
        "xray": relative_refs(images[:xray_count]),
        "ekg": relative_refs(images[xray_count:xray_count + ekg_count]),
        "blood_test": relative_refs(snapshot.blood_paths),
    }


def build_report_export_bundle(patient_id, request, generated_at, original_text,
                               edited_text, completion, review_acknowledged):
    """Build versioned, content-only report metadata without source bytes or prompts."""
    request_id = str(getattr(request, "request_id", ""))
    if not re.fullmatch(r"[A-Fa-f0-9]{32}", request_id):
        raise ValueError("The report request identifier is invalid.")
    allowed_completion = completion_metadata(completion or {})
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "template_version": REPORT_TEMPLATE_VERSION,
        "report_id": request_id.lower(),
        "patient": {"id": str(patient_id)},
        "generation": {
            "model": str(getattr(request, "model", "")),
            "generated_at": str(generated_at),
            "completion_status": "completed",
            "completion_details": allowed_completion,
        },
        "review": {
            "status": "acknowledged_locally_not_a_signature" if review_acknowledged
            else "not_acknowledged",
            "label": "Review acknowledged locally (not a signature)" if review_acknowledged
            else "Review not acknowledged",
        },
        "input_references": report_input_reference_data(request),
        "drafts": {
            "original_ai": str(original_text),
            "user_edited": str(edited_text),
        },
    }


def render_report_export_text(bundle):
    """Render a readable plain-text report bundle with both draft versions."""
    generation = bundle["generation"]
    references = bundle["input_references"]
    ref_lines = []
    for label, key in (("X-ray", "xray"), ("EKG", "ekg"), ("Blood test", "blood_test")):
        values = references.get(key, [])
        ref_lines.append(f"{label}: " + (", ".join(values) if values else "None"))
    details = json.dumps(generation["completion_details"], ensure_ascii=False, sort_keys=True)
    return (
        "MEDGEMMA REPORT EXPORT\n"
        f"Report ID: {bundle['report_id']}\n"
        f"Patient: {bundle['patient']['id']}\n"
        f"Model: {generation['model']}\n"
        f"Generated: {generation['generated_at']}\n"
        f"Completion: {generation['completion_status']} ({details})\n"
        f"Review: {bundle['review']['label']}\n"
        f"Schema version: {bundle['schema_version']}\n"
        f"Template version: {bundle['template_version']}\n\n"
        "INPUTS ACTUALLY SENT (paths relative to patient record root)\n"
        + "\n".join(ref_lines) + "\n\n"
        "ORIGINAL AI DRAFT\n"
        f"{bundle['drafts']['original_ai']}\n\n"
        "USER-EDITED DRAFT\n"
        f"{bundle['drafts']['user_edited']}\n"
    )


def render_report_export_html(bundle):
    """Render a standalone HTML report while escaping all dynamic text."""
    generation = bundle["generation"]

    def escaped(value):
        return html.escape(str(value), quote=True)

    ref_rows = []
    for label, key in (("X-ray", "xray"), ("EKG", "ekg"), ("Blood test", "blood_test")):
        refs = bundle["input_references"].get(key, [])
        ref_rows.append(
            f"<tr><th>{escaped(label)}</th><td>{escaped(', '.join(refs) if refs else 'None')}</td></tr>"
        )
    completion_json = json.dumps(generation["completion_details"], ensure_ascii=False, sort_keys=True)
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>MedGemma report {escaped(bundle['report_id'])}</title>"
        "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#182b3a}"
        "h1,h2{color:#174b67}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f8fa;padding:1rem;border-radius:.4rem}"
        "table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid #ccd8df;padding:.45rem}</style>"
        "</head><body><h1>MedGemma report</h1><dl>"
        f"<dt>Report ID</dt><dd>{escaped(bundle['report_id'])}</dd>"
        f"<dt>Patient</dt><dd>{escaped(bundle['patient']['id'])}</dd>"
        f"<dt>Model</dt><dd>{escaped(generation['model'])}</dd>"
        f"<dt>Generated</dt><dd>{escaped(generation['generated_at'])}</dd>"
        f"<dt>Completion</dt><dd>{escaped(generation['completion_status'])}; {escaped(completion_json)}</dd>"
        f"<dt>Review</dt><dd>{escaped(bundle['review']['label'])}</dd>"
        f"<dt>Schema / template</dt><dd>{escaped(bundle['schema_version'])} / {escaped(bundle['template_version'])}</dd>"
        "</dl><h2>Inputs actually sent</h2><table><tbody>" + "".join(ref_rows) + "</tbody></table>"
        f"<h2>Original AI draft</h2><pre>{escaped(bundle['drafts']['original_ai'])}</pre>"
        f"<h2>User-edited draft</h2><pre>{escaped(bundle['drafts']['user_edited'])}</pre>"
        "</body></html>\n"
    )


def report_output_directory_is_safe(directory, records_root):
    """Require an existing chosen output directory outside the patient root."""
    destination = Path(directory).expanduser().resolve(strict=True)
    patient_root = Path(records_root).resolve(strict=True)
    if not destination.is_dir():
        raise ValueError("Choose an existing output directory.")
    if destination == patient_root or destination.is_relative_to(patient_root):
        raise ValueError("Reports cannot be saved inside patient_records.")
    return destination


def atomic_write_report_exports(directory, bundle, records_root):
    """Atomically write TXT, standalone HTML and JSON files to the chosen directory.

    Each complete file is first flushed to a uniquely named sibling temporary
    file and then replaced into place. Final names use only the validated opaque
    request ID, so patient names and generated content cannot affect the path.
    """
    destination = report_output_directory_is_safe(directory, records_root)
    report_id = str(bundle.get("report_id", ""))
    if not re.fullmatch(r"[a-f0-9]{32}", report_id):
        raise ValueError("The report identifier is invalid.")
    files = {
        f"report-{report_id}.txt": render_report_export_text(bundle),
        f"report-{report_id}.html": render_report_export_html(bundle),
        f"report-{report_id}.json": json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
    }
    written = []
    for filename, content in files.items():
        target = destination / filename
        if target.exists() or target.is_symlink():
            raise FileExistsError("A report with this identifier already exists in the chosen directory.")
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=destination,
                prefix=f".{filename}.", suffix=".tmp", delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(content)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, target)
            written.append(target)
        except Exception:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            # If a later format fails, remove earlier outputs from this invocation
            # so the user never sees a partial three-format export set.
            for completed_path in written:
                try:
                    completed_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
    return tuple(written)


def validate_saved_report_bundle(bundle, expected_patient_id, filename=None):
    """Validate a saved bundle without mutating it or trusting its file contents.

    History accepts only the currently supported export schema and completed
    reports. Unknown/newer schemas and malformed data are skipped read-only.
    """
    if not isinstance(bundle, dict) or bundle.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported report schema")
    report_id = bundle.get("report_id")
    if not isinstance(report_id, str) or not re.fullmatch(r"[a-f0-9]{32}", report_id):
        raise ValueError("invalid report identifier")
    if filename is not None and Path(filename).name != f"report-{report_id}.json":
        raise ValueError("report filename does not match its identifier")
    patient = bundle.get("patient")
    if (not isinstance(patient, dict) or not isinstance(patient.get("id"), str)
            or patient.get("id") != expected_patient_id):
        raise ValueError("report belongs to a different patient")
    generation = bundle.get("generation")
    if (not isinstance(generation, dict) or generation.get("completion_status") != "completed"
            or not isinstance(generation.get("model"), str)
            or not isinstance(generation.get("generated_at"), str)
            or not isinstance(generation.get("completion_details"), dict)):
        raise ValueError("report is incomplete or malformed")
    review = bundle.get("review")
    if not isinstance(review, dict) or review.get("status") not in (
            "acknowledged_locally_not_a_signature", "not_acknowledged"):
        raise ValueError("review status is unsupported")
    drafts = bundle.get("drafts")
    if (not isinstance(drafts, dict) or not isinstance(drafts.get("original_ai"), str)
            or not isinstance(drafts.get("user_edited"), str)):
        raise ValueError("draft text is malformed")
    references = bundle.get("input_references")
    if not isinstance(references, dict):
        raise ValueError("input references are malformed")
    for key in ("xray", "ekg", "blood_test"):
        paths = references.get(key)
        if not isinstance(paths, list) or any(not isinstance(item, str) for item in paths):
            raise ValueError("input references are malformed")
        for item in paths:
            # Exported references are patient-relative POSIX paths. Keep the
            # unavailable marker, but reject absolute paths and traversal.
            if item == "[record reference unavailable]":
                continue
            normalized = item.replace("\\", "/")
            if (normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized)
                    or any(part == ".." for part in normalized.split("/"))):
                raise ValueError("input reference is not patient-relative")
    if not isinstance(bundle.get("template_version"), str):
        raise ValueError("template version is missing")
    return bundle


def read_saved_report_bundle(path, expected_patient_id):
    """Read and validate one exported JSON bundle without ever rewriting it."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("saved report is not a regular file")
    if not re.fullmatch(r"report-[a-f0-9]{32}\.json", path.name):
        raise ValueError("saved report filename is unsupported")
    raw = path.read_text(encoding="utf-8")
    bundle = json.loads(raw)
    return validate_saved_report_bundle(bundle, expected_patient_id, path.name)


def choose_saved_report_history():
    """Let the user choose an existing history directory outside patient data."""
    global report_history_directory
    selected = filedialog.askdirectory(
        parent=window, title="Choose a folder with saved report bundles", mustexist=True,
    )
    if not selected:
        return
    try:
        report_history_directory = report_output_directory_is_safe(
            selected, PATIENT_RECORDS_DIR,
        )
    except (OSError, ValueError) as error:
        messagebox.showerror("History folder unavailable", str(error), parent=window)
        return
    refresh_saved_report_history()


def refresh_saved_report_history():
    """List only valid explicitly exported bundles belonging to the active patient."""
    global report_history_paths, report_history_directory
    if "saved_report_list" not in globals():
        return
    saved_report_list.delete(0, tk.END)
    report_history_paths = []
    if report_history_directory is None:
        history_folder_label.config(text="No history folder selected.")
        history_status_label.config(text="Choose a folder containing exported JSON bundles.")
        return
    try:
        safe_directory = report_output_directory_is_safe(
            report_history_directory, PATIENT_RECORDS_DIR,
        )
        history_folder_label.config(text=f"History folder: {safe_directory.name or str(safe_directory)}")
        if active_patient is None:
            history_status_label.config(text="Select a patient to view that patient's saved reports.")
            return
        skipped = 0
        matches = []
        for path in sorted(safe_directory.glob("report-*.json"), key=lambda item: item.name.casefold()):
            if path.is_symlink() or not re.fullmatch(r"report-[a-f0-9]{32}\.json", path.name):
                continue
            try:
                bundle = read_saved_report_bundle(path, active_patient.name)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                # Bad, unsupported, and other-patient bundles are not exposed.
                # Count only malformed/unsupported files, never disclose that
                # another patient's report exists in this folder.
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict) and raw.get("patient", {}).get("id") != active_patient.name:
                        continue
                except Exception:
                    pass
                skipped += 1
                continue
            matches.append((path, bundle))
        report_history_paths = [path for path, _ in matches]
        for path, bundle in matches:
            generated_at = bundle["generation"]["generated_at"]
            report_id = bundle["report_id"]
            saved_report_list.insert(tk.END, f"{generated_at}  ·  {report_id[:8]}")
        history_status_label.config(
            text=(f"{len(matches)} saved report(s) for {active_patient.name}."
                  + (f" Skipped {skipped} malformed or unsupported file(s)." if skipped else ""))
            if matches or skipped else f"No saved reports for {active_patient.name}."
        )
    except (OSError, ValueError) as error:
        history_folder_label.config(text="History folder unavailable.")
        history_status_label.config(text=f"History could not be read ({type(error).__name__}).")
        report_history_directory = None


def reopen_saved_report(event=None):
    """Reopen the selected saved bundle for the active patient, with no model call."""
    global report_original_draft, report_patient, report_snapshot
    global report_history_bundle, report_completed, report_review_acknowledged, report_rendering
    if active_patient is None or "saved_report_list" not in globals():
        return
    selection = saved_report_list.curselection()
    if not selection or selection[0] >= len(report_history_paths):
        history_status_label.config(text="Select a saved report first.")
        return
    path = report_history_paths[selection[0]]
    try:
        safe_directory = report_output_directory_is_safe(path.parent, PATIENT_RECORDS_DIR)
        if safe_directory != report_history_directory or path.is_symlink():
            raise ValueError("saved report location changed")
        bundle = read_saved_report_bundle(path, active_patient.name)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        history_status_label.config(text="This saved report is unavailable, malformed, unsupported, or belongs to another patient.")
        refresh_saved_report_history()
        return
    clear_report_review()
    report_original_draft = bundle["drafts"]["original_ai"]
    report_patient = active_patient.name
    report_snapshot = None
    report_history_bundle = bundle
    report_completed = bundle["generation"]["completion_status"] == "completed"
    report_review_acknowledged = bundle["review"]["status"] == "acknowledged_locally_not_a_signature"
    report_rendering = True
    original_box.configure(state=tk.NORMAL)
    insert_highlighted_report(report_original_draft, original_box)
    original_box.configure(state=tk.DISABLED)
    answer_box.configure(state=tk.NORMAL)
    answer_box.edit_modified(False)
    insert_highlighted_report(bundle["drafts"]["user_edited"], answer_box)
    answer_box.edit_modified(False)
    report_rendering = False
    review_label = "Review acknowledged locally (not a signature)" if report_review_acknowledged else "Review not acknowledged"
    report_meta_label.config(text=f"SAVED · COMPLETED · {review_label}")
    report_provenance_label.config(text=report_provenance_text())
    report_review_var.set(report_review_acknowledged)
    review_button.config(state=tk.NORMAL, text="Review acknowledged" if report_review_acknowledged else "Acknowledge review")
    restore_original_button.config(state=tk.NORMAL)
    export_report_button.config(state=tk.NORMAL)
    save_quick_button.config(state=tk.NORMAL)
    report_notebook.hide(0)
    report_notebook.tab(1, state="normal")
    report_notebook.select(1)
    update_report_version_label()
    history_status_label.config(text=f"Reopened saved report {bundle['report_id'][:8]} for {active_patient.name}.")


def export_current_report():
    """Ask for a destination and deliberately export all supported formats."""
    global report_history_bundle
    if (not report_completed or (report_snapshot is None and report_history_bundle is None)
            or report_patient != (active_patient.name if active_patient else None)):
        messagebox.showerror("No completed report", "Complete a report for the active patient before exporting.", parent=window)
        return
    selected_directory = filedialog.askdirectory(
        parent=window, title="Choose a report output directory", mustexist=True,
    )
    if not selected_directory:
        return
    try:
        if report_history_bundle is not None:
            # Re-exporting a reopened bundle retains its original provenance and
            # completion metadata while recording only the current user draft.
            bundle = dict(report_history_bundle)
            bundle["drafts"] = dict(bundle["drafts"])
            bundle["drafts"]["user_edited"] = answer_box.get("1.0", "end-1c")
            acknowledged = bool(report_review_acknowledged)
            bundle["review"] = {
                "status": "acknowledged_locally_not_a_signature" if acknowledged else "not_acknowledged",
                "label": "Review acknowledged locally (not a signature)" if acknowledged
                else "Review not acknowledged",
            }
        else:
            request = report_snapshot["request"]
            bundle = build_report_export_bundle(
                patient_id=report_patient,
                request=request,
                generated_at=report_snapshot["generated_at"],
                original_text=report_original_draft,
                edited_text=answer_box.get("1.0", "end-1c"),
                completion=report_snapshot.get("metadata", {}),
                review_acknowledged=bool(report_review_acknowledged),
            )
        written = atomic_write_report_exports(selected_directory, bundle, PATIENT_RECORDS_DIR)
    except (OSError, ValueError) as error:
        messagebox.showerror("Report export failed", str(error), parent=window)
        return
    report_history_bundle = bundle
    messagebox.showinfo(
        "Report exported",
        "Saved TXT, standalone HTML, and versioned JSON report files to the chosen directory.",
        parent=window,
    )
    if report_history_directory is not None and Path(selected_directory).resolve() == report_history_directory:
        refresh_saved_report_history()


def report_provenance_text():
    """Describe patient, model, time and inputs from the actual completed request."""
    if report_history_bundle is not None:
        bundle = report_history_bundle
        generation = bundle["generation"]
        references = bundle["input_references"]
        lines = []
        for label, key in (("X-ray", "xray"), ("EKG", "ekg"), ("Blood test", "blood_test")):
            values = references.get(key, [])
            if values:
                lines.append(f"{label}: " + ", ".join(values))
        return (
            f"Saved report for: {bundle['patient']['id']}   |   Model: {generation['model']}   |   "
            f"Generated: {generation['generated_at']}\n"
            "Inputs actually sent:\n" + ("\n".join(lines) if lines else "No records were supplied (text-only request).")
        )
    if not report_completed or not report_snapshot:
        return "No completed request provenance."
    request = report_snapshot["request"]
    timestamp = report_snapshot["generated_at"]
    return (
        f"Patient: {request.patient.name}   |   Model: {request.model}   |   "
        f"Generated: {timestamp}\n"
        f"Inputs actually sent:\n{report_input_references(request)}"
    )


def update_report_version_label(event=None):
    """Label which immutable or editable report version is currently visible."""
    if not report_completed:
        report_version_label.config(text="Report view: no completed draft")
        return
    tab = report_notebook.index(report_notebook.select())
    report_version_label.config(
        text="Displayed version: Original AI draft" if tab == 0
        else "Displayed version: User-edited draft"
    )


def on_report_modified(event=None):
    """Track edits separately and invalidate acknowledgement after any change."""
    global report_review_acknowledged, report_rendering
    if report_rendering:
        answer_box.edit_modified(False)
        return
    if not report_completed or report_original_draft is None:
        answer_box.edit_modified(False)
        return
    edited = answer_box.get("1.0", "end-1c")
    if edited != report_original_draft:
        report_review_acknowledged = False
        report_review_var.set(False)
        report_meta_label.config(text="COMPLETED · User-edited draft · Review not acknowledged")
    else:
        report_meta_label.config(text="COMPLETED · Original AI draft · Review not acknowledged")
    review_button.config(text="Acknowledge review")
    answer_box.edit_modified(False)
    update_report_version_label()


def acknowledge_report_review():
    """Record local clinician acknowledgement for the complete displayed report."""
    global report_review_acknowledged
    if (not report_completed or report_original_draft is None or active_patient is None
            or report_patient != active_patient.name):
        report_review_var.set(False)
        return
    report_review_acknowledged = bool(report_review_var.get())
    version = "Original AI draft" if report_notebook.index(report_notebook.select()) == 0 else "User-edited draft"
    saved = "SAVED · " if report_history_bundle is not None else ""
    if report_review_acknowledged:
        report_meta_label.config(text=f"{saved}COMPLETED · {version} review acknowledged locally (not a signature)")
        review_button.config(text="Review acknowledged")
    else:
        report_meta_label.config(text=f"{saved}COMPLETED · {version} · Review not acknowledged")
        review_button.config(text="Acknowledge review")


def restore_original_draft():
    """Restore the editable pane from the preserved original AI wording."""
    global report_review_acknowledged, report_rendering
    if not report_completed or report_original_draft is None:
        return
    report_review_acknowledged = False
    report_review_var.set(False)
    report_rendering = True
    answer_box.configure(state=tk.NORMAL)
    answer_box.delete("1.0", tk.END)
    answer_box.insert(tk.END, report_original_draft)
    answer_box.edit_modified(False)
    report_rendering = False
    report_notebook.select(1)
    review_button.config(text="Acknowledge review")
    report_meta_label.config(text="COMPLETED · Original wording restored · Review not acknowledged")
    update_report_version_label()


def clear_report_review():
    """Clear report content and all completion/review state for patient changes."""
    global report_original_draft, report_patient, report_snapshot, report_history_bundle
    global report_completed, report_review_acknowledged, report_rendering
    report_original_draft = None
    report_patient = None
    report_snapshot = None
    report_history_bundle = None
    report_completed = False
    report_review_acknowledged = False
    report_rendering = True
    for widget in (original_box, answer_box):
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.configure(state=tk.DISABLED)
    report_rendering = False
    report_meta_label.config(text="No completed report.")
    report_provenance_label.config(text="Provenance appears here for a completed request.")
    report_review_var.set(False)
    review_button.config(state=tk.DISABLED, text="Acknowledge review")
    restore_original_button.config(state=tk.DISABLED)
    export_report_button.config(state=tk.DISABLED)
    save_quick_button.config(state=tk.DISABLED)
    report_notebook.tab(0, state="disabled")
    report_notebook.tab(1, state="disabled")


def offer_report_recovery():
    """Ask before losing edits on patient change; optionally save a local TXT copy."""
    answer = messagebox.askyesnocancel(
        "Unsaved report edits",
        "This patient's report has unsaved edits. Save a recovery copy before switching?\n\n"
        "Yes: choose a file outside patient_records.\nNo: discard these edits.\nCancel: stay with this patient.",
        parent=window,
    )
    if answer is None:
        return False
    if answer is False:
        return True
    destination = filedialog.asksaveasfilename(
        parent=window,
        title="Save report recovery copy",
        defaultextension=".txt",
        filetypes=(("Text files", "*.txt"),),
        initialfile=f"{re.sub(r'[^A-Za-z0-9._-]+', '-', report_patient or 'report')}-recovery.txt",
    )
    if not destination:
        return False
    path = Path(destination).expanduser().resolve()
    try:
        if path.is_relative_to(PATIENT_RECORDS_DIR.resolve()):
            messagebox.showerror(
                "Choose a separate report location",
                "Recovery copies cannot be saved inside patient_records.",
                parent=window,
            )
            return False
        provenance = report_provenance_text()
        content = (
            "MEDGEMMA REPORT RECOVERY COPY\n"
            f"Patient: {report_patient or 'unknown'}\n"
            f"Completion: {'completed' if report_completed else 'incomplete'}\n"
            f"Review acknowledged: {'yes' if report_review_acknowledged and not report_has_unsaved_edits() else 'no'}\n\n"
            f"{provenance}\n\n"
            "ORIGINAL AI DRAFT\n"
            f"{report_original_draft or '[no completed AI draft]'}\n\n"
            "USER-EDITED DRAFT\n"
            f"{answer_box.get('1.0', 'end-1c')}\n"
        )
        path.write_text(content, encoding="utf-8")
    except OSError as error:
        messagebox.showerror("Recovery copy could not be saved", str(error), parent=window)
        return False
    return True


def apply_report_preset(event=None):
    """Place the chosen editable starter request into the question editor."""
    preset = preset_name.get()
    template = REPORT_PRESETS.get(preset)
    if template is None:
        return
    prompt_box.delete("1.0", tk.END)
    prompt_box.insert(tk.END, template)
    prompt_box.focus_set()


def update_selection_workspace():
    """Refresh visible patient/modality counts and the removable selection tray."""
    if active_patient is None:
        active_label.config(text="Active patient: none")
        totals = (0, 0, 0)
    else:
        totals = (
            len(patient_files("xrays", IMAGE_EXTENSIONS)[0]),
            len(patient_files("ekgs", IMAGE_EXTENSIONS)[0]),
            len(blood_test_records),
        )
        active_label.config(text=f"Active patient: {patient_name.get()}")
    counts = (
        f"X-ray {len(selected_images[0])}/{totals[0]} selected",
        f"EKG {len(selected_images[1])}/{totals[1]} selected",
        f"Blood {len(selected_blood_test_indices)}/{totals[2]} selected",
    )
    counts_label.config(text="   |   ".join(counts))
    selected_xrays = tuple(selected_images[0])
    selected_ekgs = tuple(selected_images[1])
    selected_blood_paths = tuple(
        blood_test_paths[index] for index in sorted(selected_blood_test_indices)
        if index < len(blood_test_paths)
    )
    context_label.config(text=build_input_context_summary(
        active_patient, selected_xrays, selected_ekgs, selected_blood_paths
    ))
    for label, text in zip(gallery_count_labels, counts[:2]):
        label.config(text=text)
    blood_count_label.config(text=counts[2])
    tray_list.delete(0, tk.END)
    tray_items.clear()
    if selected_images[0] or selected_images[1] or selected_blood_test_indices:
        for column, modality in ((0, "X-ray"), (1, "EKG")):
            for path in selected_images[column]:
                tray_items.append(("image", column, path))
                tray_list.insert(tk.END, f"{modality}: {path.name}")
        for index in sorted(selected_blood_test_indices):
            path = blood_test_paths[index]
            record = blood_test_records[index]
            label = lab_record_title(record, path)
            tray_items.append(("blood", index, path))
            tray_list.insert(tk.END, f"Blood test: {label} ({path.name})")
    else:
        tray_list.insert(tk.END, "No records selected — text-only request")


def remove_tray_selection(event=None):
    """Remove the selected tray row from its current modality and patient."""
    rows = tray_list.curselection()
    if not rows or rows[0] >= len(tray_items):
        return
    kind, key, path = tray_items[rows[0]]
    if kind == "image":
        column = key
        button, tile, status = selected_images[column].pop(path)
        button.config(relief=tk.RAISED, background=default_button_color)
        tile.config(background=TILE_BACKGROUND)
        status.config(text="Not selected")
    else:
        selected_blood_test_indices.discard(key)
        blood_test_view.tag_configure(f"blood_record_{key}", background="white")
        blood_test_view.tag_configure(f"blood_record_{key}", foreground=TEXT_COLOR)
        set_blood_row_marker(key, False)
    update_selection_workspace()


def set_modality_selection(modality, selected):
    """Select or clear only one modality for the active patient."""
    if active_patient is None:
        return
    if modality in (0, 1):
        selection = selected_images[modality]
        if selected:
            paths, _ = patient_files(("xrays", "ekgs")[modality], IMAGE_EXTENSIONS)
            for path in paths:
                for gallery in galleries:
                    for tile in gallery["grid"].winfo_children():
                        for button in tile.winfo_children():
                            if getattr(button, "record_path", None) == path and path not in selection:
                                status = getattr(button, "selection_status")
                                selection[path] = (button, tile, status)
                                button.config(relief=tk.SUNKEN, background=DATASET_COLORS[modality][1])
                                tile.config(background=DATASET_COLORS[modality][0])
                                status.config(text="Selected")
        else:
            for path, (button, tile, status) in tuple(selection.items()):
                button.config(relief=tk.RAISED, background=default_button_color)
                tile.config(background=TILE_BACKGROUND)
                status.config(text="Not selected")
                del selection[path]
    else:
        if selected:
            selected_blood_test_indices.update(range(len(blood_test_records)))
        else:
            selected_blood_test_indices.clear()
        for index in range(len(blood_test_records)):
            set_blood_row_marker(index, index in selected_blood_test_indices)
            blood_test_view.tag_configure(
                f"blood_record_{index}",
                background="#bbdefb" if index in selected_blood_test_indices else "white",
                foreground=TEXT_COLOR,
            )
    update_selection_workspace()


def toggle_demo_data():
    """Update the session-only demo indicator from the user's explicit choice."""
    demo_status.config(text="Demo data: ON (this session)" if demo_data.get() else
                       "Demo data: OFF (this session)")


def set_blood_row_marker(index, selected):
    """Write a textual checkbox marker so blood selection is not color-only."""
    if index >= len(blood_test_records):
        return
    line = blood_record_header_lines[index]
    marker = "[x] " if selected else "[ ] "
    blood_test_view.config(state=tk.NORMAL)
    blood_test_view.delete(f"{line}.0", f"{line}.4")
    blood_test_view.insert(f"{line}.0", marker, (f"blood_record_{index}",))
    blood_test_view.config(state=tk.NORMAL)


def _display_scalar(value):
    """Render one lab cell on a single line without hiding unknown values."""
    if value is None or value == "":
        return "Unknown"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("\r", " ").replace("\n", " ")


def lab_record_title(record, path):
    """Choose a short display label without requiring a particular schema."""
    if isinstance(record, dict):
        return str(record.get("title") or record.get("panel") or path.stem).replace("\r", " ").replace("\n", " ")
    return path.stem


def _lab_number(value):
    """Return a finite numeric lab value, excluding booleans and labels."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _lab_reference_bounds(reference):
    """Parse only explicit numeric reference bounds; never infer a range."""
    if isinstance(reference, dict):
        low = _lab_number(reference.get("low", reference.get("min")))
        high = _lab_number(reference.get("high", reference.get("max")))
        unit = reference.get("unit")
        if low is not None or high is not None:
            return (low, high, unit, None)
        return None
    if isinstance(reference, (list, tuple)) and len(reference) == 2:
        low, high = _lab_number(reference[0]), _lab_number(reference[1])
        if low is not None or high is not None:
            return (low, high, None, None)
        return None
    if not isinstance(reference, str):
        return None
    text = reference.strip()
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    match = re.fullmatch(rf"({number})\s*(?:-|–|—|\bto\b)\s*({number})(?:\s*(.*))?", text, re.IGNORECASE)
    if match:
        return (float(match.group(1)), float(match.group(2)), match.group(3).strip() or None, None)
    match = re.fullmatch(rf"(<=|>=|<|>)\s*({number})(?:\s*(.*))?", text)
    if match:
        return (None, float(match.group(2)), match.group(3).strip() or None, match.group(1))
    return None


def _lab_flag(value, unit, reference):
    """Flag a numeric value only when supplied bounds and units are compatible."""
    number = _lab_number(value)
    bounds = _lab_reference_bounds(reference)
    if number is None or bounds is None:
        return "Unknown"
    low, high, range_unit, comparator = bounds
    value_unit = str(unit).strip() if unit not in (None, "") else None
    range_unit = str(range_unit).strip() if range_unit not in (None, "") else None
    if ((value_unit is None) != (range_unit is None) or
            (value_unit is not None and value_unit.casefold() != range_unit.casefold())):
        return "Unknown (unit mismatch)"
    if comparator:
        in_range = {
            "<": number < high, "<=": number <= high,
            ">": number > high, ">=": number >= high,
        }[comparator]
        return "In range" if in_range else ("High" if comparator.startswith("<") else "Low")
    if low is not None and number < low:
        return "Low"
    if high is not None and number > high:
        return "High"
    return "In range"


def lab_view_rows(record):
    """Normalize a selectable source object into readable rows and keep raw data elsewhere.

    Supports structured results arrays, mapping-style legacy results, and
    records with no recognized schema. The source object is never modified.
    """
    results = record.get("results") if isinstance(record, dict) else None
    rows = []
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                test = item.get("test", item.get("name", item.get("analyte", "Unknown test")))
                has_value = any(key in item for key in ("value", "result", "reading"))
                value = item.get("value", item.get("result", item.get("reading"))) if has_value else item
                unit = item.get("unit")
                reference = item.get("reference_range", item.get("reference", item.get(
                    "normal_range", item.get("ref_range", item.get("range")))))
                rows.append((test, value, unit, reference))
            else:
                rows.append(("Result", item, None, None))
    elif isinstance(results, dict):
        for test, item in results.items():
            if isinstance(item, dict):
                has_value = any(key in item for key in ("value", "result", "reading"))
                value = item.get("value", item.get("result", item.get("reading"))) if has_value else item
                unit = item.get("unit")
                reference = item.get("reference_range", item.get("reference", item.get(
                    "normal_range", item.get("ref_range", item.get("range")))))
            else:
                value, unit, reference = item, None, None
            rows.append((test, value, unit, reference))
    elif results not in (None, ""):
        rows.append(("Results", results, None, None))
    elif isinstance(record, dict) and any(key in record for key in ("test", "name", "analyte", "value", "result")):
        test = record.get("test", record.get("name", record.get("analyte", "Unknown test")))
        value = record.get("value", record.get("result"))
        unit = record.get("unit")
        reference = record.get("reference_range", record.get("reference", record.get(
            "normal_range", record.get("ref_range", record.get("range")))))
        rows.append((test, value, unit, reference))
    else:
        # A legacy free-form result, or metadata-only record, remains visible.
        payload = record.get("result") if isinstance(record, dict) else record
        if payload in (None, ""):
            payload = record
        rows.append(("Result", payload, None, None))
    return [
        (_display_scalar(test), _display_scalar(value), _display_scalar(unit),
         _display_scalar(reference), _lab_flag(value, unit, reference))
        for test, value, unit, reference in rows
    ]


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
MUTED_TEXT = "#4b5d6e"
PRIMARY_BLUE = "#165d9e"
PRIMARY_BLUE_DARK = "#104a80"
TEAL = "#00695c"
CORAL = "#b23d32"
SOFT_BLUE = "#d9efff"
SOFT_TEAL = "#d8f5ef"
SOFT_CORAL = "#ffe2dd"
TILE_BACKGROUND = "#f5f9fc"
DATASET_COLORS = ((PRIMARY_BLUE, SOFT_BLUE), (TEAL, SOFT_TEAL))
GALLERY_NAMES = ("XRAY", "EKG/ECG")
REPORT_PRESETS = {
    "Draft report": (
        "Draft a concise report from only the records explicitly selected below. "
        "Describe observable evidence and state uncertainty; do not infer a diagnosis."
    ),
    "Summarize selected records": (
        "Summarize only the explicitly selected records below. Keep each observation "
        "under its matching modality and state when the supplied information is limited."
    ),
    "Explain selected results": (
        "Explain the explicitly selected blood-test results in plain language, using "
        "only supplied values, units, and reference ranges. Do not infer missing ranges "
        "or diagnose. Keep image sections separate and say when no data was supplied."
    ),
}
# Remote MedGemma can take several minutes when full-resolution images are sent.
# Keep the quick server/model check short, but allow inference much longer.
SERVER_CHECK_TIMEOUT = 15


def build_report_request(user_text, preset, xray_paths=(), ekg_paths=(), blood_tests=()):
    """Build one grounded request plus a human-readable summary of exact inputs.

    The caller passes only the user's current explicit selections. Image ordering
    matches the base64 array: all selected X-rays in click order, then selected
    ECG/EKG images in click order. Blood entries are (source path, parsed JSON).
    """
    xray_paths = tuple(Path(path) for path in xray_paths)
    ekg_paths = tuple(Path(path) for path in ekg_paths)
    blood_tests = tuple((Path(path), record) for path, record in blood_tests)
    image_paths = xray_paths + ekg_paths
    image_manifest = []
    for modality, paths in (("X-RAY", xray_paths), ("ECG/EKG", ekg_paths)):
        for path in paths:
            image_manifest.append(f"{len(image_manifest) + 1}. {modality} image: {path.name}")

    modality_summary = []
    for modality, paths in (("X-RAY", xray_paths), ("ECG/EKG", ekg_paths)):
        names = ", ".join(path.name for path in paths)
        modality_summary.append(
            f"{modality}: {len(paths)} selected" + (f" ({names})" if names else "")
        )
    blood_names = ", ".join(path.name for path, _ in blood_tests)
    modality_summary.append(
        f"BLOOD TEST: {len(blood_tests)} selected" + (f" ({blood_names})" if blood_names else "")
    )
    selected_count = len(image_paths) + len(blood_tests)
    summary = "Inputs to send: " + " | ".join(modality_summary)
    if not selected_count:
        summary += " | No records selected — text-only request."
    else:
        summary += " | Only the listed selections will be attached."

    sections = [
        f"Selected report preset: {preset}",
        "Doctor's editable request:",
        user_text.strip(),
        "Explicit selected-input manifest (attached images follow this exact order):",
        "\n".join(image_manifest) if image_manifest else "No image data is attached.",
    ]
    if blood_tests:
        sections.extend((
            "Selected BLOOD TEST records (only these selected JSON records are included):",
            json.dumps(
                [{"source": path.name, "data": record} for path, record in blood_tests],
                indent=2,
                ensure_ascii=False,
            ),
        ))
    else:
        sections.append("BLOOD TEST input: No data provided; no blood-test record is attached.")
    sections.extend((
        "Required final report format; keep the three sections separate:",
        "X-RAY\n[Evidence-based findings, or No data provided]",
        "ECG/EKG\n[Evidence-based findings, or No data provided]",
        "BLOOD TEST\n[Evidence-based findings, or No data provided]",
        "Do not present a missing modality as a finding. Distinguish supplied evidence "
        "from uncertainty. If a requested conclusion is unsupported or cannot be "
        "verified from these inputs, mark it 'Needs clinician review: not verifiable "
        "from supplied inputs.' Do not invent citations or numerical confidence.",
    ))
    return "\n\n".join(sections), summary, image_paths


def build_input_context_summary(patient, xray_paths=(), ekg_paths=(), blood_paths=()):
    """Summarize the active patient and exact explicit selections for the UI."""
    if patient is None:
        return "Inputs to send: no active patient."
    groups = (
        ("X-RAY", tuple(Path(path) for path in xray_paths)),
        ("ECG/EKG", tuple(Path(path) for path in ekg_paths)),
        ("BLOOD TEST", tuple(Path(path) for path in blood_paths)),
    )
    parts = []
    count = 0
    for modality, paths in groups:
        names = ", ".join(path.name for path in paths)
        count += len(paths)
        parts.append(f"{modality}: {len(paths)} selected" + (f" ({names})" if names else ""))
    summary = "Inputs to send: " + " | ".join(parts)
    summary += " | " + (
        "Only the listed selections will be attached."
        if count else "No records selected — text-only request."
    )
    return f"Active patient: {Path(patient).name}  |  {summary}"


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


def fit_dialog_to_display(dialog, width_fraction, height_fraction, minimum_width, minimum_height):
    """Size and center a secondary window within the current display.

    Hard-coded dialog geometries make the app either tiny on high-DPI screens
    or larger than a laptop display.  Every auxiliary window uses this shared
    display-relative policy instead.
    """
    dialog.update_idletasks()
    screen_width = dialog.winfo_screenwidth()
    screen_height = dialog.winfo_screenheight()
    width = min(max(minimum_width, round(screen_width * width_fraction)), screen_width - 60)
    height = min(max(minimum_height, round(screen_height * height_fraction)), screen_height - 100)
    x = max(20, (screen_width - width) // 2)
    y = max(20, (screen_height - height) // 2)
    dialog.geometry(f"{width}x{height}+{x}+{y}")


def select_image(path, button, tile, column):
    """Toggle one image with the selection color of its gallery."""
    selection = selected_images[column]
    if path in selection:
        _, _, status = selection.pop(path)
        button.config(relief=tk.RAISED, background=default_button_color)
        tile.config(background=TILE_BACKGROUND)
        status.config(text="Not selected")
    else:
        status = button.selection_status
        selection[path] = (button, tile, status)
        border, fill = DATASET_COLORS[column]
        button.config(relief=tk.SUNKEN, background=fill)
        tile.config(background=border)
        status.config(text="Selected")
    update_selection_workspace()


def encode_image(path):
    """Encode original source bytes; gallery resizing never changes uploads."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")



def load_blood_tests():
    """Load original selectable lab records and render their normalized rows."""
    set_snapshot_heading(blood_test_panel.heading_name, blood_test_panel.heading_date,
                         blood_test_panel.heading_time, "BLOOD TEST")
    blood_test_view.config(state=tk.NORMAL)
    blood_test_view.delete("1.0", tk.END)
    blood_test_records.clear()
    blood_test_paths.clear()
    blood_record_header_lines.clear()
    blood_line_record_indices.clear()
    selected_blood_test_indices.clear()
    paths, message = patient_files("blood_tests", {".json"})
    errors = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            records = data if isinstance(data, list) else [data]
            for record in records:
                if record == {} or record == []:
                    raise ValueError("Expected a nonempty JSON record")
                if not record_matches_date_range(record, path):
                    continue
                index = len(blood_test_records)
                blood_test_records.append(record)
                blood_test_paths.append(path)
                tag = f"blood_record_{index}"
                title = lab_record_title(record, path)
                blood_record_header_lines.append(int(blood_test_view.index(tk.END).split(".")[0]))
                blood_test_view.insert(tk.END, "[ ] " + title.replace("\r", " ").replace("\n", " ") + "\n", (tag, "blood_title"))
                blood_line_record_indices.append(index)
                blood_test_view.insert(tk.END, "    Test                 | Value          | Unit       | Reference range       | Flag\n", (tag, "blood_columns"))
                blood_line_record_indices.append(index)
                for test, value, unit, reference, flag in lab_view_rows(record):
                    cells = (test[:20], value[:14], unit[:10], reference[:22], flag)
                    line = "    " + f"{cells[0]:20} | {cells[1]:14} | {cells[2]:10} | {cells[3]:22} | {cells[4]}\n"
                    blood_test_view.insert(tk.END, line, (tag,))
                    blood_line_record_indices.append(index)
                blood_test_view.insert(tk.END, "\n", (tag,))
                blood_line_record_indices.append(index)
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(f"{path.name}: {error}")
    global blood_focus_record_index
    blood_focus_record_index = None
    if not blood_test_records:
        if active_date_start is not None or active_date_end is not None:
            empty_message = "No blood-test records match the selected date range; undated records are excluded."
        else:
            empty_message = message or "No valid blood-test JSON records."
        blood_test_view.insert(tk.END, empty_message + "\n")
    if errors:
        blood_test_view.insert(tk.END, "Invalid JSON: " + "; ".join(errors))
        print("Invalid blood-test JSON: " + "; ".join(errors))
    blood_test_view.config(state=tk.NORMAL)


def show_blood_details(event=None):
    """Open the complete original JSON object for the focused/selected record."""
    index = blood_focus_record_index
    if index is None or not 0 <= index < len(blood_test_records):
        index = next(iter(sorted(selected_blood_test_indices)), None)
    if index is None:
        show_status("Select or focus a blood-test record to inspect its original details.")
        return
    dialog = tk.Toplevel(window)
    dialog.title("Original blood-test record details")
    fit_dialog_to_display(dialog, 0.52, 0.64, 520, 360)
    dialog.transient(window)
    dialog.columnconfigure(0, weight=1)
    dialog.rowconfigure(1, weight=1)
    record = blood_test_records[index]
    label = blood_test_paths[index].name
    ttk.Label(dialog, text=f"{label} — original selectable record", padding=8).grid(
        row=0, column=0, sticky="ew")
    details = scrolledtext.ScrolledText(dialog, wrap=tk.NONE, font=("Consolas", 10))
    details.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
    details.insert("1.0", json.dumps(record, ensure_ascii=False, indent=2))
    details.configure(state=tk.DISABLED)
    ttk.Button(dialog, text="Close", command=dialog.destroy).grid(row=2, column=0, sticky="e", padx=8, pady=8)


def blood_record_index_for_line(line_number):
    """Map any rendered title/table line back to its original selectable object."""
    if 1 <= line_number <= len(blood_line_record_indices):
        return blood_line_record_indices[line_number - 1]
    return None



def toggle_blood_test_record(event):
    """Select the original record represented by any of its displayed rows."""
    line_number = int(blood_test_view.index(f"@{event.x},{event.y}").split(".")[0])
    record_index = blood_record_index_for_line(line_number)
    if record_index is None:
        return "break"
    global blood_focus_record_index
    blood_focus_record_index = record_index
    blood_test_view.mark_set("insert", f"{line_number}.0")
    return toggle_blood_test_index(record_index)


def toggle_blood_test_index(record_index):
    """Toggle one blood-test record and refresh the explicit selection state."""
    if not 0 <= record_index < len(blood_test_records):
        return "break"
    line_tag = f"blood_record_{record_index}"
    if record_index in selected_blood_test_indices:
        selected_blood_test_indices.remove(record_index)
    else:
        selected_blood_test_indices.add(record_index)
    set_blood_row_marker(record_index, record_index in selected_blood_test_indices)
    blood_test_view.tag_configure(
        line_tag,
        background="#bbdefb" if record_index in selected_blood_test_indices else "white",
    )
    blood_test_view.tag_configure(line_tag, foreground=TEXT_COLOR)
    blood_test_view.tag_raise("blood_title")
    update_selection_workspace()
    return "break"


def blood_test_keypress(event):
    """Allow keyboard navigation and Space/Enter selection without editing text."""
    if event.keysym in ("space", "Return", "KP_Enter"):
        line_number = int(blood_test_view.index("insert").split(".")[0])
        record_index = blood_record_index_for_line(line_number)
        if record_index is not None:
            global blood_focus_record_index
            blood_focus_record_index = record_index
            return toggle_blood_test_index(blood_focus_record_index)
        return "break"
    if event.keysym == "Tab":
        target = blood_test_view.tk_focusPrev() if event.state & 0x1 else blood_test_view.tk_focusNext()
        if target is not None:
            target.focus_set()
        return "break"
    if event.keysym in ("Up", "Down", "Left", "Right", "Home", "End",
                        "Prior", "Next"):
        return None
    return "break"


def blood_test_keyup(event):
    """Keep the Details button targeted at the record under keyboard focus."""
    global blood_focus_record_index
    line_number = int(blood_test_view.index("insert").split(".")[0])
    record_index = blood_record_index_for_line(line_number)
    if record_index is not None:
        blood_focus_record_index = record_index


def update_gallery_scrollbars(canvas, canvas_window, grid):
    """Keep the canvas viewport scrollable when image content is oversized."""
    grid.update_idletasks()
    # Fill the visible canvas when content is small, but retain the complete
    # requested content width when an image is wider than the physical window.
    content_width = grid.winfo_reqwidth()
    viewport_width = max(canvas.winfo_width(), 1)
    canvas.itemconfigure(canvas_window, width=max(viewport_width, content_width))
    canvas.configure(scrollregion=canvas.bbox("all"))


def decode_gallery_thumbnail(request):
    """Decode one bounded display copy off-thread; never retain source bytes."""
    generation, column, item = request
    path = item["path"]
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
            original_width, original_height = source.size
            configured_width, configured_height = GALLERY_SIZES[column]
            width = configured_width or original_width
            height = configured_height or original_height
            # A zero dimension means preserve the source dimension, but never
            # let a display setting turn a gallery thumbnail into a huge bitmap.
            scale = min(1.0, THUMBNAIL_MAX_DIMENSION / max(width, height, 1))
            target_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            if column == 1:
                picture = source.resize(target_size, Image.Resampling.LANCZOS)
            else:
                picture = ImageOps.pad(
                    source,
                    target_size,
                    method=Image.Resampling.LANCZOS,
                    color="white",
                )
            picture.load()
        return generation, column, item, picture, None
    except Exception as error:
        return generation, column, item, None, type(error).__name__


def thumbnail_decode_worker():
    """Run serial Pillow decoding with bounded input/output queues."""
    while not thumbnail_worker_shutdown.is_set():
        try:
            request = thumbnail_requests.get(timeout=0.2)
        except queue.Empty:
            continue
        if request is None:
            break
        result = decode_gallery_thumbnail(request)
        while not thumbnail_worker_shutdown.is_set():
            try:
                thumbnail_results.put(result, timeout=0.1)
                break
            except queue.Full:
                continue


def ensure_thumbnail_worker():
    """Start the single background decoder lazily on first gallery load."""
    global thumbnail_worker_thread
    if thumbnail_worker_thread is None or not thumbnail_worker_thread.is_alive():
        thumbnail_worker_thread = threading.Thread(
            target=thumbnail_decode_worker,
            name="medgemma-thumbnail-decoder",
            daemon=True,
        )
        thumbnail_worker_thread.start()


def request_visible_thumbnails(column):
    """Queue only thumbnails in or near a gallery's current viewport."""
    if not galleries or column >= len(galleries):
        return
    gallery = galleries[column]
    canvas = gallery["canvas"]
    items = thumbnail_items[column]
    if not items or not canvas.winfo_exists():
        return
    try:
        top_fraction, bottom_fraction = canvas.yview()
        content_height = max(gallery["grid"].winfo_reqheight(), canvas.winfo_height(), 1)
        margin = canvas.winfo_height() * 0.35
        top = top_fraction * content_height - margin
        bottom = bottom_fraction * content_height + margin
    except tk.TclError:
        return
    ensure_thumbnail_worker()
    for item in items:
        if not item["tile"].winfo_exists():
            continue
        y0 = item["tile"].winfo_y()
        y1 = y0 + max(item["tile"].winfo_height(), 1)
        if y1 < top or y0 > bottom:
            continue
        key = (thumbnail_generation, column, str(item["path"]))
        if key in thumbnail_cache:
            thumbnail_cache.move_to_end(key)
            continue
        if key in thumbnail_pending:
            continue
        if item.get("failed"):
            continue
        thumbnail_pending.add(key)
        try:
            thumbnail_requests.put_nowait((thumbnail_generation, column, item))
        except queue.Full:
            thumbnail_pending.discard(key)
            break


def drain_thumbnail_results(limit=3):
    """Create Tk images on the UI thread and evict beyond the fixed LRU cap."""
    global thumbnail_first_paint, thumbnail_decoded_count
    try:
        from PIL import ImageTk
    except ImportError:
        return
    for _ in range(limit):
        try:
            generation, column, item, picture, error_name = thumbnail_results.get_nowait()
        except queue.Empty:
            break
        key = (generation, column, str(item["path"]))
        thumbnail_pending.discard(key)
        if generation != thumbnail_generation:
            continue
        if error_name or picture is None or not item["button"].winfo_exists():
            item["failed"] = True
            item["status"].config(text="Thumbnail unavailable")
            continue
        photo = ImageTk.PhotoImage(picture, master=window)
        item["button"].config(image=photo, text=item["path"].name)
        thumbnail_cache[key] = (photo, item)
        thumbnail_cache.move_to_end(key)
        thumbnail_decoded_count += 1
        if thumbnail_first_paint is None:
            thumbnail_first_paint = time.perf_counter() - thumbnail_started_at
            # Timing is intentionally aggregate only; filenames and record data are omitted.
            print(
                f"[gallery] first thumbnail visible in {thumbnail_first_paint:.3f}s "
                f"({THUMBNAIL_CACHE_LIMIT}-image / ~39.3 MB cache cap, "
                f"{THUMBNAIL_MAX_DIMENSION}px max edge)"
            )
        while len(thumbnail_cache) > THUMBNAIL_CACHE_LIMIT:
            _old_key, (old_photo, old_item) = thumbnail_cache.popitem(last=False)
            if old_item["button"].winfo_exists():
                old_item["button"].config(image="", text=old_item["path"].name)
                old_item["status"].config(text="Thumbnail unloaded — scroll to reload")
        if len(thumbnail_cache) <= THUMBNAIL_CACHE_LIMIT:
            item["status"].config(text="Thumbnail ready — Enter/Space selects")
    # The queue limits decoded Pillow objects waiting for Tk to <= four.


def invalidate_thumbnail_work():
    """Drop stale patient work and free cached Tk images before rebuilding."""
    global thumbnail_generation
    thumbnail_generation += 1
    thumbnail_cache.clear()
    thumbnail_pending.clear()
    for work_queue in (thumbnail_requests, thumbnail_results):
        while True:
            try:
                work_queue.get_nowait()
            except queue.Empty:
                break


def current_viewer_records():
    """Return the active patient's safe image paths in gallery order."""
    records = []
    for modality in ("xrays", "ekgs"):
        paths, _ = patient_files(modality, IMAGE_EXTENSIONS)
        records.extend((modality, path) for path in paths)
    return records


def close_image_viewer():
    """Close the detail window and release its display-only image references."""
    global image_viewer
    state = image_viewer
    image_viewer = None
    if state is not None:
        try:
            state["window"].destroy()
        except tk.TclError:
            pass


def viewer_fit_scale(source_size, viewport_size):
    """Calculate an aspect-preserving fit scale for any image orientation."""
    width, height = source_size
    viewport_width, viewport_height = viewport_size
    if min(width, height, viewport_width, viewport_height) <= 0:
        return 1.0
    return min(viewport_width / width, viewport_height / height)


def viewer_render_size(source_size, viewport_size, zoom=1.0, max_dimension=12000):
    """Return bounded display dimensions for fit-relative zoom levels."""
    width, height = source_size
    fit = viewer_fit_scale(source_size, viewport_size)
    scale = min(8.0, max(0.1, fit * zoom))
    if max(width * scale, height * scale) > max_dimension:
        scale *= max_dimension / max(width * scale, height * scale)
    return max(1, round(width * scale)), max(1, round(height * scale))


def render_viewer_image(state, center=True):
    """Render a transformed display copy; never write it back to the source."""
    if image_viewer is not state or not state["window"].winfo_exists():
        return
    from PIL import Image, ImageTk

    canvas = state["canvas"]
    canvas.update_idletasks()
    viewport = (max(1, canvas.winfo_width()), max(1, canvas.winfo_height()))
    source = state["source"]
    if source is None:
        return
    fit = viewer_fit_scale(source.size, viewport)
    state["fit_scale"] = fit
    size = viewer_render_size(source.size, viewport, state["zoom"])
    picture = source.resize(size, Image.Resampling.LANCZOS)
    state["photo"] = ImageTk.PhotoImage(picture, master=canvas)
    canvas.delete("all")
    x = max((viewport[0] - size[0]) // 2, 0)
    y = max((viewport[1] - size[1]) // 2, 0)
    canvas.create_image(x, y, anchor="nw", image=state["photo"])
    canvas.configure(scrollregion=(0, 0, max(size[0], viewport[0]), max(size[1], viewport[1])))
    if center:
        canvas.xview_moveto(0.5)
        canvas.yview_moveto(0.5)
    state["status"].config(
        text=f"{state['path'].name}  |  {source.width} × {source.height}  |  {state['zoom']:.0%} of fit"
    )


def show_viewer_record(state, index):
    """Load one record belonging to the patient's viewer snapshot."""
    from PIL import Image, ImageOps

    if image_viewer is not state or state["patient"] != active_patient:
        close_image_viewer()
        return
    index %= len(state["records"])
    modality, path = state["records"][index]
    state["index"] = index  # Keep navigation moving past unreadable files.
    state["path"] = path
    try:
        safe_path(path, state["patient"])
        with Image.open(path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
            source.load()
    except (OSError, ValueError) as error:
        state["source"] = None
        state["photo"] = None
        state["canvas"].delete("all")
        state["title"].config(text=f"{modality.upper()} — {path.name} (unavailable)")
        state["status"].config(text=f"Could not open image for inspection ({type(error).__name__}).")
        return
    state["source"] = source
    state["zoom"] = 1.0
    state["title"].config(text=f"{modality.upper()} — {path.name}")
    render_viewer_image(state)


def navigate_viewer(delta):
    """Move to the previous or next image without changing its selection."""
    if image_viewer is not None:
        show_viewer_record(image_viewer, image_viewer["index"] + delta)


def zoom_viewer(multiplier):
    """Zoom relative to fit while keeping the current image available to pan."""
    if image_viewer is None:
        return
    state = image_viewer
    state["zoom"] = min(8.0, max(0.1, state["zoom"] * multiplier))
    render_viewer_image(state)


def reset_viewer():
    """Return the current image to the full-view fit and centered position."""
    if image_viewer is not None:
        image_viewer["zoom"] = 1.0
        render_viewer_image(image_viewer)


def open_image_viewer(path):
    """Open a separate, nonselecting detail viewer for an active-patient image."""
    global image_viewer
    if active_patient is None:
        return
    try:
        safe_path(path, active_patient)
    except (OSError, ValueError):
        show_status("This image is no longer available for the active patient.")
        return
    records = current_viewer_records()
    match = next((index for index, (_, record_path) in enumerate(records) if record_path == path), None)
    if match is None:
        show_status("This image is no longer available for the active patient.")
        return
    close_image_viewer()
    viewer = tk.Toplevel(window)
    viewer.title("Image detail viewer")
    fit_dialog_to_display(viewer, 0.78, 0.80, 640, 480)
    viewer.minsize(420, 320)
    viewer.transient(window)
    viewer.protocol("WM_DELETE_WINDOW", close_image_viewer)
    toolbar = ttk.Frame(viewer)
    toolbar.pack(fill=tk.X, padx=8, pady=6)
    title = ttk.Label(toolbar, text="", style="Field.TLabel")
    title.pack(side=tk.LEFT, padx=(0, 12))
    ttk.Button(toolbar, text="Previous  ‹", command=lambda: navigate_viewer(-1)).pack(side=tk.LEFT, padx=2)
    ttk.Button(toolbar, text="Next  ›", command=lambda: navigate_viewer(1)).pack(side=tk.LEFT, padx=2)
    ttk.Button(toolbar, text="Fit", command=reset_viewer).pack(side=tk.LEFT, padx=(10, 2))
    ttk.Button(toolbar, text="−", command=lambda: zoom_viewer(1 / 1.25)).pack(side=tk.LEFT, padx=2)
    ttk.Button(toolbar, text="+", command=lambda: zoom_viewer(1.25)).pack(side=tk.LEFT, padx=2)
    ttk.Button(toolbar, text="Reset", command=reset_viewer).pack(side=tk.LEFT, padx=2)
    status = ttk.Label(viewer, text="", anchor="w")
    status.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 6))
    canvas_frame = ttk.Frame(viewer)
    canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))
    canvas = tk.Canvas(canvas_frame, background="#17222d", highlightthickness=0, takefocus=True)
    canvas.grid(row=0, column=0, sticky="nsew")
    yscroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
    xscroll.grid(row=1, column=0, sticky="ew")
    canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    canvas_frame.rowconfigure(0, weight=1)
    canvas_frame.columnconfigure(0, weight=1)
    state = {
        "window": viewer, "canvas": canvas, "title": title, "status": status,
        "records": records, "index": match, "patient": active_patient,
        "source": None, "photo": None, "path": path, "zoom": 1.0,
        "fit_scale": 1.0,
    }
    image_viewer = state
    show_viewer_record(state, match)
    canvas.bind("<Configure>", lambda _event: render_viewer_image(state), add="+")
    canvas.bind("<ButtonPress-1>", lambda event: canvas.scan_mark(event.x, event.y))
    canvas.bind("<B1-Motion>", lambda event: canvas.scan_dragto(event.x, event.y, gain=1))
    viewer.bind("<Left>", lambda _event: navigate_viewer(-1))
    viewer.bind("<Right>", lambda _event: navigate_viewer(1))
    viewer.bind("<Prior>", lambda _event: navigate_viewer(-1))
    viewer.bind("<Next>", lambda _event: navigate_viewer(1))
    viewer.bind("<plus>", lambda _event: zoom_viewer(1.25))
    viewer.bind("<equal>", lambda _event: zoom_viewer(1.25))
    viewer.bind("<minus>", lambda _event: zoom_viewer(1 / 1.25))
    viewer.bind("<0>", lambda _event: reset_viewer())
    viewer.bind("<Escape>", lambda _event: close_image_viewer())
    canvas.focus_set()


def load_images():
    """Build lightweight tiles, then decode visible thumbnails asynchronously."""
    global thumbnail_items, thumbnail_started_at, thumbnail_first_paint, thumbnail_decoded_count
    invalidate_thumbnail_work()
    thumbnail_started_at = time.perf_counter()
    thumbnail_first_paint = None
    thumbnail_decoded_count = 0
    thumbnail_items = [[], []]
    for gallery in galleries:
        for widget in gallery["grid"].winfo_children():
            widget.destroy()
    for selection in selected_images:
        selection.clear()

    try:
        from PIL import Image, ImageOps
    except ImportError:
        Image = ImageOps = None

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
        set_snapshot_heading(gallery["heading_name"], gallery["heading_date"],
                             gallery["heading_time"], GALLERY_NAMES[column])
        if not paths:
            ttk.Label(grid, text=message, wraplength=400).grid(row=0, column=0, padx=5, pady=12)
        if Image is None and paths:
            ttk.Label(grid, text="Install Pillow: py -m pip install pillow", wraplength=220).grid(row=0, column=0, padx=5, pady=12)
            continue
        displayed = len(paths)
        for index, path in enumerate(paths):
            tile = tk.Frame(grid, background=TILE_BACKGROUND, padx=5, pady=5)
            # X-Ray uses three images per row. EKG uses a vertical list with
            # exactly one image per row. Horizontal scrolling handles wide
            # originals when a configured dimension is zero.
            columns_per_row = 1 if column == 1 else 3
            tile.grid(
                row=index // columns_per_row,
                column=index % columns_per_row,
                padx=2,
                pady=5,
                sticky="n",
            )
            button = tk.Button(
                tile, image="", relief=tk.RAISED, borderwidth=2,
                takefocus=True, text=f"{path.name}\nLoading thumbnail…", compound=tk.TOP,
                highlightthickness=2, highlightbackground="#526b80",
                highlightcolor=PRIMARY_BLUE_DARK, foreground=TEXT_COLOR,
                activeforeground=TEXT_COLOR,
            )
            button.record_path = path
            button.selection_status = ttk.Label(tile, text="Not selected", anchor="center")
            button.config(command=lambda p=path, b=button, t=tile, c=column: select_image(p, b, t, c))
            wheel_handler = lambda event, c=column: (
                galleries[c]["canvas"].yview_scroll(int(-event.delta / 120), "units"),
                "break",
            )
            button.bind("<MouseWheel>", wheel_handler, add="+")
            tile.bind("<MouseWheel>", wheel_handler, add="+")
            button.pack()
            button.selection_status.pack()
            if not streamlined_layout:
                ttk.Button(tile, text="View details", takefocus=True,
                           command=lambda p=path: open_image_viewer(p)).pack(pady=(3, 0))
            # The button label and explicit state text make selection clear without color.
            thumbnail_items[column].append({
                "path": path, "button": button, "tile": tile,
                "status": button.selection_status, "index": index,
            })
        gallery["title"].config(text=f"{GALLERY_NAMES[column]} — {len(paths)} image(s)")
        set_snapshot_heading(gallery["heading_name"], gallery["heading_date"],
                             gallery["heading_time"], GALLERY_NAMES[column])
        update_gallery_scrollbars(
            gallery["canvas"], gallery["canvas_window"], gallery["grid"]
        )
        gallery["canvas"].bind(
            "<MouseWheel>",
            lambda event, c=gallery["canvas"]: (c.yview_scroll(int(-event.delta / 120), "units"), "break"),
            add="+",
        )
        gallery["grid"].bind(
            "<MouseWheel>",
            lambda event, c=gallery["canvas"]: (c.yview_scroll(int(-event.delta / 120), "units"), "break"),
            add="+",
        )
        gallery["canvas"].bind(
            "<Configure>", lambda _event, c=column: request_visible_thumbnails(c), add="+"
        )
    ensure_thumbnail_worker()
    for column in range(2):
        request_visible_thumbnails(column)


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
    try:
        with urlopen(f"{url}/api/tags", timeout=SERVER_CHECK_TIMEOUT) as response:
            data = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(server_error_category(error)) from None
    if not isinstance(data, dict) or not isinstance(data.get("models", []), list):
        raise RuntimeError("invalid server response")
    return [item["name"] for item in data.get("models", [])
            if isinstance(item, dict) and item.get("name")]


def model_capabilities(url, model):
    """Read optional Ollama model capabilities; unknown metadata stays unknown."""
    request = Request(
        f"{url}/api/show",
        data=json.dumps({"model": model}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=SERVER_CHECK_TIMEOUT) as response:
            data = json.load(response)
        if not isinstance(data, dict):
            return None
        capabilities = data.get("capabilities")
        if isinstance(capabilities, list) and all(isinstance(item, str) for item in capabilities):
            return set(capabilities)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        # Older servers may not implement /api/show or may omit capabilities.
        # Do not turn optional metadata into a valid request failure.
        return None
    return None


def set_request_stage(request_state, stage):
    """Publish an honest request phase without exposing response content."""
    if request_state is None:
        return
    request_state.stage = stage
    request_id = request_state.snapshot.request_id
    post_ui(lambda rid=request_id, value=stage: show_status(value)
            if request_is_current(rid) else None)


def final_only_answer(response_text):
    """Strip supported reasoning wrappers and reject reasoning-only output.

    Joining streamed response fragments before this function makes delimiters
    split across arbitrary network chunks indistinguishable from whole tokens.
    The separate Ollama ``thinking`` field is intentionally never passed here.
    """
    answer = response_text.strip()
    final_patterns = (
        r"<unused95>\s*(?:final\s*)?",
        r"<\|channel\|>\s*final\s*",
        r"</think\s*>\s*",
    )
    for pattern in final_patterns:
        matches = list(re.finditer(pattern, answer, flags=re.IGNORECASE))
        if matches:
            answer = answer[matches[-1].end():].strip()
            break
    # A start marker without its matching final delimiter is reasoning-only.
    thinking_pattern = r"(?:<unused94>\s*thought|<think>|<\|channel\|>\s*analysis)"
    if re.search(thinking_pattern, answer, flags=re.IGNORECASE):
        raise RuntimeError(
            "The model returned no final report (reasoning-only response). "
            "Retry explicitly; if this repeats, check that the selected model "
            "template produces a final answer."
        )
    # Remove stray control tokens if a template emits them around final text.
    answer = re.sub(r"</?(?:think|analysis|final)>|<\|(?:channel|end)\|>", "", answer,
                    flags=re.IGNORECASE).strip()
    if not answer:
        raise RuntimeError("The model completed without a final answer. Retry the request.")
    return answer


def parse_stream_events(lines):
    """Parse synthetic or live Ollama NDJSON events into response and metadata."""
    response_parts = []
    metadata = {}
    done = False
    for raw_line in lines:
        if isinstance(raw_line, bytes):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise RuntimeError("Malformed response event (invalid UTF-8).") from error
        else:
            line = str(raw_line)
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError("Malformed response event (invalid JSON).") from error
        if not isinstance(event, dict):
            raise RuntimeError("Malformed response event (expected a JSON object).")
        if event.get("error"):
            raise RuntimeError("Ollama reported a generation error. Check the selected model and server.")
        part = event.get("response", "")
        if part is not None:
            if not isinstance(part, str):
                raise RuntimeError("Malformed response event (response text was not a string).")
            response_parts.append(part)
        # Do not retain, log, or display `thinking`; only its existence is ignored.
        if event.get("done") is True:
            done = True
            for key in ("done_reason", "total_duration", "load_duration", "prompt_eval_count",
                        "prompt_eval_duration", "eval_count", "eval_duration"):
                value = event.get(key)
                if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                    metadata[key] = value
    if not done:
        raise RuntimeError("Truncated response: Ollama disconnected before its final event. Retry the request.")
    if metadata.get("done_reason") == "length":
        raise RuntimeError("Truncated response: Ollama reached its output limit. Retry with a larger limit.")
    return "".join(response_parts), metadata


def completion_metadata(data):
    """Return actual Ollama completion fields, excluding all generated text."""
    allowed = ("done_reason", "total_duration", "load_duration", "prompt_eval_count",
               "prompt_eval_duration", "eval_count", "eval_duration")
    return {key: data[key] for key in allowed
            if isinstance(data.get(key), (str, int, float)) and not isinstance(data.get(key), bool)}


def _post_generate(url, payload, request_state, stream):
    """POST one request and retain the response handle for local cancellation."""
    outgoing = dict(payload)
    outgoing["stream"] = stream
    request = Request(
        f"{url}/api/generate", data=json.dumps(outgoing).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    response = urlopen(request, timeout=GENERATION_TIMEOUT)
    if request_state is not None:
        with request_state.connection_lock:
            request_state.response = response
            if request_state.cancel_event.is_set():
                response.close()
    return response


def generate(url, model, question, image_paths=(), request_state=None, stream=True):
    """POST selected full-resolution images and return final text plus metadata."""
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
            "sections. Use only the explicitly selected inputs and their manifest. "
            "Separate supplied evidence from uncertainty, do not infer a diagnosis, "
            "and do not claim more than the inputs support. If a section has no "
            "supplied data, write 'No data provided' under that section. Mark a "
            "requested conclusion that is unsupported or unverifiable as 'Needs "
            "clinician review: not verifiable from supplied inputs.' Do not invent "
            "citations or numerical confidence."
        ),
        # The user's question, for example: "What do these images show?"
        "prompt": question,
        # Generation length, consistency, and repetition controls come from .env.
        "options": {
            "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE,
            "repeat_penalty": REPEAT_PENALTY,
            "repeat_last_n": REPEAT_LAST_N,
        },
    }
    capabilities = model_capabilities(url, model)
    # Only send a thinking control when /api/show explicitly advertises support.
    # Older/unknown servers get the normal request without this optional field.
    if capabilities is not None and "thinking" in capabilities:
        payload["think"] = THINK
    set_request_stage(request_state, "Preparing selected records")
    if image_paths:
        # encode_image returns one Base64 string for each selected source file.
        # image_paths contains selected X-rays followed by ECG/EKG images,
        # preserving click order within each. The gallery's
        # resized PhotoImage objects are never sent. With no selected images,
        # omit the "images" key entirely and submit a text-only request.
        payload["images"] = [encode_image(path) for path in image_paths]
    # json.dumps turns the dictionary into JSON text. UTF-8 then converts that
    # text into bytes for HTTP. Example: {"prompt": "My question", "images":
    # ["<base64 image 1>", "<base64 image 2>"], ...}.
    try:
        set_request_stage(request_state, "Waiting for model")
        try:
            response = _post_generate(url, payload, request_state, stream)
        except HTTPError as error:
            # Some older Ollama versions reject streaming. Retry the same model
            # and payload once as a regular JSON response; never change models.
            if stream and error.code in (400, 404):
                error.close()
                set_request_stage(request_state, "Retrying with nonstreaming response compatibility")
                response = _post_generate(url, payload, request_state, False)
                stream = False
            else:
                raise
        try:
            set_request_stage(request_state, "Receiving report")
            if stream:
                response_text, metadata = parse_stream_events(response)
            else:
                try:
                    data = json.load(response)
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                    raise RuntimeError("Malformed nonstreaming response (invalid JSON).") from error
                if not isinstance(data, dict):
                    raise RuntimeError("Malformed nonstreaming response (expected a JSON object).")
                response_text = data.get("response", "")
                metadata = completion_metadata(data)
                if data.get("done") is False:
                    raise RuntimeError("Truncated response: Ollama marked the response incomplete. Retry the request.")
                if metadata.get("done_reason") == "length":
                    raise RuntimeError("Truncated response: Ollama reached its output limit. Retry with a larger limit.")
                if not isinstance(response_text, str):
                    raise RuntimeError("Malformed nonstreaming response (response text was not a string).")
        finally:
            if request_state is not None:
                with request_state.connection_lock:
                    if request_state.response is response:
                        request_state.response = None
            response.close()
    except HTTPError as error:
        raise RuntimeError(server_error_category(error)) from None
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
        raise RuntimeError(server_error_category(error)) from None
    except (IncompleteRead, RemoteDisconnected, ConnectionResetError, BrokenPipeError):
        raise RuntimeError("Connection disconnected before completion. Retry the request.") from None

    if request_state is not None and request_state.cancel_event.is_set():
        raise RuntimeError("Request cancelled.")

    answer = final_only_answer(response_text)
    if request_state is not None:
        request_state.completion = dict(metadata)
    return GenerationResult(answer=answer, metadata=metadata)


def show_status(message):
    """Only the Tkinter main thread calls this function."""
    status_label.config(text=message)


def set_readiness(server, model):
    """Update independent, non-content readiness indicators on the Tk thread."""
    global server_readiness, model_readiness
    server_readiness = server
    model_readiness = model
    generation = (
        "READY" if active_patient is not None and server == "READY"
        and model.startswith("INSTALLED") else "NOT READY"
    )
    readiness_label.config(
        text=(f"Records: {records_readiness}   |   Server: {server}   |   "
              f"Model: {model}   |   Generation: {generation}")
    )


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


def finish_request(request_id, answer=None, metadata=None, error=None, cancelled=False):
    """Finish a request once; obsolete and duplicate worker callbacks are ignored."""
    global current_request, report_original_draft, report_patient, report_snapshot
    global report_completed, report_review_acknowledged, report_rendering
    state = current_request
    if state is None or state.snapshot.request_id != request_id or state.terminal:
        return
    # A stale callback must never place one patient's report in another patient's view.
    if active_patient is None or active_patient.resolve() != state.snapshot.patient.resolve():
        state.terminal = True
        current_request = None
        patient_picker.config(state="readonly")
        send_button.config(state=tk.NORMAL)
        cancel_button.config(state=tk.DISABLED)
        retry_button.config(state=tk.DISABLED)
        report_meta_label.config(text="DISCARDED · Patient changed; no report or review was retained")
        show_status("Discarded response because the active patient changed.")
        return
    state.terminal = True
    current_request = None
    patient_picker.config(state="readonly")
    send_button.config(state=tk.NORMAL)
    cancel_button.config(state=tk.DISABLED)
    retry_button.config(state=tk.NORMAL if error is not None and not cancelled else tk.DISABLED)
    elapsed_label.config(text=f"Elapsed: {int(time.monotonic() - state.started_at)}s")
    if cancelled:
        show_status("Request cancelled. The server may continue processing.")
    else:
        show_status("Completed" if error is None else f"Request failed: {error}")
    clear_report_review()
    if cancelled:
        report_meta_label.config(text="CANCELLED · No completed draft; review unavailable")
        answer_box.configure(state=tk.NORMAL)
        answer_box.insert(tk.END, "Request cancelled.")
        answer_box.configure(state=tk.DISABLED)
    elif error is None:
        report_original_draft = str(answer or "")
        report_patient = state.snapshot.patient.name
        report_snapshot = {
            "request": state.snapshot,
            "generated_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
            "metadata": dict(metadata or {}),
        }
        report_completed = True
        report_review_acknowledged = False
        report_rendering = True
        original_box.configure(state=tk.NORMAL)
        insert_highlighted_report(report_original_draft, original_box)
        original_box.configure(state=tk.DISABLED)
        answer_box.configure(state=tk.NORMAL)
        answer_box.edit_modified(False)
        summary = completion_summary(metadata or {})
        insert_highlighted_report(report_original_draft, answer_box)
        answer_box.edit_modified(False)
        report_rendering = False
        report_meta_label.config(text="COMPLETED · Original AI draft · Review not acknowledged"
                                 + (f" · {summary}" if summary else ""))
        report_provenance_label.config(text=report_provenance_text())
        review_button.config(state=tk.NORMAL, text="Acknowledge review")
        restore_original_button.config(state=tk.NORMAL)
        export_report_button.config(state=tk.NORMAL)
        save_quick_button.config(state=tk.NORMAL)
        report_notebook.hide(0)
        report_notebook.tab(1, state="normal")
        report_notebook.select(1)
        report_version_label.config(text="Displayed version: User-edited draft")
        report_review_var.set(False)
    else:
        report_meta_label.config(text=("CANCELLED" if cancelled else "INCOMPLETE")
                                 + " · No completed draft; review unavailable")
        answer_box.configure(state=tk.NORMAL)
        answer_box.insert(tk.END, f"Error: {error}")
        answer_box.configure(state=tk.DISABLED)
    if not cancelled and error is None and voice_enabled:
        speak_response(answer)


def completion_summary(metadata):
    """Format actual Ollama completion fields without including generated text."""
    parts = []
    reason = metadata.get("done_reason")
    if reason:
        parts.append(f"reason {reason}")
    prompt_tokens = metadata.get("prompt_eval_count")
    output_tokens = metadata.get("eval_count")
    if isinstance(prompt_tokens, int):
        parts.append(f"prompt {prompt_tokens} tokens")
    if isinstance(output_tokens, int):
        parts.append(f"output {output_tokens} tokens")
    duration = metadata.get("total_duration")
    if isinstance(duration, (int, float)):
        parts.append(f"server {duration / 1_000_000_000:.1f}s")
    eval_duration = metadata.get("eval_duration")
    if isinstance(eval_duration, (int, float)) and eval_duration > 0 and isinstance(output_tokens, int):
        parts.append(f"{output_tokens / (eval_duration / 1_000_000_000):.1f} tokens/s")
    return " · ".join(parts)


def update_elapsed(request_id):
    """Refresh elapsed time while the same request remains active."""
    if request_is_current(request_id):
        elapsed = int(time.monotonic() - current_request.started_at)
        elapsed_label.config(text=f"Elapsed: {elapsed}s")
        window.after(250, lambda rid=request_id: update_elapsed(rid))


def cancel_request():
    """Stop accepting output immediately and close an established HTTP response."""
    state = current_request
    if state is None or state.terminal:
        return
    state.cancel_event.set()
    with state.connection_lock:
        response = state.response
    if response is not None:
        try:
            response.close()
        except OSError:
            pass
    finish_request(state.snapshot.request_id, cancelled=True)


def close_window():
    """Cancel locally without joining workers or blocking the Tk event loop."""
    if report_has_unsaved_edits() and not offer_report_recovery():
        return
    state = current_request
    if state is not None and not state.terminal:
        state.cancel_event.set()
        with state.connection_lock:
            response = state.response
        if response is not None:
            try:
                response.close()
            except OSError:
                pass
        state.terminal = True
    thumbnail_worker_shutdown.set()
    window.destroy()


def insert_highlighted_report(report, widget=None):
    """Display the report and color medical-section names red."""
    widget = widget or answer_box
    # Match common spellings regardless of capitalization:
    # X-ray/Xray/X ray, EKG, ECG, and Blood Test/Blood-Test.
    pattern = re.compile(
        r"\b(?:x[\s-]?ray|ecg\s*/\s*ekg|ekg\s*/\s*ecg|ekg|ecg|blood[\s-]?test)\b",
        flags=re.IGNORECASE,
    )
    position = 0
    for match in pattern.finditer(report):
        widget.insert(tk.END, report[position:match.start()])
        widget.insert(tk.END, match.group(0), ("report_medical_heading",))
        position = match.end()
    widget.insert(tk.END, report[position:])


def run_request(state):
    """Run only the frozen request inputs and tag every UI callback with its ID."""
    snapshot = state.snapshot
    request_id = snapshot.request_id
    try:
        set_request_stage(state, "Checking server and selected model")
        for path in snapshot.image_paths:
            safe_path(path, snapshot.patient)
        # Give a useful message before trying generation with a missing model.
        models = list_models(snapshot.url)
        if state.cancel_event.is_set():
            return
        if snapshot.model not in models and f"{snapshot.model}:latest" not in models:
            post_ui(lambda rid=request_id: set_readiness("READY", f"MISSING ({snapshot.model})")
                    if request_is_current(rid) else None)
            raise RuntimeError(
                f"Model is not installed on the selected server ({snapshot.model}). "
                "Check the model name or install it on that server."
            )
        post_ui(lambda rid=request_id: set_readiness("READY", f"INSTALLED ({snapshot.model})")
                if request_is_current(rid) else None)
        if state.cancel_event.is_set():
            return
        result = generate(
            snapshot.url, snapshot.model, snapshot.question,
            snapshot.image_paths, request_state=state,
        )
        post_ui(lambda rid=request_id, value=result: finish_request(
            rid, answer=value.answer, metadata=value.metadata)
                if request_is_current(rid) else None)
    except Exception as error:
        message = str(error) if isinstance(error, (ValueError, RuntimeError)) else server_error_category(error)
        if state.cancel_event.is_set():
            return
        if (message.startswith("HTTP error ") or message in (
                "request timed out", "connection failed", "connection disconnected before completion",
                "malformed response event", "server check failed")):
            post_ui(lambda rid=request_id: set_readiness("OFFLINE", "UNKNOWN (server check failed)")
                    if request_is_current(rid) else None)
        post_ui(lambda rid=request_id, detail=message: finish_request(rid, error=detail)
                if request_is_current(rid) else None)


def send_question():
    """Validate GUI values and send one question to the selected model."""
    global current_request
    if current_request is not None:
        return
    patient = active_patient
    if patient is None:
        show_status("Select a patient first.")
        return
    if report_has_unsaved_edits() and not offer_report_recovery():
        show_status("Send cancelled; the current report draft was kept open.")
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
    selected_blood_paths = tuple(
        blood_test_paths[index]
        for index in sorted(selected_blood_test_indices)
        if index < len(blood_test_paths)
    )
    xray_count = len(selected_images[0])
    ekg_count = len(selected_images[1])
    model = model_name.get().strip()
    try:
        url = base_url(server_url.get())
        if not user_question or not model:
            raise ValueError("Enter a question and a model name.")
    except ValueError as error:
        show_status(str(error))
        return
    snapshot = RequestSnapshot(
        request_id=uuid.uuid4().hex,
        patient=Path(patient),
        url=url,
        model=model,
        question=str(question),
        image_paths=tuple(Path(path) for path in image_paths),
        blood_paths=tuple(Path(path) for path in selected_blood_paths),
        xray_count=xray_count,
        ekg_count=ekg_count,
    )
    state = RequestState(snapshot=snapshot)
    current_request = state
    patient_picker.config(state=tk.DISABLED)
    send_button.config(state=tk.DISABLED)
    cancel_button.config(state=tk.NORMAL)
    retry_button.config(state=tk.DISABLED)
    clear_report_review()
    report_meta_label.config(text="IN PROGRESS · No completed draft; review unavailable")
    answer_box.configure(state=tk.DISABLED)
    attachments = []
    if image_paths:
        attachments.append(f"{len(image_paths)} image(s)")
    if selected_blood_tests:
        attachments.append(f"{len(selected_blood_tests)} blood-test record(s)")
    show_status(
        f"Checking server for {model}"
        + (f" with {', '.join(attachments)}" if attachments else " (text only)")
        + "..."
    )
    # Pass the question and image snapshot together to the background worker.
    # run_request calls generate(), which builds and sends one HTTP POST.
    elapsed_label.config(text="Elapsed: 0s")
    window.after(250, lambda rid=snapshot.request_id: update_elapsed(rid))
    threading.Thread(target=run_request, args=(state,), daemon=True).start()


def retry_request():
    """Explicitly start a fresh request from the current prompt and selections."""
    if current_request is not None:
        return
    show_status("Retrying with the current prompt and current record selection...")
    send_question()


def refresh_models():
    """Read the model list from the selected local or remote Ollama server."""
    try:
        url = base_url(server_url.get())
    except ValueError as error:
        show_status(str(error))
        return
    refresh_button.config(state=tk.DISABLED)
    set_readiness("CHECKING", "CHECKING")
    show_status("Checking Ollama...")

    def worker():
        try:
            models = list_models(url)
            def complete():
                refresh_button.config(state=tk.NORMAL)
                model_picker.configure(values=models or DEFAULT_MODELS)
                selected_model = model_name.get().strip()
                installed = selected_model in models or f"{selected_model}:latest" in models
                set_readiness(
                    "READY",
                    f"INSTALLED ({selected_model})" if installed else f"MISSING ({selected_model})",
                )
                show_status(
                    "Ollama server reachable. "
                    + (f"{len(models)} installed model(s) found." if models else "No models installed.")
                )
            post_ui(complete)
        except Exception as error:
            message = str(error) if isinstance(error, RuntimeError) else server_error_category(error)
            def failed():
                refresh_button.config(state=tk.NORMAL)
                set_readiness("OFFLINE", "UNKNOWN (server check failed)")
                show_status(f"Server check failed ({message}). Records remain available; use Check Connection to retry.")
            post_ui(failed)

    threading.Thread(target=worker, daemon=True).start()


def main():
    """Validate startup data, then construct the preserved Tkinter interface."""
    global PATIENT_RECORDS_DIR, patients, window, ui_style, server_url, model_name, server_entry, model_picker, refresh_button, patient_name, patient_picker, active_label, demo_data, demo_status, counts_label, tray_list, tray_items, preset_name, prompt_box, context_label, send_button, cancel_button, retry_button, elapsed_label, voice_toggle_button, status_label, answer_box, original_box, report_notebook, report_meta_label, report_provenance_label, report_version_label, report_review_var, review_button, restore_original_button, export_report_button, save_quick_button, saved_report_list, history_folder_label, history_status_label, selected_images, default_button_color, galleries, gallery_count_labels, blood_test_panel, blood_count_label, blood_test_records, blood_test_paths, selected_blood_test_indices, blood_test_view, blood_record_header_lines, blood_line_record_indices, blood_focus_record_index, gallery_viewport_width, actions_placeholder, readiness_label, diagnostics_label, records_readiness, streamlined_layout, snapshot_selection_label
    # Fatal data-root validation precedes construction of any Tk widget.
    try:
        PATIENT_RECORDS_DIR = patient_records_root()
    except (RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
    patients = discover_patients(PATIENT_RECORDS_DIR)
    records_readiness = (
        f"READY ({len(patients)} patient(s))" if patients
        else "EMPTY (no patient directories)"
    )

    # Build the Tkinter desktop interface on the main thread.
    window = tk.Tk()
    window.protocol("WM_DELETE_WINDOW", close_window)
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
    # Clam exposes a clear border color on focus; classic tk.Buttons below use
    # a two-pixel system-colored highlight ring for the same keyboard cue.
    ui_style.map("TButton", bordercolor=[("focus", PRIMARY_BLUE_DARK)])
    ui_style.map("TEntry", bordercolor=[("focus", PRIMARY_BLUE_DARK)])
    ui_style.map("TCombobox", bordercolor=[("focus", PRIMARY_BLUE_DARK)])
    window.option_add("*Button.highlightThickness", 2)
    window.option_add("*Button.highlightBackground", "#526b80")
    window.option_add("*Button.highlightColor", PRIMARY_BLUE_DARK)
    # Reserve one full-width image area containing two stacked galleries.
    # Start at a laptop-friendly size; users can drag the panes to allocate space.
    window_width = window_dimension("WINDOW_WIDTH", 1360)
    window_height = window_dimension("WINDOW_HEIGHT", 900)
    # Keep the clinical workspace readable on ultrawide displays as well as
    # laptops.  A giant, edge-to-edge form makes each panel harder to scan.
    visible_width = min(window_width, 1600, window.winfo_screenwidth() - 30)
    visible_height = min(window_height, 900, window.winfo_screenheight() - 80)
    window.geometry(f"{visible_width}x{visible_height}")
    try:
        tk_scale = float(window.tk.call("tk", "scaling"))
    except (tk.TclError, ValueError):
        tk_scale = 1.0
    # Only genuinely narrow windows use the two-row header fallback.  Display
    # scaling alone must not separate the patient menu from its date filters.
    compact_layout = visible_width < 900
    # Keep the clinical screen focused on selecting records and asking a
    # question.  Background callback support remains intact, but optional
    # report-management and demo controls are not shown in the main workflow.
    streamlined_layout = True
    qa_wraplength = max(250, min(440, round(visible_width * 0.34)))
    window.minsize(min(880, visible_width), min(560, visible_height))
    # X-ray and EKG share one row on normal desktop displays, so calculate a
    # per-gallery viewport instead of requesting nearly the entire data area.
    gallery_viewport_width = max(220, round((visible_width - 45) * 0.58))

    settings = ttk.LabelFrame(
        window, text="Connection & Model", style="Settings.TLabelframe"
    )
    settings.pack(fill=tk.X, padx=14, pady=(10, 5))
    settings.columnconfigure(1, weight=1 if compact_layout else 0)
    settings.columnconfigure(3, weight=1 if compact_layout else 0)
    server_url = tk.StringVar(value=DEFAULT_SERVER_URL)
    model_name = tk.StringVar(value=DEFAULT_MODEL)
    ttk.Label(settings, text="Server URL", style="Field.TLabel").grid(row=0, column=0, padx=(10, 5), pady=7, sticky="w")
    server_entry = ttk.Entry(
        settings,
        textvariable=server_url,
        width=max(24, min(42, len(DEFAULT_SERVER_URL) + 2)),
    )
    server_entry.grid(row=0, column=1, padx=4, pady=3, sticky="ew" if compact_layout else "w")
    model_field_label = ttk.Label(settings, text="Model", style="Field.TLabel")
    model_field_label.grid(row=0, column=2, padx=(14, 5), pady=7, sticky="w")
    model_picker = ttk.Combobox(
        settings,
        textvariable=model_name,
        values=DEFAULT_MODELS,
        width=max(16, min(28, len(DEFAULT_MODEL) + 2)),
    )
    model_picker.grid(row=1 if compact_layout else 0, column=1 if compact_layout else 3,
                      padx=4, pady=3, sticky="ew" if compact_layout else "w")
    if compact_layout:
        model_field_label.grid_forget()
        model_field_label.grid(row=1, column=0, padx=(10, 5), pady=7, sticky="w")
        # Move the model label and server/model inputs to a two-row layout so
        # 125–150% scaling does not push the connection check off-screen.
    refresh_button = ttk.Button(
        settings,
        text="Check Connection",
        command=refresh_models,
        style="Secondary.TButton",
    )
    refresh_button.grid(row=1 if compact_layout else 0, column=2 if compact_layout else 4,
                        padx=(10, 8), pady=3, sticky="w")

    readiness_label = ttk.Label(
        settings,
        text=f"Records: {records_readiness}   |   Server: CHECKING   |   Model: CHECKING   |   Generation: NOT READY",
        style="Status.TLabel",
    )
    readiness_label.grid(row=2 if compact_layout else 1, column=0,
                         columnspan=3 if compact_layout else 5,
                         padx=10, pady=(1, 5), sticky="w")
    settings.pack_forget()

    patient_bar = ttk.LabelFrame(window, text="Patient Selection and Date Range", style="Settings.TLabelframe")
    patient_bar.pack(fill=tk.X, padx=14, pady=5)
    patient_name = tk.StringVar()
    ttk.Label(patient_bar, text="Patient ID", style="Field.TLabel").grid(row=0, column=0, padx=(10, 4), pady=5)
    patient_picker = ttk.Combobox(patient_bar, textvariable=patient_name,
                                  values=tuple(patients), state="readonly", width=30)
    patient_picker.grid(row=0, column=1, padx=6, pady=5, sticky="w")
    patient_picker.bind("<<ComboboxSelected>>", select_patient)
    active_label = ttk.Label(patient_bar, text="Active patient: none", style="Field.TLabel")
    if compact_layout:
        active_label.grid(row=2, column=0, columnspan=2, padx=(10, 5), pady=5, sticky="w")
    else:
        active_label.grid(row=0, column=4, padx=8, pady=5, sticky="w")
    demo_data = tk.BooleanVar(value=False)
    demo_toggle = ttk.Checkbutton(patient_bar, text="Demo data for this session", variable=demo_data,
                                  command=toggle_demo_data)
    if not streamlined_layout:
        demo_toggle.grid(
            row=1 if compact_layout else 0, column=2 if compact_layout else 4,
            columnspan=2 if compact_layout else 1, padx=8, pady=5, sticky="w"
        )
    demo_status = ttk.Label(patient_bar, text="Demo data: OFF (this session)", style="Status.TLabel")
    if not streamlined_layout:
        demo_status.grid(row=1 if compact_layout else 0, column=4 if compact_layout else 5,
                         padx=(0, 10), pady=5, sticky="w")
    counts_label = ttk.Label(patient_bar, text="X-ray 0/0 selected   |   EKG 0/0 selected   |   Blood 0/0 selected",
                             style="Status.TLabel")
    if not streamlined_layout:
        counts_label.grid(row=2 if compact_layout else 1, column=0, columnspan=6,
                          padx=10, pady=(1, 4), sticky="w")
    tray_label = ttk.Label(patient_bar, text="Selected-record tray", style="Field.TLabel")
    if not streamlined_layout:
        tray_label.grid(row=3 if compact_layout else 2, column=0, padx=10, pady=(3, 4), sticky="nw")
    tray_list = tk.Listbox(patient_bar, height=2, exportselection=False,
                           activestyle="underline", takefocus=True)
    if not streamlined_layout:
        tray_list.grid(row=3 if compact_layout else 2, column=1,
                       columnspan=4, padx=4, pady=(3, 5), sticky="ew")
    tray_list.bind("<Double-Button-1>", remove_tray_selection)
    remove_tray_button = ttk.Button(patient_bar, text="Remove selected", command=remove_tray_selection)
    if not streamlined_layout:
        remove_tray_button.grid(row=3 if compact_layout else 2, column=5, padx=(4, 10), pady=3, sticky="n")
    patient_bar.columnconfigure(3, weight=0)
    patient_bar.columnconfigure(4, weight=0)
    patient_bar.columnconfigure(1, weight=1)
    tray_items = []
    tray_list.bind("<Return>", remove_tray_selection)
    tray_list.bind("<Delete>", remove_tray_selection)

    date_filter_bar = ttk.Frame(patient_bar, style="App.TFrame")
    date_filter_bar.grid(row=0 if not compact_layout else 1, column=2 if not compact_layout else 0,
                         columnspan=3 if compact_layout else 2, padx=(8, 10), pady=5,
                         sticky="e" if not compact_layout else "w")
    date_from_var = tk.StringVar(value="")
    date_to_var = tk.StringVar(value="")
    ttk.Button(date_filter_bar, text="Choose date & time", command=choose_snapshot,
               style="Secondary.TButton").pack(side=tk.LEFT, padx=(4, 8), pady=5)
    snapshot_selection_label = ttk.Label(
        date_filter_bar, text="Choose a date and time to load data.", style="Status.TLabel"
    )
    snapshot_selection_label.pack(side=tk.LEFT, padx=(0, 4), pady=5)
    # Retained only for compatibility with old saved-session helper callbacks;
    # the visible workflow uses exactly one date/time picker.
    date_filter_status = snapshot_selection_label

    # Horizontal and vertical sash panes keep all core panels user-resizable.
    body = ttk.Panedwindow(window, orient=tk.HORIZONTAL)
    body.pack(fill=tk.BOTH, expand=True, padx=14, pady=(5, 14))
    data_panes = ttk.Panedwindow(body, orient=tk.VERTICAL)

    # One unified question-and-answer window.
    qa_panel = ttk.LabelFrame(
        body,
        text="AI Clinical Assistant Center",
        labelanchor="n",
        style="Assistant.TLabelframe",
    )
    body.add(qa_panel, weight=35)
    body.add(data_panes, weight=65)
    ttk.Label(qa_panel, text="Your Question", style="Field.TLabel").pack(anchor="w", padx=10, pady=(10, 5))
    preset_row = ttk.Frame(qa_panel, style="App.TFrame")
    preset_row.pack(fill=tk.X, padx=10, pady=(0, 4))
    ttk.Label(preset_row, text="Preset", style="Field.TLabel").pack(side=tk.LEFT, padx=(0, 6))
    preset_name = tk.StringVar(value="Draft report")
    preset_picker = ttk.Combobox(
        preset_row,
        textvariable=preset_name,
        values=tuple(REPORT_PRESETS),
        state="readonly",
        width=30,
    )
    preset_picker.pack(side=tk.LEFT)
    preset_picker.bind("<<ComboboxSelected>>", apply_report_preset)
    ttk.Label(
        preset_row,
        text="Choosing a preset replaces this text; edit it before sending.",
        style="Status.TLabel",
        wraplength=min(260, qa_wraplength),
    ).pack(side=tk.LEFT, padx=7)
    if streamlined_layout:
        preset_row.pack_forget()
    context_label = ttk.Label(
        qa_panel,
        text="Inputs to send: no active patient.",
        style="Status.TLabel",
        wraplength=qa_wraplength,
        justify=tk.LEFT,
    )
    # Hide verbose context from Question panel; counts remain in medical dataset panels.
    # The label object stays available to existing status-update routines.
    prompt_box = scrolledtext.ScrolledText(
        qa_panel,
        width=18,
        height=6,
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
    if compact_layout:
        send_button.grid(row=0, column=0, padx=3, pady=2, sticky="ew")
    else:
        send_button.pack(side=tk.LEFT, padx=(0, 8))
    cancel_button = ttk.Button(
        action_bar, text="Cancel", command=cancel_request, state=tk.DISABLED
    )
    if compact_layout:
        cancel_button.grid(row=0, column=1, padx=3, pady=2, sticky="ew")
    else:
        cancel_button.pack(side=tk.LEFT, padx=(0, 8))
    save_quick_button = ttk.Button(
        action_bar, text="Save report", command=export_current_report, state=tk.DISABLED
    )
    if compact_layout:
        save_quick_button.grid(row=1, column=0, padx=3, pady=2, sticky="ew")
    else:
        save_quick_button.pack(side=tk.LEFT, padx=(0, 8))
    retry_button = ttk.Button(
        action_bar, text="Retry", command=retry_request, state=tk.DISABLED
    )
    if compact_layout:
        retry_button.grid(row=0, column=2, padx=3, pady=2, sticky="ew")
    else:
        retry_button.pack(side=tk.LEFT, padx=(0, 8))
    voice_toggle_button = ttk.Button(
        action_bar,
        text="Voice: OFF",
        command=toggle_voice,
        style="VoiceOff.TButton",
    )
    if compact_layout:
        voice_toggle_button.grid(row=1, column=1, columnspan=2, padx=3, pady=2, sticky="ew")
        for column in range(3):
            action_bar.columnconfigure(column, weight=1, uniform="action_buttons")
    else:
        voice_toggle_button.pack(side=tk.LEFT)
    close_gui_button = ttk.Button(action_bar, text="Close GUI", command=close_window)
    if compact_layout:
        close_gui_button.grid(row=2, column=0, columnspan=3, padx=3, pady=(5, 2), sticky="ew")
    else:
        close_gui_button.pack(side=tk.LEFT, padx=(8, 0))
    status_label = ttk.Label(
        qa_panel,
        text="Ready. Select a server and model.",
        wraplength=min(300, qa_wraplength),
        style="Status.TLabel",
    )
    status_label.pack(anchor="w")
    elapsed_label = ttk.Label(qa_panel, text="", style="Status.TLabel")
    elapsed_label.pack(anchor="w")
    diagnostics_label = ttk.Label(
        qa_panel,
        text=f"Diagnostics: {runtime_diagnostics()}",
        wraplength=qa_wraplength,
        style="Status.TLabel",
    )
    if not streamlined_layout:
        diagnostics_label.pack(anchor="w", padx=2, pady=(2, 0))
    actions_placeholder = None
    # Avoid cluttering the question and response layout with a redundant banner.
    report_review_heading = ttk.Label(qa_panel, text="Report review", style="Field.TLabel")
    report_review_heading.pack(anchor="w", padx=10, pady=(8, 3))
    report_meta_label = ttk.Label(
        qa_panel, text="No completed report.", style="Status.TLabel", wraplength=qa_wraplength
    )
    report_meta_label.pack(anchor="w", padx=10, pady=(0, 3))
    report_provenance_label = ttk.Label(
        qa_panel,
        text="Provenance appears here for a completed request.",
        style="Status.TLabel",
        wraplength=qa_wraplength,
        justify=tk.LEFT,
    )
    report_provenance_label.pack(anchor="w", padx=10, pady=(0, 4))
    review_controls = ttk.Frame(qa_panel, style="App.TFrame")
    review_controls.pack(fill=tk.X, padx=10, pady=(0, 3))
    report_review_var = tk.BooleanVar(value=False)
    review_button = ttk.Checkbutton(
        review_controls,
        text="I reviewed the displayed report draft",
        variable=report_review_var,
        command=acknowledge_report_review,
        state=tk.DISABLED,
    )
    review_button.pack(side=tk.LEFT)
    export_report_button = ttk.Button(
        review_controls,
        text="Export TXT + HTML + JSON",
        command=export_current_report,
        state=tk.DISABLED,
    )
    export_report_button.pack(side=tk.LEFT, padx=(8, 0))
    restore_original_button = ttk.Button(
        review_controls,
        text="Restore original AI draft",
        command=restore_original_draft,
        state=tk.DISABLED,
    )
    restore_original_button.pack(side=tk.LEFT, padx=(8, 0))
    history_frame = ttk.LabelFrame(qa_panel, text="Saved report history", style="Settings.TLabelframe")
    history_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
    history_toolbar = ttk.Frame(history_frame, style="App.TFrame")
    history_toolbar.pack(fill=tk.X, padx=5, pady=(4, 2))
    ttk.Button(history_toolbar, text="Choose history folder…", command=choose_saved_report_history).pack(side=tk.LEFT)
    ttk.Button(history_toolbar, text="Refresh", command=refresh_saved_report_history).pack(side=tk.LEFT, padx=(6, 0))
    ttk.Button(history_toolbar, text="Reopen selected", command=reopen_saved_report).pack(side=tk.LEFT, padx=(6, 0))
    history_folder_label = ttk.Label(history_frame, text="No history folder selected.", style="Status.TLabel")
    history_folder_label.pack(anchor="w", padx=6, pady=(0, 2))
    saved_report_list = tk.Listbox(history_frame, height=3, exportselection=False)
    saved_report_list.pack(fill=tk.X, padx=5, pady=(0, 2))
    saved_report_list.bind("<Double-Button-1>", reopen_saved_report)
    history_status_label = ttk.Label(
        history_frame, text="Choose a folder containing exported JSON bundles.",
        style="Status.TLabel", wraplength=qa_wraplength,
    )
    history_status_label.pack(anchor="w", padx=6, pady=(0, 4))
    report_version_label = ttk.Label(
        qa_panel, text="Report view: no completed draft", style="Status.TLabel"
    )
    report_version_label.pack(anchor="w", padx=10, pady=(0, 2))
    ttk.Label(qa_panel, text="AI Response", style="Field.TLabel").pack(anchor="w", padx=10, pady=(8, 3))
    # Retain the notebook internally for report-safety callbacks, but hide
    # its tabs so clinicians see one response area, not draft variants.
    ui_style.layout("Response.TNotebook.Tab", [])
    report_notebook = ttk.Notebook(qa_panel, style="Response.TNotebook")
    report_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    original_box = scrolledtext.ScrolledText(
        report_notebook,
        width=18,
        wrap=tk.WORD,
        font=("Segoe UI", 11),
        background="#f8fcff",
        foreground=TEXT_COLOR,
        relief=tk.FLAT,
        borderwidth=6,
        state=tk.DISABLED,
    )
    answer_box = scrolledtext.ScrolledText(
        report_notebook,
        width=18,
        wrap=tk.WORD,
        font=("Segoe UI", 11),
        background="#fbfdff",
        foreground=TEXT_COLOR,
        insertbackground=PRIMARY_BLUE,
        selectbackground=SOFT_BLUE,
        relief=tk.FLAT,
        borderwidth=6,
    )
    for widget in (original_box, answer_box):
        widget.tag_configure(
            "report_medical_heading",
            foreground="#d32f2f",
            font=("Segoe UI", 11, "bold"),
        )
    report_notebook.add(original_box, text="Original AI draft")
    report_notebook.add(answer_box, text="Response")
    report_notebook.tab(0, state="disabled")
    report_notebook.tab(1, state="disabled")
    # The original draft remains available to internal report-export logic,
    # while clinicians see one clear response surface rather than editor tabs.
    report_notebook.hide(0)
    report_notebook.bind("<<NotebookTabChanged>>", update_report_version_label)
    answer_box.bind("<<Modified>>", on_report_modified)
    if streamlined_layout:
        # Hide non-requested report-management controls while retaining the
        # answer editor and its existing callback wiring.
        for widget in (
            cancel_button, save_quick_button, retry_button, report_meta_label,
            report_review_heading, report_provenance_label, review_controls, history_frame,
            report_version_label,
        ):
            widget.pack_forget()
            widget.grid_remove()

    image_panel = ttk.Frame(body, style="App.TFrame")
    data_panes.add(image_panel, weight=70)
    image_panel.rowconfigure(0, weight=1, uniform="image_rows")
    image_panel.rowconfigure(2, weight=1, uniform="image_rows")
    image_panel.columnconfigure(0, weight=1)
    selected_images = [{}, {}]  # Blue first, then orange; each keeps click order.
    sample_button = tk.Button(image_panel)
    default_button_color = sample_button.cget("background")
    sample_button.destroy()

    galleries = []
    gallery_count_labels = []
    for gallery_index in range(2):
        heading = tk.Frame(image_panel, background=APP_BACKGROUND)
        heading_name = tk.Label(heading, text=GALLERY_NAMES[gallery_index], background=APP_BACKGROUND,
                                foreground=TEAL, font=("Segoe UI", 12, "bold"))
        heading_date = tk.Label(heading, text="", background=APP_BACKGROUND,
                                foreground=TEAL, font=("Segoe UI", 12, "bold"))
        heading_time = tk.Label(heading, text="", background=APP_BACKGROUND,
                                foreground="#6a1b9a", font=("Segoe UI", 12, "bold"))
        heading_name.pack(side=tk.LEFT)
        heading_date.pack(side=tk.LEFT)
        heading_time.pack(side=tk.LEFT)
        pane = ttk.LabelFrame(
            image_panel,
            labelwidget=heading,
            style="Gallery.TLabelframe",
        )
        pane.grid(
            row=gallery_index * 2,
            column=0,
            sticky="nsew",
            pady=(0, 3) if gallery_index == 0 else (3, 0),
        )
        pane.columnconfigure(0, weight=1)
        pane.rowconfigure(1, weight=1)
        pane.rowconfigure(2, weight=0)
        control_row = ttk.Frame(pane, style="App.TFrame")
        control_row.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(3, 0))
        if not streamlined_layout:
            ttk.Button(control_row, text="Select all", command=lambda m=gallery_index: set_modality_selection(m, True)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(control_row, text="Clear", command=lambda m=gallery_index: set_modality_selection(m, False)).pack(side=tk.LEFT)
        modality_count = ttk.Label(control_row, text="0/0 selected", style="Status.TLabel")
        if not streamlined_layout:
            modality_count.pack(side=tk.LEFT, padx=10)
        gallery_count_labels.append(modality_count)
        canvas = tk.Canvas(
            pane,
            width=gallery_viewport_width,
            height=220,
            background=TILE_BACKGROUND,
            highlightthickness=0,
        )
        canvas.grid(row=1, column=0, sticky="nsew")
        vscroll = ttk.Scrollbar(pane, orient=tk.VERTICAL, command=canvas.yview)
        vscroll.grid(row=1, column=1, sticky="ns")
        hscroll = ttk.Scrollbar(pane, orient=tk.HORIZONTAL, command=canvas.xview)
        hscroll.grid(row=2, column=0, sticky="ew")
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
                "heading_name": heading_name,
                "heading_date": heading_date,
                "heading_time": heading_time,
                "canvas": canvas,
                "canvas_window": canvas_window,
                "grid": grid,
            }
        )

    # Keep the selectable Blood Test Data list below the image galleries.
    blood_heading = tk.Frame(body, background=APP_BACKGROUND)
    blood_heading_name = tk.Label(blood_heading, text="BLOOD TEST", background=APP_BACKGROUND,
                                  foreground=TEAL, font=("Segoe UI", 12, "bold"))
    blood_heading_date = tk.Label(blood_heading, text="", background=APP_BACKGROUND,
                                  foreground=TEAL, font=("Segoe UI", 12, "bold"))
    blood_heading_time = tk.Label(blood_heading, text="", background=APP_BACKGROUND,
                                  foreground="#6a1b9a", font=("Segoe UI", 12, "bold"))
    blood_heading_name.pack(side=tk.LEFT)
    blood_heading_date.pack(side=tk.LEFT)
    blood_heading_time.pack(side=tk.LEFT)
    blood_test_panel = ttk.LabelFrame(
        body, labelwidget=blood_heading, style="Gallery.TLabelframe",
    )
    blood_test_panel.heading_name = blood_heading_name
    blood_test_panel.heading_date = blood_heading_date
    blood_test_panel.heading_time = blood_heading_time
    data_panes.add(blood_test_panel, weight=30)
    blood_test_panel.rowconfigure(1, weight=1)
    blood_test_panel.columnconfigure(0, weight=1)
    blood_controls = ttk.Frame(blood_test_panel, style="App.TFrame")
    blood_controls.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(4, 0))
    if not streamlined_layout:
        ttk.Button(blood_controls, text="Select all", command=lambda: set_modality_selection(2, True)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(blood_controls, text="Clear", command=lambda: set_modality_selection(2, False)).pack(side=tk.LEFT)
        ttk.Button(blood_controls, text="Details", command=show_blood_details).pack(side=tk.LEFT, padx=(5, 0))
    blood_count_label = ttk.Label(blood_controls, text="0/0 selected", style="Status.TLabel")
    if not streamlined_layout:
        blood_count_label.pack(side=tk.LEFT, padx=10)
    blood_test_records = []
    blood_test_paths = []
    blood_record_header_lines = []
    blood_line_record_indices = []
    blood_focus_record_index = None
    selected_blood_test_indices = set()
    blood_test_view = tk.Text(
        blood_test_panel,
        wrap=tk.NONE,
        font=("Consolas", 10),
        cursor="hand2",
        state=tk.NORMAL,
        background="#fffdfb",
        foreground=TEXT_COLOR,
        selectbackground=SOFT_CORAL,
        selectforeground=TEXT_COLOR,
        relief=tk.FLAT,
        borderwidth=5,
        takefocus=True,
    )
    blood_test_view.tag_configure("blood_title", foreground=CORAL, font=("Consolas", 10, "bold"))
    blood_test_view.tag_configure("blood_columns", foreground=MUTED_TEXT, font=("Consolas", 10, "bold"))
    blood_test_view.bind("<Button-1>", toggle_blood_test_record)
    blood_test_view.bind("<KeyPress>", blood_test_keypress)
    blood_test_view.bind("<KeyRelease>", blood_test_keyup)
    blood_test_view.grid(row=1, column=0, sticky="nsew", padx=(5, 0), pady=(5, 0))
    blood_test_scroll = ttk.Scrollbar(
        blood_test_panel,
        orient=tk.VERTICAL,
        command=blood_test_view.yview,
    )
    blood_test_scroll.grid(row=1, column=1, sticky="ns", padx=(0, 5), pady=(5, 0))
    blood_test_hscroll = ttk.Scrollbar(
        blood_test_panel,
        orient=tk.HORIZONTAL,
        command=blood_test_view.xview,
    )
    blood_test_hscroll.grid(row=2, column=0, sticky="ew", padx=(5, 0), pady=(0, 5))
    blood_test_view.configure(
        yscrollcommand=blood_test_scroll.set,
        xscrollcommand=blood_test_hscroll.set,
    )

    # Explicit initial sash positions prevent the large blood-test text widget
    # from squeezing the X-ray and EKG previews into a narrow horizontal strip.
    # Users can still drag both sashes after startup.
    def establish_workspace_proportions():
        try:
            window.update_idletasks()
            width = body.winfo_width()
            height = data_panes.winfo_height()
            if width > 500:
                body.sashpos(0, int(width * 0.35))
            if height > 250:
                data_panes.sashpos(0, int(height * 0.70))
        except tk.TclError:
            pass

    window.after(180, establish_workspace_proportions)

    # Select and load the first available patient as soon as the UI is ready.
    if patients:
        patient_picker.current(0)

    # Load records and check the server after the complete window is assembled.
    window.after(50, drain_ui)
    window.after(100, select_patient)
    window.after(250, refresh_models)
    # The startup check reads /api/tags only; it never generates or sends records.

    window.mainloop()


if __name__ == "__main__":
    main()
