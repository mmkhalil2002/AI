"""Desktop record selector and MedGemma/Ollama report workspace."""
from __future__ import annotations

import json
import os
import threading
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import messagebox, ttk

from mcp_client import MCPRecords

PYTHON_DIR = Path(__file__).resolve().parent


def load_env() -> dict[str, str]:
    """Read local .env values without replacing explicit OS environment values."""
    values = {}
    env_file = PYTHON_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = os.environ.get(key.strip(), value.strip())
    return values


def patient_records_root(settings: dict[str, str]) -> Path:
    """Resolve and validate the GUI-relative patient-record directory."""
    raw = os.environ.get("PATIENT_RECORDS_DIR", settings.get("PATIENT_RECORDS_DIR", "patient_records"))
    root = Path(raw)
    root = (PYTHON_DIR / root).resolve() if not root.is_absolute() else root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Patient records directory does not exist: {root}")
    return root


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.settings = load_env(); self.root_dir = patient_records_root(self.settings)
        self.records: MCPRecords | None = None
        self.title("MedGemma patient records"); self.geometry(self.settings.get("WINDOW_WIDTH", "1400") + "x" + self.settings.get("WINDOW_HEIGHT", "850"))
        self.protocol("WM_DELETE_WINDOW", self.close_window)
        self.patient = tk.StringVar(); self.status = tk.StringVar(value="Choose a patient and records.")
        top = ttk.Frame(self, padding=10); top.pack(fill="x")
        ttk.Label(top, text="Patient ID").pack(side="left")
        self.patient_box = ttk.Combobox(top, textvariable=self.patient, state="readonly", values=self.patients())
        self.patient_box.pack(side="left", padx=6); self.patient_box.bind("<<ComboboxSelected>>", lambda _: self.load_records())
        ttk.Button(top, text="Refresh", command=self.load_records).pack(side="left")
        self.tree = ttk.Treeview(self, columns=("path",), show="tree", selectmode="extended")
        self.tree.pack(fill="both", expand=True, padx=10, pady=5)
        ttk.Label(self, text="Doctor's request").pack(anchor="w", padx=10)
        self.prompt = tk.Text(self, height=5); self.prompt.pack(fill="x", padx=10)
        controls = ttk.Frame(self, padding=10); controls.pack(fill="x")
        ttk.Button(controls, text="Send to MedGemma", command=self.send).pack(side="left")
        ttk.Button(controls, text="Close", command=self.close_window).pack(side="right")
        ttk.Label(controls, textvariable=self.status).pack(side="left", padx=12)
        self.output = tk.Text(self, height=18, wrap="word"); self.output.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def patients(self) -> list[str]:
        return sorted(p.name for p in self.root_dir.iterdir() if p.is_dir() and not p.name.startswith("."))

    def load_records(self):
        self.tree.delete(*self.tree.get_children())
        patient = self.patient.get()
        if not patient: return
        base = self.root_dir / patient
        for modality in ("xrays", "ekgs", "blood_tests"):
            folders = list(base.rglob(modality))
            for folder in folders:
                parent = self.tree.insert("", "end", text=folder.relative_to(self.root_dir).as_posix(), open=True)
                suffixes = {".json"} if modality == "blood_tests" else {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
                for item in sorted(folder.iterdir()):
                    if item.is_file() and item.suffix.lower() in suffixes:
                        self.tree.insert(parent, "end", text=item.name, values=(item.relative_to(self.root_dir).as_posix(),))

    def record_client(self) -> MCPRecords:
        if self.records is None: self.records = MCPRecords(self.root_dir)
        return self.records

    def send(self):
        paths = [self.tree.item(i, "values")[0] for i in self.tree.selection() if self.tree.item(i, "values")]
        request = self.prompt.get("1.0", "end").strip()
        if not paths or not request:
            messagebox.showwarning("Missing selection", "Select at least one record and enter a request."); return
        self.status.set("Retrieving records and contacting Ollama…")
        threading.Thread(target=self._send_worker, args=(paths, request), daemon=True).start()

    def _send_worker(self, paths, request):
        try:
            blood = [self.record_client().read_blood_record(p) for p in paths if "/blood_tests/" in f"/{p}"]
            images = [self.record_client().read_record_bytes(p) for p in paths if "/blood_tests/" not in f"/{p}"]
            payload = {"model": self.settings.get("MEDGEMMA_MODEL", "medgemma1.5:latest"), "prompt": request + "\nBlood records: " + json.dumps(blood), "images": [__import__("base64").b64encode(i).decode() for i in images], "stream": False,
                       "options": {"num_predict": int(self.settings.get("NUM_PREDICT", 4048)), "temperature": float(self.settings.get("TEMPERATURE", .2))}}
            url = self.settings.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/") + "/api/generate"
            response = urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}), timeout=int(self.settings.get("GENERATION_TIMEOUT", 1200)))
            report = json.loads(response.read())["response"]
            self.after(0, lambda: (self.output.delete("1.0", "end"), self.output.insert("end", report), self.status.set("Report received; clinician review required.")))
        except Exception as exc:
            message = f"Request failed: {exc}"
            self.after(0, lambda: self.status.set(message))

    def close_window(self):
        if self.records: self.records.close()
        self.destroy()


if __name__ == "__main__":
    try: App().mainloop()
    except Exception as error: messagebox.showerror("MedGemma", str(error))
