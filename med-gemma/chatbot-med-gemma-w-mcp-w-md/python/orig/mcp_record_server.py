"""Shared, read-only MCP tools for one patient-record modality."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

MAX_BYTES = 64 * 1024 * 1024
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def make_server(modality: str) -> FastMCP:
    """Create a server constrained to paths containing exactly *modality*."""
    root = Path(os.environ["PATIENT_RECORDS_DIR"]).resolve()
    mcp = FastMCP(f"medgemma-{modality}")

    def checked(relative: str) -> Path:
        candidate = (root / relative).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("Path is outside the patient records directory") from exc
        if any(part.startswith(".") for part in Path(relative).parts):
            raise ValueError("Hidden paths are not available")
        if modality not in candidate.relative_to(root).parts:
            raise ValueError(f"Path does not belong to {modality}")
        if candidate.is_symlink():
            raise ValueError("Linked records are not available")
        return candidate

    @mcp.tool()
    def list_records(relative_folder: str) -> str:
        """List supported records below one folder, returning root-relative paths."""
        folder = checked(relative_folder)
        if not folder.is_dir():
            raise ValueError("Record folder does not exist")
        suffixes = {".json"} if modality == "blood_tests" else IMAGE_EXTENSIONS
        paths = [str(p.relative_to(root)).replace("\\", "/") for p in folder.rglob("*")
                 if p.is_file() and not p.is_symlink() and p.suffix.lower() in suffixes]
        return json.dumps({"paths": sorted(paths)})

    @mcp.tool()
    def read_record(relative_path: str) -> str:
        """Read an image as Base64 or a blood record as parsed JSON."""
        record = checked(relative_path)
        suffixes = {".json"} if modality == "blood_tests" else IMAGE_EXTENSIONS
        if not record.is_file() or record.suffix.lower() not in suffixes:
            raise ValueError("Unsupported record type")
        if record.stat().st_size > MAX_BYTES:
            raise ValueError("Record exceeds the 64 MiB limit")
        if modality == "blood_tests":
            with record.open("r", encoding="utf-8") as handle:
                return json.dumps({"record": json.load(handle)})
        return json.dumps({"base64": base64.b64encode(record.read_bytes()).decode("ascii")})

    return mcp
