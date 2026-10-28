"""Synchronous facade over the three local stdio MCP record servers."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import threading
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPRecords:
    """Start the read-only servers once and serialize record-tool requests.

    Example: ``MCPRecords(root).read_record_bytes('P0001/xrays/chest.png')``.
    """
    def __init__(self, records_root: Path):
        self.root = Path(records_root).resolve()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.sessions: dict[str, ClientSession] = {}
        self.stack: AsyncExitStack | None = None
        self.lock = threading.Lock()
        self._call(self._start())

    def _call(self, awaitable):
        return asyncio.run_coroutine_threadsafe(awaitable, self.loop).result(timeout=60)

    async def _start(self):
        self.stack = AsyncExitStack()
        scripts = {"xrays": "mcp_xray_server.py", "ekgs": "mcp_ekg_server.py", "blood_tests": "mcp_blood_server.py"}
        for modality, script in scripts.items():
            params = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).with_name(script))],
                                          env={**os.environ, "PATIENT_RECORDS_DIR": str(self.root)})
            read, write = await self.stack.enter_async_context(stdio_client(params))
            session = await self.stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[modality] = session

    @staticmethod
    def record_location(path: str | Path) -> tuple[str, str]:
        """Return the modality and normalized root-relative record path."""
        parts = Path(path).as_posix().split("/")
        for modality in ("xrays", "ekgs", "blood_tests"):
            if modality in parts:
                return modality, "/".join(parts)
        raise ValueError("Record path has no recognized modality")

    async def _tool(self, modality: str, name: str, arguments: dict) -> dict:
        result = await self.sessions[modality].call_tool(name, arguments=arguments)
        text = "".join(block.text for block in result.content if hasattr(block, "text"))
        return json.loads(text)

    def list_records(self, relative_folder: str) -> list[str]:
        modality, _ = self.record_location(relative_folder)
        with self.lock:
            return self._call(self._tool(modality, "list_records", {"relative_folder": relative_folder}))["paths"]

    def read_record_bytes(self, relative_path: str) -> bytes:
        modality, _ = self.record_location(relative_path)
        if modality == "blood_tests":
            raise ValueError("Use read_blood_record for blood-test JSON")
        with self.lock:
            payload = self._call(self._tool(modality, "read_record", {"relative_path": relative_path}))
        return base64.b64decode(payload["base64"], validate=True)

    def read_blood_record(self, relative_path: str):
        modality, _ = self.record_location(relative_path)
        if modality != "blood_tests":
            raise ValueError("Expected a blood_tests record")
        with self.lock:
            return self._call(self._tool(modality, "read_record", {"relative_path": relative_path}))["record"]

    def close(self):
        if self.stack:
            try:
                self._call(self.stack.aclose())
            finally:
                self.stack = None
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2)
