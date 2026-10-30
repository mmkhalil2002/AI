"""MCP transport and record access, independent of the GUI and Ollama.

Create MCPRecords(records_root, server_directory, server_ids); call read_record_bytes,
read_blood_record, or call. Call close() when the GUI exits.
"""
# =============================================================================
# HOW THE GUI USES THIS MCP CLIENT
# =============================================================================
#
# The GUI imports this class:
#
#     from mcp_client import MCPRecords
#
# The GUI's record_client() function creates one MCPRecords instance and reuses
# it. record_client() is defined in the GUI file, not in this module.
#
# Creating MCPRecords starts three local MCP servers in a background thread:
#
#     Folder name       Server script
#     ------------      --------------------
#     xrays             mcp_xray_server.py
#     ekgs              mcp_ekg_server.py
#     blood_tests       mcp_blood_server.py
#
# Each server connection is stored in self.sessions under its folder name.
# Requests travel through subprocess stdin/stdout; no TCP ports are needed.
#
# =============================================================================
# EXAMPLE 1: READ AN X-RAY
# =============================================================================
#
# GUI code:
#
#     xray_path = Path(
#         "C:/AI/med-gemma/patient_records/P0001/xrays/chest.png"
#     )
#     xray_bytes = record_client().read_record_bytes(xray_path)
#
# Inside this MCP client:
#
#     record_location(xray_path) identifies:
#         modality = "xrays"
#         relative = "P0001/xrays/chest.png"
#
#     read_record_bytes() calls:
#
#         self.call(
#             "xrays",
#             "read_record",
#             {"relative_path": "P0001/xrays/chest.png"}
#         )
#
#     _call() selects self.sessions["xrays"] and invokes its read_record tool.
#     The X-ray server reads the file and returns Base64-encoded image data.
#     This client decodes the Base64 and returns the original image bytes.
#
# =============================================================================
# EXAMPLE 2: READ AN EKG
# =============================================================================
#
# GUI code:
#
#     ekg_path = Path(
#         "C:/AI/med-gemma/patient_records/P0001/ekgs/ekg.png"
#     )
#     ekg_bytes = record_client().read_record_bytes(ekg_path)
#
# Inside this MCP client:
#
#     record_location(ekg_path) identifies:
#         modality = "ekgs"
#         relative = "P0001/ekgs/ekg.png"
#
#     read_record_bytes() calls:
#
#         self.call(
#             "ekgs",
#             "read_record",
#             {"relative_path": "P0001/ekgs/ekg.png"}
#         )
#
#     _call() selects self.sessions["ekgs"] and invokes its read_record tool.
#     The EKG server reads the file and returns Base64-encoded image data.
#     This client decodes the Base64 and returns the original image bytes.
#
# =============================================================================
# EXAMPLE 3: READ A BLOOD TEST
# =============================================================================
#
# GUI code:
#
#     blood_path = Path(
#         "C:/AI/med-gemma/patient_records/P0001/blood_tests/blood.json"
#     )
#     blood_data = record_client().read_blood_record(blood_path)
#
# Inside this MCP client:
#
#     record_location(blood_path) identifies:
#         modality = "blood_tests"
#         relative = "P0001/blood_tests/blood.json"
#
#     read_blood_record() calls:
#
#         self.call(
#             "blood_tests",
#             "read_record",
#             {"relative_path": "P0001/blood_tests/blood.json"}
#         )
#
#     _call() selects self.sessions["blood_tests"] and calls read_record.
#     The blood server reads and parses the JSON file.
#     This client returns the parsed value under the response's "record" key,
#     normally a Python dictionary or list. No Base64 decoding is needed.
#
# =============================================================================
# ROUTING, DISPLAY, AND SHUTDOWN
# =============================================================================
#
# Routing depends on the folder name, not the filename or image contents.
# All three servers expose a tool named read_record; the modality selects
# which server receives that tool request.
#
# The examples assume records_root is C:/AI/med-gemma/patient_records.
# Replace the example paths with your actual patient record locations.
#
# The servers read records; this client transfers them; the GUI displays them.
# The GUI's orchestration code separately sends selected data to MedGemma.
# This module does not call MedGemma or Qwen.
#
# call() waits for a response, so it blocks whichever thread calls it.
# Calling it on the GUI thread can temporarily pause the interface.
#
# When the GUI exits, it calls the existing MCP client's close() method to
# request shutdown of the connections and server processes.
# =============================================================================
# =============================================================================
# WHAT THE SERVER'S read_record() TOOL DOES
# =============================================================================
#
# read_record() reads ONE specific patient file identified by relative_path.
# It is defined on each MCP server, not in this MCP client module.
#
# Example request:
#
#     self.call(
#         "xrays",
#         "read_record",
#         {"relative_path": "P0001/xrays/chest.png"}
#     )
#
# The X-ray server locates this file beneath its configured records root:
#
#     patient_records/P0001/xrays/chest.png
#
# Before reading, the server checks:
#   - The path stays within the patient directory and correct modality.
#   - The path does not use prohibited symbolic links or junctions.
#   - The file extension is supported.
#   - The file size does not exceed 64 MiB.
#
# The response depends on the server:
#
#   X-ray server:
#       Reads the requested image and returns Base64-encoded image bytes.
#
#   EKG server:
#       Reads the requested image and returns Base64-encoded image bytes.
#
#   Blood-test server:
#       Reads the requested JSON file and returns its parsed contents
#       under the "record" key.
#
# Records are currently stored in folders and files, not in a database.
# read_record() retrieves data only; it does not perform medical analysis.
#
# Use list_records to discover available files.
# Use read_record to retrieve the contents of one specific file.
# =============================================================================
import asyncio
import base64
import json
import os
import sys
import threading
from pathlib import Path
from contextlib import AsyncExitStack
from concurrent.futures import TimeoutError as FutureTimeout

# Public interface used by the GUI. One instance manages all three servers.
# This module contains no Tkinter widgets, Ollama requests, or model reasoning.
# Example GUI usage (keep this file named mcp_client.py beside the GUI):
#   from mcp_client import MCPRecords
#   records = MCPRecords(patient_records_directory, python_files_directory)
#   image_bytes = records.read_record_bytes(xray_path)
#   blood_data = records.read_blood_record(blood_json_path)
#   records.close()
# Merely importing this module does not launch servers; constructing the class does.
class MCPRecords:
    # records_root: directory containing patient folders, e.g. patient_records/.
    # server_directory: directory containing the three server Python scripts.
    # The GUI supplies these paths; this module does not read .env itself.
    # Relative constructor paths resolve against the process working directory.
    # This constructor launches a daemon thread and returns without waiting for
    # initialization. The first call() waits for the ready event.
    def __init__(self, records_root, server_directory=None, server_ids=None):
        # Store an absolute normalized root used for path calculations and passed
        # to every server. resolve() alone does not verify that the root exists.
        self.records_root = Path(records_root).resolve()
        # With no explicit directory, use the directory containing this module.
        # An explicit server_directory must be a directory, not a script filename.
        self.server_directory = Path(server_directory or __file__).resolve()
        if server_directory is None:
            self.server_directory = self.server_directory.parent
        # Keep record-folder names separate from MCP connection identifiers.
        # The GUI loads these values from .env, but defaults retain compatibility
        # for callers that construct this client directly.
        defaults = {
            'xrays': 'xrays',
            'ekgs': 'ekgs',
            'blood_tests': 'blood_tests',
        }
        if server_ids is not None and set(server_ids) - set(defaults):
            raise ValueError('MCP server IDs may only be set for known modalities.')
        supplied_ids = server_ids or {}
        self.server_ids = {}
        for modality, default in defaults.items():
            value = supplied_ids.get(modality) or default
            if not isinstance(value, str):
                raise ValueError('MCP server IDs must be strings.')
            # Direct callers get the same blank-means-default behavior as the
            # GUI's .env configuration.
            self.server_ids[modality] = value.strip() or default
        if len(set(self.server_ids.values())) != len(self.server_ids):
            raise ValueError('MCP server IDs must be non-empty and unique for each modality.')
        # Shared lifecycle state:
        # loop: asyncio event loop owned exclusively by the background thread.
        # ready: threading event notifying callers of success OR startup failure.
        # error: background exception, if any; callers check it after ready.
        # sessions: modality name -> initialized MCP ClientSession.
        # stop_event: asyncio event requesting orderly shutdown.
        self.loop = None
        self.ready = threading.Event()
        self.error = None
        self.sessions = {}
        self.stop_event = None
        # A daemon thread does not keep Python alive after the GUI exits.
        # Call close() during normal shutdown so async contexts can clean up.
        # A daemon thread alone does not guarantee cleanup on abrupt process exit.
        threading.Thread(target=self._thread, daemon=True).start()

    # Thread entry point: create, run, and eventually close one asyncio loop.
    # It hosts MCP communication independently of Tkinter's event loop.
    def _thread(self):
        try:
            asyncio.run(self._serve())
        # Capture failures even during server initialization or context cleanup.
        # Waking ready prevents a waiting caller from unnecessarily waiting the
        # full startup timeout. The generic message in call() calls this a startup
        # failure even if the background exception occurred later.
        except BaseException as error:
            self.error = error
            # Signal success only once ALL three sessions are initialized.
            # Keep their contexts open until close() sets stop_event.
            self.ready.set()

    # Own the subprocess transports and sessions for their entire lifetime.
    # The mcp package must provide the v1 ClientSession/stdio APIs (mcp<2).
    # Imports occur here, so a missing package is captured by _thread().
    async def _serve(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        self.loop = asyncio.get_running_loop()
        self.stop_event = asyncio.Event()
        # Serialize tool requests across ALL modalities. This avoids overlapping
        # calls but also means one slow server delays requests to other servers.
        self.lock = asyncio.Lock()
        # Register each async context as it opens. Leaving this block unwinds
        # sessions/transports in reverse order, including after partial startup.
        # The SDK's transport context manages its subprocess cleanup.
        async with AsyncExitStack() as stack:
            # Start and initialize the three servers sequentially. Exact modality
            # strings are also used by the GUI and by record_location().
            for modality, script in [('xrays', 'mcp_xray_server.py'),
                                     ('ekgs', 'mcp_ekg_server.py'),
                                     ('blood_tests', 'mcp_blood_server.py')]:
                # Inherit the current environment, overriding only the records
                # root so every child server uses the same absolute directory.
                env = dict(
                    os.environ,
                    PATIENT_RECORDS_DIR=str(self.records_root),
                    MCP_SERVER_ID=self.server_ids[modality],
                )
                # Use the SAME Python interpreter as the GUI. Communication is
                # through subprocess stdin/stdout; no listening TCP port is used.
                # Absolute script paths allow launches from another directory.
                params = StdioServerParameters(command=sys.executable,
                    args=[str(self.server_directory / script)], env=env)
                # Spawn the server and obtain its MCP input/output streams.
                # Servers must reserve stdout for MCP protocol messages.
                read, write = await stack.enter_async_context(stdio_client(params))
                # Wrap those streams in the SDK session, then perform the MCP
                # initialization handshake before allowing tool calls.
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self.sessions[self.server_ids[modality]] = session
            # Signal success only once ALL three sessions are initialized.
            # Keep their contexts open until close() sets stop_event.
            self.ready.set()
            await self.stop_event.wait()

    # Internal async operation; run on the background loop, never directly
    # called by the GUI. modality selects a session; tool and arguments are
    # forwarded to that server. An unknown modality raises KeyError.
    async def _call(self, modality, tool, arguments):
        async with self.lock:
            result = await self.sessions[self.server_ids[modality]].call_tool(tool, arguments)
            # These project servers return JSON in text content blocks. Ignore
            # other MCP content types; this is not a generic result decoder for
            # arbitrary third-party servers or structured-only responses.
            texts = [item.text for item in result.content if item.type == 'text']
            # Tool execution errors become Python exceptions for the GUI.
            # Otherwise concatenate text blocks and parse their JSON payload.
            # Invalid JSON propagates as a JSON decoding exception.
            if result.isError:
                raise RuntimeError('MCP record error: ' + ' '.join(texts))
            return json.loads(''.join(texts))

    # Synchronous bridge for GUI code or its worker threads.
    # Example:
    #   call('xrays', 'list_records', {'relative_folder': 'P0001/xrays'})
    # Returns the decoded JSON result, normally a Python dictionary.
    # IMPORTANT: although I/O runs on another thread, THIS method blocks its
    # caller while waiting. Calling it on the Tkinter thread can pause the UI.
    # Do not call it from the MCP background loop itself (it would deadlock).
    def call(self, modality, tool, arguments):
        # Wait up to 30 seconds for initialization/failure notification.
        # A startup timeout does not stop the initialization thread.
        if not self.ready.wait(30):
            raise RuntimeError('MCP servers did not start within 30 seconds.')
        if self.error is not None:
            raise RuntimeError('MCP startup failed. Install requirements.txt and check server files. ' + str(self.error))
        # Safely schedule the coroutine from the caller's thread onto the MCP
        # event loop. A concurrent.futures.Future bridges the two threads.
        future = asyncio.run_coroutine_threadsafe(self._call(modality, tool, arguments), self.loop)
        try:
            # Wait up to 60 seconds, including time waiting for the shared lock.
            # Exceptions raised inside _call() are re-raised in this thread.
            return future.result(timeout=60)
        except FutureTimeout as error:
            # Request coroutine cancellation after timeout; this does not prove
            # that server-side execution has already stopped.
            future.cancel()
            raise RuntimeError('MCP record request timed out.') from error

    # Request shutdown without waiting for completion or joining the thread.
    # call_soon_threadsafe schedules Event.set on the event's own loop.
    # That wakes _serve(), which exits its AsyncExitStack and closes sessions.
    # This is intended for normal GUI exit AFTER startup; a close() called before
    # loop/stop_event exist does nothing. There is no restart mechanism here.
    def close(self):
        if self.loop is not None and self.loop.is_running() and self.stop_event is not None:
            self.loop.call_soon_threadsafe(self.stop_event.set)


    # Convert a GUI file path to (modality, root-relative POSIX path).
    # Example: C:/app/patient_records/P0001/xrays/chest.png becomes
    # ('xrays', 'P0001/xrays/chest.png'). Date/time intermediate folders are OK.
    # The first relative component is treated as the patient directory.
    def record_location(self, path):
        root = self.records_root
        # Normalize the path and reject paths resolving outside records_root.
        # The server separately enforces patient/modality confinement, allowed
        # file types and link restrictions; this client check is not sufficient
        # by itself to enforce every server access rule.
        relative = Path(path).resolve().relative_to(root)
        # Find exactly one recognized modality component after the patient ID.
        # Missing, duplicate, or mixed modality components are rejected below.
        modalities = [part for part in relative.parts[1:] if part in ('xrays', 'ekgs', 'blood_tests')]
        if len(modalities) != 1:
            raise ValueError('Record must belong to one modality folder.')
        # Forward slashes give the server a consistent relative path format,
        # including when the GUI is running on Windows.
        return modalities[0], relative.as_posix()

    # Convenience method for X-ray/EKG images. Route to the modality server's
    # read_record tool, then turn its base64 text back into original file bytes.
    # The GUI can pass those bytes to Pillow through io.BytesIO for display,
    # or base64-encode them for the existing Ollama image request.
    # This method expects an image-server response with a 'base64' field; use
    # read_blood_record() for blood JSON. No image resizing happens here.
    def read_record_bytes(self, path):
        modality, relative = self.record_location(path)
        result = self.call(modality, 'read_record', {'relative_path': relative})
        # validate=True rejects malformed base64 rather than silently discarding
        # invalid characters. Missing response fields also raise an exception.
        return base64.b64decode(result['base64'], validate=True)

    # Convenience method for blood JSON. Reject another modality before calling
    # its server. The blood server parses the JSON and returns it under 'record'.
    # Return that value unchanged (typically a dict or list); the GUI controls
    # display, selection, and which results are included in the model prompt.
    def read_blood_record(self, path):
        modality, relative = self.record_location(path)
        if modality != 'blood_tests':
            raise ValueError('Expected a blood-test record.')
        return self.call(modality, 'read_record', {'relative_path': relative})['record']


