# Integrated release verification

Application: `../python/client_medgemma_patients.py`
Verification date: 2026-10-25
Runtime: bundled Python 3.12.14, Tkinter available, Pillow 12.3.0.

The T13 run used temporary synthetic records for two ordinary patient names. It did not read, create, change, or transmit production patient data. The app source, `.env`, generator source, launcher scripts, and endpoint/send implementation were not changed for T13.

## T13 checks run and passed

- Compiled the application source in memory and imported it without starting the GUI.
- Exercised patient-root validation with a missing root, a regular file, an empty directory, and a relative root beneath a moved/script-directory fixture. Process-level startup with a missing root and a regular-file root exited with status 1 and printed the resolved path and stop message before constructing Tk.
- Ran the actual Tk application under a withdrawn (hidden) Tk root against temporary `Aster Patient` and `Boreal Patient` folders. Both patient records loaded; the title-less CBC panel was found. The Actions placeholder had no children.
- Selected X-ray, EKG, and CBC records for patient one, submitted through the actual `send_question` / request worker / `generate` path, and intercepted HTTP at a local in-process mock for `/api/tags`, `/api/show`, and `/api/generate`. The payload contained only the active patient's selected lab record. Both images' encoded payload strings exactly matched the original synthetic file bytes; no thumbnail data was substituted.
- Cancelled a request while the mocked stream was blocked. The UI reached its cancelled terminal state in under one second, the partial response was not retained as a completed draft, and the later stream result could not update the report.
- Completed a second mocked request, verified the final response omitted private reasoning, edited the user draft, acknowledged review, and exported TXT/HTML/JSON outside `patient_records`. Bundle checks confirmed the patient, separate original and edited drafts, local acknowledgement wording, and patient-relative X-ray/EKG references.
- Switched to patient two and confirmed selections, prompt/report context, and visible history were patient-specific. A second mock request used only patient two's selected lab record and no images; a second report was saved. Switching back showed only patient one's saved report. Reopening restored original text, edited text, and review state without another generation call.
- Compared SHA-256 hashes of every synthetic source image and JSON file before and after the workflow; all were unchanged.
- Parser fixtures covered reasoning markers split across response events, separate `thinking` content, valid completion metadata, truncated stream, and malformed JSON event.
- Generator fixtures confirmed equal seed and inventory produce equal assignments; `--dry-run` left its destination absent and source-file hashes unchanged.
- The two-image-per-patient synthetic GUI fixture reported first visible gallery content in approximately 0.10–0.13 seconds in this run. This is a small-fixture observation, not a large-gallery or cross-machine performance benchmark.

## Earlier recorded checks retained

The previous verification record also reports programmatic widget checks for `MohamedLjalil`, `P0002`, and `Patient With Spaces`; hidden-directory filtering; invalid image/JSON and missing-modality behavior; mocked timeout and unreachable-server handling; and a launcher/config-directory path check. Those checks were not all repeated as part of the T13 scenario above. The T13 process-level startup test used an unrelated workspace working directory. In this execution environment, setting the process working directory to the Public project `config/` directory was denied, so that launch-path check was not repeated here.

## Separate visual launch and live-service observation

A separate visual check by the coordinating agent launched the desktop app with bundled Python and captured the AI Clinical Assistant Center window. The screen showed **Records READY (5 patient(s))**, a selected patient, three X-ray tiles, one ECG tile, a readable blood-test table, the preset/question area, Cancel/Save/Retry controls, and an empty Actions placeholder. This confirms the startup screen and a populated-record layout only; it is not a full visual, keyboard, or scaling review. No Send action was taken and no medical files or images were transmitted.

The configured Ollama `/api/tags` check was intermittent at first because the installed Ollama executable was not on PATH. Starting the existing executable at `C:\Users\mmkha_s1\AppData\Local\Programs\Ollama\ollama.exe` restored the local service; `/api/tags` then returned `qwen3:8b` and `medgemma1.5:latest`. A text-only synthetic `/api/generate` request completed with `done_reason=stop`. The response included a model reasoning wrapper; the app's `final_only_answer` helper removed it and retained only the requested X-RAY, ECG/EKG, and BLOOD TEST sections. No patient records or images were sent. This verifies the live local endpoint and final-answer filtering, while the complete GUI Send path with patient attachments remains covered by the in-process mock scenario above.

## Not verified / limitations

- The T13 Tk window was withdrawn for its interaction checks. A separate screenshot-based visual check confirmed the startup layout and visible records as described above; full visual review at common display scaling, keyboard navigation, and every report dialog remains unverified.
- No live patient-record generation was performed. The local Ollama service was started and exercised with a text-only synthetic request after the visual check; live attachment generation remains covered only by the in-process mock scenario.
- The installed `python` command was a non-runnable Windows Store alias; checks used the bundled Python runtime. Launcher execution itself was not repeated in this session.
- Unreadable-directory ACL behavior was not tested. Permission behavior varies by platform; missing and non-directory roots were tested.
- The real `python/patient_records` directory exists and the visual check found five patient folders. Their contents were not read for T13, and no record files were changed. The synthetic integration fixtures used a separate temporary root.
- Clinical accuracy, diagnostic safety, image interpretation quality, and ECG measurement validity have not been evaluated. These software checks establish only request, state, persistence, and data-handling behavior. MedGemma output remains research assistance requiring clinician review.

## User launch

Use the explicit named launcher from `config/` (`run.bat` on Windows or `bash run.sh` on Ubuntu), or run `python client_medgemma_patients.py` / `python3 client_medgemma_patients.py` from `python/`. Workflow, report export/history, and troubleshooting steps are in `README.md`.
