#!/usr/bin/env python3
"""
dlc_setup_and_prepare.py  (Windows / Linux / macOS)  ✅ UPDATED FOR WINDOWS GUI + ERROR MESSAGES

WHAT'S UPDATED (per your request):
✅ Installs DeepLabCut GUI reliably on Windows by installing wxPython via conda-forge first
✅ Prints clear error messages if ANY package install step fails
✅ Writes full logs to: deeplabcut/<DATASET_FOLDER_NAME>/logs/install_log.txt
✅ Still creates DLC project + extracts frames + writes OUTPUT_SUMMARY.txt

RUN (Windows PowerShell):
  .\dlc_setup_and_prepare.py
  # If direct execution doesn't work:
  py .\dlc_setup_and_prepare.py

RUN (Linux/macOS):
  chmod +x dlc_setup_and_prepare.py
  ./dlc_setup_and_prepare.py

-------------------------------------------------------
EDIT THESE TWO LINES (YOU ASKED THIS):
  DATA_PATH
  DATASET_FOLDER_NAME
-------------------------------------------------------
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# ============================================================
# ✅ USER SETTINGS (EDIT ONLY THESE TWO REQUIRED)
# ============================================================

# (1) Path to your dataset root folder
DATA_PATH = r"C:\path\to\your\dataset"  # <-- CHANGE THIS

# (2) Name of output folder under ./deeplabcut/
DATASET_FOLDER_NAME = "datasetname"  # <-- CHANGE THIS

# ============================================================
# OPTIONAL SETTINGS (safe to keep defaults)
# ============================================================
PROJECT_NAME = "DogMotionDLC"
EXPERIMENTER = "labeler"
MODE = "auto"  # "auto" | "2d" | "3d"

ENV_NAME = "dlc310"
PYTHON_VERSION = "3.10"

# Install extras
INSTALL_TF = True
INSTALL_GUI = True  # ✅ set True to install GUI

# Frame extraction settings
EXTRACT_MODE = "automatic"   # "automatic" | "manual"
EXTRACT_ALGO = "kmeans"      # "kmeans" | "uniform"
COPY_VIDEOS_INTO_PROJECT = True

# 2D image->video settings
VIDEO_FPS = 30
MAX_IMAGES_FOR_TEMP_VIDEO = 800

# Bodyparts (2D)
BODYPARTS = [
    "nose",
    "left_ear", "right_ear",
    "withers",
    "tail_base",
    "left_front_paw", "right_front_paw",
    "left_hind_paw", "right_hind_paw",
]

# ============================================================
# END SETTINGS
# ============================================================


# -----------------------------
# Logging + robust command runner
# -----------------------------
LOG_FILE: Optional[Path] = None


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if LOG_FILE:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8", errors="ignore") as f:
            f.write(line + "\n")


def run(cmd: List[str], env=None, step_name: str = "command") -> None:
    """
    Runs a command and prints a helpful error message if it fails.
    Captures stdout/stderr on failure and writes to log.
    """
    cmd_str = " ".join(str(c) for c in cmd)
    log(f"[RUN:{step_name}] {cmd_str}")

    try:
        # capture output for logging if it fails
        subprocess.run([str(c) for c in cmd], env=env, check=True, text=True)
    except subprocess.CalledProcessError as e:
        log(f"[ERROR:{step_name}] Command failed with exit code {e.returncode}")
        log(f"[ERROR:{step_name}] Command: {cmd_str}")

        # Try to re-run capturing stdout/stderr for clearer diagnostics
        try:
            p = subprocess.run([str(c) for c in cmd], env=env, check=False, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.stdout:
                log(f"[STDOUT:{step_name}]\n{p.stdout}")
            if p.stderr:
                log(f"[STDERR:{step_name}]\n{p.stderr}")
        except Exception as ee:
            log(f"[WARN:{step_name}] Could not capture stdout/stderr: {ee}")

        # Raise a clean error message to stop script
        raise RuntimeError(
            f"{step_name} failed.\n"
            f"Command: {cmd_str}\n"
            f"See log file: {LOG_FILE}"
        ) from e


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"[OK] Already downloaded: {out_path}")
        return
    log(f"[DL] {url} -> {out_path}")
    try:
        with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
            f.write(r.read())
    except Exception as e:
        log(f"[ERROR] Download failed: {url}")
        raise RuntimeError(f"Download failed: {url}\nSee log file: {LOG_FILE}") from e
    log("[OK] Download complete")


# -----------------------------
# micromamba setup helpers
# -----------------------------
def detect_micromamba_platform() -> str:
    sysname = platform.system().lower()
    machine = platform.machine().lower()

    if "windows" in sysname:
        return "win-64"
    if "linux" in sysname:
        if machine in ("x86_64", "amd64"):
            return "linux-64"
        if machine in ("aarch64", "arm64"):
            return "linux-aarch64"
        if "ppc64le" in machine:
            return "linux-ppc64le"
        raise RuntimeError(f"Unsupported Linux arch: {machine}")
    if "darwin" in sysname or "mac" in sysname:
        if machine in ("x86_64", "amd64"):
            return "osx-64"
        if machine in ("arm64", "aarch64"):
            return "osx-arm64"
        raise RuntimeError(f"Unsupported macOS arch: {machine}")

    raise RuntimeError(f"Unsupported OS: {platform.system()}")


def extract_micromamba(tar_bz2: Path, extract_dir: Path) -> Path:
    """
    Expected layouts:
      - Linux/macOS: bin/micromamba
      - Windows: Library/bin/micromamba.exe
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    log(f"[EXTRACT] {tar_bz2} -> {extract_dir}")
    try:
        with tarfile.open(tar_bz2, "r:bz2") as tf:
            tf.extractall(extract_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to extract micromamba archive: {tar_bz2}\nSee log: {LOG_FILE}") from e

    candidates = [
        extract_dir / "bin" / "micromamba",
        extract_dir / "Library" / "bin" / "micromamba.exe",
    ]
    for p in candidates:
        if p.exists():
            return p

    for p in extract_dir.rglob("micromamba*"):
        if p.is_file() and p.name.lower() in ("micromamba", "micromamba.exe"):
            return p

    raise RuntimeError(f"Could not locate micromamba executable after extraction.\nSee log: {LOG_FILE}")


def ensure_exec_bit(path: Path) -> None:
    if platform.system().lower().startswith("win"):
        return
    try:
        path.chmod(path.stat().st_mode | 0o111)
    except Exception:
        pass


# -----------------------------
# Dataset helpers
# -----------------------------
def find_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in exts]


def find_videos(folder: Path) -> List[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v"}
    return [p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in exts]


def pick_two_cameras(videos: List[Path]) -> Tuple[Path, Path]:
    if len(videos) < 2:
        raise RuntimeError("Need at least 2 videos for 3D (two cameras).")

    p0 = re.compile(r"(cam|camera|view)[ _-]?0", re.IGNORECASE)
    p1 = re.compile(r"(cam|camera|view)[ _-]?1", re.IGNORECASE)
    cand0 = [v for v in videos if p0.search(v.name)]
    cand1 = [v for v in videos if p1.search(v.name)]
    if cand0 and cand1:
        return cand0[0], cand1[0]
    return videos[0], videos[1]


# -----------------------------
# Summary helpers
# -----------------------------
def count_extracted_frames(project_dir: Path) -> int:
    labeled = project_dir / "labeled-data"
    if not labeled.exists():
        return 0
    exts = {".png", ".jpg", ".jpeg"}
    return sum(1 for p in labeled.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def find_latest_project_dir(projects_dir: Path, project_prefix: str) -> Optional[Path]:
    if not projects_dir.exists():
        return None
    candidates = [p for p in projects_dir.iterdir() if p.is_dir() and p.name.startswith(project_prefix + "-")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    global LOG_FILE

    dataset_root = Path(DATA_PATH).expanduser().resolve()
    if not dataset_root.exists():
        raise RuntimeError(f"DATA_PATH does not exist: {dataset_root}")

    datasetname = DATASET_FOLDER_NAME.strip()
    if not datasetname:
        raise RuntimeError("DATASET_FOLDER_NAME is empty. Set it at top of the script.")

    base_out = Path("deeplabcut") / datasetname
    micromamba_dir = base_out / "micromamba"
    env_root = base_out / "env_root"
    projects_dir = base_out / "projects"
    temp_videos_dir = base_out / "temp_videos"
    logs_dir = base_out / "logs"
    summary_path = base_out / "OUTPUT_SUMMARY.txt"

    for d in [micromamba_dir, env_root, projects_dir, temp_videos_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    LOG_FILE = logs_dir / "install_log.txt"
    # clear old log
    try:
        if LOG_FILE.exists():
            LOG_FILE.unlink()
    except Exception:
        pass

    log("============================================================")
    log("DLC SETUP + PREP (UPDATED WINDOWS GUI + ERROR MESSAGES)")
    log("============================================================")
    log(f"Dataset path: {dataset_root}")
    log(f"Output root:  {base_out.resolve()}")
    log(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    log(f"GUI requested: {INSTALL_GUI}")
    log(f"TF extras requested: {INSTALL_TF}")

    # -----------------------------
    # 1) Install micromamba
    # -----------------------------
    os_tag = detect_micromamba_platform()
    mm_exe = micromamba_dir / ("micromamba.exe" if os_tag.startswith("win") else "micromamba")

    url = f"https://micro.mamba.pm/api/micromamba/{os_tag}/latest"
    tar_path = micromamba_dir / f"micromamba-{os_tag}.tar.bz2"
    extract_dir = micromamba_dir / f"extract-{os_tag}"

    if not mm_exe.exists():
        download(url, tar_path)
        extracted = extract_micromamba(tar_path, extract_dir)
        shutil.copy2(extracted, mm_exe)
        ensure_exec_bit(mm_exe)
        log(f"[OK] micromamba installed: {mm_exe}")
    else:
        log(f"[OK] micromamba already present: {mm_exe}")

    env = os.environ.copy()
    env["MAMBA_ROOT_PREFIX"] = str(env_root)

    run([mm_exe, "--version"], env=env, step_name="micromamba_version")

    # -----------------------------
    # 2) Create env if missing
    # -----------------------------
    try:
        env_list = subprocess.check_output([str(mm_exe), "env", "list"], env=env, text=True, errors="ignore")
    except Exception as e:
        raise RuntimeError(f"Failed to list micromamba environments.\nSee log: {LOG_FILE}") from e

    if ENV_NAME in env_list:
        log(f"[OK] Env exists: {ENV_NAME}")
    else:
        run(
            [mm_exe, "create", "-y", "-n", ENV_NAME, "-c", "conda-forge", f"python={PYTHON_VERSION}", "pip"],
            env=env,
            step_name="create_env",
        )

    # -----------------------------
    # 3) Install packages (Windows-safe GUI)
    # -----------------------------
    # Always upgrade pip basics
    run([mm_exe, "run", "-n", ENV_NAME, "python", "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        env=env, step_name="pip_upgrade")

    # ✅ Windows GUI reliability: install wxpython from conda-forge first (if GUI requested)
    if INSTALL_GUI:
        if platform.system().lower().startswith("win"):
            run([mm_exe, "install", "-y", "-n", ENV_NAME, "-c", "conda-forge", "wxpython"],
                env=env, step_name="install_wxpython_conda_windows")
        else:
            # On Linux/macOS, pip is often OK; still let DLC extras handle it
            log("[INFO] Non-Windows OS: wxPython will be handled by deeplabcut[gui] if needed.")

    extras = []
    if INSTALL_GUI:
        extras.append("gui")
    if INSTALL_TF:
        extras.append("tf")
    extra_str = f"[{','.join(extras)}]" if extras else ""

    # Install DeepLabCut
    run([mm_exe, "run", "-n", ENV_NAME, "python", "-m", "pip", "install", f"deeplabcut{extra_str}"],
        env=env, step_name="pip_install_deeplabcut")

    # Ensure OpenCV
    run([mm_exe, "run", "-n", ENV_NAME, "python", "-m", "pip", "install", "opencv-python"],
        env=env, step_name="pip_install_opencv")

    # Sanity checks
    run([mm_exe, "run", "-n", ENV_NAME, "python", "-c",
         "import deeplabcut; print('DeepLabCut:', deeplabcut.__version__)"],
        env=env, step_name="check_deeplabcut_import")

    if INSTALL_GUI:
        # Check GUI entrypoint without opening window (import wx)
        run([mm_exe, "run", "-n", ENV_NAME, "python", "-c", "import wx; print('wxPython OK')"],
            env=env, step_name="check_wxpython_import")

    # -----------------------------
    # 4) Decide 2D vs 3D
    # -----------------------------
    vids = find_videos(dataset_root)
    imgs = find_images(dataset_root)

    mode = MODE.lower().strip()
    if mode == "auto":
        if len(vids) >= 2:
            mode = "3d"
        elif len(vids) == 1 or len(imgs) > 0:
            mode = "2d"
        else:
            raise RuntimeError("AUTO mode: no videos or images found in dataset folder.")

    if mode not in ("2d", "3d"):
        raise RuntimeError(f"Invalid MODE: {MODE} (must be auto/2d/3d)")

    log(f"[INFO] Videos found: {len(vids)}")
    log(f"[INFO] Images found: {len(imgs)}")
    log(f"[INFO] Selected mode: {mode}")

    if len(BODYPARTS) < 2:
        raise RuntimeError("BODYPARTS must include at least 2 points.")

    selected_videos: List[str] = []
    if mode == "3d":
        if len(vids) < 2:
            raise RuntimeError("3D mode requires at least 2 videos in DATA_PATH.")
        v0, v1 = pick_two_cameras(vids)
        selected_videos = [str(v0), str(v1)]
    else:
        if vids:
            selected_videos = [str(vids[0])]

    # -----------------------------
    # 5) Runner inside env to create DLC project + extract frames
    # -----------------------------
    runner_path = base_out / "dlc_runner_prepare.py"

    runner_code = f"""\
from pathlib import Path

DATASET_ROOT = Path(r\"{str(dataset_root)}\").resolve()
PROJECTS_DIR = Path(r\"{str(projects_dir)}\").resolve()
TEMP_VIDEOS_DIR = Path(r\"{str(temp_videos_dir)}\").resolve()

MODE = {mode!r}
PROJECT = {PROJECT_NAME!r}
EXPERIMENTER = {EXPERIMENTER!r}
COPY_VIDEOS = {bool(COPY_VIDEOS_INTO_PROJECT)!r}
EXTRACT_MODE = {EXTRACT_MODE!r}
ALGO = {EXTRACT_ALGO!r}
FPS = {int(VIDEO_FPS)!r}
MAX_FRAMES_VIDEO = {int(MAX_IMAGES_FOR_TEMP_VIDEO)!r}
BODYPARTS = {list(BODYPARTS)!r}
SELECTED_VIDEOS = {selected_videos!r}

def find_images(folder: Path):
    exts = {{".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}}
    return [p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in exts]

def images_to_video(images, out_mp4: Path, fps=30, max_frames=None):
    import cv2
    if not images:
        raise RuntimeError("No images found to build a video.")
    if max_frames is not None:
        images = images[:max_frames]
    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first image: {{images[0]}}")
    h, w = first.shape[:2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
    for p in images:
        im = cv2.imread(str(p))
        if im is None:
            continue
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        vw.write(im)
    vw.release()
    if not out_mp4.exists() or out_mp4.stat().st_size == 0:
        raise RuntimeError("Video creation failed (output empty).")
    return out_mp4

def main():
    import deeplabcut

    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    if MODE == "2d":
        if SELECTED_VIDEOS:
            video_path = Path(SELECTED_VIDEOS[0]).resolve()
            print(f"[INFO] Using existing video: {{video_path}}")
        else:
            imgs = find_images(DATASET_ROOT)
            print(f"[INFO] Found {{len(imgs)}} images. Building temp MP4 for DLC...")
            video_path = (TEMP_VIDEOS_DIR / f"{{PROJECT}}_from_images.mp4").resolve()
            images_to_video(imgs, video_path, fps=FPS, max_frames=MAX_FRAMES_VIDEO)
            print(f"[OK] Created video: {{video_path}}")

        print("[INFO] Creating DLC 2D project...")
        config_path = deeplabcut.create_new_project(
            PROJECT,
            EXPERIMENTER,
            [str(video_path)],
            working_directory=str(PROJECTS_DIR),
            copy_videos=COPY_VIDEOS,
            multianimal=False,
            bodyparts=BODYPARTS,
        )
        print(f"[OK] 2D Project config: {{config_path}}")

        deeplabcut.add_new_videos(config_path, [str(video_path)], copy_videos=COPY_VIDEOS)

        print("[INFO] Extracting frames...")
        if EXTRACT_MODE == "automatic":
            deeplabcut.extract_frames(config_path, mode="automatic", algo=ALGO, userfeedback=False)
        else:
            deeplabcut.extract_frames(config_path, mode="manual")

        print("\\n✅ DONE (2D). Config path:", config_path)
        return

    # 3D mode
    if len(SELECTED_VIDEOS) < 2:
        raise RuntimeError("3D mode requires 2 selected videos.")
    v0 = Path(SELECTED_VIDEOS[0]).resolve()
    v1 = Path(SELECTED_VIDEOS[1]).resolve()
    print(f"[INFO] Selected 3D camera videos:\\n  cam0: {{v0}}\\n  cam1: {{v1}}")

    print("[INFO] Creating DLC 3D project...")
    config3d_path = deeplabcut.create_new_project_3d(
        PROJECT,
        EXPERIMENTER,
        num_cameras=2,
        working_directory=str(PROJECTS_DIR),
        copy_videos=COPY_VIDEOS,
    )
    print(f"[OK] 3D Project config: {{config3d_path}}")

    deeplabcut.add_new_videos(config3d_path, [str(v0), str(v1)], copy_videos=COPY_VIDEOS)

    print("[INFO] Extracting frames...")
    if EXTRACT_MODE == "automatic":
        deeplabcut.extract_frames(config3d_path, mode="automatic", algo=ALGO, userfeedback=False)
    else:
        deeplabcut.extract_frames(config3d_path, mode="manual")

    print("\\n✅ DONE (3D). Config path:", config3d_path)

if __name__ == "__main__":
    main()
"""
    runner_path.write_text(runner_code, encoding="utf-8")
    log(f"[INFO] Runner written: {runner_path}")

    run([mm_exe, "run", "-n", ENV_NAME, "python", str(runner_path)], env=env, step_name="dlc_project_prepare")

    # -----------------------------
    # 6) OUTPUT_SUMMARY.txt
    # -----------------------------
    proj_dir = find_latest_project_dir(projects_dir, PROJECT_NAME)
    config_path = (proj_dir / "config.yaml") if proj_dir else None
    extracted_frames = count_extracted_frames(proj_dir) if proj_dir else 0

    used_videos = selected_videos[:]
    temp_video_guess = temp_videos_dir / f"{PROJECT_NAME}_from_images.mp4"
    if mode == "2d" and (not used_videos) and temp_video_guess.exists():
        used_videos = [str(temp_video_guess.resolve())]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("DeepLabCut Preparation Summary")
    lines.append("=" * 32)
    lines.append(f"Timestamp: {now}")
    lines.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    lines.append("")
    lines.append("Input")
    lines.append("-" * 6)
    lines.append(f"DATA_PATH: {str(dataset_root)}")
    lines.append(f"Mode: {mode}")
    lines.append(f"Videos found: {len(vids)}")
    lines.append(f"Images found: {len(imgs)}")
    lines.append("")
    lines.append("Output")
    lines.append("-" * 6)
    lines.append(f"Output root: {str(base_out.resolve())}")
    lines.append(f"Projects dir: {str(projects_dir.resolve())}")
    lines.append(f"Log file: {str(LOG_FILE.resolve())}")
    lines.append("")
    lines.append("Environment")
    lines.append("-" * 11)
    lines.append(f"Micromamba: {str(mm_exe.resolve())}")
    lines.append(f"MAMBA_ROOT_PREFIX: {str(env_root.resolve())}")
    lines.append(f"Env name: {ENV_NAME}")
    lines.append(f"GUI installed: {INSTALL_GUI}")
    lines.append(f"TF extras installed: {INSTALL_TF}")
    lines.append("")
    lines.append("DeepLabCut Project")
    lines.append("-" * 17)
    if proj_dir and proj_dir.exists():
        lines.append(f"Project folder: {str(proj_dir.resolve())}")
    else:
        lines.append("Project folder: (NOT FOUND)")
    if config_path and config_path.exists():
        lines.append(f"Config path: {str(config_path.resolve())}")
    else:
        lines.append("Config path: (NOT FOUND)")
    lines.append(f"Extracted frames (count): {extracted_frames}")
    lines.append("")
    lines.append("Selected/Used Videos")
    lines.append("-" * 20)
    if used_videos:
        for v in used_videos:
            lines.append(f"- {v}")
    else:
        lines.append("(None detected; check project/videos or temp_videos)")
    lines.append("")
    lines.append("Next Commands")
    lines.append("-" * 13)
    lines.append("Check DLC in env:")
    lines.append(f'  "{str(mm_exe.resolve())}" run -n {ENV_NAME} python -c "import deeplabcut; print(deeplabcut.__version__)"')
    lines.append("")
    if INSTALL_GUI:
        lines.append("Launch DLC GUI:")
        lines.append(f'  "{str(mm_exe.resolve())}" run -n {ENV_NAME} python -m deeplabcut')
        lines.append("")
    lines.append("Notes")
    lines.append("-" * 5)
    lines.append(" - Frames are stored under: <project_folder>/labeled-data/")
    lines.append(" - After labeling: create_training_dataset -> train_network -> analyze_videos")
    lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    log("============================================================")
    log("✅ DONE")
    log("============================================================")
    log(f"Summary: {summary_path.resolve()}")
    if config_path and config_path.exists():
        log(f"Config.yaml: {config_path.resolve()}")
    else:
        log("Config.yaml not found (something likely failed during project creation).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORTED] by user.")
        sys.exit(130)
    except Exception as e:
        # Always print the error and point to log file (if created)
        print("\n[FAILED]", str(e))
        try:
            if LOG_FILE:
                print("See log file:", str(LOG_FILE.resolve()))
        except Exception:
            pass
        # print stack trace for debugging
        traceback.print_exc()
        sys.exit(1)
