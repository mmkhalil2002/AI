#!/usr/bin/env python3
# ============================================================
# install-dog-2D-csv-text-dataset.py
# (CLEAN + SELECT + OPTIONAL _v + DROP INVALID BY % OF 'Y' KEYS)
# ============================================================
# Dog-Pose (Ultralytics) -> ONE CLEAN CSV + ONE aligned CLEAN TXT
#
# ✅ Downloads dog-pose.zip automatically if missing
# ✅ Extracts automatically if raw/ is empty
# ✅ Auto-discovers dataset root (images/ + labels/) even with nested folders
# ✅ Parses YOLO-Pose labels into pixel coordinates (x_px, y_px)
#
# ✅ INVALID-DATA POLICY (YOUR FINAL REQUEST):
#   Keep a row only if the % of VALID keypoints among the OUTPUT-SELECTED ("Y") keypoints
#   is >= X.
#
#   You set X via:
#      -y X
#
#   Examples:
#     -y 80   => keep row if >= 80% of the Y keypoints are valid
#     -y 100  => keep row only if ALL Y keypoints are valid
#
#   "Valid keypoint" means:
#     - x,y are not NaN
#     - (x,y) is not (0,0)    <-- common placeholder for missing
#     - if REJECT_OUT_OF_BOUNDS=True: x,y must be within image bounds
#     - if --include-v: v must be > VISIBILITY_THRESHOLD
#
# OUTPUT (both CSV and TXT have THE SAME COLUMNS):
#   ./dog-2D-csv-text-dataset/dog2d_keypoints_clean.csv
#   ./dog-2D-csv-text-dataset/dog2d_keypoints_clean_aligned.txt
#
# OPTIONAL:
#   --include-v     Include *_v visibility columns (otherwise excluded)
#
# READ MODE:
#   -r              Read TXT line-by-line and print to screen (build first if missing)
#   -f N            With -r: read only N lines and also write them to:
#                   ./dog-2D-csv-text-dataset/dog-N
#
# HELP:
#   -h / --help     Show help (argparse provides this automatically)
#
# EXAMPLES:
#   python3 install-dog-2D-csv-text-dataset.py
#   python3 install-dog-2D-csv-text-dataset.py -y 80
#   python3 install-dog-2D-csv-text-dataset.py -y 100
#   python3 install-dog-2D-csv-text-dataset.py --include-v -y 70
#   python3 install-dog-2D-csv-text-dataset.py -r -f 50
# ============================================================

import sys
import subprocess
import importlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import zipfile
import argparse
import math

# ----------------------------
# Auto-install needed packages
# ----------------------------
REQUIRED_PACKAGES = ["pandas", "requests", "tqdm", "certifi", "numpy", "Pillow"]

def ensure_packages() -> None:
    for pkg in REQUIRED_PACKAGES:
        mod = pkg if pkg != "Pillow" else "PIL"
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"[INSTALL] Installing missing package: {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

ensure_packages()

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import certifi
from PIL import Image

# =============================================================================
# SETTINGS
# =============================================================================
OUT_DIR = Path("./dog-2D-csv-text-dataset").resolve()
ARCHIVES_DIR = OUT_DIR / "archives"
RAW_DIR = OUT_DIR / "raw"

OUT_CSV = OUT_DIR / "dog2d_keypoints_clean.csv"
OUT_TXT = OUT_DIR / "dog2d_keypoints_clean_aligned.txt"

DOGPOSE_ZIP_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/dog-pose.zip"
DOGPOSE_ZIP_NAME = "dog-pose.zip"

VERIFY_SSL = True
CHUNK_SIZE = 1024 * 1024

# =============================================================================
# FILTER DEFAULTS (overridden by CLI args)
# =============================================================================
DROP_INVALID_ROWS_DEFAULT = True

# ✅ YOUR REQUEST: -y X means % of VALID among the 'Y' elements
Y_VALID_PERCENT_REQUIRED_DEFAULT = 80.0

# If True: keypoints outside image dimensions are invalid
REJECT_OUT_OF_BOUNDS_DEFAULT = True

# If include_v: v <= threshold => invalid
VISIBILITY_THRESHOLD_DEFAULT = 0.0

# =============================================================================
# FIELD SELECTION TABLES (Y/N controls output inclusion)
# =============================================================================
KPT_MAP: Dict[str, Tuple[str, str]] = {
    # FRONT LEFT
    "front_left_paw":   ("FL_PW", "Y"),
    "front_left_knee":  ("FL_KN", "Y"),
    "front_left_elbow": ("FL_EL", "Y"),

    # REAR LEFT
    "rear_left_paw":    ("RL_PW", "Y"),
    "rear_left_knee":   ("RL_KN", "Y"),
    "rear_left_elbow":  ("RL_EL", "Y"),

    # FRONT RIGHT
    "front_right_paw":   ("FR_PW", "Y"),
    "front_right_knee":  ("FR_KN", "Y"),
    "front_right_elbow": ("FR_EL", "Y"),

    # REAR RIGHT
    "rear_right_paw":    ("RR_PW", "Y"),
    "rear_right_knee":   ("RR_KN", "Y"),
    "rear_right_elbow":  ("RR_EL", "Y"),

    # TAIL
    "tail_start": ("TL_S", "N"),
    "tail_end":   ("TL_E", "N"),

    # HEAD
    "left_ear_base":  ("LE_B", "N"),
    "right_ear_base": ("RE_B", "N"),
    "left_ear_tip":   ("LE_T", "N"),
    "right_ear_tip":  ("RE_T", "N"),
    "left_eye":       ("LEY",  "N"),
    "right_eye":      ("REY",  "N"),
    "nose":           ("NS",   "N"),
    "chin":           ("CH",   "N"),

    # BODY CORE
    "withers": ("SH_C", "N"),
    "throat":  ("THR",  "N"),
}

META_MAP: Dict[str, Tuple[str, str]] = {
    "split":        ("split",  "Y"),  # train/val/test/all
    "image_file":   ("img",    "N"),
    "image_path":   ("path",   "N"),
    "object_index": ("obj",    "N"),
    "image_width":  ("W",      "N"),
    "image_height": ("H",      "N"),
    "bbox_xc_px":   ("bb_xc",  "N"),
    "bbox_yc_px":   ("bb_yc",  "N"),
    "bbox_w_px":    ("bb_w",   "N"),
    "bbox_h_px":    ("bb_h",   "N"),
}

# The dataset keypoints order in Dog-Pose (Ultralytics) is 24 keypoints:
KPT_NAMES = [
    "front_left_paw",
    "front_left_knee",
    "front_left_elbow",
    "rear_left_paw",
    "rear_left_knee",
    "rear_left_elbow",
    "front_right_paw",
    "front_right_knee",
    "front_right_elbow",
    "rear_right_paw",
    "rear_right_knee",
    "rear_right_elbow",
    "tail_start",
    "tail_end",
    "left_ear_base",
    "right_ear_base",
    "nose",
    "chin",
    "left_ear_tip",
    "right_ear_tip",
    "left_eye",
    "right_eye",
    "withers",
    "throat",
]

def selected_Y_abbrs() -> List[str]:
    """Return ABBRs for keypoints marked 'Y' in the same order as KPT_NAMES."""
    out: List[str] = []
    for kpt_name in KPT_NAMES:
        if kpt_name not in KPT_MAP:
            continue
        abbr, yn = KPT_MAP[kpt_name]
        if yn.strip().upper() == "Y":
            out.append(abbr)
    return out

# =============================================================================
# Download helper
# =============================================================================
def download_file(url: str, dst: Path, timeout: int = 60) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    resume_pos = tmp.stat().st_size if tmp.exists() else 0

    headers = {}
    if resume_pos > 0:
        headers["Range"] = f"bytes={resume_pos}-"

    verify_arg = False if not VERIFY_SSL else certifi.where()

    print("============================================================")
    print("[DOWNLOAD]")
    print(f"URL    : {url}")
    print(f"TARGET : {dst}")
    print(f"RESUME : {resume_pos} bytes" if resume_pos > 0 else "RESUME : no")
    print("============================================================")

    with requests.get(url, stream=True, headers=headers, timeout=timeout, verify=verify_arg) as r:
        r.raise_for_status()
        total = r.headers.get("Content-Length")
        total_size = int(total) + resume_pos if total else None

        mode = "ab" if resume_pos > 0 else "wb"
        with open(tmp, mode) as f, tqdm(
            total=total_size,
            initial=resume_pos,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {dst.name}",
        ) as pbar:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    tmp.rename(dst)
    print(f"[OK] Downloaded: {dst.resolve()}")

# =============================================================================
# Extract zip
# =============================================================================
def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Extracting {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    print("[OK] Extraction done.")

# =============================================================================
# Find dataset root robustly
# =============================================================================
def find_dataset_root(raw_dir: Path) -> Tuple[Path, Path]:
    images_dirs = [p for p in raw_dir.rglob("images") if p.is_dir()]
    labels_dirs = [p for p in raw_dir.rglob("labels") if p.is_dir()]

    best: Optional[Tuple[Path, Path]] = None
    best_score = -1

    for img in images_dirs:
        for lbl in labels_dirs:
            if img.parent == lbl.parent:
                score = len(img.parts)
                try:
                    score += 10 if any(img.iterdir()) else 0
                    score += 10 if any(lbl.iterdir()) else 0
                except Exception:
                    pass
                if score > best_score:
                    best_score = score
                    best = (img, lbl)

    if best is not None:
        return best

    img_pick = next((p for p in images_dirs if any(p.iterdir())), None) or (images_dirs[0] if images_dirs else None)
    lbl_pick = next((p for p in labels_dirs if any(p.iterdir())), None) or (labels_dirs[0] if labels_dirs else None)

    if img_pick and lbl_pick:
        return (img_pick, lbl_pick)

    raise RuntimeError(
        "Could not find dataset 'images' and 'labels' folders under raw/. "
        "Open raw/ and confirm extracted structure."
    )

def detect_splits(images_dir: Path, labels_dir: Path) -> List[Tuple[str, Path, Path]]:
    img_subdirs = {p.name: p for p in images_dir.iterdir() if p.is_dir()}
    lbl_subdirs = {p.name: p for p in labels_dir.iterdir() if p.is_dir()}

    common = sorted(set(img_subdirs.keys()) & set(lbl_subdirs.keys()))
    splits: List[Tuple[str, Path, Path]] = []

    if common:
        for name in common:
            splits.append((name, img_subdirs[name], lbl_subdirs[name]))
        return splits

    img_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
    lbl_files = list(labels_dir.glob("*.txt"))

    if img_files and lbl_files:
        return [("all", images_dir, labels_dir)]

    alias = {"val": "valid", "valid": "val"}
    for a, b in alias.items():
        if a in img_subdirs and b in lbl_subdirs:
            splits.append((a, img_subdirs[a], lbl_subdirs[b]))
        if a in lbl_subdirs and b in img_subdirs:
            splits.append((b, img_subdirs[b], lbl_subdirs[a]))

    if splits:
        return splits

    raise RuntimeError(
        f"Could not detect splits.\n"
        f"Images dir: {images_dir}\n"
        f"Labels dir: {labels_dir}\n"
        f"Found image subdirs: {sorted(img_subdirs.keys())}\n"
        f"Found label subdirs: {sorted(lbl_subdirs.keys())}"
    )

# =============================================================================
# Write aligned TXT (same columns as CSV)
# =============================================================================
def write_aligned_text(df: pd.DataFrame, out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    formatted = df.copy()
    for c in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[c]):
            formatted[c] = formatted[c].map(lambda v: f"{v: .6f}" if pd.notnull(v) else " NaN")

    widths: List[int] = []
    for c in formatted.columns:
        col_vals = formatted[c].astype(str).tolist()
        w = max(len(str(c)), *(len(v) for v in col_vals))
        widths.append(w)

    def fmt_row(vals: List[str]) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))

    lines_out: List[str] = []
    lines_out.append(fmt_row([str(c) for c in formatted.columns]))
    lines_out.append(fmt_row(["-" * len(str(c)) for c in formatted.columns]))

    for _, row in formatted.iterrows():
        lines_out.append(fmt_row([str(v) for v in row.tolist()]))

    out_txt.write_text("\n".join(lines_out) + "\n", encoding="utf-8")

# =============================================================================
# YOLO pose parsing
# =============================================================================
def parse_yolo_pose_line(tokens: List[str], k: int):
    if len(tokens) < 5 + 3 * k:
        raise ValueError(f"Line too short: got {len(tokens)} tokens, expected >= {5 + 3*k}")

    cls = int(float(tokens[0]))
    xc = float(tokens[1]); yc = float(tokens[2]); bw = float(tokens[3]); bh = float(tokens[4])

    kpts = []
    base = 5
    for i in range(k):
        x = float(tokens[base + 3*i + 0])
        y = float(tokens[base + 3*i + 1])
        v = float(tokens[base + 3*i + 2])
        kpts.append((x, y, v))
    return cls, xc, yc, bw, bh, kpts

def to_pixels(xn: float, yn: float, w: int, h: int):
    return xn * w, yn * h

# =============================================================================
# Build output columns list (CSV and TXT identical)
# =============================================================================
def build_output_columns(include_v: bool) -> List[str]:
    cols: List[str] = []
    cols.append("motion_type")
    cols.append("seq")

    for internal_name, (abbr, yn) in META_MAP.items():
        if yn.strip().upper() == "Y":
            cols.append(abbr)

    for kpt_name in KPT_NAMES:
        if kpt_name not in KPT_MAP:
            continue
        abbr, yn = KPT_MAP[kpt_name]
        if yn.strip().upper() != "Y":
            continue
        cols.append(f"{abbr}_x")
        cols.append(f"{abbr}_y")
        if include_v:
            cols.append(f"{abbr}_v")

    return cols

# =============================================================================
# INVALID-DATA FILTER HELPERS
# =============================================================================
def _is_nan(x: Any) -> bool:
    try:
        return pd.isna(x) or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return True

def _valid_xy(x: float, y: float, W: Optional[int], H: Optional[int], reject_oob: bool) -> bool:
    if _is_nan(x) or _is_nan(y):
        return False
    if abs(float(x)) < 1e-12 and abs(float(y)) < 1e-12:
        return False
    if reject_oob and W is not None and H is not None:
        if x < 0 or y < 0 or x > W or y > H:
            return False
    return True

def y_valid_stats_for_row(
    row: Dict[str, Any],
    include_v: bool,
    W: Optional[int],
    H: Optional[int],
    reject_oob: bool,
    vis_thr: float,
) -> Tuple[int, int, float, List[str]]:
    """
    Compute validity only for the 'Y' keypoints (the output-selected ones).

    Returns:
      valid_count, total_y, percent_valid, bad_abbr_list
    """
    y_abbrs = selected_Y_abbrs()
    total = len(y_abbrs)
    if total == 0:
        return 0, 0, 0.0, ["(no Y keypoints selected)"]

    valid = 0
    bad: List[str] = []

    for abbr in y_abbrs:
        x = row.get(f"{abbr}_x", np.nan)
        y = row.get(f"{abbr}_y", np.nan)
        ok_xy = _valid_xy(float(x), float(y), W, H, reject_oob=reject_oob)

        ok_v = True
        if include_v:
            v = row.get(f"{abbr}_v", np.nan)
            if _is_nan(v) or float(v) <= vis_thr:
                ok_v = False

        if ok_xy and ok_v:
            valid += 1
        else:
            bad.append(abbr)

    pct = 100.0 * (valid / total)
    return valid, total, pct, bad

# =============================================================================
# Build dataset (CSV + TXT)
# =============================================================================
def build_dataset(
    include_v: bool,
    y_percent_required: float,
    drop_invalid_rows: bool,
    reject_oob: bool,
    vis_thr: float,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = ARCHIVES_DIR / DOGPOSE_ZIP_NAME

    print("============================================================")
    print("[PIPELINE] dog-2D-csv-text-dataset")
    print("------------------------------------------------------------")
    print(f"RAW DIR     : {RAW_DIR}")
    print(f"OUT DIR     : {OUT_DIR}")
    print(f"CLEAN CSV   : {OUT_CSV}")
    print(f"CLEAN TXT   : {OUT_TXT}")
    print(f"include _v  : {include_v}")
    print("------------------------------------------------------------")
    print("[INVALID FILTER SETTINGS]")
    print(f"DROP_INVALID_ROWS      : {drop_invalid_rows}")
    print(f"-y (Y% required)       : {y_percent_required:.2f}%  (applies to Y keypoints only)")
    print(f"REJECT_OUT_OF_BOUNDS   : {reject_oob}")
    print(f"VISIBILITY_THRESHOLD   : {vis_thr}  (only used when --include-v)")
    print(f"Y keypoints selected   : {len(selected_Y_abbrs())} -> {selected_Y_abbrs()}")
    print("============================================================\n")

    if not zip_path.exists():
        download_file(DOGPOSE_ZIP_URL, zip_path)
    else:
        print(f"[INFO] Archive already exists: {zip_path}")

    if not any(RAW_DIR.iterdir()):
        extract_zip(zip_path, RAW_DIR)
    else:
        print(f"[INFO] raw/ not empty; skipping extraction: {RAW_DIR}")

    images_dir, labels_dir = find_dataset_root(RAW_DIR)
    print(f"[INFO] Found images dir: {images_dir}")
    print(f"[INFO] Found labels dir: {labels_dir}")

    splits = detect_splits(images_dir, labels_dir)
    print("[INFO] Detected splits:")
    for sname, idir, ldir in splits:
        print(f"  - {sname}:")
        print(f"      images: {idir}")
        print(f"      labels: {ldir}")

    rows_raw: List[Dict[str, Any]] = []
    seq_counter = 0

    kept = 0
    dropped = 0
    drop_examples: List[str] = []

    for split, img_dir, lbl_dir in splits:
        label_files = sorted(lbl_dir.glob("*.txt"))
        print(f"[INFO] Split={split}: {len(label_files)} label files")

        for lf in tqdm(label_files, desc=f"Parsing {split}", unit="file"):
            img_path = None
            for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
                cand = img_dir / (lf.stem + ext)
                if cand.exists():
                    img_path = cand
                    break
            if img_path is None:
                continue

            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception:
                continue

            text = lf.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue

            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            for obj_i, ln in enumerate(lines):
                toks = ln.split()
                try:
                    _, xc, yc, bw, bh, kpts = parse_yolo_pose_line(toks, k=len(KPT_NAMES))
                except Exception:
                    continue

                seq_id = f"sq{seq_counter}"
                seq_counter += 1

                r: Dict[str, Any] = {}
                r["motion_type"] = "unknown"
                r["seq"] = seq_id

                # Meta
                r["split"] = split
                r["image_file"] = img_path.name
                r["image_path"] = img_path.as_posix()
                r["object_index"] = obj_i
                r["image_width"] = W
                r["image_height"] = H

                # bbox pixels
                xcp, ycp = to_pixels(xc, yc, W, H)
                r["bbox_xc_px"] = xcp
                r["bbox_yc_px"] = ycp
                r["bbox_w_px"] = bw * W
                r["bbox_h_px"] = bh * H

                # Keypoints internal storage: store for ALL keypoints so Y validity can be checked
                for name, (xn, yn, v) in zip(KPT_NAMES, kpts):
                    if name not in KPT_MAP:
                        continue
                    abbr, _ = KPT_MAP[name]
                    xp, yp = to_pixels(xn, yn, W, H)
                    r[f"{abbr}_x"] = xp
                    r[f"{abbr}_y"] = yp
                    r[f"{abbr}_v"] = v

                # ✅ Drop invalid by % of Y keypoints
                if drop_invalid_rows:
                    valid_k, total_k, pct, bad_list = y_valid_stats_for_row(
                        r,
                        include_v=include_v,
                        W=W,
                        H=H,
                        reject_oob=reject_oob,
                        vis_thr=vis_thr,
                    )
                    if pct + 1e-9 < y_percent_required:  # numeric tolerance
                        dropped += 1
                        if len(drop_examples) < 15:
                            drop_examples.append(
                                f"drop seq={seq_id} split={split} Y_valid={valid_k}/{total_k} ({pct:.1f}%) bad={bad_list}"
                            )
                        continue

                rows_raw.append(r)
                kept += 1

    print("============================================================")
    print("[INVALID FILTER RESULTS]")
    print(f"Kept rows   : {kept}")
    print(f"Dropped rows: {dropped}")
    if drop_examples:
        print("Examples of dropped rows (first 15):")
        for ex in drop_examples:
            print("  -", ex)
    print("============================================================\n")

    if not rows_raw:
        raise RuntimeError(
            "No rows parsed AFTER filtering.\n"
            "Try lowering -y (required percent), or disable -y filtering by setting -y 0.\n"
            "Also consider running without --include-v (less strict).\n"
        )

    df_raw = pd.DataFrame(rows_raw)

    out_cols = build_output_columns(include_v=include_v)

    clean = pd.DataFrame()
    clean["motion_type"] = df_raw["motion_type"] if "motion_type" in df_raw.columns else "unknown"
    clean["seq"] = df_raw["seq"] if "seq" in df_raw.columns else [f"sq{i}" for i in range(len(df_raw))]

    # Meta fields included
    for internal_name, (abbr, yn) in META_MAP.items():
        if yn.strip().upper() != "Y":
            continue
        clean[abbr] = df_raw[internal_name] if internal_name in df_raw.columns else np.nan

    # Keypoint fields included (ONLY Y)
    for kpt_name in KPT_NAMES:
        if kpt_name not in KPT_MAP:
            continue
        abbr, yn = KPT_MAP[kpt_name]
        if yn.strip().upper() != "Y":
            continue

        for suffix in ["x", "y"]:
            col = f"{abbr}_{suffix}"
            clean[col] = df_raw[col] if col in df_raw.columns else np.nan

        if include_v:
            vcol = f"{abbr}_v"
            clean[vcol] = df_raw[vcol] if vcol in df_raw.columns else np.nan

    # Strict reorder
    for c in out_cols:
        if c not in clean.columns:
            clean[c] = np.nan
    clean = clean[out_cols]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUT_CSV, index=False)
    write_aligned_text(clean, OUT_TXT)

    print("\n[DONE]")
    print(f"CLEAN CSV : {OUT_CSV}")
    print(f"CLEAN TXT : {OUT_TXT}")
    print(f"ROWS      : {len(clean)}")
    print("\n[NOTE] Filtering was applied BEFORE writing output.")

# =============================================================================
# Read TXT line-by-line
# =============================================================================
def read_txt_line_by_line(txt_path: Path, limit_lines: Optional[int], dump_to: Optional[Path]) -> None:
    if not txt_path.exists():
        raise FileNotFoundError(f"TXT file not found: {txt_path}")

    out_fh = None
    try:
        if dump_to is not None:
            dump_to.parent.mkdir(parents=True, exist_ok=True)
            out_fh = open(dump_to, "w", encoding="utf-8")

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, start=1):
                print(line.rstrip("\n"))
                if out_fh:
                    out_fh.write(line)
                if limit_lines is not None and i >= limit_lines:
                    break
    finally:
        if out_fh:
            out_fh.close()

# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        prog="install-dog-2D-csv-text-dataset.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Build a clean Dog-Pose CSV + aligned TXT.\n"
            "Filter rows by % of VALID keypoints among SELECTED ('Y') keypoints.\n\n"
            "Examples:\n"
            "  python3 install-dog-2D-csv-text-dataset.py -y 80\n"
            "  python3 install-dog-2D-csv-text-dataset.py -y 100 --include-v\n"
            "  python3 install-dog-2D-csv-text-dataset.py -r -f 50\n"
        ),
    )

    ap.add_argument("--include-v", action="store_true", help="Include *_v columns (visibility) in output")

    # ✅ YOUR REQUEST: -y %x  (percentage threshold applied to ONLY Y elements)
    ap.add_argument(
        "-y",
        type=float,
        default=Y_VALID_PERCENT_REQUIRED_DEFAULT,
        help=(
            "Minimum percent of VALID keypoints among SELECTED ('Y') keypoints required per row.\n"
            "Example: -y 80 keeps rows with >=80%% valid among Y keypoints.\n"
            "         -y 100 keeps only rows where all Y keypoints are valid.\n"
            "Range: 0..100"
        ),
    )

    ap.add_argument(
        "--no-drop",
        action="store_true",
        help="Disable filtering (keep all rows even if invalid).",
    )

    ap.add_argument(
        "--no-oob",
        action="store_true",
        help="Do NOT reject out-of-bounds points (only rejects NaN and (0,0)).",
    )

    ap.add_argument(
        "--vis-thr",
        type=float,
        default=VISIBILITY_THRESHOLD_DEFAULT,
        help="Visibility threshold used ONLY with --include-v. Invalid if v <= threshold. Default: 0.0",
    )

    ap.add_argument("-r", action="store_true", help="Read existing aligned TXT line-by-line (build first if missing)")
    ap.add_argument(
        "-f",
        type=int,
        default=0,
        help="With -r: number of lines to read; also write them to ./dog-2D-csv-text-dataset/dog-N",
    )

    # -h / --help is provided automatically by argparse
    args = ap.parse_args()

    # Validate -y
    if args.y < 0 or args.y > 100:
        raise ValueError("'-y' must be between 0 and 100")

    if args.r:
        if not OUT_TXT.exists():
            build_dataset(
                include_v=args.include_v,
                y_percent_required=args.y,
                drop_invalid_rows=(not args.no_drop),
                reject_oob=(not args.no_oob),
                vis_thr=args.vis_thr,
            )

        limit = args.f if args.f and args.f > 0 else None
        dump_to = (OUT_DIR / f"dog-{args.f}") if (args.f and args.f > 0) else None
        read_txt_line_by_line(OUT_TXT, limit_lines=limit, dump_to=dump_to)

        if dump_to:
            print(f"\n[INFO] Wrote first {args.f} lines to: {dump_to}")
        return

    build_dataset(
        include_v=args.include_v,
        y_percent_required=args.y,
        drop_invalid_rows=(not args.no_drop),
        reject_oob=(not args.no_oob),
        vis_thr=args.vis_thr,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print("\n[ERROR] Script failed.")
        print(f"Reason: {ex}")
        sys.exit(1)