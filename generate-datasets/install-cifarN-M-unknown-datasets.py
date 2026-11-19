# ==========================================================
# COMPLETE CROSS-PLATFORM AUTO-INSTALL ROUTINE
# ==========================================================
# PURPOSE
# -------
# This section:
#
#   1) Checks whether pip is available
#   2) Installs pip if needed
#   3) Installs required third-party Python packages
#   4) Verifies imports after installation
#cd P
# It is designed to work on:
#
#   • Windows
#   • Ubuntu / Linux
#   • macOS
#
# IMPORTANT
# ---------
# Put this section at the TOP of your script BEFORE importing
# third-party packages such as:
#
#   import requests
#   import torch
#   import torchvision
#
# Standard library modules do NOT need pip installation.
# ==========================================================

import sys
import subprocess
import importlib
import urllib.request
import tempfile
import os


def ensure_pip_available():
    """
    Ensure that pip is available for the CURRENT Python interpreter.

    CHECK ORDER
    -----------
    1) Try:
           python -m pip --version
       If this works, pip already exists.

    2) Try:
           python -m ensurepip --upgrade
       This works on many Python installations.

    3) Fallback:
           download get-pip.py and run it
       This requires internet access.

    RETURNS
    -------
    True  -> pip is available
    False -> pip could not be installed
    """

    # ------------------------------------------------------
    # STEP 1 — Check if pip already exists
    # ------------------------------------------------------
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )

        if result.returncode == 0:
            print("[OK] pip is already available.")
            return True

    except Exception:
        pass

    print("[INFO] pip was not found.")
    print("[INFO] Trying to install pip using ensurepip...")

    # ------------------------------------------------------
    # STEP 2 — Try ensurepip
    # ------------------------------------------------------
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            check=False
        )

        if result.returncode == 0:
            verify = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )

            if verify.returncode == 0:
                print("[OK] pip installed successfully using ensurepip.")
                return True

    except Exception as e:
        print(f"[WARNING] ensurepip failed: {e}")

    print("[INFO] ensurepip did not work.")
    print("[INFO] Trying fallback installation using get-pip.py ...")

    # ------------------------------------------------------
    # STEP 3 — Fallback to get-pip.py
    # ------------------------------------------------------
    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp_file:
            temp_path = tmp_file.name

        print(f"[INFO] Downloading get-pip.py from: {get_pip_url}")
        urllib.request.urlretrieve(get_pip_url, temp_path)

        print("[INFO] Running get-pip.py ...")
        result = subprocess.run(
            [sys.executable, temp_path],
            check=False
        )

        if result.returncode != 0:
            print("[ERROR] get-pip.py failed.")
            return False

        verify = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )

        if verify.returncode == 0:
            print("[OK] pip installed successfully using get-pip.py.")
            return True

        print("[ERROR] pip still not available after get-pip.py.")
        return False

    except Exception as e:
        print(f"[ERROR] Failed to install pip automatically: {e}")
        return False

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def ensure_python_package(import_name, pip_name=None):
    """
    Ensure that one Python package is installed and importable.

    PARAMETERS
    ----------
    import_name : str
        Module name used by Python import.

        Example:
            import_name = "requests"
            import_name = "PIL"
            import_name = "cv2"

    pip_name : str or None
        Package name used by pip.

        Example:
            pip_name = "requests"
            pip_name = "pillow"
            pip_name = "opencv-python"

        If omitted, pip_name defaults to import_name.

    RETURNS
    -------
    True  -> package is available
    False -> installation failed
    """

    if pip_name is None:
        pip_name = import_name

    # ------------------------------------------------------
    # STEP 1 — Try importing first
    # ------------------------------------------------------
    try:
        importlib.import_module(import_name)
        print(f"[OK] Python package already installed: {pip_name}")
        return True

    except ImportError:
        print(f"[INFO] Missing Python package: {pip_name}")

    # ------------------------------------------------------
    # STEP 2 — Make sure pip exists
    # ------------------------------------------------------
    if not ensure_pip_available():
        print("[ERROR] pip is not available, so package installation cannot continue.")
        return False

    print(f"[INFO] Installing Python package: {pip_name}")

    # ------------------------------------------------------
    # STEP 3 — Install package using current Python
    # ------------------------------------------------------
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", pip_name],
            check=False
        )

        if result.returncode != 0:
            print(f"[ERROR] Failed to install Python package: {pip_name}")
            return False

    except Exception as e:
        print(f"[ERROR] Exception while installing package '{pip_name}': {e}")
        return False

    # ------------------------------------------------------
    # STEP 4 — Verify import after installation
    # ------------------------------------------------------
    try:
        importlib.import_module(import_name)
        print(f"[OK] Installed Python package successfully: {pip_name}")
        return True

    except ImportError:
        print(f"[ERROR] Package installed but still cannot be imported: {pip_name}")
        return False


def ensure_required_python_packages():
    """
    Add ALL third-party packages required by your script here.

    IMPORTANT
    ---------
    Add only packages that need pip.
    Do NOT add standard library modules such as:
        os, sys, json, time, shutil, subprocess, re, math, tempfile

    EXAMPLES
    --------
    For a simple chatbot script:
        ("requests", "requests")

    For computer vision:
        ("cv2", "opencv-python")
        ("PIL", "pillow")

    For PyTorch:
        ("torch", "torch")
        ("torchvision", "torchvision")
        ("torchaudio", "torchaudio")
    """

    required_packages = [
        ("requests", "requests"),
    ]

    for import_name, pip_name in required_packages:
        if not ensure_python_package(import_name, pip_name):
            print("[ERROR] Cannot continue because a required package is missing.")
            sys.exit(1)


# ----------------------------------------------------------
# RUN INSTALLATION NOW
# ----------------------------------------------------------
# This should execute BEFORE importing third-party packages.
# ----------------------------------------------------------
ensure_required_python_packages()

# ============================================================
# export_cifar100_as_raw_png_cifar_style_with_stats.py
# ============================================================
# AUTO-INSTALL:
#   If packages are missing, this script auto-installs them
#   into the SAME Python interpreter running this script.
#
# EXPORTS CIFAR-100 into CIFAR-style folders (RANGE MODE)
# WITH AN EXTRA CLASS CALLED: "unknown"
#
#   User selects a class RANGE:
#     - Beginning class index: N
#     - Ending class index:     M
#   (Indices are in the OFFICIAL CIFAR-100 class order, 0..99)
#
# NEW "unknown" CLASS BEHAVIOR:
#   • A new class folder named "unknown" is added to:
#         train/
#         test/
#   • The unknown class contains X% of images sampled from CIFAR-100
#     classes that are NOT included between N and M.
#   • Default unknown percentage = 10%
#
# OUTPUT DIRECTORY (UPDATED AS REQUESTED):
#   ./cifar_N_M_unknown/
#
#   Inside it:
#     train/<selected_class_name>/*.png
#     train/unknown/*.png
#
#     test/<selected_class_name>/*.png
#     test/unknown/*.png
#
#     nottrained_test/<NOT-selected_class_name>/*.png
#
#     class_counts.json
#     class_counts.csv
#
# IMPORTANT BEHAVIOR:
#   • "train" contains ONLY selected classes (N..M) + unknown
#   • "test" contains ONLY selected classes (N..M) + unknown
#   • "nottrained_test" contains ALL OTHER CIFAR-100 classes NOT in (N..M),
#     exported from the CIFAR-100 TEST split only
#   • "unknown" is built from a sampled subset of NON-selected classes
#
# STATS:
#   Saves per-class counts + FIRST image size (W×H) per class
#   across ALL exported folders (train/test/nottrained_test)
#
# CONSOLE SUMMARY:
#   Prints only the selected classes plus "unknown"
#   in a simple format:
#
#       class_name : train=X  test=Y  total=Z
#
# ============================================================

import os
import sys
import json
import csv
import math
import random
import subprocess
from collections import defaultdict

# ============================================================
# AUTO-INSTALL HELPERS
# ============================================================

def _pip_install(packages):
    """
    Install packages into the CURRENT Python interpreter environment.
    This ensures there is no mismatch between 'pip' and 'python'.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    print("\n[AUTO-INSTALL] Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def _ensure_import(import_name, pip_name=None):
    """
    Try import. If missing, auto-install and import again.
    """
    try:
        __import__(import_name)
    except Exception:
        pip_name = pip_name or import_name
        print(f"[AUTO-INSTALL] Missing '{import_name}'. Installing '{pip_name}' ...")
        _pip_install([pip_name])
        __import__(import_name)


# Required packages
_ensure_import("PIL", "pillow")
_ensure_import("torch", "torch")
_ensure_import("torchvision", "torchvision")

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # optional

from PIL import Image
from torchvision import datasets

# ============================================================
# CONFIG
# ============================================================

CACHE_ROOT = "torchvision_cache"   # download/cache here (inside script folder)
DEFAULT_UNKNOWN_PERCENT = 10.0     # default requested value
RANDOM_SEED = 42                   # reproducible unknown sampling

# ============================================================
# HELPERS
# ============================================================

def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def safe_name(s):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(s))


def to_rgb_pil(img):
    """
    Forces RGB output (safety).
    CIFAR-100 images are already RGB PIL images, but we keep this consistent.
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(img).convert("RGB")


def prompt_class_range(max_classes=100):
    """
    Prompt user for beginning and ending class indices (inclusive).
    - Indices are in official CIFAR-100 order: 0..(max_classes-1)
    - Clamps and fixes ordering if needed.
    - Empty input defaults to full range [0..max_classes-1]
    """
    while True:
        try:
            raw_n = input(
                f"\n[INPUT] Beginning class index N (0-{max_classes-1}) [default=0]: "
            ).strip()
            raw_m = input(
                f"[INPUT] Ending class index M (0-{max_classes-1}) [default={max_classes-1}]: "
            ).strip()

            n = 0 if raw_n == "" else int(raw_n)
            m = (max_classes - 1) if raw_m == "" else int(raw_m)

            if n < 0:
                print("[WARN] N < 0. Using 0.")
                n = 0
            if m < 0:
                print("[WARN] M < 0. Using 0.")
                m = 0
            if n > max_classes - 1:
                print(f"[WARN] N > {max_classes-1}. Using {max_classes-1}.")
                n = max_classes - 1
            if m > max_classes - 1:
                print(f"[WARN] M > {max_classes-1}. Using {max_classes-1}.")
                m = max_classes - 1

            if n > m:
                print(f"[WARN] N ({n}) > M ({m}). Swapping.")
                n, m = m, n

            return n, m
        except Exception:
            print("[WARN] Invalid input. Please enter integers (e.g., 0, 9, 20, 99).")


def prompt_unknown_percent(default_percent=10.0):
    """
    Prompt user for percentage of NON-selected images to place into
    the new class called 'unknown'.

    Returns:
      float percent in [0,100]
    """
    while True:
        try:
            raw = input(
                f"[INPUT] Percentage for UNKNOWN class from NON-selected classes [default={default_percent}%]: "
            ).strip()

            pct = default_percent if raw == "" else float(raw)

            if pct < 0:
                print("[WARN] Percentage cannot be negative. Using 0.")
                pct = 0.0
            if pct > 100:
                print("[WARN] Percentage cannot exceed 100. Using 100.")
                pct = 100.0

            return pct
        except Exception:
            print("[WARN] Invalid input. Please enter a number such as 10 or 12.5.")


def save_counts(ds_root, class_names_exported, train_counts, test_counts, nottrained_counts, first_sizes):
    """
    Saves:
      - class_counts.json
      - class_counts.csv

    Includes:
      - train/test/nottrained_test/total counts
      - first image size (W×H) seen for each class
    """
    stats = {}
    for cname_raw in class_names_exported:
        cname = safe_name(cname_raw)

        tr  = int(train_counts.get(cname, 0))
        te  = int(test_counts.get(cname, 0))
        ntt = int(nottrained_counts.get(cname, 0))
        w_h = first_sizes.get(cname, None)

        if w_h is None:
            fw, fh, fwh = None, None, None
        else:
            fw, fh = int(w_h[0]), int(w_h[1])
            fwh = f"{fw}x{fh}"

        stats[cname] = {
            "train": tr,
            "test": te,
            "nottrained_test": ntt,
            "total": tr + te + ntt,
            "first_w": fw,
            "first_h": fh,
            "first_wh": fwh
        }

    with open(os.path.join(ds_root, "class_counts.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    with open(os.path.join(ds_root, "class_counts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "train", "test", "nottrained_test", "total", "first_w", "first_h", "first_wh"])
        for c, v in stats.items():
            w.writerow([c, v["train"], v["test"], v["nottrained_test"], v["total"], v["first_w"], v["first_h"], v["first_wh"]])

    return stats


def collect_indices_by_selection(ds, selected_label_set, keep_selected=True):
    """
    Build a list of dataset indices by label membership.

    Parameters:
      - keep_selected:
            True  -> keep only selected labels
            False -> keep only NON-selected labels
    """
    out = []
    for i in range(len(ds)):
        _, label = ds[i]
        y = int(label)
        in_set = (y in selected_label_set)

        if keep_selected:
            if in_set:
                out.append(i)
        else:
            if not in_set:
                out.append(i)

    return out


def sample_unknown_indices(ds, selected_label_set, unknown_percent):
    """
    Sample indices for the synthetic "unknown" class from the complement set
    (classes NOT in N..M).
    """
    complement_indices = collect_indices_by_selection(
        ds=ds,
        selected_label_set=selected_label_set,
        keep_selected=False
    )

    total_complement = len(complement_indices)

    if total_complement == 0 or unknown_percent <= 0:
        return [], total_complement, 0

    sample_count = int(math.ceil((unknown_percent / 100.0) * total_complement))
    sample_count = max(0, min(sample_count, total_complement))

    rng = random.Random(RANDOM_SEED)
    sampled = rng.sample(complement_indices, sample_count)

    return sampled, total_complement, sample_count


# ============================================================
# EXPORT LOGIC
# ============================================================

def export_split_by_label_set(split_name, ds, out_root, class_names_all, allowed_label_set,
                             counter, first_sizes, prefix, skip_if_label_not_allowed=True):
    """
    Export images based on label filtering.

    Parameters:
      - split_name: "train" / "test" / "nottrained_test"
      - skip_if_label_not_allowed:
          True  -> export ONLY labels in allowed_label_set
          False -> export ONLY labels NOT in allowed_label_set
    """
    it = range(len(ds))
    if tqdm is not None:
        it = tqdm(it, desc=f"{prefix}:{split_name}", unit="img")

    for i in it:
        img, label = ds[i]
        y = int(label)

        in_set = (y in allowed_label_set)
        if skip_if_label_not_allowed:
            if not in_set:
                continue
        else:
            if in_set:
                continue

        cname_raw = class_names_all[y]
        cname = safe_name(cname_raw)

        img = to_rgb_pil(img)

        if cname not in first_sizes:
            w, h = img.size
            first_sizes[cname] = (w, h)

        fname = f"{prefix}_{split_name}_{i:06d}.png"
        img.save(os.path.join(out_root, cname, fname))

        counter[cname] += 1


def export_unknown_split(split_name, ds, out_root, sampled_indices, counter, first_sizes, prefix):
    """
    Export a sampled subset of NON-selected images into a synthetic class
    named "unknown".
    """
    unknown_cname = safe_name("unknown")

    it = sampled_indices
    if tqdm is not None:
        it = tqdm(it, desc=f"{prefix}:{split_name}:unknown", unit="img")

    for idx in it:
        img, _ = ds[idx]
        img = to_rgb_pil(img)

        if unknown_cname not in first_sizes:
            w, h = img.size
            first_sizes[unknown_cname] = (w, h)

        fname = f"{prefix}_{split_name}_unknown_{idx:06d}.png"
        img.save(os.path.join(out_root, unknown_cname, fname))

        counter[unknown_cname] += 1


def print_selected_plus_unknown_counts(class_names_selected, train_counts, test_counts):
    """
    Print only the selected classes plus unknown in a simple format:

        class_name : train=X  test=Y  total=Z
    """
    print("\n" + "=" * 70)
    print("SELECTED CLASSES INCLUDING UNKNOWN")
    print("=" * 70)

    for cname_raw in class_names_selected:
        cname = safe_name(cname_raw)
        tr = int(train_counts.get(cname, 0))
        te = int(test_counts.get(cname, 0))
        total_selected = tr + te
        print(f"{cname:20s} : train={tr:<6d} test={te:<6d} total={total_selected}")

    unknown_name = safe_name("unknown")
    unknown_tr = int(train_counts.get(unknown_name, 0))
    unknown_te = int(test_counts.get(unknown_name, 0))
    unknown_total = unknown_tr + unknown_te
    print(f"{unknown_name:20s} : train={unknown_tr:<6d} test={unknown_te:<6d} total={unknown_total}")

    print("=" * 70)


def main():
    base = script_dir()
    cache_root = os.path.join(base, CACHE_ROOT)
    ensure_dir(cache_root)

    print("[INFO] Python executable:", sys.executable)
    print("[INFO] Cache root:", cache_root)

    # ------------------------------------------------------------
    # DOWNLOAD CIFAR-100 (train + test)
    # ------------------------------------------------------------
    train_ds = datasets.CIFAR100(root=cache_root, train=True, download=True)
    test_ds  = datasets.CIFAR100(root=cache_root, train=False, download=True)

    # Class names (official CIFAR-100 order)
    class_names_all = list(train_ds.classes)  # length = 100

    # ------------------------------------------------------------
    # PROMPT USER
    # ------------------------------------------------------------
    n, m = prompt_class_range(max_classes=len(class_names_all))
    unknown_percent = prompt_unknown_percent(default_percent=DEFAULT_UNKNOWN_PERCENT)

    # ------------------------------------------------------------
    # Selected classes = indices N..M (inclusive)
    # Not-trained classes = all others
    # ------------------------------------------------------------
    selected_label_set = set(range(n, m + 1))
    class_names_selected = [class_names_all[i] for i in range(n, m + 1)]
    class_names_nottrained = [
        class_names_all[i]
        for i in range(len(class_names_all))
        if i not in selected_label_set
    ]

    # ------------------------------------------------------------
    # OUTPUT DIRECTORY NAMING
    #   ./cifar_N_M_unknown/
    # ------------------------------------------------------------
    export_root_name = f"cifar_{n}_{m}_unknown"
    export_root = os.path.join(base, export_root_name)

    print("\n" + "=" * 100)
    print(f"[INFO] Exporting CIFAR-100 classes in range: N={n} .. M={m} (inclusive)")
    print(f"[INFO] Unknown percentage from NON-selected classes: {unknown_percent:.2f}%")
    print(f"[INFO] Output directory: {export_root}")
    print("-" * 100)
    print("[INFO] Selected classes:")
    for idx in range(n, m + 1):
        print(f"  {idx:02d}: {class_names_all[idx]}")
    print("-" * 100)
    print("[INFO] Extra class:")
    print("  unknown")
    print("=" * 100 + "\n")

    ensure_dir(export_root)

    # ------------------------------------------------------------
    # Build folder tree:
    #   train/ selected + unknown
    #   test/ selected + unknown
    #   nottrained_test/ NOT selected
    # ------------------------------------------------------------
    train_root = os.path.join(export_root, "train")
    test_root  = os.path.join(export_root, "test")
    ntt_root   = os.path.join(export_root, "nottrained_test")

    ensure_dir(train_root)
    ensure_dir(test_root)
    ensure_dir(ntt_root)

    # Create class folders for selected classes (train/test)
    for cname_raw in class_names_selected:
        cname = safe_name(cname_raw)
        ensure_dir(os.path.join(train_root, cname))
        ensure_dir(os.path.join(test_root,  cname))

    # Create unknown folder (train/test)
    ensure_dir(os.path.join(train_root, safe_name("unknown")))
    ensure_dir(os.path.join(test_root,  safe_name("unknown")))

    # Create class folders for NOT-trained classes (nottrained_test)
    for cname_raw in class_names_nottrained:
        cname = safe_name(cname_raw)
        ensure_dir(os.path.join(ntt_root, cname))

    # ------------------------------------------------------------
    # Counters + size tracking
    # ------------------------------------------------------------
    train_counts = defaultdict(int)
    test_counts  = defaultdict(int)
    nottrained_counts = defaultdict(int)
    first_sizes  = {}

    # ------------------------------------------------------------
    # Export selected classes
    # ------------------------------------------------------------
    export_split_by_label_set(
        split_name="train",
        ds=train_ds,
        out_root=train_root,
        class_names_all=class_names_all,
        allowed_label_set=selected_label_set,
        counter=train_counts,
        first_sizes=first_sizes,
        prefix="cifar100",
        skip_if_label_not_allowed=True
    )

    export_split_by_label_set(
        split_name="test",
        ds=test_ds,
        out_root=test_root,
        class_names_all=class_names_all,
        allowed_label_set=selected_label_set,
        counter=test_counts,
        first_sizes=first_sizes,
        prefix="cifar100",
        skip_if_label_not_allowed=True
    )

    # ------------------------------------------------------------
    # Export all NON-selected classes to nottrained_test
    # ------------------------------------------------------------
    export_split_by_label_set(
        split_name="nottrained_test",
        ds=test_ds,
        out_root=ntt_root,
        class_names_all=class_names_all,
        allowed_label_set=selected_label_set,
        counter=nottrained_counts,
        first_sizes=first_sizes,
        prefix="cifar100",
        skip_if_label_not_allowed=False
    )

    # ------------------------------------------------------------
    # Build unknown from NON-selected classes
    # ------------------------------------------------------------
    train_unknown_indices, _, _ = sample_unknown_indices(
        ds=train_ds,
        selected_label_set=selected_label_set,
        unknown_percent=unknown_percent
    )

    test_unknown_indices, _, _ = sample_unknown_indices(
        ds=test_ds,
        selected_label_set=selected_label_set,
        unknown_percent=unknown_percent
    )

    export_unknown_split(
        split_name="train",
        ds=train_ds,
        out_root=train_root,
        sampled_indices=train_unknown_indices,
        counter=train_counts,
        first_sizes=first_sizes,
        prefix="cifar100"
    )

    export_unknown_split(
        split_name="test",
        ds=test_ds,
        out_root=test_root,
        sampled_indices=test_unknown_indices,
        counter=test_counts,
        first_sizes=first_sizes,
        prefix="cifar100"
    )

    # ------------------------------------------------------------
    # Save stats files
    # ------------------------------------------------------------
    class_names_exported = class_names_selected + class_names_nottrained + ["unknown"]
    save_counts(
        export_root,
        class_names_exported,
        train_counts,
        test_counts,
        nottrained_counts,
        first_sizes
    )

    # ------------------------------------------------------------
    # SIMPLE SUMMARY:
    # Print ONLY selected classes + unknown
    # Format:
    #   class_name : train=X  test=Y  total=Z
    # ------------------------------------------------------------
    print_selected_plus_unknown_counts(
        class_names_selected=class_names_selected,
        train_counts=train_counts,
        test_counts=test_counts
    )

    print("\n📁 Output folder:", export_root)
    print("✅ Done.")


if __name__ == "__main__":
    main()