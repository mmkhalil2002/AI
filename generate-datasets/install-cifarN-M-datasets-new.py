# ============================================================
# export_cifar100_as_raw_png_cifar_style_with_stats.py
# ============================================================
# ✅ AUTO-INSTALL:
#   If packages are missing, this script auto-installs them
#   into the SAME Python interpreter running this script.
#
# ✅ EXPORTS CIFAR-100 into CIFAR-style folders (RANGE MODE):
#
#   User selects a class RANGE:
#     - Beginning class index: N
#     - Ending class index:     M
#   (Indices are in the OFFICIAL CIFAR-100 class order, 0..99)
#
# ✅ OUTPUT DIRECTORY (AS REQUESTED):
#   ./data_N_M/
#
#   Inside it:
#     train/<selected_class_name>/*.png
#     test/<selected_class_name>/*.png
#     nottrained_test/<NOT-selected_class_name>/*.png
#     class_counts.json
#     class_counts.csv
#
# ✅ IMPORTANT BEHAVIOR:
#   • "train" contains ONLY selected classes (N..M).
#   • "test" contains ONLY selected classes (N..M).
#   • "nottrained_test" contains ALL OTHER CIFAR-100 classes NOT in (N..M),
#     exported from the CIFAR-100 TEST split only.
#
# ✅ STATS:
#   Saves per-class counts + FIRST image size (W×H) per class
#   across ALL exported folders (train/test/nottrained_test).
# ============================================================

import os
import sys
import json
import csv
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

            # Clamp to valid range
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

            # Ensure N <= M
            if n > m:
                print(f"[WARN] N ({n}) > M ({m}). Swapping.")
                n, m = m, n

            return n, m
        except Exception:
            print("[WARN] Invalid input. Please enter integers (e.g., 0, 9, 20, 99).")

def save_counts(ds_root, class_names_exported, train_counts, test_counts, nottrained_counts, first_sizes):
    """
    Saves:
      - class_counts.json
      - class_counts.csv

    Includes:
      - train/test/nottrained_test/total counts
      - first image size (W×H) seen for each class

    NOTE:
      class_names_exported is a list of RAW class names that appear in ANY exported folder.
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

    # JSON
    with open(os.path.join(ds_root, "class_counts.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    # CSV
    with open(os.path.join(ds_root, "class_counts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "train", "test", "nottrained_test", "total", "first_w", "first_h", "first_wh"])
        for c, v in stats.items():
            w.writerow([c, v["train"], v["test"], v["nottrained_test"], v["total"], v["first_w"], v["first_h"], v["first_wh"]])

    return stats

# ============================================================
# EXPORT LOGIC
# ============================================================

def export_split_by_label_set(split_name, ds, out_root, class_names_all, allowed_label_set,
                             counter, first_sizes, prefix, skip_if_label_not_allowed=True):
    """
    Export images based on label filtering.

    Parameters:
      - split_name: "train" / "test" / "nottrained_test"
      - ds: torchvision dataset split
      - out_root: folder where class subfolders exist
      - class_names_all: full CIFAR-100 class list (len=100)
      - allowed_label_set: labels to include
      - skip_if_label_not_allowed:
          True  -> export ONLY labels in allowed_label_set
          False -> export ONLY labels NOT in allowed_label_set (i.e., complement set)

      - counter: dict counting images per class (safe_name(class_name) -> count)
      - first_sizes: dict storing first seen (W,H) per class (safe_name(class_name) -> (W,H))
    """
    it = range(len(ds))
    if tqdm is not None:
        it = tqdm(it, desc=f"{prefix}:{split_name}", unit="img")

    for i in it:
        img, label = ds[i]
        y = int(label)

        # --------------------------------------------------------
        # FILTER LOGIC:
        #   - train/test: export ONLY selected labels
        #   - nottrained_test: export ONLY NON-selected labels
        # --------------------------------------------------------
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

        # record FIRST size for this class (one time only)
        if cname not in first_sizes:
            w, h = img.size  # PIL: (W,H)
            first_sizes[cname] = (w, h)

        fname = f"{prefix}_{split_name}_{i:06d}.png"
        img.save(os.path.join(out_root, cname, fname))

        counter[cname] += 1

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

    # Class names (CIFAR-100 built-in) - official order
    class_names_all = list(train_ds.classes)  # length = 100

    # ------------------------------------------------------------
    # PROMPT USER: class range [N..M]
    # ------------------------------------------------------------
    n, m = prompt_class_range(max_classes=len(class_names_all))

    # ------------------------------------------------------------
    # Selected classes = indices N..M (inclusive)
    # Not-trained classes = all others
    # ------------------------------------------------------------
    selected_label_set = set(range(n, m + 1))
    class_names_selected = [class_names_all[i] for i in range(n, m + 1)]
    class_names_nottrained = [class_names_all[i] for i in range(len(class_names_all)) if i not in selected_label_set]

    # ------------------------------------------------------------
    # OUTPUT DIRECTORY NAMING (AS REQUESTED):
    #   ./data_N_M/
    # ------------------------------------------------------------
    export_root_name = f"cifar_{n}_{m}"
    export_root = os.path.join(base, export_root_name)

    print("\n" + "=" * 90)
    print(f"[INFO] Exporting CIFAR-100 classes in range: N={n} .. M={m} (inclusive)")
    print(f"[INFO] Output directory: {export_root}")
    print("-" * 90)
    print("[INFO] Selected (TRAINED) classes:")
    for idx in range(n, m + 1):
        print(f"  {idx:02d}: {class_names_all[idx]}")
    print("-" * 90)
    print("[INFO] Not-trained classes will be exported from TEST split into: nottrained_test/")
    print("=" * 90 + "\n")

    ensure_dir(export_root)

    # ------------------------------------------------------------
    # Build folder tree:
    #   train/ selected
    #   test/ selected
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
    first_sizes  = {}  # safe_name(class_name) -> (W,H), from first seen image in that class

    # ------------------------------------------------------------
    # Export:
    #   - train: selected labels only
    #   - test:  selected labels only
    #   - nottrained_test: NON-selected labels only (from test split)
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

    export_split_by_label_set(
        split_name="nottrained_test",
        ds=test_ds,
        out_root=ntt_root,
        class_names_all=class_names_all,
        allowed_label_set=selected_label_set,
        counter=nottrained_counts,
        first_sizes=first_sizes,
        prefix="cifar100",
        # export complement set:
        skip_if_label_not_allowed=False
    )

    # ------------------------------------------------------------
    # Save stats files (ALL exported classes)
    # ------------------------------------------------------------
    class_names_exported = class_names_selected + class_names_nottrained
    stats = save_counts(export_root, class_names_exported, train_counts, test_counts, nottrained_counts, first_sizes)

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------
    selected_total_train = sum(train_counts.values())
    selected_total_test  = sum(test_counts.values())
    nottrained_total     = sum(nottrained_counts.values())

    print("\n" + "=" * 90)
    print(f"✅ EXPORTED: CIFAR-100 RANGE MODE  (N={n} .. M={m})")
    print("-" * 90)
    print(f"[SELECTED]     train images: {selected_total_train}")
    print(f"[SELECTED]     test  images: {selected_total_test}")
    print(f"[NOT-TRAINED]  test  images: {nottrained_total}")
    print("-" * 90)
    print("Per-class summary (train/test/nottrained_test/total):")
    for c, v in stats.items():
        print(
            f"{c:20s} | train={v['train']:6d} | test={v['test']:6d} | "
            f"nottrained={v['nottrained_test']:6d} | total={v['total']:6d} | first_WxH={v['first_wh']}"
        )
    print("=" * 90)
    print("📁 Output folder:", export_root)
    print("✅ Done.")

if __name__ == "__main__":
    main()
