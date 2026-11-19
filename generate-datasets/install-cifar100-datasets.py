# ============================================================
# export_cifar100_as_raw_png_cifar_style_with_stats.py
# ============================================================
# ✅ AUTO-INSTALL:
#   If packages are missing, this script auto-installs them
#   into the SAME Python interpreter running this script.
#
# ✅ EXPORTS CIFAR-100 into CIFAR-style folders:
#
#   ./cifar100/
#       train/<class_name>/*.png
#       test/<class_name>/*.png
#       class_counts.json
#       class_counts.csv
#
# ✅ Saves per-class counts + FIRST image size (W×H) per class.
# ✅ Converts images to RGB (CIFAR-100 is already RGB, but kept for safety).
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

EXPORT_ROOT = "cifar100"               # <- you asked: directory called "cifar100"
CACHE_ROOT  = "torchvision_cache"      # download/cache here (inside script folder)

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
    CIFAR-100 images are already RGB PIL images, but we keep this consistent with your other exporters.
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(img).convert("RGB")

def save_counts(ds_root, class_names, train_counts, test_counts, first_sizes):
    """
    Saves:
      - class_counts.json
      - class_counts.csv
    Includes:
      - train/test/total counts
      - first image size (W×H) seen for each class
    """
    stats = {}
    for cname in class_names:
        c = safe_name(cname)
        tr = int(train_counts.get(c, 0))
        te = int(test_counts.get(c, 0))
        w_h = first_sizes.get(c, None)

        if w_h is None:
            fw, fh, fwh = None, None, None
        else:
            fw, fh = int(w_h[0]), int(w_h[1])
            fwh = f"{fw}x{fh}"

        stats[c] = {
            "train": tr,
            "test": te,
            "total": tr + te,
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
        w.writerow(["class", "train", "test", "total", "first_w", "first_h", "first_wh"])
        for c, v in stats.items():
            w.writerow([c, v["train"], v["test"], v["total"], v["first_w"], v["first_h"], v["first_wh"]])

    return stats

# ============================================================
# EXPORT LOGIC
# ============================================================

def export_split(split_name, ds, out_root, class_names, counter, first_sizes, prefix):
    it = range(len(ds))
    if tqdm is not None:
        it = tqdm(it, desc=f"{prefix}:{split_name}", unit="img")

    for i in it:
        img, label = ds[i]
        y = int(label)

        cname_raw = class_names[y]
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
    cache_root  = os.path.join(base, CACHE_ROOT)
    export_root = os.path.join(base, EXPORT_ROOT)

    ensure_dir(cache_root)
    ensure_dir(export_root)

    print("[INFO] Python executable:", sys.executable)
    print("[INFO] Cache root:", cache_root)
    print("[INFO] Export root:", export_root)

    # ------------------------------------------------------------
    # DOWNLOAD CIFAR-100 (train + test)
    # ------------------------------------------------------------
    train_ds = datasets.CIFAR100(root=cache_root, train=True, download=True)
    test_ds  = datasets.CIFAR100(root=cache_root, train=False, download=True)

    # Class names (CIFAR-100 built-in)
    class_names = list(train_ds.classes)

    # Build folder tree
    train_root = os.path.join(export_root, "train")
    test_root  = os.path.join(export_root, "test")
    ensure_dir(train_root)
    ensure_dir(test_root)

    for cname in class_names:
        ensure_dir(os.path.join(train_root, safe_name(cname)))
        ensure_dir(os.path.join(test_root,  safe_name(cname)))

    # Counters + size tracking
    train_counts = defaultdict(int)
    test_counts  = defaultdict(int)
    first_sizes  = {}  # class_name -> (W,H), from first seen image in that class

    # Export images
    export_split("train", train_ds, train_root, class_names, train_counts, first_sizes, prefix="cifar100")
    export_split("test",  test_ds,  test_root,  class_names, test_counts,  first_sizes, prefix="cifar100")

    # Save stats files
    stats = save_counts(export_root, class_names, train_counts, test_counts, first_sizes)

    # Print summary
    print("\n" + "=" * 80)
    print("✅ EXPORTED: CIFAR-100")
    for c, v in stats.items():
        print(
            f"{c:20s} | train={v['train']:6d} | test={v['test']:6d} | total={v['total']:6d} | first_WxH={v['first_wh']}"
        )
    print("=" * 80)
    print("📁 Output folder:", export_root)
    print("✅ Done.")

if __name__ == "__main__":
    main()
