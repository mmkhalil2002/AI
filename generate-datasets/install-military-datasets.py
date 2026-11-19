# ============================================================
# military-datasets.py  (AUTO-INSTALL VERSION)
# ============================================================
# ✅ If packages are missing, this script auto-installs them
#   into the SAME Python interpreter running this script.
#
# Exports "military/defense-relevant" remote-sensing datasets
# (classification: 1 label per image) into CIFAR-style folders:
#
#   <EXPORT_ROOT>/<dataset_name>/
#       train/<class_name>/*.png
#       test/<class_name>/*.png
#       class_counts.json
#       class_counts.csv
#
# ✅ Converts grayscale images to RGB.
# ✅ Saves class counts (JSON + CSV).
# ✅ Creates train/test split if only one split exists.
#
# ✅ NEW (YOUR REQUEST):
#   • For EACH CLASS, record the FIRST image size (W×H) seen in that class
#   • Save it inside the statistics:
#       - class_counts.json
#       - class_counts.csv
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
    This ensures: no mismatch between 'pip' and 'python'.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    print("\n[AUTO-INSTALL] Running:", " ".join(cmd))
    subprocess.check_call(cmd)

def _ensure_import(pkg_import_name, pip_name=None):
    """
    Try import. If missing, auto-install and import again.
    """
    try:
        __import__(pkg_import_name)
        return
    except Exception:
        pip_name = pip_name or pkg_import_name
        print(f"[AUTO-INSTALL] Missing '{pkg_import_name}'. Installing '{pip_name}' ...")
        _pip_install([pip_name])
        __import__(pkg_import_name)

# Ensure required packages
_ensure_import("PIL", "pillow")
_ensure_import("datasets", "datasets")
_ensure_import("tqdm", "tqdm")


from PIL import Image
from datasets import load_dataset
from tqdm import tqdm


# ============================================================
# DATASETS (HUGGING FACE HUB)
# ============================================================
# NOTE:
# - These are remote-sensing scene classification datasets often used
#   in aerial/satellite surveillance contexts.
# - All are classification (one label per image).
# ============================================================
MILITARY_DATASETS = [
    # UC Merced Land Use
    {"hub": "blanchon/UC_Merced", "name": "uc_merced", "label_col": "label", "image_col": "image"},

    # RESISC45
    {"hub": "timm/resisc45", "name": "resisc45", "label_col": "label", "image_col": "image"},

    # PatternNet
    {"hub": "blanchon/PatternNet", "name": "patternnet", "label_col": "label", "image_col": "image"},
]

EXPORT_ROOT = "raw_military_export"


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
    Forces RGB output.
    - If image is grayscale, convert to RGB
    - If already RGB, keep as RGB
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(img).convert("RGB")

def get_class_names(ds, label_col):
    """
    HF datasets usually store label names in the 'label' feature.
    If names exist, use them. Otherwise fallback to numeric strings.
    """
    feat = ds.features.get(label_col, None)
    if feat is not None and hasattr(feat, "names") and feat.names:
        return list(feat.names)

    # fallback if names are missing
    max_label = int(max(ds[label_col]))
    return [str(i) for i in range(max_label + 1)]


# ============================================================
# EXPORT SPLIT
# ============================================================

def export_split(
    ds,
    split_name,
    out_root,
    image_col,
    label_col,
    class_names,
    counter,
    first_size_map,
    prefix,
):
    """
    Export one split (train or test).

    NEW:
    ----
    first_size_map[class_name] = (W, H) for the FIRST image we see in that class.
    """
    it = tqdm(range(len(ds)), desc=f"{prefix}:{split_name}", unit="img")

    for i in it:
        ex = ds[i]
        y = int(ex[label_col])
        cname_raw = class_names[y]
        cname = safe_name(cname_raw)

        img = to_rgb_pil(ex[image_col])

        # ✅ NEW: record FIRST W×H for this class (only once)
        if cname not in first_size_map:
            w, h = img.size  # PIL gives (W, H)
            first_size_map[cname] = (int(w), int(h))

        fname = f"{prefix}_{split_name}_{i:06d}.png"
        img.save(os.path.join(out_root, cname, fname))

        counter[cname] += 1


# ============================================================
# SAVE COUNTS + FIRST IMAGE SIZE PER CLASS
# ============================================================

def save_counts(ds_root, class_names, train_counts, test_counts, first_size_map):
    """
    Saves:
      - class_counts.json
      - class_counts.csv

    NEW fields per class:
      - first_w
      - first_h
      - first_wh  (string "W×H")
    """
    stats = {}
    for cname in class_names:
        c = safe_name(cname)

        tr = int(train_counts.get(c, 0))
        te = int(test_counts.get(c, 0))

        # ✅ NEW: first seen size for this class (may be None if somehow no images exported)
        wh = first_size_map.get(c, None)
        if wh is None:
            first_w, first_h = None, None
            first_wh = ""
        else:
            first_w, first_h = int(wh[0]), int(wh[1])
            first_wh = f"{first_w}x{first_h}"

        stats[c] = {
            "train": tr,
            "test": te,
            "total": tr + te,
            "first_w": first_w,
            "first_h": first_h,
            "first_wh": first_wh,
        }

    # JSON
    with open(os.path.join(ds_root, "class_counts.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    # CSV
    with open(os.path.join(ds_root, "class_counts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # ✅ NEW columns for W×H
        w.writerow(["class", "train", "test", "total", "first_w", "first_h", "first_wh"])
        for c, v in stats.items():
            w.writerow([c, v["train"], v["test"], v["total"], v["first_w"], v["first_h"], v["first_wh"]])

    return stats


# ============================================================
# EXPORT LOGIC
# ============================================================

def export_one_dataset(spec, export_root):
    hub = spec["hub"]
    name = spec["name"]
    label_col = spec["label_col"]
    image_col = spec["image_col"]

    print("\n" + "=" * 80)
    print(f"📥 Loading dataset from HF: {hub}")
    print("=" * 80)

    ds_dict = load_dataset(hub)

    # If dataset already has train/test, use them
    if "train" in ds_dict and "test" in ds_dict:
        train_ds = ds_dict["train"]
        test_ds  = ds_dict["test"]
    else:
        # Create stratified train/test split from first available split
        first_split = list(ds_dict.keys())[0]
        full = ds_dict[first_split]

        # stratify_by_column keeps class balance in train/test
        split = full.train_test_split(test_size=0.2, seed=42, stratify_by_column=label_col)
        train_ds = split["train"]
        test_ds  = split["test"]

    class_names = get_class_names(train_ds, label_col)

    ds_root = os.path.join(export_root, name)
    train_root = os.path.join(ds_root, "train")
    test_root  = os.path.join(ds_root, "test")

    ensure_dir(train_root)
    ensure_dir(test_root)

    for cname in class_names:
        ensure_dir(os.path.join(train_root, safe_name(cname)))
        ensure_dir(os.path.join(test_root,  safe_name(cname)))

    train_counts = defaultdict(int)
    test_counts  = defaultdict(int)

    # ✅ NEW: store first image size per class
    first_size_map = {}

    export_split(
        train_ds, "train", train_root,
        image_col, label_col, class_names,
        train_counts, first_size_map,
        prefix=name
    )

    export_split(
        test_ds, "test", test_root,
        image_col, label_col, class_names,
        test_counts, first_size_map,
        prefix=name
    )

    stats = save_counts(ds_root, class_names, train_counts, test_counts, first_size_map)

    print("\n✅ EXPORTED:", name)
    for c, v in stats.items():
        wh = v.get("first_wh", "")
        wh_txt = f" | first_WxH={wh}" if wh else ""
        print(f"{c:25s} | train={v['train']:6d} | test={v['test']:6d} | total={v['total']:6d}{wh_txt}")

    print("📁 Output folder:", ds_root)


def main():
    base = script_dir()
    export_root = os.path.join(base, EXPORT_ROOT)
    ensure_dir(export_root)

    print("[INFO] Python executable:", sys.executable)
    print("[INFO] Export root:", export_root)

    for spec in MILITARY_DATASETS:
        export_one_dataset(spec, export_root)

    print("\n✅ ALL MILITARY-STYLE DATASETS EXPORTED")
    print("📁 Root output folder:", export_root)


if __name__ == "__main__":
    main()
