# ============================================================
# export_medmnist_as_raw_png_cifar_style_with_class_counts_and_size.py
# ============================================================
# MILITARY (MINIMAL OUTPUT) VERSION:
#   • Auto-installs missing packages into the SAME Python running this script
#   • Converts ALL images (including grayscale) → RGB
#   • Exports CIFAR-style folder hierarchy (train/test/class_name/*.png)
#   • Counts images per class
#   • ALSO captures image size info and stores it INSIDE class_counts.json + class_counts.csv
#
# OUTPUT FILES (ONLY THESE TWO):
#   - class_counts.json   ✅ counts + size info per class + dataset-level size
#   - class_counts.csv    ✅ counts + size info per class
#
# NO OTHER FILES ARE WRITTEN.
# ============================================================

import os
import sys
import json
import csv
import subprocess
from collections import defaultdict, Counter

# ============================================================
# AUTO-INSTALL HELPERS (SAFE + SAME PYTHON)
# ============================================================

def _pip_install(packages):
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + list(packages)
    print("\n[AUTO-INSTALL] Running:", " ".join(cmd))
    subprocess.check_call(cmd)

def _ensure_import(import_name, pip_name=None):
    try:
        __import__(import_name)
        return
    except Exception:
        pip_name = pip_name or import_name
        print(f"[AUTO-INSTALL] Missing '{import_name}'. Installing '{pip_name}' ...")
        _pip_install([pip_name])
        __import__(import_name)

_ensure_import("PIL", "pillow")
_ensure_import("medmnist", "medmnist")
_ensure_import("tqdm", "tqdm")

from PIL import Image
import medmnist
from medmnist import INFO
from tqdm import tqdm

# ============================================================
# DATASETS (EACH IS A SEPARATE DATASET)
# ============================================================

DATASETS = [
    "dermamnist",
    "pneumoniamnist",
    "pathmnist",
    "bloodmnist",
    "retinamnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
]

EXPORT_ROOT = "raw_medical_export"
CACHE_ROOT  = "medmnist_cache"

# ============================================================
# HELPERS
# ============================================================

def script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def safe_name(s):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(s))

def label_to_int(label):
    # MedMNIST labels are often numpy arrays like shape (1,) or (1,1)
    if hasattr(label, "shape"):
        return int(label.reshape(-1)[0])
    return int(label)

def class_names_from_info(info):
    # INFO[name]["label"] is a dict like {"0": "xxx", "1": "yyy", ...}
    lab = info.get("label", {})
    items = [(int(k), v) for k, v in lab.items()]
    items.sort()
    return [str(v) for _, v in items]

def to_pil(img):
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(img)

def to_rgb_keep_size(img: Image.Image) -> Image.Image:
    # Converts grayscale/RGBA -> RGB without resizing.
    return img.convert("RGB")

def most_common_hw(counter: Counter):
    """
    Return the most common (H,W) from a Counter({(H,W): count}).
    If empty, return None.
    """
    if not counter:
        return None
    (h, w), cnt = counter.most_common(1)[0]
    return {"h": int(h), "w": int(w), "count": int(cnt)}

# ============================================================
# EXPORT ONE DATASET
# ============================================================

def export_dataset(name, cache_root, export_root):

    if name not in INFO:
        print(f"❌ Dataset not found: {name}")
        return

    info = INFO[name]
    task = info.get("task", "")

    # Only export classification datasets (1 label per image)
    if task not in ("multi-class", "binary-class"):
        print(f"⚠️ Skipped {name} (task={task})")
        return

    DataClass = getattr(medmnist, info["python_class"])

    # download=True ensures MedMNIST gets downloaded into cache_root
    train_ds = DataClass(split="train", root=cache_root, download=True)
    test_ds  = DataClass(split="test",  root=cache_root, download=True)

    class_names_raw = class_names_from_info(info)
    class_names = [safe_name(c) for c in class_names_raw]

    ds_root = os.path.join(export_root, name)
    train_root = os.path.join(ds_root, "train")
    test_root  = os.path.join(ds_root, "test")

    ensure_dir(train_root)
    ensure_dir(test_root)

    # Create class folders
    for c in class_names:
        ensure_dir(os.path.join(train_root, c))
        ensure_dir(os.path.join(test_root,  c))

    # ------------------------------------------------------------
    # COUNTS
    # ------------------------------------------------------------
    train_counts = defaultdict(int)
    test_counts  = defaultdict(int)

    # ------------------------------------------------------------
    # SIZE STATS (IN-MEMORY ONLY)
    # We will store only:
    #   • dataset-level most common size per split
    #   • per-class most common size per split
    # Then write those fields INSIDE class_counts.json + .csv
    # ------------------------------------------------------------
    train_size_hist = Counter()
    test_size_hist  = Counter()

    train_class_size_hist = defaultdict(Counter)   # class -> Counter((H,W)->count)
    test_class_size_hist  = defaultdict(Counter)

    def export_split(split_name, ds, out_root, counter_counts, split_size_hist, class_size_hist):
        it = tqdm(range(len(ds)), desc=f"{name}:{split_name}", unit="img")
        for i in it:
            img, label = ds[i]
            y = label_to_int(label)

            img = to_pil(img)
            img = to_rgb_keep_size(img)

            w, h = img.size
            hw = (h, w)

            cname = class_names[y]
            fname = f"{name}_{split_name}_{i:06d}.png"
            saved_path = os.path.join(out_root, cname, fname)

            img.save(saved_path)

            counter_counts[cname] += 1
            split_size_hist[hw] += 1
            class_size_hist[cname][hw] += 1

    export_split("train", train_ds, train_root, train_counts, train_size_hist, train_class_size_hist)
    export_split("test",  test_ds,  test_root,  test_counts,  test_size_hist,  test_class_size_hist)

    # ------------------------------------------------------------
    # BUILD ONE OUTPUT OBJECT: class_counts.json
    # Includes counts + size info in the same file
    # ------------------------------------------------------------
    dataset_train_common = most_common_hw(train_size_hist)
    dataset_test_common  = most_common_hw(test_size_hist)

    out_json = {
        "dataset_name": name,
        "task": task,
        "num_classes": int(len(class_names)),
        "class_names": class_names,
        "dataset_most_common_size": {
            "train": dataset_train_common,  # e.g. {"h":28,"w":28,"count":7000}
            "test":  dataset_test_common
        },
        "classes": {}
    }

    for cname in class_names:
        tr = int(train_counts.get(cname, 0))
        te = int(test_counts.get(cname, 0))
        out_json["classes"][cname] = {
            "train": tr,
            "test": te,
            "total": tr + te,
            "most_common_size": {
                "train": most_common_hw(train_class_size_hist.get(cname, Counter())),
                "test":  most_common_hw(test_class_size_hist.get(cname, Counter())),
            }
        }

    # ------------------------------------------------------------
    # WRITE ONLY TWO FILES
    # ------------------------------------------------------------
    json_path = os.path.join(ds_root, "class_counts.json")
    csv_path  = os.path.join(ds_root, "class_counts.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=4)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class",
            "train", "test", "total",
            "train_common_h", "train_common_w",
            "test_common_h", "test_common_w"
        ])

        for cname in class_names:
            cinfo = out_json["classes"][cname]
            tr_common = cinfo["most_common_size"]["train"] or {}
            te_common = cinfo["most_common_size"]["test"] or {}

            writer.writerow([
                cname,
                cinfo["train"], cinfo["test"], cinfo["total"],
                tr_common.get("h", ""), tr_common.get("w", ""),
                te_common.get("h", ""), te_common.get("w", ""),
            ])

    # ------------------------------------------------------------
    # MINIMAL CONSOLE OUTPUT
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"✅ EXPORTED: {name}  | task={task} | classes={len(class_names)} | train={len(train_ds)} | test={len(test_ds)}")
    print(f"Most common TRAIN size (H,W): {dataset_train_common}")
    print(f"Most common TEST  size (H,W): {dataset_test_common}")
    print("📁 Output folder:", ds_root)
    print("📄 Wrote ONLY:")
    print("    - class_counts.json")
    print("    - class_counts.csv")
    print("=" * 70)

# ============================================================
# MAIN
# ============================================================

def main():
    base = script_dir()
    cache_root  = os.path.join(base, CACHE_ROOT)
    export_root = os.path.join(base, EXPORT_ROOT)

    ensure_dir(cache_root)
    ensure_dir(export_root)

    print("[INFO] Python executable:", sys.executable)
    print("[INFO] Cache root:", cache_root)
    print("[INFO] Export root:", export_root)

    for d in DATASETS:
        export_dataset(d, cache_root, export_root)

    print("\n✅ ALL DATASETS EXPORTED")
    print("📁 Output folder:", export_root)

if __name__ == "__main__":
    main()
