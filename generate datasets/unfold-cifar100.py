#!/usr/bin/env python3
"""
CIFAR-100 → PNG extractor with REAL fine class names
(Optionally extract ONLY selected classes)

BASE PATH:
    DATA_ROOT = "../../../data"

Expected input:
    ../../../data/cifar-100-python.tar.gz

Output:
    ../../../data/mydata100/
        train/<fine_class_name>/*.png
        test/<fine_class_name>/*.png

Dependencies:
    pip install pillow numpy
"""

import os
import tarfile
import shutil
import pickle
import numpy as np
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = "./"
ARCHIVE_DIR = DATA_ROOT

TEMP_EXTRACT_DIR = os.path.join(DATA_ROOT, "cifar100_temp")
MYDATA_ROOT = os.path.join(DATA_ROOT, "cifar-100")

TRAIN_ROOT = os.path.join(MYDATA_ROOT, "train")
TEST_ROOT  = os.path.join(MYDATA_ROOT, "test")

DEBUG = True


def debug_print(msg):
    if DEBUG:
        print(msg)


# ============================================================
# OPTIONAL: SELECT SUBSET OF *FINE* CLASSES (CIFAR-100)
# ============================================================
# CIFAR-100 has:
#   • 20 coarse classes
#   • 100 fine classes  <-- we export fine labels here
#
# If you want ALL 100 fine classes:
#   SELECTED_CLASSES = None
#
# If you want only a subset, list fine class names like:
#   ["apple", "baby", "bicycle", ...]
SELECTED_CLASSES = None  # <-- set to None for all 100


# ============================================================
# CIFAR HELPERS
# ============================================================

def load_cifar100_meta(meta_file):
    """
    CIFAR-100 meta file contains:
        b'fine_label_names': [b'apple', b'aquarium_fish', ...]  (100)
        b'coarse_label_names': [b'aquatic_mammals', ...]        (20)
    """
    with open(meta_file, "rb") as f:
        meta = pickle.load(f, encoding="bytes")

    fine_names = [x.decode("utf-8") for x in meta[b"fine_label_names"]]
    coarse_names = [x.decode("utf-8") for x in meta[b"coarse_label_names"]]
    return fine_names, coarse_names


def load_cifar100_split(split_file):
    """
    CIFAR-100 train/test files contain:
        b"data": N x 3072
        b"fine_labels": list length N
        b"coarse_labels": list length N
    """
    with open(split_file, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    data = batch[b"data"]
    fine_labels = batch[b"fine_labels"]
    coarse_labels = batch[b"coarse_labels"]
    return data, fine_labels, coarse_labels


# ============================================================
# EXTRACT ARCHIVE
# ============================================================

def extract_cifar100_archive():
    print("\n[STEP] Searching for CIFAR-100 archive in:")
    print(f"       {ARCHIVE_DIR}")

    archives = [
        os.path.join(ARCHIVE_DIR, f)
        for f in os.listdir(ARCHIVE_DIR)
        if f.lower().endswith((".tar.gz", ".tgz", ".tar"))
    ]

    if not archives:
        print("[ERROR] No CIFAR archive found!")
        print("        Expected: cifar-100-python.tar.gz")
        return False

    # Prefer the CIFAR-100 archive if multiple tar.gz exist
    archive = None
    for a in archives:
        if "cifar-100" in os.path.basename(a).lower():
            archive = a
            break
    if archive is None:
        archive = archives[0]

    print(f"[INFO] Using archive: {archive}")

    if os.path.exists(TEMP_EXTRACT_DIR):
        shutil.rmtree(TEMP_EXTRACT_DIR)

    os.makedirs(TEMP_EXTRACT_DIR, exist_ok=True)

    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(TEMP_EXTRACT_DIR)

    print("[OK] Archive extracted.")
    return True


# ============================================================
# SAVE IMAGES
# ============================================================

def ensure_dirs(root, class_names):
    os.makedirs(root, exist_ok=True)
    for c in class_names:
        os.makedirs(os.path.join(root, c), exist_ok=True)


def save_split_pngs(split_file, split_root, split_name,
                    fine_label_names, selected_set):
    """
    Convert CIFAR-100 split (train/test) to PNG files under:
        split_root/<fine_class_name>/*.png
    """
    if not os.path.exists(split_file):
        print(f"[ERROR] Missing split file: {split_file}")
        return 0

    data, fine_labels, _coarse_labels = load_cifar100_split(split_file)

    # CIFAR raw: N x 3072 -> N x 3 x 32 x 32 -> N x 32 x 32 x 3
    data = data.reshape(-1, 3, 32, 32)
    data = data.transpose(0, 2, 3, 1)

    export_classes = fine_label_names if selected_set is None else sorted(selected_set)
    ensure_dirs(split_root, export_classes)

    saved = 0
    skipped = 0

    for i, img_arr in enumerate(data):
        lbl = int(fine_labels[i])
        if lbl < 0 or lbl >= len(fine_label_names):
            skipped += 1
            continue

        class_name = fine_label_names[lbl]

        # Optional subset filter
        if selected_set is not None and class_name not in selected_set:
            skipped += 1
            continue

        img = Image.fromarray(img_arr.astype(np.uint8))

        out_path = os.path.join(
            split_root,
            class_name,
            f"{split_name}_{i:06d}_{class_name}.png"
        )
        img.save(out_path)
        saved += 1

    print(f"[OK] {split_name}: saved={saved}, skipped={skipped}")
    return saved


# ============================================================
# MAIN
# ============================================================

def main():

    if not extract_cifar100_archive():
        return

    cifar_dir = os.path.join(TEMP_EXTRACT_DIR, "cifar-100-python")
    meta_file = os.path.join(cifar_dir, "meta")
    train_file = os.path.join(cifar_dir, "train")
    test_file = os.path.join(cifar_dir, "test")

    if not os.path.exists(meta_file):
        print(f"[ERROR] meta not found: {meta_file}")
        return

    fine_names, coarse_names = load_cifar100_meta(meta_file)

    print(f"[INFO] CIFAR-100 fine classes:  {len(fine_names)}")
    print(f"[INFO] CIFAR-100 coarse classes: {len(coarse_names)}")

    # Decide selected classes (fine classes)
    if SELECTED_CLASSES is None:
        selected_set = None
        print("[INFO] Exporting ALL 100 fine classes")
    else:
        selected_set = set(SELECTED_CLASSES)

        unknown = sorted([x for x in selected_set if x not in fine_names])
        if unknown:
            print("[ERROR] Unknown fine class names in SELECTED_CLASSES:")
            for u in unknown:
                print("   -", u)
            print("\n[INFO] Example valid fine classes (first 25):")
            for x in fine_names[:25]:
                print("   -", x)
            return

        print("[INFO] Exporting ONLY selected fine classes:", sorted(selected_set))

    os.makedirs(TRAIN_ROOT, exist_ok=True)
    os.makedirs(TEST_ROOT, exist_ok=True)

    total_train = save_split_pngs(train_file, TRAIN_ROOT, "train", fine_names, selected_set)
    total_test  = save_split_pngs(test_file,  TEST_ROOT,  "test",  fine_names, selected_set)

    print("\n[DONE]")
    print(f"Train images: {total_train}")
    print(f"Test images : {total_test}")
    print(f"Output path : {MYDATA_ROOT}")

    # Cleanup
    shutil.rmtree(TEMP_EXTRACT_DIR)
    print("[CLEANUP] Temp folder removed")


if __name__ == "__main__":
    main()
