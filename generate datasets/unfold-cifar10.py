#!/usr/bin/env python3
"""
CIFAR-10 → PNG extractor with REAL class names
(Optionally extract ONLY selected classes)

BASE PATH:
    DATA_ROOT = "../../../data"

Expected input:
    ../../../data/cifar-10-python.tar.gz

Output:
    ../../../data/mydata/
        train/<class>/*.png
        test/<class>/*.png

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

TEMP_EXTRACT_DIR = os.path.join(DATA_ROOT, "cifar_temp")
MYDATA_ROOT = os.path.join(DATA_ROOT, "cifar-10")

TRAIN_ROOT = os.path.join(MYDATA_ROOT, "train")
TEST_ROOT  = os.path.join(MYDATA_ROOT, "test")

DEBUG = True


def debug_print(msg):
    if DEBUG:
        print(msg)


# ============================================================
# OPTIONAL: SELECT SUBSET OF CLASSES
# ============================================================

# Set to None → export ALL CIFAR-10 classes
# Or list any subset you want
SELECTED_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ============================================================
# CIFAR HELPERS
# ============================================================

def load_cifar_batch(batch_file):
    with open(batch_file, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    return batch[b"data"], batch[b"labels"]


def load_cifar10_label_names(meta_file):
    with open(meta_file, "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    return [x.decode("utf-8") for x in meta[b"label_names"]]


# ============================================================
# EXTRACT ARCHIVE
# ============================================================

def extract_cifar_archive():
    print("\n[STEP] Searching for CIFAR archive in:")
    print(f"       {ARCHIVE_DIR}")

    archives = [
        os.path.join(ARCHIVE_DIR, f)
        for f in os.listdir(ARCHIVE_DIR)
        if f.lower().endswith((".tar.gz", ".tgz", ".tar"))
    ]

    if not archives:
        print("[ERROR] No CIFAR archive found!")
        print("        Expected: cifar-10-python.tar.gz")
        return False

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


def save_batch(batch_path, split_root, split_name,
               batch_name, label_names, selected_set):

    if not os.path.exists(batch_path):
        return 0

    data, labels = load_cifar_batch(batch_path)

    data = data.reshape(-1, 3, 32, 32)
    data = data.transpose(0, 2, 3, 1)  # HWC

    export_classes = label_names if selected_set is None else sorted(selected_set)
    ensure_dirs(split_root, export_classes)

    saved = 0
    skipped = 0

    for i, img in enumerate(data):
        lbl = int(labels[i])
        if lbl < 0 or lbl >= len(label_names):
            skipped += 1
            continue

        name = label_names[lbl]
        if selected_set is not None and name not in selected_set:
            skipped += 1
            continue

        img = Image.fromarray(img.astype(np.uint8))
        out = os.path.join(
            split_root,
            name,
            f"{split_name}_{batch_name}_{i:05d}_{name}.png"
        )
        img.save(out)
        saved += 1

    print(f"[OK] {split_name}/{batch_name}: saved={saved}, skipped={skipped}")
    return saved


# ============================================================
# MAIN
# ============================================================

def main():

    if not extract_cifar_archive():
        return

    cifar_dir = os.path.join(TEMP_EXTRACT_DIR, "cifar-10-batches-py")
    meta_file = os.path.join(cifar_dir, "batches.meta")

    if not os.path.exists(meta_file):
        print("[ERROR] batches.meta not found")
        return

    label_names = load_cifar10_label_names(meta_file)
    print("[INFO] CIFAR-10 labels:", label_names)

    if SELECTED_CLASSES is None:
        selected_set = None
        print("[INFO] Exporting ALL classes")
    else:
        selected_set = set(SELECTED_CLASSES)
        print("[INFO] Exporting ONLY:", sorted(selected_set))

    os.makedirs(TRAIN_ROOT, exist_ok=True)
    os.makedirs(TEST_ROOT, exist_ok=True)

    total_train = 0
    for i in range(1, 6):
        total_train += save_batch(
            os.path.join(cifar_dir, f"data_batch_{i}"),
            TRAIN_ROOT,
            "train",
            f"data_batch_{i}",
            label_names,
            selected_set
        )

    total_test = save_batch(
        os.path.join(cifar_dir, "test_batch"),
        TEST_ROOT,
        "test",
        "test_batch",
        label_names,
        selected_set
    )

    print("\n[DONE]")
    print(f"Train images: {total_train}")
    print(f"Test images : {total_test}")
    print(f"Output path : {MYDATA_ROOT}")

    shutil.rmtree(TEMP_EXTRACT_DIR)
    print("[CLEANUP] Temp folder removed")


if __name__ == "__main__":
    main()
