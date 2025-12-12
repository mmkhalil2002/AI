#!/usr/bin/env python3
"""
CIFAR-10 → mydata PNG generator with REAL class names.

BASE PATH:
    DATA_ROOT = "../../../data"

This script assumes:

    • A CIFAR-10 archive (e.g., cifar-10-python.tar.gz) exists in:
          ../../../data

It will:

    1) Extract the CIFAR archive to a temporary folder:
          ../../../data/cifar_temp/...

       Inside that temp folder, the original CIFAR layout is:

          cifar-10-batches-py/
              batches.meta
              data_batch_1
              data_batch_2
              data_batch_3
              data_batch_4
              data_batch_5
              test_batch

    2) Convert all CIFAR images to PNG and save them under:

          ../../../data/mydata

       with the following final output structure USING REAL LABEL NAMES:

    --------------------------------------------------------
    # EXPECTED FOLDER STRUCTURE:
    #   mydata/
    #       train/
    #           airplane/
    #           automobile/
    #           bird/
    #           cat/
    #           deer/
    #           dog/
    #           frog/
    #           horse/
    #           ship/
    #           truck/
    #       test/
    #           airplane/
    #           automobile/
    #           bird/
    #           cat/
    #           deer/
    #           dog/
    #           frog/
    #           horse/
    #           ship/
    #           truck/
    # 
    #   (folder names match CIFAR-10 original class names)
    --------------------------------------------------------

       That is:

          ../../../data/mydata/train/airplane/*.png
          ../../../data/mydata/train/automobile/*.png
          ...
          ../../../data/mydata/test/truck/*.png

    3) Remove the temporary extraction directory at the end.

Dependencies:
    pip install pillow numpy
"""

import os
import tarfile
import shutil
import pickle
import numpy as np
from PIL import Image


# ------------------------------------------------------------
# BASE PATH: start from "../../../data"
# ------------------------------------------------------------
DATA_ROOT = "../../../data"          # Root containing CIFAR tar.gz

# Where we search for the CIFAR archive
ARCHIVE_DIR = DATA_ROOT

# Temporary extraction directory (will be removed at the end)
TEMP_EXTRACT_DIR = os.path.join(DATA_ROOT, "cifar_temp")

# Final output root directory (contains "mydata")
MYDATA_ROOT = os.path.join(DATA_ROOT, "mydata")

# Train and test root folders (as requested)
TRAIN_ROOT = os.path.join(MYDATA_ROOT, "train")
TEST_ROOT  = os.path.join(MYDATA_ROOT, "test")

# ------------------------------------------------------------
# CIFAR-10 CLASS NAMES (in the original order)
# ------------------------------------------------------------
# Index → Name mapping:
#   0: airplane
#   1: automobile
#   2: bird
#   3: cat
#   4: deer
#   5: dog
#   6: frog
#   7: horse
#   8: ship
#   9: truck
CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# ------------------------------------------------------------
# Helper: load a CIFAR batch file
# ------------------------------------------------------------
def load_cifar_batch(batch_file):
    """
    Load one CIFAR batch file.

    Args:
        batch_file (str): Path to data_batch_X or test_batch

    Returns:
        data   (ndarray): shape (N, 3072)
        labels (list)   : length N
    """
    with open(batch_file, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    return batch[b"data"], batch[b"labels"]


# ------------------------------------------------------------
# Helper: find and extract CIFAR archive into TEMP_EXTRACT_DIR
# ------------------------------------------------------------
def extract_cifar_archive():
    """
    Find a .tar/.tar.gz/.tgz CIFAR archive in ARCHIVE_DIR and
    extract it into TEMP_EXTRACT_DIR.

    Returns:
        bool: True on success, False otherwise.
    """
    print("---------------------------------------------------")
    print("[STEP] Searching for CIFAR archive in:")
    print(f"       {ARCHIVE_DIR}")
    print("---------------------------------------------------")

    if not os.path.isdir(ARCHIVE_DIR):
        print(f"[ERROR] Archive directory does NOT exist: {ARCHIVE_DIR}")
        return False

    archives = [
        os.path.join(ARCHIVE_DIR, f)
        for f in os.listdir(ARCHIVE_DIR)
        if f.lower().endswith((".tar.gz", ".tar", ".tgz"))
    ]

    if not archives:
        print(f"[ERROR] No CIFAR archive (.tar/.tar.gz/.tgz) found in {ARCHIVE_DIR}")
        print("        Please place cifar-10-python.tar.gz there.")
        return False

    archive = archives[0]
    print(f"[INFO] Using archive: {archive}")

    # Clean previous temp directory if it exists
    if os.path.exists(TEMP_EXTRACT_DIR):
        shutil.rmtree(TEMP_EXTRACT_DIR)

    os.makedirs(TEMP_EXTRACT_DIR, exist_ok=True)

    print(f"[INFO] Extracting archive into: {TEMP_EXTRACT_DIR}")
    try:
        with tarfile.open(archive, "r:*") as tar:
            tar.extractall(TEMP_EXTRACT_DIR)
    except Exception as e:
        print(f"[ERROR] Failed to extract archive: {e}")
        return False

    print("[OK] Archive extracted successfully.")
    return True


# ------------------------------------------------------------
# Helper: ensure class dirs exist using REAL LABEL NAMES
# ------------------------------------------------------------
def ensure_class_dirs_with_labels(root_dir):
    """
    Create class directories under root_dir using CIFAR10_LABELS:

        root_dir/airplane/
        root_dir/automobile/
        ...
        root_dir/truck/
    """
    os.makedirs(root_dir, exist_ok=True)
    for label_name in CIFAR10_LABELS:
        cls_dir = os.path.join(root_dir, label_name)
        os.makedirs(cls_dir, exist_ok=True)


# ------------------------------------------------------------
# Helper: save all images from one batch into:
#         (TRAIN_ROOT or TEST_ROOT)/<label_name>/*.png
# ------------------------------------------------------------
def save_batch_images_to_label_dirs(batch_path, split_root, split_name, batch_name):
    """
    Convert a CIFAR batch file to PNG images and save them into:

        split_root/<label_name>/*.png

    where label_name is one of:
        airplane, automobile, bird, cat, deer,
        dog, frog, horse, ship, truck

    Args:
        batch_path (str): path to CIFAR batch (data_batch_X or test_batch)
        split_root (str): TRAIN_ROOT or TEST_ROOT
        split_name (str): "train" or "test" (for filename prefix)
        batch_name (str): e.g. "data_batch_1", "test_batch"

    Returns:
        int: number of images saved from this batch
    """
    if not os.path.exists(batch_path):
        print(f"[WARN] CIFAR batch file not found: {batch_path}")
        return 0

    print(f"[INFO] Converting batch '{batch_name}' for split '{split_name}'")
    print(f"       Source: {batch_path}")

    # Load raw CIFAR data
    data, labels = load_cifar_batch(batch_path)

    # CIFAR raw format: N x 3072 → reshape to N x 3 x 32 x 32
    data = data.reshape(-1, 3, 32, 32)
    # Reorder to N x 32 x 32 x 3 (HWC) for PIL
    data = data.transpose(0, 2, 3, 1)

    # Ensure label-named folders exist
    ensure_class_dirs_with_labels(split_root)

    count = 0
    for i, img_array in enumerate(data):
        label_idx = labels[i]

        # Defensive: make sure index is within range
        if not (0 <= label_idx < len(CIFAR10_LABELS)):
            print(f"[WARN] Invalid label index {label_idx}; skipping image {i}")
            continue

        label_name = CIFAR10_LABELS[label_idx]
        img = Image.fromarray(img_array)

        # Class directory, e.g. "airplane", "cat", "truck"
        cls_dir = os.path.join(split_root, label_name)

        # Example filename:
        #   train_data_batch_1_img_00000_label_3_cat.png
        filename = (
            f"{split_name}_{batch_name}_img_{i:05d}_"
            f"label_{label_idx}_{label_name}.png"
        )
        save_path = os.path.join(cls_dir, filename)
        img.save(save_path)
        count += 1

    print(f"[OK] Saved {count} images for split '{split_name}' into:")
    print(f"     {split_root}")
    return count


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    print("===================================================")
    print(" CIFAR-10 → mydata PNG Generator (Named Classes)")
    print("===================================================")
    print(f"[INFO] DATA_ROOT         : {DATA_ROOT}")
    print(f"[INFO] TEMP_EXTRACT_DIR  : {TEMP_EXTRACT_DIR}")
    print(f"[INFO] MYDATA_ROOT       : {MYDATA_ROOT}")
    print(f"[INFO] TRAIN_ROOT (out)  : {TRAIN_ROOT}")
    print(f"[INFO] TEST_ROOT  (out)  : {TEST_ROOT}")
    print("===================================================\n")

    # Step 1: Extract CIFAR archive
    if not extract_cifar_archive():
        print("[FATAL] Could not extract CIFAR archive. Exiting.")
        return

    # Step 2: Find the directory that contains "data_batch_1"
    cifar_root = None
    for root, dirs, files in os.walk(TEMP_EXTRACT_DIR):
        if "data_batch_1" in files:
            cifar_root = root
            break

    if cifar_root is None:
        print("[ERROR] Could not find CIFAR batch files (data_batch_1) after extraction.")
        return

    # Example: cifar_root = .../cifar_temp/cifar-10-batches-py
    print(f"[INFO] Located CIFAR batch directory: {cifar_root}\n")

    # Step 3: Explain + create mydata/train and mydata/test structure
    print("---------------------------------------------------")
    print("[STEP] Ensuring output folder structure exists:")
    print("---------------------------------------------------")
    print("   mydata/")
    print("       train/")
    print("           airplane/")
    print("           automobile/")
    print("           bird/")
    print("           cat/")
    print("           deer/")
    print("           dog/")
    print("           frog/")
    print("           horse/")
    print("           ship/")
    print("           truck/")
    print("       test/")
    print("           airplane/")
    print("           automobile/")
    print("           bird/")
    print("           cat/")
    print("           deer/")
    print("           dog/")
    print("           frog/")
    print("           horse/")
    print("           ship/")
    print("           truck/")
    print("---------------------------------------------------\n")

    # Ensure root and split dirs exist
    os.makedirs(MYDATA_ROOT, exist_ok=True)
    ensure_class_dirs_with_labels(TRAIN_ROOT)
    ensure_class_dirs_with_labels(TEST_ROOT)

    # Step 4: Process training batches (data_batch_1..5)
    print("---------------------------------------------------")
    print("[STEP] Generating TRAIN images from data_batch_1..5")
    print("---------------------------------------------------")

    train_batches = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]

    total_train = 0
    for bname in train_batches:
        batch_path = os.path.join(cifar_root, bname)
        total_train += save_batch_images_to_label_dirs(
            batch_path=batch_path,
            split_root=TRAIN_ROOT,
            split_name="train",
            batch_name=bname,
        )

    print(f"[INFO] Total TRAIN images saved: {total_train}\n")

    # Step 5: Process test batch (test_batch)
    print("---------------------------------------------------")
    print("[STEP] Generating TEST images from test_batch")
    print("---------------------------------------------------")

    test_batch_name = "test_batch"
    test_batch_path = os.path.join(cifar_root, test_batch_name)
    total_test = save_batch_images_to_label_dirs(
        batch_path=test_batch_path,
        split_root=TEST_ROOT,
        split_name="test",
        batch_name=test_batch_name,
    )

    print(f"[INFO] Total TEST images saved: {total_test}\n")

    # Step 6: Remove temporary extraction directory
    print("---------------------------------------------------")
    print("[STEP] Cleaning up temporary extraction directory...")
    print("---------------------------------------------------")

    shutil.rmtree(TEMP_EXTRACT_DIR, ignore_errors=True)
    print("[OK] Temporary directory removed.\n")

    print("===================================================")
    print(" DONE ✅  All CIFAR images written under mydata/")
    print("---------------------------------------------------")
    print(" FINAL OUTPUT STRUCTURE (USING REAL CLASS NAMES):")
    print("   mydata/")
    print("       train/")
    print("           airplane/*.png")
    print("           automobile/*.png")
    print("           bird/*.png")
    print("           cat/*.png")
    print("           deer/*.png")
    print("           dog/*.png")
    print("           frog/*.png")
    print("           horse/*.png")
    print("           ship/*.png")
    print("           truck/*.png")
    print("       test/")
    print("           airplane/*.png")
    print("           automobile/*.png")
    print("           bird/*.png")
    print("           cat/*.png")
    print("           deer/*.png")
    print("           dog/*.png")
    print("           frog/*.png")
    print("           horse/*.png")
    print("           ship/*.png")
    print("           truck/*.png")
    print("===================================================\n")


if __name__ == "__main__":
    main()
