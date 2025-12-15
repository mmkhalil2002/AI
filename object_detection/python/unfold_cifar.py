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
DEBUG = True   # set to False to silence debug messages


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

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------------------------------------
    # ASSUME GLOBAL DATA_PATH IS ALREADY DEFINED
    # --------------------------------------------------------
    # Example (outside this function):
    #   DATA_PATH = "../../../data/mydata"
    #
    # Expected structure:
    #   ../../../data/mydata/
    #       train/
    #           classA/
    #           classB/
    #           ...
    #       test/
    #           classA/
    #           classB/
    #           ...
    # --------------------------------------------------------
    debug_print(f"[main] Global DATA_PATH = {DATA_PATH!r}")

    # Build train and test directories from the global DATA_PATH
    train_path = os.path.join(DATA_PATH, "train")
    test_path  = os.path.join(DATA_PATH, "test")

    debug_print(f"[main] Computed train_path = {train_path}")
    debug_print(f"[main] Computed test_path  = {test_path}")

    print("Training images from:", train_path)
    print("Testing  images from:", test_path)

    # --------------------------------------------------------
    # DATA TRANSFORMS FOR YOUR DATA
    # --------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((32, 32)),        # 128x128 → 32x32 (if needed)
        transforms.ToTensor(),              # convert to [C, H, W] in [0, 1]
        transforms.Normalize(               # normalize to [-1, 1]
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # ------------------------------------------------------------------
    # LOAD DATASETS USING ImageFolder
    # ------------------------------------------------------------------
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_path,
        transform=transform
    )

    debug_print(f"[main] Loaded train_dataset with {len(train_dataset)} images")
    debug_print(f"[main] Loaded test_dataset  with {len(test_dataset)} images")

    # --------------------------------------------------------
    # DYNAMIC CLASS NAMES (NO HARDCODED CIFAR-10 LABELS)
    # --------------------------------------------------------
    # ImageFolder automatically builds:
    #   train_dataset.classes → ["airplane", "automobile", ...] or any custom folders
    #
    # We rely ONLY on these dynamic names instead of CIFAR10_LABELS.
    # This works for:
    #   • Original CIFAR-10 extracted into folders
    #   • Any custom dataset with class subdirectories
    # --------------------------------------------------------
    global GLOBAL_LABELS
    GLOBAL_LABELS = train_dataset.classes  # dynamic label list
    debug_print("[main] Class index → name mapping (from train_dataset.classes):")
    for idx, name in enumerate(GLOBAL_LABELS):
        debug_print(f"   {idx}: {name}")

    # Optionally show first few training samples to verify labels
    max_show = min(5, len(train_dataset))
    for i in range(max_show):
        _, lbl = train_dataset[i]                  # (image_tensor, label_index)
        cls_name = GLOBAL_LABELS[lbl]
        debug_print(f"[main] Sample train index {i} → label {lbl} ('{cls_name}')")

    # ============================================================
    # DATALOADERS (with RANDOMIZATION)
    # ============================================================
    # train_loader:
    #   • shuffle=True  → random order every epoch (good for training)
    #
    # test_loader:
    #   • shuffle=True  → random order at evaluation time
    #     (does NOT change labels, just which index comes first)
    # ============================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=True,      # full randomization for training
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=True,      # randomization for test iteration
        num_workers=2
    )

    # --------------------------------------------------------
    # DETERMINE NUMBER OF CLASSES FROM DATASET
    # --------------------------------------------------------
    num_classes = len(train_dataset.classes)
    print("Number of classes detected in train:", num_classes)
    print("Class names:", train_dataset.classes)

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------
    model = StaticInitLearnableCNN(num_classes=num_classes)

    # --------------------------------------------------------
    # LOAD OR TRAIN MODEL
    # --------------------------------------------------------
    model_filename = os.path.join(MODEL_PATH, MODEL_FILENAME)
    debug_print(f"[main] Model file path = {model_filename}")

    if os.path.exists(model_filename):
        print(f"Loading trained weights from: {model_filename}")
        state_dict = torch.load(model_filename, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("No saved model found. Training a new model...")
        model = train_model(model, train_loader, device,
                            num_epochs=NUM_EPOCHS, lr=1e-3)
        print(f"Saving trained model to: {model_filename}")
        torch.save(model.state_dict(), model_filename)

    # ------------------------------------------------------------
    # INTERACTIVE LOOP FOR USER-DRIVEN DETECTION
    # ------------------------------------------------------------
    import msvcrt

    print("\n--------------------------------------------------")
    print("Interactive Image Detection Mode")
    print("Press:")
    print("   d  → detect on an image index")
    print("   e  → exit program")
    print("--------------------------------------------------\n")

    while True:
        print("Enter command (d = detect, e = exit): ", end="", flush=True)

        # READ ONE CHARACTER WITHOUT PRESSING ENTER
        key = msvcrt.getch().decode().lower()
        print(key)   # echo the key

        if key == 'e':
            print("Exiting program. Goodbye!")
            break

        elif key == 'd':
            idx_str = input(f"Enter image index (0 – {len(test_dataset)-1}): ").strip()

            if not idx_str.isdigit():
                print("❌ Invalid index. Must be a number.")
                continue

            idx = int(idx_str)

            if idx < 0 or idx >= len(test_dataset):
                print("❌ Index out of range. Try again.")
                continue

            print(f"\nRunning detection on test image index {idx} ...")
            detect_single_image(model, test_dataset, device, index=idx)

        else:
            print("❌ Unknown command. Use 'd' for detect or 'e' to exit.")



if __name__ == "__main__":
    main()
