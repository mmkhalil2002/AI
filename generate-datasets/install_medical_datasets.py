import os
import sys
import shutil
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

RAW_DATASET_DIRNAME = "raw_medical_export"
OUTPUT_DATASET_DIRNAME = "medical_set"

TRAIN_DIRNAME = "train"
TEST_DIRNAME = "test"

CLEAR_OUTPUT_FIRST = True

ALLOWED_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

# ============================================================
# HELPERS
# ============================================================

def base_dir():
    # 🔥 current working directory
    return os.getcwd()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def safe_name(name):
    return "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in str(name)
    )

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTS

def unique_path(dst_dir, filename):
    base, ext = os.path.splitext(filename)
    dst = os.path.join(dst_dir, filename)

    if not os.path.exists(dst):
        return dst

    i = 1
    while True:
        new_name = f"{base}_{i:06d}{ext}"
        dst = os.path.join(dst_dir, new_name)
        if not os.path.exists(dst):
            return dst
        i += 1

# ============================================================
# CORE COPY LOGIC
# ============================================================

def copy_split(src_root, dst_root, split_name, counts):
    if not os.path.isdir(src_root):
        return

    for cname_raw in os.listdir(src_root):
        src_class = os.path.join(src_root, cname_raw)
        if not os.path.isdir(src_class):
            continue

        cname = safe_name(cname_raw)
        dst_class = os.path.join(dst_root, cname)
        ensure_dir(dst_class)

        for root, _, files in os.walk(src_class):
            for f in files:
                if not is_image_file(f):
                    continue

                src_file = os.path.join(root, f)
                safe_file = safe_name(os.path.splitext(f)[0]) + os.path.splitext(f)[1].lower()
                dst_file = unique_path(dst_class, safe_file)

                shutil.copy2(src_file, dst_file)
                counts[split_name][cname] += 1

def merge_all(raw_root, train_out, test_out):
    counts = {
        "train": defaultdict(int),
        "test": defaultdict(int)
    }

    found_train = 0
    found_test = 0

    for root, dirs, _ in os.walk(raw_root):
        base = os.path.basename(root).lower()

        if base == "train":
            print(f"[INFO] Found train: {root}")
            copy_split(root, train_out, "train", counts)
            found_train += 1
            dirs[:] = []

        elif base == "test":
            print(f"[INFO] Found test: {root}")
            copy_split(root, test_out, "test", counts)
            found_test += 1
            dirs[:] = []

    return counts, found_train, found_test

# ============================================================
# SUMMARY
# ============================================================

def print_summary(counts, output_root):
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total_train = sum(counts["train"].values())
    total_test  = sum(counts["test"].values())

    print("\nTRAIN:")
    for c in sorted(counts["train"]):
        print(f"{c:25s} {counts['train'][c]}")

    print("\nTEST:")
    for c in sorted(counts["test"]):
        print(f"{c:25s} {counts['test'][c]}")

    print("\nTOTAL:")
    print(f"train = {total_train}")
    print(f"test  = {total_test}")
    print("="*60)

# ============================================================
# MAIN
# ============================================================

def main():
    base = base_dir()

    raw_root = os.path.join(base, RAW_DATASET_DIRNAME)
    out_root = os.path.join(base, OUTPUT_DATASET_DIRNAME)

    train_out = os.path.join(out_root, TRAIN_DIRNAME)
    test_out  = os.path.join(out_root, TEST_DIRNAME)

    print("[INFO] Base dir :", base)
    print("[INFO] Source   :", raw_root)
    print("[INFO] Output   :", out_root)

    if not os.path.isdir(raw_root):
        raise FileNotFoundError(f"Missing: {raw_root}")

    if CLEAR_OUTPUT_FIRST and os.path.exists(out_root):
        print("[INFO] Removing old output...")
        shutil.rmtree(out_root)

    ensure_dir(train_out)
    ensure_dir(test_out)

    counts, ft, fs = merge_all(raw_root, train_out, test_out)

    if ft == 0 and fs == 0:
        raise RuntimeError("No train/test folders found!")

    print_summary(counts, out_root)

    print("\n✅ medical_set created successfully")

# ============================================================

if __name__ == "__main__":
    main()