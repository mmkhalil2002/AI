#!/usr/bin/env python3
# ============================================================
# 2D Dog Dataset Range Subset Generator
# Output folder format:
#     2D-dog-N-M-datasets
# Where:
#     N = beginning class index
#     M = last class index
# ============================================================

import os
import shutil
import traceback

# 🔵 CHANGE THIS TO YOUR ORIGINAL DATASET LOCATION
SOURCE_DATASET = r"C:\Users\Public\mkhalil\AI\data\2D-dog-datasets"


def pause():
    input("\nPress ENTER to exit...")


def read_int(prompt):
    while True:
        value = input(prompt).strip()
        if value.isdigit():
            return int(value)
        print("❌ Please enter a valid integer.")


def main():

    print("\n====================================================")
    print(" 2D Dog Dataset Range Generator")
    print("====================================================\n")

    print("Working directory:", os.getcwd())
    print("Source dataset:", SOURCE_DATASET, "\n")

    # --------------------------------------------------------
    # Validate source dataset
    # --------------------------------------------------------
    if not os.path.exists(SOURCE_DATASET):
        print("❌ SOURCE_DATASET does not exist.")
        pause()
        return

    train_src = os.path.join(SOURCE_DATASET, "train")
    test_src  = os.path.join(SOURCE_DATASET, "test")

    if not os.path.exists(train_src) or not os.path.exists(test_src):
        print("❌ train/ or test/ folder missing.")
        pause()
        return

    # --------------------------------------------------------
    # Get shared classes
    # --------------------------------------------------------
    train_classes = sorted([
        d for d in os.listdir(train_src)
        if os.path.isdir(os.path.join(train_src, d))
    ])

    test_classes = set([
        d for d in os.listdir(test_src)
        if os.path.isdir(os.path.join(test_src, d))
    ])

    classes = [c for c in train_classes if c in test_classes]

    total_classes = len(classes)

    if total_classes == 0:
        print("❌ No shared classes found.")
        pause()
        return

    print(f"Total shared classes: {total_classes}\n")

    for i, name in enumerate(classes):
        print(f"{i:>3} → {name}")

    print()

    # --------------------------------------------------------
    # Prompt user
    # --------------------------------------------------------
    first = read_int("Enter FIRST class index: ")
    last  = read_int("Enter LAST  class index: ")

    if first < 0 or last >= total_classes or first > last:
        print("\n❌ Invalid range.")
        pause()
        return

    selected = classes[first:last + 1]

    print("\nSelected classes:")
    for cls in selected:
        print(" -", cls)

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        pause()
        return

    # --------------------------------------------------------
    # Create output folder with N-M format
    # --------------------------------------------------------
    output_name = f"2D-dog-{first}-{last}-datasets"
    output_path = os.path.abspath(output_name)

    train_dst = os.path.join(output_path, "train")
    test_dst  = os.path.join(output_path, "test")

    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(test_dst, exist_ok=True)

    print("\nCreating dataset:", output_name)
    print("Copying classes...\n")

    for cls in selected:
        shutil.copytree(
            os.path.join(train_src, cls),
            os.path.join(train_dst, cls),
            dirs_exist_ok=True
        )
        shutil.copytree(
            os.path.join(test_src, cls),
            os.path.join(test_dst, cls),
            dirs_exist_ok=True
        )
        print("→", cls)

    print("\n====================================================")
    print("✅ DONE")
    print("Created:", output_path)
    print("====================================================")

    pause()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n❌ ERROR OCCURRED:")
        traceback.print_exc()
        pause()