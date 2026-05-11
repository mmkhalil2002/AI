import os
import sys
import json
import csv
import math
import random
import subprocess
import importlib
import urllib.request
import tempfile
from collections import defaultdict


# ==========================================================
# AUTO-INSTALL ROUTINE
# ==========================================================

def ensure_pip_available():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
        if result.returncode == 0:
            print("[OK] pip is already available.")
            return True
    except Exception:
        pass

    try:
        result = subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            check=False
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass

    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp_file:
            temp_path = tmp_file.name

        urllib.request.urlretrieve(get_pip_url, temp_path)

        result = subprocess.run(
            [sys.executable, temp_path],
            check=False
        )

        return result.returncode == 0

    except Exception as e:
        print(f"[ERROR] Failed to install pip: {e}")
        return False

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def ensure_python_package(import_name, pip_name=None):
    if pip_name is None:
        pip_name = import_name

    try:
        importlib.import_module(import_name)
        print(f"[OK] Package already installed: {pip_name}")
        return True
    except ImportError:
        print(f"[INFO] Missing package: {pip_name}")

    if not ensure_pip_available():
        return False

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", pip_name],
            check=False
        )

        if result.returncode != 0:
            return False

        importlib.import_module(import_name)
        print(f"[OK] Installed package: {pip_name}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed installing {pip_name}: {e}")
        return False


def ensure_required_python_packages():
    required_packages = [
        ("PIL", "pillow"),
    ]

    for import_name, pip_name in required_packages:
        if not ensure_python_package(import_name, pip_name):
            print("[ERROR] Required package missing. Cannot continue.")
            sys.exit(1)


ensure_required_python_packages()

from PIL import Image

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ============================================================
# CONFIG
# ============================================================

RAW_DATASET_DIRNAME = "raw_military_export"

SOURCE_TRAIN_DIRNAME = "train"
SOURCE_TEST_DIRNAME = "test"

DEFAULT_UNKNOWN_PERCENT = 10.0
RANDOM_SEED = 42

OUTPUT_PREFIX = "RME10"
OUTPUT_IMAGE_SIZE = (32, 32)

ALLOWED_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# HELPERS
# ============================================================

def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_name(s):
    return "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in str(s)
    )


def load_split_dataset(split_root):
    """
    Load one split from this structure:

        split_root/
            class1/
                image1.jpg
            class2/
                image2.png

    Example:

        raw_military_export/train/tank/img1.jpg
        raw_military_export/test/tank/img2.jpg

    Returns:
        data:
            [(image_path, label_index), ...]

        class_names:
            sorted list of class folder names
    """

    if not os.path.isdir(split_root):
        raise FileNotFoundError(
            f"Required split directory not found: {split_root}"
        )

    class_names = sorted([
        d for d in os.listdir(split_root)
        if os.path.isdir(os.path.join(split_root, d))
    ])

    if not class_names:
        raise RuntimeError(
            f"No class folders found inside: {split_root}"
        )

    class_to_idx = {
        cname: idx
        for idx, cname in enumerate(class_names)
    }

    data = []

    for cname in class_names:
        class_dir = os.path.join(split_root, cname)

        for current_dir, _, files in os.walk(class_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()

                if ext not in ALLOWED_EXTS:
                    continue

                image_path = os.path.join(current_dir, fname)
                data.append((image_path, class_to_idx[cname]))

    if not data:
        raise RuntimeError(
            f"No image files found inside: {split_root}"
        )

    return data, class_names


def align_test_labels_to_train_classes(test_data, test_class_names, train_class_names):
    """
    Align test labels to the train class index order.

    This prevents label mismatch if train/test class folders are sorted
    differently or if one split has fewer folders.
    """

    train_class_to_idx = {
        cname: idx
        for idx, cname in enumerate(train_class_names)
    }

    test_idx_to_class = {
        idx: cname
        for idx, cname in enumerate(test_class_names)
    }

    aligned_data = []

    for image_path, old_label in test_data:
        cname = test_idx_to_class[old_label]

        if cname not in train_class_to_idx:
            print(f"[WARN] Test class not found in train classes. Skipping: {cname}")
            continue

        aligned_label = train_class_to_idx[cname]
        aligned_data.append((image_path, aligned_label))

    return aligned_data


def load_and_resize_image(path):
    """
    Load image, convert to RGB, resize to 32x32.
    """

    img = Image.open(path).convert("RGB")
    img = img.resize(OUTPUT_IMAGE_SIZE)
    return img


def prompt_class_range(max_classes):
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

            n = max(0, min(n, max_classes - 1))
            m = max(0, min(m, max_classes - 1))

            if n > m:
                print(f"[WARN] N ({n}) > M ({m}). Swapping.")
                n, m = m, n

            return n, m

        except Exception:
            print("[WARN] Invalid input. Please enter integer values.")


def prompt_unknown_percent(default_percent=10.0):
    while True:
        try:
            raw = input(
                f"[INPUT] Percentage for UNKNOWN class from NON-selected classes [default={default_percent}%]: "
            ).strip()

            pct = default_percent if raw == "" else float(raw)
            pct = max(0.0, min(pct, 100.0))

            return pct

        except Exception:
            print("[WARN] Invalid input. Please enter a number.")


def collect_indices_by_selection(ds, selected_label_set, keep_selected=True):
    out = []

    for i in range(len(ds)):
        _, label = ds[i]
        label = int(label)

        in_set = label in selected_label_set

        if keep_selected and in_set:
            out.append(i)

        if not keep_selected and not in_set:
            out.append(i)

    return out


def sample_unknown_indices(ds, selected_label_set, unknown_percent):
    """
    Sample unknown images from NON-selected classes.
    """

    complement_indices = collect_indices_by_selection(
        ds=ds,
        selected_label_set=selected_label_set,
        keep_selected=False
    )

    total_complement = len(complement_indices)

    if total_complement == 0 or unknown_percent <= 0:
        return [], total_complement, 0

    sample_count = int(
        math.ceil((unknown_percent / 100.0) * total_complement)
    )

    sample_count = max(0, min(sample_count, total_complement))

    rng = random.Random(RANDOM_SEED)
    sampled = rng.sample(complement_indices, sample_count)

    return sampled, total_complement, sample_count


def save_counts(
    ds_root,
    class_names_exported,
    train_counts,
    test_counts,
    nottrained_counts,
    first_sizes
):
    stats = {}

    for cname_raw in class_names_exported:
        cname = safe_name(cname_raw)

        tr = int(train_counts.get(cname, 0))
        te = int(test_counts.get(cname, 0))
        ntt = int(nottrained_counts.get(cname, 0))

        size = first_sizes.get(cname)

        if size is None:
            fw, fh, fwh = None, None, None
        else:
            fw, fh = size
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

    with open(
        os.path.join(ds_root, "class_counts.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(stats, f, indent=4)

    with open(
        os.path.join(ds_root, "class_counts.csv"),
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "class",
            "train",
            "test",
            "nottrained_test",
            "total",
            "first_w",
            "first_h",
            "first_wh"
        ])

        for c, v in stats.items():
            writer.writerow([
                c,
                v["train"],
                v["test"],
                v["nottrained_test"],
                v["total"],
                v["first_w"],
                v["first_h"],
                v["first_wh"]
            ])

    return stats


# ============================================================
# EXPORT LOGIC
# ============================================================

def export_split_by_label_set(
    split_name,
    ds,
    out_root,
    class_names_all,
    allowed_label_set,
    counter,
    first_sizes,
    prefix,
    skip_if_label_not_allowed=True
):
    """
    Export images based on selected or non-selected label filtering.

    All exported images are saved as PNG and resized to 32x32.
    """

    it = range(len(ds))

    if tqdm is not None:
        it = tqdm(it, desc=f"{prefix}:{split_name}", unit="img")

    for i in it:
        img_path, label = ds[i]
        label = int(label)

        in_set = label in allowed_label_set

        if skip_if_label_not_allowed:
            if not in_set:
                continue
        else:
            if in_set:
                continue

        cname_raw = class_names_all[label]
        cname = safe_name(cname_raw)

        img = load_and_resize_image(img_path)

        if cname not in first_sizes:
            first_sizes[cname] = img.size

        fname = f"{prefix}_{split_name}_{i:06d}.png"
        img.save(os.path.join(out_root, cname, fname))

        counter[cname] += 1


def export_unknown_split(
    split_name,
    ds,
    out_root,
    sampled_indices,
    counter,
    first_sizes,
    prefix
):
    """
    Export sampled NON-selected images into synthetic unknown class.
    """

    unknown_cname = safe_name("unknown")

    it = sampled_indices

    if tqdm is not None:
        it = tqdm(it, desc=f"{prefix}:{split_name}:unknown", unit="img")

    for idx in it:
        img_path, _ = ds[idx]

        img = load_and_resize_image(img_path)

        if unknown_cname not in first_sizes:
            first_sizes[unknown_cname] = img.size

        fname = f"{prefix}_{split_name}_unknown_{idx:06d}.png"
        img.save(os.path.join(out_root, unknown_cname, fname))

        counter[unknown_cname] += 1


def print_selected_plus_unknown_counts(
    class_names_selected,
    train_counts,
    test_counts
):
    print("\n" + "=" * 70)
    print("SELECTED CLASSES INCLUDING UNKNOWN")
    print("=" * 70)

    for cname_raw in class_names_selected:
        cname = safe_name(cname_raw)

        tr = int(train_counts.get(cname, 0))
        te = int(test_counts.get(cname, 0))
        total = tr + te

        print(
            f"{cname:20s} : train={tr:<6d} test={te:<6d} total={total}"
        )

    unknown_name = safe_name("unknown")

    tr = int(train_counts.get(unknown_name, 0))
    te = int(test_counts.get(unknown_name, 0))
    total = tr + te

    print(
        f"{unknown_name:20s} : train={tr:<6d} test={te:<6d} total={total}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    base = script_dir()

    raw_dataset_root = os.path.join(base, RAW_DATASET_DIRNAME)

    source_train_root = os.path.join(
        raw_dataset_root,
        SOURCE_TRAIN_DIRNAME
    )

    source_test_root = os.path.join(
        raw_dataset_root,
        SOURCE_TEST_DIRNAME
    )

    print("[INFO] Python executable:", sys.executable)
    print("[INFO] Source root directory:", raw_dataset_root)
    print("[INFO] Source train directory:", source_train_root)
    print("[INFO] Source test directory:", source_test_root)

    # ------------------------------------------------------------
    # LOAD RAW MILITARY TRAIN AND TEST SPLITS
    # ------------------------------------------------------------
    train_ds, train_class_names = load_split_dataset(source_train_root)
    test_ds_raw, test_class_names = load_split_dataset(source_test_root)

    class_names_all = train_class_names

    test_ds = align_test_labels_to_train_classes(
        test_data=test_ds_raw,
        test_class_names=test_class_names,
        train_class_names=train_class_names
    )

    print(f"[INFO] Train images found: {len(train_ds)}")
    print(f"[INFO] Test images found: {len(test_ds)}")
    print(f"[INFO] Total train classes found: {len(class_names_all)}")

    # ------------------------------------------------------------
    # PROMPT USER FOR CLASS RANGE
    # ------------------------------------------------------------
    n, m = prompt_class_range(max_classes=len(class_names_all))

    unknown_percent = prompt_unknown_percent(
        default_percent=DEFAULT_UNKNOWN_PERCENT
    )

    # ------------------------------------------------------------
    # SELECTED CLASS RANGE
    # ------------------------------------------------------------
    selected_label_set = set(range(n, m + 1))

    class_names_selected = [
        class_names_all[i]
        for i in range(n, m + 1)
    ]

    class_names_nottrained = [
        class_names_all[i]
        for i in range(len(class_names_all))
        if i not in selected_label_set
    ]

    # ------------------------------------------------------------
    # OUTPUT DIRECTORY
    # Example:
    #   RME10_0_9_unknown
    # ------------------------------------------------------------
    export_root_name = f"{OUTPUT_PREFIX}_{n}_{m}_unknown"
    export_root = os.path.join(base, export_root_name)

    print("\n" + "=" * 100)
    print(f"[INFO] Exporting raw military classes in range: N={n} .. M={m}")
    print(f"[INFO] Output image size: {OUTPUT_IMAGE_SIZE[0]}x{OUTPUT_IMAGE_SIZE[1]}")
    print(f"[INFO] Unknown percentage from NON-selected classes: {unknown_percent:.2f}%")
    print(f"[INFO] Output directory: {export_root}")
    print("-" * 100)

    print("[INFO] Selected classes:")
    for idx in range(n, m + 1):
        print(f"  {idx:02d}: {class_names_all[idx]}")

    print("-" * 100)
    print("[INFO] Extra class:")
    print("  unknown")
    print("=" * 100 + "\n")

    ensure_dir(export_root)

    # ------------------------------------------------------------
    # CREATE OUTPUT FOLDER TREE
    # ------------------------------------------------------------
    train_root = os.path.join(export_root, "train")
    test_root = os.path.join(export_root, "test")
    ntt_root = os.path.join(export_root, "nottrained_test")

    ensure_dir(train_root)
    ensure_dir(test_root)
    ensure_dir(ntt_root)

    for cname_raw in class_names_selected:
        cname = safe_name(cname_raw)
        ensure_dir(os.path.join(train_root, cname))
        ensure_dir(os.path.join(test_root, cname))

    ensure_dir(os.path.join(train_root, safe_name("unknown")))
    ensure_dir(os.path.join(test_root, safe_name("unknown")))

    for cname_raw in class_names_nottrained:
        cname = safe_name(cname_raw)
        ensure_dir(os.path.join(ntt_root, cname))

    # ------------------------------------------------------------
    # COUNTERS
    # ------------------------------------------------------------
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    nottrained_counts = defaultdict(int)
    first_sizes = {}

    # ------------------------------------------------------------
    # EXPORT SELECTED TRAIN CLASSES
    # Source:
    #   raw_military_export/train/<selected_class>/
    #
    # Output:
    #   RME10_N_M_unknown/train/<selected_class>/
    # ------------------------------------------------------------
    export_split_by_label_set(
        split_name="train",
        ds=train_ds,
        out_root=train_root,
        class_names_all=class_names_all,
        allowed_label_set=selected_label_set,
        counter=train_counts,
        first_sizes=first_sizes,
        prefix=OUTPUT_PREFIX,
        skip_if_label_not_allowed=True
    )

    # ------------------------------------------------------------
    # EXPORT SELECTED TEST CLASSES
    # Source:
    #   raw_military_export/test/<selected_class>/
    #
    # Output:
    #   RME10_N_M_unknown/test/<selected_class>/
    # ------------------------------------------------------------
    export_split_by_label_set(
        split_name="test",
        ds=test_ds,
        out_root=test_root,
        class_names_all=class_names_all,
        allowed_label_set=selected_label_set,
        counter=test_counts,
        first_sizes=first_sizes,
        prefix=OUTPUT_PREFIX,
        skip_if_label_not_allowed=True
    )

    # ------------------------------------------------------------
    # EXPORT NON-SELECTED TEST CLASSES TO nottrained_test
    # Source:
    #   raw_military_export/test/<non_selected_class>/
    #
    # Output:
    #   RME10_N_M_unknown/nottrained_test/<non_selected_class>/
    # ------------------------------------------------------------
    export_split_by_label_set(
        split_name="nottrained_test",
        ds=test_ds,
        out_root=ntt_root,
        class_names_all=class_names_all,
        allowed_label_set=selected_label_set,
        counter=nottrained_counts,
        first_sizes=first_sizes,
        prefix=OUTPUT_PREFIX,
        skip_if_label_not_allowed=False
    )

    # ------------------------------------------------------------
    # BUILD UNKNOWN CLASS
    #
    # Train unknown:
    #   sampled from raw_military_export/train/<non_selected_class>/
    #
    # Test unknown:
    #   sampled from raw_military_export/test/<non_selected_class>/
    # ------------------------------------------------------------
    train_unknown_indices, _, _ = sample_unknown_indices(
        ds=train_ds,
        selected_label_set=selected_label_set,
        unknown_percent=unknown_percent
    )

    test_unknown_indices, _, _ = sample_unknown_indices(
        ds=test_ds,
        selected_label_set=selected_label_set,
        unknown_percent=unknown_percent
    )

    export_unknown_split(
        split_name="train",
        ds=train_ds,
        out_root=train_root,
        sampled_indices=train_unknown_indices,
        counter=train_counts,
        first_sizes=first_sizes,
        prefix=OUTPUT_PREFIX
    )

    export_unknown_split(
        split_name="test",
        ds=test_ds,
        out_root=test_root,
        sampled_indices=test_unknown_indices,
        counter=test_counts,
        first_sizes=first_sizes,
        prefix=OUTPUT_PREFIX
    )

    # ------------------------------------------------------------
    # SAVE STATISTICS
    # ------------------------------------------------------------
    class_names_exported = (
        class_names_selected
        + class_names_nottrained
        + ["unknown"]
    )

    save_counts(
        export_root,
        class_names_exported,
        train_counts,
        test_counts,
        nottrained_counts,
        first_sizes
    )

    # ------------------------------------------------------------
    # PRINT SUMMARY
    # ------------------------------------------------------------
    print_selected_plus_unknown_counts(
        class_names_selected=class_names_selected,
        train_counts=train_counts,
        test_counts=test_counts
    )

    print("\n📁 Output folder:", export_root)
    print("✅ Done.")


if __name__ == "__main__":
    main()