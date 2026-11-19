import os
import sys
import json
import csv
import math
import random
from collections import defaultdict
from PIL import Image

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ============================================================
# CONFIG
# ============================================================

SOURCE_DATASET_DIRNAME = "medical_datasets"

SOURCE_TRAIN_DIRNAME = "train"
SOURCE_TEST_DIRNAME = "test"

OUTPUT_PREFIX = "MED10"

DEFAULT_UNKNOWN_PERCENT = 10.0
RANDOM_SEED = 42

OUTPUT_IMAGE_SIZE = (32, 32)

ALLOWED_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


# ============================================================
# HELPERS
# ============================================================

def base_dir():
    return os.getcwd()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_name(s):
    return "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in str(s)
    )


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTS


def format_unknown_percent_for_name(pct):
    if float(pct).is_integer():
        return str(int(pct))
    return str(pct).replace(".", "p")


def load_and_resize_image(path):
    img = Image.open(path).convert("RGB")  # keep RGB (no grayscale)
    img = img.resize((32, 32), Image.Resampling.LANCZOS)  # high-quality resize
    return img

def load_split_dataset(split_root):
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"Required directory not found: {split_root}")

    class_names = sorted([
        d for d in os.listdir(split_root)
        if os.path.isdir(os.path.join(split_root, d))
    ])

    if not class_names:
        raise RuntimeError(f"No class folders found inside: {split_root}")

    class_to_idx = {cname: idx for idx, cname in enumerate(class_names)}

    data = []

    for cname in class_names:
        class_dir = os.path.join(split_root, cname)

        for current_dir, _, files in os.walk(class_dir):
            for fname in files:
                if not is_image_file(fname):
                    continue

                image_path = os.path.join(current_dir, fname)
                data.append((image_path, class_to_idx[cname]))

    if not data:
        raise RuntimeError(f"No images found inside: {split_root}")

    return data, class_names


def align_test_labels_to_train_classes(test_data, test_class_names, train_class_names):
    train_class_to_idx = {
        cname: idx for idx, cname in enumerate(train_class_names)
    }

    test_idx_to_class = {
        idx: cname for idx, cname in enumerate(test_class_names)
    }

    aligned_data = []

    for image_path, old_label in test_data:
        cname = test_idx_to_class[old_label]

        if cname not in train_class_to_idx:
            print(f"[WARN] Test class not found in train. Skipping: {cname}")
            continue

        aligned_data.append((image_path, train_class_to_idx[cname]))

    return aligned_data


def prompt_class_range(max_classes):
    while True:
        try:
            raw_n = input(
                f"\n[INPUT] Beginning class index N (0-{max_classes - 1}) [default=0]: "
            ).strip()

            raw_m = input(
                f"[INPUT] Ending class index M (0-{max_classes - 1}) [default={max_classes - 1}]: "
            ).strip()

            n = 0 if raw_n == "" else int(raw_n)
            m = max_classes - 1 if raw_m == "" else int(raw_m)

            n = max(0, min(n, max_classes - 1))
            m = max(0, min(m, max_classes - 1))

            if n > m:
                print(f"[WARN] N ({n}) > M ({m}). Swapping.")
                n, m = m, n

            return n, m

        except Exception:
            print("[WARN] Invalid input. Enter integer values.")


def prompt_unknown_percent(default_percent):
    while True:
        try:
            raw = input(
                f"[INPUT] Percentage for UNKNOWN from non-selected classes [default={default_percent}%]: "
            ).strip()

            pct = default_percent if raw == "" else float(raw)
            pct = max(0.0, min(pct, 100.0))

            return pct

        except Exception:
            print("[WARN] Invalid input. Enter a number.")


def collect_indices_by_selection(ds, selected_label_set, keep_selected=True):
    out = []

    for i, (_, label) in enumerate(ds):
        label = int(label)
        in_set = label in selected_label_set

        if keep_selected and in_set:
            out.append(i)

        if not keep_selected and not in_set:
            out.append(i)

    return out


def sample_unknown_indices(ds, selected_label_set, unknown_percent):
    complement_indices = collect_indices_by_selection(
        ds=ds,
        selected_label_set=selected_label_set,
        keep_selected=False
    )

    total_complement = len(complement_indices)

    if total_complement == 0:
        print("[WARN] No non-selected images are available for UNKNOWN.")
        return []

    if unknown_percent <= 0:
        print("[WARN] UNKNOWN percentage is 0, so no UNKNOWN images will be exported.")
        return []

    sample_count = int(math.ceil((unknown_percent / 100.0) * total_complement))

    # Guarantee at least 1 unknown image if non-selected images exist.
    sample_count = max(1, min(sample_count, total_complement))

    rng = random.Random(RANDOM_SEED)
    return rng.sample(complement_indices, sample_count)


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
        nt = int(nottrained_counts.get(cname, 0))

        size = first_sizes.get(cname)

        if size is None:
            fw, fh, fwh = None, None, None
        else:
            fw, fh = size
            fwh = f"{fw}x{fh}"

        stats[cname] = {
            "train": tr,
            "test": te,
            "nottrained_test": nt,
            "total": tr + te + nt,
            "first_w": fw,
            "first_h": fh,
            "first_wh": fwh,
        }

    json_path = os.path.join(ds_root, "class_counts.json")
    csv_path = os.path.join(ds_root, "class_counts.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "class",
            "train",
            "test",
            "nottrained_test",
            "total",
            "first_w",
            "first_h",
            "first_wh",
        ])

        for cname, values in stats.items():
            writer.writerow([
                cname,
                values["train"],
                values["test"],
                values["nottrained_test"],
                values["total"],
                values["first_w"],
                values["first_h"],
                values["first_wh"],
            ])

    print(f"[INFO] Saved counts: {json_path}")
    print(f"[INFO] Saved counts: {csv_path}")


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

        class_out_dir = os.path.join(out_root, cname)
        ensure_dir(class_out_dir)

        img = load_and_resize_image(img_path)

        if cname not in first_sizes:
            first_sizes[cname] = img.size

        fname = f"{prefix}_{split_name}_{i:06d}.png"
        img.save(os.path.join(class_out_dir, fname))

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
    if not sampled_indices:
        print(f"[WARN] No UNKNOWN images for {split_name}. UNKNOWN folder will NOT be created.")
        return

    unknown_cname = safe_name("unknown")
    unknown_dir = os.path.join(out_root, unknown_cname)
    ensure_dir(unknown_dir)

    it = sampled_indices

    if tqdm is not None:
        it = tqdm(it, desc=f"{prefix}:{split_name}:unknown", unit="img")

    for idx in it:
        img_path, _ = ds[idx]

        img = load_and_resize_image(img_path)

        if unknown_cname not in first_sizes:
            first_sizes[unknown_cname] = img.size

        fname = f"{prefix}_{split_name}_unknown_{idx:06d}.png"
        img.save(os.path.join(unknown_dir, fname))

        counter[unknown_cname] += 1


def print_selected_plus_unknown_counts(class_names_selected, train_counts, test_counts):
    print("\n" + "=" * 70)
    print("SELECTED CLASSES INCLUDING UNKNOWN")
    print("=" * 70)

    for cname_raw in class_names_selected:
        cname = safe_name(cname_raw)

        tr = int(train_counts.get(cname, 0))
        te = int(test_counts.get(cname, 0))

        print(f"{cname:25s} : train={tr:<6d} test={te:<6d} total={tr + te}")

    unknown_name = safe_name("unknown")

    tr = int(train_counts.get(unknown_name, 0))
    te = int(test_counts.get(unknown_name, 0))

    print(f"{unknown_name:25s} : train={tr:<6d} test={te:<6d} total={tr + te}")
    print("=" * 70)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    base = base_dir()

    source_dataset_root = os.path.join(base, SOURCE_DATASET_DIRNAME)
    source_train_root = os.path.join(source_dataset_root, SOURCE_TRAIN_DIRNAME)
    source_test_root = os.path.join(source_dataset_root, SOURCE_TEST_DIRNAME)

    print("=" * 80)
    print("[INFO] Medical dataset installer")
    print("=" * 80)
    print("[INFO] Python executable:", sys.executable)
    print("[INFO] Base directory    :", base)
    print("[INFO] Source root       :", source_dataset_root)
    print("[INFO] Source train      :", source_train_root)
    print("[INFO] Source test       :", source_test_root)
    print("=" * 80)

    train_ds, train_class_names = load_split_dataset(source_train_root)
    test_ds_raw, test_class_names = load_split_dataset(source_test_root)

    class_names_all = train_class_names

    test_ds = align_test_labels_to_train_classes(
        test_data=test_ds_raw,
        test_class_names=test_class_names,
        train_class_names=train_class_names
    )

    print(f"[INFO] Train images found       : {len(train_ds)}")
    print(f"[INFO] Test images found        : {len(test_ds)}")
    print(f"[INFO] Total train classes found: {len(class_names_all)}")

    print("\n[INFO] Class index map:")
    for idx, cname in enumerate(class_names_all):
        print(f"  {idx:03d}: {cname}")

    n, m = prompt_class_range(max_classes=len(class_names_all))

    unknown_percent = prompt_unknown_percent(
        default_percent=DEFAULT_UNKNOWN_PERCENT
    )

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

    unknown_pct_name = format_unknown_percent_for_name(unknown_percent)

    export_root_name = f"{OUTPUT_PREFIX}-{n:02d}-{m:02d}-unknown{unknown_pct_name}"
    export_root = os.path.join(base, export_root_name)

    train_root = os.path.join(export_root, "train")
    test_root = os.path.join(export_root, "test")
    ntt_root = os.path.join(export_root, "nottrained_test")

    print("\n" + "=" * 100)
    print(f"[INFO] Exporting selected medical classes: N={n} .. M={m}")
    print(f"[INFO] Output image size: {OUTPUT_IMAGE_SIZE[0]}x{OUTPUT_IMAGE_SIZE[1]}")
    print(f"[INFO] Unknown percentage from non-selected classes: {unknown_percent:.2f}%")
    print(f"[INFO] Output directory: {export_root}")
    print("-" * 100)

    print("[INFO] Selected classes:")
    for idx in range(n, m + 1):
        print(f"  {idx:03d}: {class_names_all[idx]}")

    print("-" * 100)
    print("[INFO] Extra class: unknown")
    print("=" * 100 + "\n")

    ensure_dir(train_root)
    ensure_dir(test_root)
    ensure_dir(ntt_root)

    for cname_raw in class_names_selected:
        cname = safe_name(cname_raw)
        ensure_dir(os.path.join(train_root, cname))
        ensure_dir(os.path.join(test_root, cname))

    for cname_raw in class_names_nottrained:
        cname = safe_name(cname_raw)
        ensure_dir(os.path.join(ntt_root, cname))

    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    nottrained_counts = defaultdict(int)
    first_sizes = {}

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

    train_unknown_indices = sample_unknown_indices(
        ds=train_ds,
        selected_label_set=selected_label_set,
        unknown_percent=unknown_percent
    )

    test_unknown_indices = sample_unknown_indices(
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

    if unknown_percent > 0:
        if len(test_unknown_indices) == 0:
            print("[WARN] test/unknown was not created because there are no non-selected test images.")
            print("[WARN] To create test/unknown, select fewer classes so non-selected test classes exist.")
        elif test_counts.get("unknown", 0) <= 0:
            raise RuntimeError("UNKNOWN was expected, but test/unknown has 0 images.")

    class_names_exported = (
        class_names_selected
        + class_names_nottrained
    )

    if train_counts.get("unknown", 0) > 0 or test_counts.get("unknown", 0) > 0:
        class_names_exported.append("unknown")

    save_counts(
        ds_root=export_root,
        class_names_exported=class_names_exported,
        train_counts=train_counts,
        test_counts=test_counts,
        nottrained_counts=nottrained_counts,
        first_sizes=first_sizes
    )

    print_selected_plus_unknown_counts(
        class_names_selected=class_names_selected,
        train_counts=train_counts,
        test_counts=test_counts
    )

    print("\n📁 Output folder:", export_root)
    print("✅ Done.")


if __name__ == "__main__":
    main()