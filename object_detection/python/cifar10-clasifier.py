# ============================================================
# multi_model_inference_directory.py
# ============================================================
# ✅ Inference-only classifier script (NO training)
#
# WHAT THIS SCRIPT DOES:
# ------------------------------------------------------------
# ✅ You define 1+ trained models in a global list (MODELS)
# ✅ Each model has:
#     - weights file path
#     - its OWN class list (the classes it was trained on)
# ✅ You provide an INPUT directory that contains images
# ✅ For each image:
#     - run ALL models
#     - compute confidence per model
#     - choose the model/class with MAX raw confidence
# ✅ Prints per-image result and a summary
# ✅ Optionally displays each tested image enlarged
# ✅ Before displaying a new image, the old displayed image
#    window is closed first
# ✅ Optional keyboard control:
#     - press ENTER to move to the next tested image
#
# NOTES:
# ------------------------------------------------------------
# • No ImageFolder dataset required
# • No train_loader / test_loader / train_model() needed
# • Works with ANY input directory that contains images
# • Uses your CNN architecture: 3→128→256→512→1024 + GAP + FC
# • Number of classes is VARIABLE per model
#   (it is determined automatically from len(cfg["classes"]))
# ============================================================


# ============================================================
# AUTO-INSTALL DEPENDENCIES (RUNS ONCE AT SCRIPT START)
# ============================================================

import sys
import subprocess
import importlib


def _pip_install(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *pkgs])


def _ensure_import(import_name, pip_name=None):
    try:
        importlib.import_module(import_name)
    except Exception:
        _pip_install([pip_name or import_name])
        importlib.import_module(import_name)


def ensure_deps_for_this_script():
    # ---- Core ML stack ----
    try:
        importlib.import_module("torch")
        importlib.import_module("torchvision")
    except Exception:
        print("[AUTO-INSTALL] Installing PyTorch stack...")
        _pip_install(["torch", "torchvision", "torchaudio"])

    # ---- Utilities ----
    _ensure_import("numpy")
    _ensure_import("PIL", "pillow")
    _ensure_import("tqdm")


# 🔥 RUN AUTO-INSTALL NOW
ensure_deps_for_this_script()


# ============================================================
# NORMAL IMPORTS (SAFE AFTER AUTO-INSTALL)
# ============================================================

import os
import time
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageTk
from tqdm import tqdm
import tkinter as tk


# ============================================================
# GLOBAL CONFIG
# ============================================================

DEBUG_FLAG = True

# ------------------------------------------------------------
# Directory containing unknown images
# ------------------------------------------------------------
TEST_IMAGE_DIR = "../../../data/cifar10_clasifier_test"

# ------------------------------------------------------------
# Process only these extensions
# ------------------------------------------------------------
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ------------------------------------------------------------
# Inference batch size
# ------------------------------------------------------------
INFER_BATCH_SIZE = 64

# ------------------------------------------------------------
# OPTIONAL DISPLAY FEATURE
# ------------------------------------------------------------
# DISPLAY_TESTED_IMAGE:
#   • If True  -> show each tested image enlarged
#   • If False -> no image display, only console output
#
# ENLARGE_FACTOR:
#   • Enlarges displayed image width and height by this factor
#   • Default requested value = 6
# ------------------------------------------------------------
DISPLAY_TESTED_IMAGE = True
ENLARGE_FACTOR = 6

# ------------------------------------------------------------
# OPTIONAL KEYBOARD CONTROL
# ------------------------------------------------------------
# If True:
#   after each tested image is shown, the script waits until
#   you press ENTER before continuing to the next image.
#
# If False:
#   the script continues automatically.
# ------------------------------------------------------------
WAIT_FOR_ENTER_BETWEEN_IMAGES = True

# ------------------------------------------------------------
# Optional directory where displayed images are saved
# ------------------------------------------------------------
DISPLAY_OUTPUT_DIR = "displayed_results"

# ------------------------------------------------------------
# Optional Tk window title
# ------------------------------------------------------------
DISPLAY_WINDOW_TITLE = "Tested Image Viewer"


# ============================================================
# MODEL ARCH CONSTANTS (YOUR ARCH)
# ============================================================

CONV1_IN_CHANNELS = 3
CONV1_OUT_CHANNELS = 128

CONV2_IN_CHANNELS = 128
CONV2_OUT_CHANNELS = 256

CONV3_IN_CHANNELS = 256
CONV3_OUT_CHANNELS = 512

CONV4_IN_CHANNELS = 512
CONV4_OUT_CHANNELS = 1024


# ============================================================
# GLOBAL CLASSES
# ============================================================
# IMPORTANT:
# ----------
# Each model must have the exact class order used in training.
# The model predicts an index [0..C-1] which maps to this list.
#
# FLEXIBILITY:
# ------------
# The current script supports VARIABLE numbers of classes.
# Each model can have a different number of classes.
#
# Example:
#   model A -> 10 classes
#   model B -> 15 classes
#   model C -> 8 classes
#
# The model output layer is created automatically using:
#
#     num_classes = len(cfg["classes"])
#
# So the FC layer becomes:
#
#     Linear(1024 -> num_classes)
#
# ------------------------------------------------------------
# Example class groups:
# ------------------------------------------------------------

# FIRST 10 CIFAR-100 CLASSES: indices 00 → 09
CIFAR_10_CLASSES_1 = [
    "apple",          # 00
    "aquarium_fish",  # 01
    "baby",           # 02
    "bear",           # 03
    "beaver",         # 04
    "bed",            # 05
    "bee",            # 06
    "beetle",         # 07
    "bicycle",        # 08
    "bottle",         # 09
]

# SECOND 10 CIFAR-100 CLASSES: indices 10 → 19
CIFAR_10_CLASSES_2 = [
    "bowl",         # 10
    "boy",          # 11
    "bridge",       # 12
    "bus",          # 13
    "butterfly",    # 14
    "camel",        # 15
    "can",          # 16
    "castle",       # 17
    "caterpillar",  # 18
    "cattle",       # 19
]

# Optional safety checks for these example lists
if len(CIFAR_10_CLASSES_1) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_1 must contain at least 2 classes.")

if len(CIFAR_10_CLASSES_2) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_2 must contain at least 2 classes.")


# ============================================================
# MODEL REGISTRY
# ============================================================
# Put the exact saved weight filename in "weights".
#
# IMPORTANT:
# ----------
# If your real files were saved with ".pth", include ".pth".
# If your real files were saved without ".pth", keep them without ".pth".
#
# FLEXIBILITY:
# ------------
# Each model can define its own class list size.
# The script automatically creates the correct FC output size.
# ============================================================

MODEL_BASE_DIR = "../../../"

MODELS: List[Dict] = [
    {
        "name": "cifar00-09",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "cifar-00-09-cnn-128-256-512-1024-576s-L5463-A1000-T8830"
        ),
        "classes": CIFAR_10_CLASSES_1,
        "temperature": 1.0,
    },

    {
        "name": "cifar10-19",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "cifar-10-19-cnn-128-256-512-1024-375s-L5232-A1000-T8610"
        ),
        "classes": CIFAR_10_CLASSES_2,
        "temperature": 1.0,
    },
]


# ============================================================
# DEBUG PRINT
# ============================================================

def debug_print(*args, **kwargs):
    if DEBUG_FLAG:
        print(*args, **kwargs)


# ============================================================
# GLOBAL DISPLAY STATE
# ============================================================
# We keep one window reference here.
# Before showing the next image, we destroy the old window.
# ============================================================

DISPLAY_ROOT = None
DISPLAY_LABEL = None
DISPLAY_PHOTO = None


# ============================================================
# MODEL DEFINITION (INFERENCE-READY)
# ============================================================

class StaticInitLearnableCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ============================================================
        # INPUT IMAGE
        # ============================================================
        # Expected input shape:
        #
        #     [B, 3, 32, 32]
        #
        # Meaning:
        #   B = batch size
        #   3 = RGB channels
        #   32x32 = resized image size
        #
        # Total input values per image:
        #
        #     3 × 32 × 32 = 3,072
        # ============================================================


        # ============================================================
        # CONVOLUTION LAYER 1
        # ============================================================
        # Conv2d(3 -> 128, kernel=3x3, padding=1)
        #
        # Output shape:
        #     128 × 32 × 32
        #
        # Total output neurons:
        #     128 × 32 × 32 = 131,072
        #
        # Input size seen by each neuron:
        #     3 × 3 × 3 = 27 inputs
        #
        # Each of the 128 filters has:
        #     27 weights + 1 bias
        # ============================================================
        self.conv1 = nn.Conv2d(CONV1_IN_CHANNELS, CONV1_OUT_CHANNELS, kernel_size=3, padding=1, bias=True)

        # BatchNorm for 128 channels
        self.bn1 = nn.BatchNorm2d(CONV1_OUT_CHANNELS)

        # MaxPool halves spatial size:
        #     128 × 32 × 32 -> 128 × 16 × 16
        self.pool = nn.MaxPool2d(2, 2)


        # ============================================================
        # CONVOLUTION LAYER 2
        # ============================================================
        # Conv2d(128 -> 256, kernel=3x3, padding=1)
        #
        # Input shape:
        #     128 × 16 × 16
        #
        # Output shape:
        #     256 × 16 × 16
        #
        # Total output neurons:
        #     256 × 16 × 16 = 65,536
        #
        # Input size seen by each neuron:
        #     3 × 3 × 128 = 1,152 inputs
        #
        # Each filter has:
        #     1,152 weights + 1 bias
        # ============================================================
        self.conv2 = nn.Conv2d(CONV2_IN_CHANNELS, CONV2_OUT_CHANNELS, kernel_size=3, padding=1, bias=True)

        # BatchNorm for 256 channels
        self.bn2 = nn.BatchNorm2d(CONV2_OUT_CHANNELS)


        # ============================================================
        # CONVOLUTION LAYER 3
        # ============================================================
        # Conv2d(256 -> 512, kernel=3x3, padding=1)
        #
        # Input shape:
        #     256 × 16 × 16
        #
        # Output shape:
        #     512 × 16 × 16
        #
        # Total output neurons:
        #     512 × 16 × 16 = 131,072
        #
        # Input size seen by each neuron:
        #     3 × 3 × 256 = 2,304 inputs
        #
        # Each filter has:
        #     2,304 weights + 1 bias
        # ============================================================
        self.conv3 = nn.Conv2d(CONV3_IN_CHANNELS, CONV3_OUT_CHANNELS, kernel_size=3, padding=1, bias=True)

        # BatchNorm for 512 channels
        self.bn3 = nn.BatchNorm2d(CONV3_OUT_CHANNELS)


        # ============================================================
        # CONVOLUTION LAYER 4
        # ============================================================
        # Conv2d(512 -> 1024, kernel=3x3, padding=1)
        #
        # Input shape:
        #     512 × 16 × 16
        #
        # Output shape:
        #     1024 × 16 × 16
        #
        # Total output neurons:
        #     1024 × 16 × 16 = 262,144
        #
        # Input size seen by each neuron:
        #     3 × 3 × 512 = 4,608 inputs
        #
        # Each filter has:
        #     4,608 weights + 1 bias
        # ============================================================
        self.conv4 = nn.Conv2d(CONV4_IN_CHANNELS, CONV4_OUT_CHANNELS, kernel_size=3, padding=1, bias=True)

        # BatchNorm for 1024 channels
        self.bn4 = nn.BatchNorm2d(CONV4_OUT_CHANNELS)


        # ============================================================
        # GLOBAL AVERAGE POOLING
        # ============================================================
        # Converts:
        #     1024 × 16 × 16
        # into:
        #     1024 × 1 × 1
        #
        # Final neuron count after GAP:
        #     1024 neurons
        # ============================================================
        self.gap = nn.AdaptiveAvgPool2d(1)


        # ============================================================
        # DROPOUT
        # ============================================================
        # Randomly disables 30% of the 1024 neurons during training.
        # Disabled automatically during inference because we use model.eval().
        # ============================================================
        self.dropout = nn.Dropout(p=0.3)


        # ============================================================
        # FINAL FULLY CONNECTED CLASSIFIER
        # ============================================================
        # Linear(1024 -> num_classes)
        #
        # IMPORTANT:
        # ----------
        # num_classes is VARIABLE and comes from:
        #
        #     len(cfg["classes"])
        #
        # So if one model has:
        #     10 classes -> Linear(1024 -> 10)
        #
        # and another model has:
        #     15 classes -> Linear(1024 -> 15)
        #
        # Each output neuron corresponds to one class in that model's
        # class list.
        #
        # Parameters:
        #     1024 × num_classes weights
        #     num_classes biases
        # ============================================================
        self.fc = nn.Linear(CONV4_OUT_CHANNELS, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


# ============================================================
# TRANSFORM (MUST MATCH TRAINING)
# ============================================================

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])


# ============================================================
# HELPERS
# ============================================================

def list_images_in_dir(root_dir: str) -> List[str]:
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Input directory not found: {root_dir}")

    paths = []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in ALLOWED_EXTS:
            paths.append(p)

    paths.sort()
    return paths


def load_image_tensor(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    x = INFER_TRANSFORM(img)
    return x


def safe_load_state_dict(
    model: nn.Module,
    weights_path: str,
    device: torch.device,
    expected_num_classes: int,
) -> None:
    """
    Loads state_dict safely and checks that the number of output
    classes in the checkpoint matches the class list length.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    state = torch.load(weights_path, map_location=device)

    # In case you saved a full checkpoint dict instead of pure state_dict
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    if "fc.weight" not in state:
        raise RuntimeError(
            f"Checkpoint does not contain 'fc.weight': {weights_path}"
        )

    checkpoint_num_classes = int(state["fc.weight"].shape[0])

    if checkpoint_num_classes != expected_num_classes:
        raise RuntimeError(
            f"Class count mismatch for weights file:\n"
            f"  {weights_path}\n"
            f"Checkpoint expects {checkpoint_num_classes} classes, "
            f"but your global class list defines {expected_num_classes} classes."
        )

    model.load_state_dict(state)


def make_safe_filename(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def wait_for_enter_to_continue():
    print()
    input("Press ENTER to continue to the next tested image...")
    print()


def close_previous_display_window():
    """
    Close/remove the old display window before showing the new one.
    """
    global DISPLAY_ROOT, DISPLAY_LABEL, DISPLAY_PHOTO

    if DISPLAY_ROOT is not None:
        try:
            DISPLAY_ROOT.update_idletasks()
            DISPLAY_ROOT.destroy()
        except Exception:
            pass

    DISPLAY_ROOT = None
    DISPLAY_LABEL = None
    DISPLAY_PHOTO = None


def display_tested_image(
    image_path: str,
    detected_class: str,
    confidence: float,
    winning_model: str,
    enlarge_factor: int = 6,
    display_enabled: bool = True,
    wait_for_enter: bool = True,
) -> None:
    """
    Optionally displays the tested image enlarged by enlarge_factor.

    Before displaying the new image, the old displayed image window
    is closed first.
    """
    global DISPLAY_ROOT, DISPLAY_LABEL, DISPLAY_PHOTO

    if not display_enabled:
        return

    if enlarge_factor < 1:
        enlarge_factor = 1

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[DISPLAY-SKIP] Could not open image for display: {image_path}  err={e}")
        return

    # Close the old image display before showing the new one
    close_previous_display_window()

    # Enlarge image
    w, h = img.size
    enlarged_w = w * enlarge_factor
    enlarged_h = h * enlarge_factor
    img_large = img.resize((enlarged_w, enlarged_h), Image.NEAREST)

    # Draw result text
    draw = ImageDraw.Draw(img_large)
    font = ImageFont.load_default()

    line1 = f"Detected: {detected_class}"
    line2 = f"Confidence: {confidence * 100:.2f}%"
    line3 = f"Model: {winning_model}"

    text_lines = [line1, line2, line3]

    padding = 10
    line_height = 16
    text_block_height = padding * 2 + line_height * len(text_lines)

    draw.rectangle(
        [(0, 0), (enlarged_w, text_block_height)],
        fill=(0, 0, 0)
    )

    y = padding
    for line in text_lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += line_height

    os.makedirs(DISPLAY_OUTPUT_DIR, exist_ok=True)

    base_name = os.path.basename(image_path)
    name_root, _ = os.path.splitext(base_name)

    out_name = (
        f"{make_safe_filename(name_root)}"
        f"__det_{make_safe_filename(detected_class)}"
        f"__conf_{int(round(confidence * 10000))}"
        f".png"
    )
    out_path = os.path.join(DISPLAY_OUTPUT_DIR, out_name)

    try:
        img_large.save(out_path)
    except Exception as e:
        print(f"[DISPLAY-SKIP] Could not save display image: {image_path}  err={e}")
        return

    # Show in Tkinter window
    try:
        DISPLAY_ROOT = tk.Tk()
        DISPLAY_ROOT.title(DISPLAY_WINDOW_TITLE)

        DISPLAY_PHOTO = ImageTk.PhotoImage(img_large)
        DISPLAY_LABEL = tk.Label(DISPLAY_ROOT, image=DISPLAY_PHOTO)
        DISPLAY_LABEL.pack()

        DISPLAY_ROOT.update_idletasks()
        DISPLAY_ROOT.update()
    except Exception as e:
        print(f"[DISPLAY-SKIP] Could not display image in window: {image_path}  err={e}")
        close_previous_display_window()
        return

    if wait_for_enter:
        wait_for_enter_to_continue()
        close_previous_display_window()


# ============================================================
# MAIN MULTI-MODEL DIRECTORY CLASSIFIER
# ============================================================

def run_directory_multi_model_classifier(
    image_paths: List[str],
    models_cfg: List[Dict],
    device: torch.device,
) -> None:
    """
    For each image:
      - run all models
      - compute raw probabilities
      - choose the model with MAX raw top1 confidence
      - print:
          • final detected class
          • final confidence
          • per-model detected class + confidence
      - optionally display the tested image enlarged
      - optionally wait for ENTER before next image
    """

    if len(image_paths) == 0:
        print("❌ No images found in input directory.")
        return

    loaded_models = []

    for cfg in models_cfg:
        classes = cfg["classes"]

        # ------------------------------------------------------------
        # FLEXIBLE CLASS COUNT
        # ------------------------------------------------------------
        # The number of classes for each model is determined
        # automatically from the length of the class list.
        #
        # This allows different models to have different numbers
        # of classes.
        #
        # Example:
        #     model A -> 10 classes
        #     model B -> 15 classes
        #     model C -> 8 classes
        #
        # The CNN output layer will automatically match.
        # ------------------------------------------------------------
        num_classes = len(classes)

        if num_classes < 2:
            raise RuntimeError(
                f"Model {cfg['name']!r} must define at least 2 classes."
            )

        model = StaticInitLearnableCNN(num_classes=num_classes)

        safe_load_state_dict(
            model=model,
            weights_path=cfg["weights"],
            device=device,
            expected_num_classes=num_classes,
        )

        model.to(device)
        model.eval()

        loaded_models.append((cfg, model))

        debug_print(f"[LOAD] model={cfg['name']!r}")
        debug_print(f"       weights={cfg['weights']!r}")
        debug_print(f"       num_classes={num_classes}")
        debug_print(f"       classes={classes}")

    pin = (device.type == "cuda")

    print("\n============================================================")
    print("MULTI-MODEL DIRECTORY CLASSIFIER (Inference Only)")
    print("============================================================")
    print(f"Device                 : {device}")
    print(f"Input dir images       : {len(image_paths)}")
    print(f"Batch size             : {INFER_BATCH_SIZE}")
    print(f"Models loaded          : {len(loaded_models)}")
    print(f"Display enabled        : {DISPLAY_TESTED_IMAGE}")
    print(f"Enlarge factor         : {ENLARGE_FACTOR}")
    print(f"Wait for ENTER         : {WAIT_FOR_ENTER_BETWEEN_IMAGES}")
    print("Selection method       : MAX raw confidence")
    print("============================================================\n")

    t0 = time.perf_counter()

    total_images_processed = 0
    per_model_wins = {cfg["name"]: 0 for cfg, _ in loaded_models}

    for start in tqdm(range(0, len(image_paths), INFER_BATCH_SIZE), desc="Classifying"):
        batch_paths = image_paths[start:start + INFER_BATCH_SIZE]

        xs = []
        ok_paths = []

        for p in batch_paths:
            try:
                xs.append(load_image_tensor(p))
                ok_paths.append(p)
            except Exception as e:
                print(f"[SKIP] failed to load image: {p}  err={e}")

        if len(xs) == 0:
            continue

        x = torch.stack(xs, dim=0).to(device, non_blocking=pin)

        # --------------------------------------------------------
        # For each image in the batch, keep the best overall result
        # across all models using raw top1 confidence.
        # --------------------------------------------------------
        best_name = [""] * len(ok_paths)
        best_pred = [-1] * len(ok_paths)
        best_conf = [-1.0] * len(ok_paths)
        best_cls = [""] * len(ok_paths)

        # Store each model result for each image
        per_image_all_results = [[] for _ in range(len(ok_paths))]

        with torch.no_grad():
            for cfg, model in loaded_models:
                logits = model(x)
                temp = float(cfg.get("temperature", 1.0) or 1.0)

                probs = torch.softmax(logits / temp, dim=1)   # [B, Cmodel]
                confs, pred_ids = torch.max(probs, dim=1)     # [B]

                pred_ids = pred_ids.detach().cpu().tolist()
                confs = confs.detach().cpu().tolist()

                classes = cfg["classes"]
                model_name = cfg["name"]

                for i in range(len(ok_paths)):
                    conf = float(confs[i])
                    pid = int(pred_ids[i])
                    cls_name = classes[pid] if 0 <= pid < len(classes) else f"class_{pid}"

                    # Save this model's result for this image
                    per_image_all_results[i].append(
                        {
                            "model_name": model_name,
                            "detected_class": cls_name,
                            "pred_id": pid,
                            "confidence": conf,
                        }
                    )

                    # ----------------------------------------------------
                    # WINNER SELECTION:
                    # ----------------------------------------------------
                    # Choose the model that has the LARGEST raw top1
                    # confidence for this image.
                    # ----------------------------------------------------
                    if conf > best_conf[i]:
                        best_conf[i] = conf
                        best_pred[i] = pid
                        best_name[i] = model_name
                        best_cls[i] = cls_name

        for i, p in enumerate(ok_paths):
            total_images_processed += 1
            per_model_wins[best_name[i]] += 1

            print("------------------------------------------------------------")
            print(f"IMAGE FILE         : {os.path.basename(p)}")
            print(f"FINAL DETECTION    : {best_cls[i]}")
            print(f"FINAL CONFIDENCE   : {best_conf[i] * 100:.2f}%")
            print(f"WINNING MODEL      : {best_name[i]}")
            print("MODEL-BY-MODEL RESULTS:")

            for result in per_image_all_results[i]:
                print(
                    f"  - {result['model_name']:<15} -> "
                    f"detected={result['detected_class']:<20} "
                    f"confidence={result['confidence'] * 100:6.2f}%"
                )

            display_tested_image(
                image_path=p,
                detected_class=best_cls[i],
                confidence=best_conf[i],
                winning_model=best_name[i],
                enlarge_factor=ENLARGE_FACTOR,
                display_enabled=DISPLAY_TESTED_IMAGE,
                wait_for_enter=WAIT_FOR_ENTER_BETWEEN_IMAGES,
            )

        print("------------------------------------------------------------")

    dt = time.perf_counter() - t0

    # Make sure any last window is closed when processing ends
    close_previous_display_window()

    print("\n============================================================")
    print("SUMMARY")
    print("============================================================")
    print(f"Total images processed : {total_images_processed}")
    print(f"Total time             : {dt:.2f} sec")
    if total_images_processed > 0:
        print(f"Avg time / image       : {dt / total_images_processed:.4f} sec")
    print("------------------------------------------------------------")
    print("Model win counts (how many times each model had max confidence):")
    for k, v in per_model_wins.items():
        print(f"  {k:<20} : {v}")
    print("============================================================\n")


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_print("Using device:", device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if not MODELS:
        print("❌ MODELS list is empty. Add at least one model in global MODELS.")
        return

    image_paths = list_images_in_dir(TEST_IMAGE_DIR)

    run_directory_multi_model_classifier(
        image_paths=image_paths,
        models_cfg=MODELS,
        device=device,
    )


if __name__ == "__main__":
    main()