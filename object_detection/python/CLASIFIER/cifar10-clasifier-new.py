# ============================================================
# multi_model_inference_directory_trained_model.py
# ============================================================
# INFERENCE-ONLY ROUTINE
#
# PURPOSE
# ------------------------------------------------------------
# This script assumes:
#   1) You already trained the model before
#   2) You already saved the trained weights to disk
#   3) This script will LOAD those trained weights
#   4) Then it will classify images from a directory
#
# IMPORTANT
# ------------------------------------------------------------
# This script DOES NOT train.
# It ONLY:
#   - creates the same CNN architecture
#   - loads the trained checkpoint into that architecture
#   - switches the model to evaluation mode
#   - runs inference on test images
#
# MODEL SELECTION
# ------------------------------------------------------------
# You can define one or more trained models in MODELS.
# For each image:
#   - every trained model runs inference
#   - we compare predictions across all models
#   - FINAL DETECTION is chosen as the highest-confidence class
#     whose name does NOT start with "unknown"
#
# DEBUG FEATURES
# ------------------------------------------------------------
# This version prints:
#   - which trained weights file was loaded
#   - checkpoint FC shape
#   - per-model top-3 predictions
#   - final winner
#
# ============================================================


# ============================================================
# AUTO-INSTALL DEPENDENCIES
# ============================================================

import sys
import subprocess
import importlib


def _pip_install(pkgs):
    """
    Install packages into the SAME Python interpreter running this script.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *pkgs])


def _ensure_import(import_name, pip_name=None):
    """
    Try import. If missing, install, then import again.
    """
    try:
        importlib.import_module(import_name)
    except Exception:
        _pip_install([pip_name or import_name])
        importlib.import_module(import_name)


def ensure_deps_for_this_script():
    """
    Ensure all required packages are available.
    """
    try:
        importlib.import_module("torch")
        importlib.import_module("torchvision")
    except Exception:
        print("[AUTO-INSTALL] Installing PyTorch stack...")
        _pip_install(["torch", "torchvision", "torchaudio"])

    _ensure_import("numpy")
    _ensure_import("PIL", "pillow")
    _ensure_import("tqdm")


ensure_deps_for_this_script()


# ============================================================
# NORMAL IMPORTS
# ============================================================

import os
import time
from typing import List, Dict, Tuple

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
# Directory containing unknown images to classify
# ------------------------------------------------------------
MODEL_BASE_DIR = "../../../../"

TEST_IMAGE_DIR = os.path.join(
    MODEL_BASE_DIR,
    "data",
    "cifar10_clasifier_test"
)

# ------------------------------------------------------------
# Allowed image types
# ------------------------------------------------------------
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ------------------------------------------------------------
# Inference batch size
# ------------------------------------------------------------
INFER_BATCH_SIZE = 64

# ------------------------------------------------------------
# Display controls
# ------------------------------------------------------------
DISPLAY_TESTED_IMAGE = True
ENLARGE_FACTOR = 6
WAIT_FOR_ENTER_BETWEEN_IMAGES = True
DISPLAY_WINDOW_TITLE = "Tested Image Viewer"

# ------------------------------------------------------------
# Optional confidence threshold
# ------------------------------------------------------------
# If enabled, any winning prediction below this threshold
# will be renamed to UNKNOWN_LABEL_BELOW_THRESHOLD.
# This is independent from any class such as "unknown1".
# ------------------------------------------------------------
USE_LOW_CONFIDENCE_UNKNOWN_RULE = False
LOW_CONFIDENCE_THRESHOLD = 0.60
UNKNOWN_LABEL_BELOW_THRESHOLD = "unknown"

# ------------------------------------------------------------
# Top-K display
# ------------------------------------------------------------
TOPK_TO_PRINT = 3


# ============================================================
# MODEL ARCH CONSTANTS
# ============================================================

CONV1_IN_CHANNELS = 3
CONV1_OUT_CHANNELS = 128

CONV2_IN_CHANNELS = 128
CONV2_OUT_CHANNELS = 256

CONV3_IN_CHANNELS = 256
CONV3_OUT_CHANNELS = 512

CONV4_IN_CHANNELS = 512
CONV4_OUT_CHANNELS = 1024


# ------------------------------------------------------------
# Example group 1: first 10 classes
# ------------------------------------------------------------
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
    "unknown1",       #
]

# ------------------------------------------------------------
# Example group 2: next 10 classes
# ------------------------------------------------------------
CIFAR_10_CLASSES_2 = [
    "bottle",      # 09
    "bowl",        # 10
    "boy",         # 11
    "bridge",      # 12
    "bus",         # 13
    "butterfly",   # 14
    "camel",       # 15
    "can",         # 16
    "castle",      # 17
    "unknown2"     #
]

CIFAR_10_CLASSES_3 = [
    "caterpillar",    # 18
    "cattle",         # 19
    "chair",          # 20
    "chimpanzee",     # 21
    "clock",          # 22
    "cloud",          # 23
    "cockroach",      # 24
    "couch",          # 25
    "crab",           # 26
    "unknown3"        #
]

CIFAR_10_CLASSES_4 = [
    "crocodile",     # 27
    "cup",           # 28
    "dinosaur",      # 29
    "dolphin",       # 30
    "elephant",      # 31
    "flatfish",      # 32
    "forest",        # 33
    "fox",           # 34
    "girl",          # 35
    "unknown4"       #
]

CIFAR_10_CLASSES_5 = [
    "hamster",       # 36
    "house",         # 37
    "kangaroo",      # 38
    "keyboard",      # 39
    "lamp",          # 40
    "lawn_mower",    # 41
    "leopard",       # 42
    "lion",          # 43
    "lizard",        # 44
    "unknown5"
]

CIFAR_10_CLASSES_6 = [
    "lobster",       # 45
    "man",           # 46
    "maple_tree",    # 47
    "motorcycle",    # 48
    "mountain",      # 49
    "mouse",         # 50
    "mushroom",      # 51
    "oak_tree",      # 52
    "orange",        # 53
    "unknown6"       #
]

CIFAR_10_CLASSES_7 = [
    "orchid",        # 54
    "otter",         # 55
    "palm_tree",     # 56
    "pear",          # 57
    "pickup_truck",  # 58
    "pine_tree",     # 59
    "plain",         # 60
    "plate",         # 61
    "poppy",         # 62
    "unknown7"       #
]

CIFAR_10_CLASSES_8 = [
   "porcupine",      # 63
   "possum",         # 64
   "rabbit",         # 65
   "raccoon",        # 66
   "ray",            # 67
   "road",           # 68
   "rocket",         # 69
   "rose",           # 70
   "sea",            # 71
   "unknown8"        #
]

CIFAR_10_CLASSES_9  = [
   "seal",           # 72
   "shark",          # 73
   "shrew",          # 74
   "skunk",          # 75
   "skyscraper",     # 76
   "snail",          # 77
   "snake",          # 78
   "spider",         # 79
   "squirrel",       # 80
   "unknown10"       #
]

# ------------------------------------------------------------
# Optional safety checks for these example lists
# ------------------------------------------------------------
if len(CIFAR_10_CLASSES_1) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_1 must contain at least 2 classes.")

if len(CIFAR_10_CLASSES_2) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_2 must contain at least 2 classes.")

if len(CIFAR_10_CLASSES_3) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_3 must contain at least 2 classes.")

if len(CIFAR_10_CLASSES_4) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_4 must contain at least 2 classes.")

if len(CIFAR_10_CLASSES_5) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_5 must contain at least 2 classes.")

if len(CIFAR_10_CLASSES_6) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_6 must contain at least 2 classes.")

if len(CIFAR_10_CLASSES_7) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_7 must contain at least 2 classes.")

if len(CIFAR_10_CLASSES_8) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_8 must contain at least 2 classes.")

if len(CIFAR_10_CLASSES_9) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_9 must contain at least 2 classes.")
# ============================================================
# MODEL REGISTRY
# ============================================================

MODELS: List[Dict] = [
    {
        "name": "cifar00-08",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "cifar-00-08-unknown30-cnn-128-256-512-1024-1744s-L5205-A9999-T8766"
        ),
        "classes": CIFAR_10_CLASSES_1,
        "temperature": 1.0,
    },

    {
        "name": "cifar09-17",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "cifar-09-17-unknown30-cnn-128-256-512-1024-1744s-L5205-A9999-T8766"
        ),
        "classes": CIFAR_10_CLASSES_2,
        "temperature": 1.0,
    },

    {
        "name": "cifar18-26",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "cifar-18-26-unknown30-cnn-128-256-512-1024-1744s-L5205-A9999-T8766"
        ),
        "classes": CIFAR_10_CLASSES_3,
        "temperature": 1.0,
    },

    {
        "name": "cifar27-35",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "cifar-27-35-unknown30-cnn-128-256-512-1024-1744s-L5205-A9999-T8766"
        ),
        "classes": CIFAR_10_CLASSES_4,
        "temperature": 1.0,
    },

    {
        "name": "cifar36-44",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "cifar-36-44-unknown30-cnn-128-256-512-1024-1744s-L5205-A9999-T8766"
        ),
        "classes": CIFAR_10_CLASSES_5,
        "temperature": 1.0,
    },

    {
        "name": "cifar45-53",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "cifar-45-53-unknown30-cnn-128-256-512-1024-1744s-L5205-A9999-T8766"
        ),
        "classes": CIFAR_10_CLASSES_6,
        "temperature": 1.0,
    },

    {
        "name": "cifar54-62",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "cifar-54-62-unknown30-cnn-128-256-512-1024-1744s-L5205-A9999-T8766"
        ),
        "classes": CIFAR_10_CLASSES_7,
        "temperature": 1.0,
    },

    {
        "name": "cifar63-71",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "cifar-63-71-unknown30-cnn-128-256-512-1024-1744s-L5205-A9999-T8766"
        ),
        "classes": CIFAR_10_CLASSES_8,
        "temperature": 1.0,
    },

    {
        "name": "cifar72-80",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "cifar-72-80-unknown30-cnn-128-256-512-1024-1744s-L5205-A9999-T8766"
        ),
        "classes": CIFAR_10_CLASSES_9,
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
# DISPLAY GLOBALS
# ============================================================

DISPLAY_ROOT = None
DISPLAY_LABEL = None
DISPLAY_PHOTO = None


# ============================================================
# CNN MODEL
# ============================================================

class StaticInitLearnableCNN(nn.Module):
    """
    CNN architecture used during training and reused here for inference.

    IMPORTANT:
    The architecture here must exactly match the architecture used
    when the checkpoint was trained.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.conv1 = nn.Conv2d(CONV1_IN_CHANNELS, CONV1_OUT_CHANNELS, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(CONV1_OUT_CHANNELS)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(CONV2_IN_CHANNELS, CONV2_OUT_CHANNELS, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(CONV2_OUT_CHANNELS)

        self.conv3 = nn.Conv2d(CONV3_IN_CHANNELS, CONV3_OUT_CHANNELS, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(CONV3_OUT_CHANNELS)

        self.conv4 = nn.Conv2d(CONV4_IN_CHANNELS, CONV4_OUT_CHANNELS, kernel_size=3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(CONV4_OUT_CHANNELS)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
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
# IMAGE TRANSFORM
# ============================================================
# IMPORTANT:
# These normalization values should match the values used in training.
# If training used different mean/std values, replace them here with
# the exact same values from training.
# ============================================================

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
])


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def list_images_in_dir(root_dir: str) -> List[str]:
    """
    Return all valid image file paths sorted alphabetically.
    """
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
    """
    Open one image, convert to RGB, apply inference transform,
    and return tensor [C,H,W].
    """
    img = Image.open(image_path).convert("RGB")
    x = INFER_TRANSFORM(img)
    return x


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def safe_load_state_dict(model, weights_path: str, device: torch.device, expected_num_classes: int):
    """
    Load a TRAINED checkpoint into the model.

    This function proves the script is using a trained checkpoint because:
      - it reads the saved checkpoint from disk
      - it checks the FC output size
      - it loads the saved parameters into the fresh model object
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Trained weights file not found: {weights_path}")

    print(f"[INFO] Loading trained model weights from: {weights_path}")

    state = torch.load(weights_path, map_location=device)

    # Common checkpoint wrappers
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if "fc.weight" not in state:
        raise RuntimeError(
            f"Checkpoint does not contain 'fc.weight'. "
            f"Cannot verify class count for: {weights_path}"
        )

    checkpoint_num_classes = int(state["fc.weight"].shape[0])
    print(f"[INFO] Checkpoint fc.weight shape: {tuple(state['fc.weight'].shape)}")
    print(f"[INFO] Checkpoint output classes: {checkpoint_num_classes}")
    print(f"[INFO] Expected output classes : {expected_num_classes}")

    if checkpoint_num_classes != expected_num_classes:
        raise RuntimeError(
            f"Class count mismatch.\n"
            f"Checkpoint file : {weights_path}\n"
            f"Checkpoint FC   : {checkpoint_num_classes}\n"
            f"Your class list : {expected_num_classes}\n"
            f"Make sure cfg['classes'] matches training exactly."
        )

    model.load_state_dict(state)
    print("[INFO] Trained checkpoint loaded successfully.\n")


# ============================================================
# TOP-K HELPER
# ============================================================

def get_topk_predictions(probs_row: torch.Tensor, classes: List[str], topk: int = 3) -> List[Tuple[str, float, int]]:
    """
    Return top-k predictions for one image from one model.

    Output item format:
        (class_name, confidence, class_index)
    """
    k = min(topk, len(classes))
    vals, inds = torch.topk(probs_row, k=k, dim=0)

    vals = vals.detach().cpu().tolist()
    inds = inds.detach().cpu().tolist()

    out = []
    for conf, idx in zip(vals, inds):
        class_name = classes[idx] if 0 <= idx < len(classes) else f"class_{idx}"
        out.append((class_name, float(conf), int(idx)))
    return out


def starts_with_unknown(class_name: str) -> bool:
    """
    Return True if the class name starts with 'unknown' (case-insensitive).
    """
    return class_name.strip().lower().startswith("unknown")


# ============================================================
# KEYBOARD CONTROL
# ============================================================

def wait_for_input():
    """
    ENTER -> continue
    E     -> exit
    """
    print()
    key = input("Press ENTER for next image or 'E' to exit: ").strip().lower()
    if key == "e":
        print("Exiting program...")
        sys.exit(0)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def close_previous_display_window():
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


def display_tested_image(image_path, detected_class, confidence, winning_model):
    """
    Enlarge and display the tested image.
    """
    global DISPLAY_ROOT, DISPLAY_LABEL, DISPLAY_PHOTO

    if not DISPLAY_TESTED_IMAGE:
        return

    close_previous_display_window()

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[DISPLAY-SKIP] Could not open image: {image_path}  err={e}")
        return

    w, h = img.size
    img = img.resize((w * ENLARGE_FACTOR, h * ENLARGE_FACTOR), Image.NEAREST)

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    line1 = f"Detected: {detected_class}"
    line2 = f"Confidence: {confidence * 100:.2f}%"
    line3 = f"Model: {winning_model}"

    text_lines = [line1, line2, line3]

    padding = 10
    line_height = 16
    text_block_height = padding * 2 + line_height * len(text_lines)

    draw.rectangle((0, 0, img.size[0], text_block_height), fill=(0, 0, 0))

    y = padding
    for line in text_lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += line_height

    try:
        DISPLAY_ROOT = tk.Tk()
        DISPLAY_ROOT.title(DISPLAY_WINDOW_TITLE)

        DISPLAY_PHOTO = ImageTk.PhotoImage(img)
        DISPLAY_LABEL = tk.Label(DISPLAY_ROOT, image=DISPLAY_PHOTO)
        DISPLAY_LABEL.pack()

        DISPLAY_ROOT.update_idletasks()
        DISPLAY_ROOT.update()

    except Exception as e:
        print(f"[DISPLAY-SKIP] Could not display image: {image_path}  err={e}")
        close_previous_display_window()
        return

    if WAIT_FOR_ENTER_BETWEEN_IMAGES:
        wait_for_input()

    close_previous_display_window()


# ============================================================
# MAIN MULTI-MODEL CLASSIFIER
# ============================================================

def run_directory_multi_model_classifier(image_paths, models_cfg, device):
    """
    For each image:
      - run all trained models
      - compute probabilities
      - collect predictions from all models
      - FINAL DETECTION is chosen as the highest-confidence class
        whose name does NOT start with 'unknown'
      - if all model winners are unknown-prefixed, fall back to the
        overall highest-confidence result
      - print per-model and final results
    """

    if len(image_paths) == 0:
        print("No images found in input directory.")
        return

    loaded_models = []

    # --------------------------------------------------------
    # LOAD TRAINED MODELS
    # --------------------------------------------------------
    for cfg in models_cfg:
        num_classes = len(cfg["classes"])

        if num_classes < 2:
            raise RuntimeError(f"Model {cfg['name']!r} must define at least 2 classes.")

        print("============================================================")
        print(f"[MODEL PREPARE] Creating architecture for model: {cfg['name']}")
        print(f"[MODEL PREPARE] Number of classes          : {num_classes}")

        model = StaticInitLearnableCNN(num_classes=num_classes)

        # This is the key part:
        # load the TRAINED checkpoint into the fresh model object
        safe_load_state_dict(
            model=model,
            weights_path=cfg["weights"],
            device=device,
            expected_num_classes=num_classes
        )

        model.to(device)

        # Set inference mode
        model.eval()

        print(f"[MODEL READY] {cfg['name']} is now in evaluation mode.\n")

        loaded_models.append((cfg, model))

        debug_print(f"[DEBUG LOAD] model name = {cfg['name']}")
        debug_print(f"[DEBUG LOAD] weights    = {cfg['weights']}")
        debug_print(f"[DEBUG LOAD] classes    = {cfg['classes']}")
        debug_print()

    print("\n============================================================")
    print("MULTI-MODEL DIRECTORY CLASSIFIER (TRAINED MODELS)")
    print("============================================================")
    print(f"Device                 : {device}")
    print(f"Input dir images       : {len(image_paths)}")
    print(f"Batch size             : {INFER_BATCH_SIZE}")
    print(f"Models loaded          : {len(loaded_models)}")
    print(f"Display enabled        : {DISPLAY_TESTED_IMAGE}")
    print(f"Enlarge factor         : {ENLARGE_FACTOR}")
    print(f"Wait for ENTER         : {WAIT_FOR_ENTER_BETWEEN_IMAGES}")
    print("Selection method       : highest-confidence NON-unknown class")
    print("Fallback method        : highest-confidence overall result")
    print("============================================================\n")

    t0 = time.perf_counter()

    total_images_processed = 0
    per_model_wins = {cfg["name"]: 0 for cfg, _ in loaded_models}
    pin = (device.type == "cuda")

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
        # We keep two best trackers:
        #
        # 1) best_non_unknown_* :
        #       best result whose class does NOT start with "unknown"
        #
        # 2) best_overall_* :
        #       best result regardless of class name
        #
        # Final selection rule:
        #   use best_non_unknown_* if available
        #   otherwise fall back to best_overall_*
        # --------------------------------------------------------
        best_non_unknown_name = [""] * len(ok_paths)
        best_non_unknown_pred = [-1] * len(ok_paths)
        best_non_unknown_conf = [-1.0] * len(ok_paths)
        best_non_unknown_cls = [""] * len(ok_paths)

        best_overall_name = [""] * len(ok_paths)
        best_overall_pred = [-1] * len(ok_paths)
        best_overall_conf = [-1.0] * len(ok_paths)
        best_overall_cls = [""] * len(ok_paths)

        per_image_all_results = [[] for _ in range(len(ok_paths))]

        with torch.no_grad():
            for cfg, model in loaded_models:
                logits = model(x)

                temp = float(cfg.get("temperature", 1.0) or 1.0)
                probs = torch.softmax(logits / temp, dim=1)

                confs, pred_ids = torch.max(probs, dim=1)

                pred_ids_cpu = pred_ids.detach().cpu().tolist()
                confs_cpu = confs.detach().cpu().tolist()
                probs_cpu = probs.detach().cpu()

                classes = cfg["classes"]
                model_name = cfg["name"]

                for i in range(len(ok_paths)):
                    conf = float(confs_cpu[i])
                    pid = int(pred_ids_cpu[i])

                    cls_name = classes[pid] if 0 <= pid < len(classes) else f"class_{pid}"
                    topk_preds = get_topk_predictions(probs_cpu[i], classes, topk=TOPK_TO_PRINT)

                    per_image_all_results[i].append(
                        {
                            "model_name": model_name,
                            "detected_class": cls_name,
                            "pred_id": pid,
                            "confidence": conf,
                            "topk": topk_preds,
                            "is_unknown": starts_with_unknown(cls_name),
                        }
                    )

                    # ----------------------------------------------------
                    # OVERALL WINNER SELECTION
                    # ----------------------------------------------------
                    if conf > best_overall_conf[i]:
                        best_overall_conf[i] = conf
                        best_overall_pred[i] = pid
                        best_overall_name[i] = model_name
                        best_overall_cls[i] = cls_name

                    # ----------------------------------------------------
                    # NON-UNKNOWN WINNER SELECTION
                    # ----------------------------------------------------
                    if (not starts_with_unknown(cls_name)) and (conf > best_non_unknown_conf[i]):
                        best_non_unknown_conf[i] = conf
                        best_non_unknown_pred[i] = pid
                        best_non_unknown_name[i] = model_name
                        best_non_unknown_cls[i] = cls_name

        # --------------------------------------------------------
        # PRINT RESULTS PER IMAGE
        # --------------------------------------------------------
        for i, p in enumerate(ok_paths):
            total_images_processed += 1

            # ----------------------------------------------------
            # Final label selection:
            #   use best non-unknown result if available
            #   otherwise use overall result
            # ----------------------------------------------------
            if best_non_unknown_conf[i] >= 0.0:
                final_label = best_non_unknown_cls[i]
                final_conf = best_non_unknown_conf[i]
                final_model = best_non_unknown_name[i]
            else:
                final_label = best_overall_cls[i]
                final_conf = best_overall_conf[i]
                final_model = best_overall_name[i]

            if USE_LOW_CONFIDENCE_UNKNOWN_RULE and final_conf < LOW_CONFIDENCE_THRESHOLD:
                final_label = UNKNOWN_LABEL_BELOW_THRESHOLD

            per_model_wins[final_model] += 1

            print("------------------------------------------------------------")
            print(f"IMAGE FILE         : {os.path.basename(p)}")
            print(f"FINAL DETECTION    : {final_label}")
            print(f"FINAL CONFIDENCE   : {final_conf * 100:.2f}%")
            print(f"WINNING MODEL      : {final_model}")
            print("MODEL-BY-MODEL RESULTS:")

            for result in per_image_all_results[i]:
                print(
                    f"  - {result['model_name']:<15} -> "
                    f"detected={result['detected_class']:<20} "
                    f"confidence={result['confidence'] * 100:6.2f}%"
                )

                print("    top predictions:")
                for rank, (class_name, conf, class_idx) in enumerate(result["topk"], start=1):
                    print(
                        f"      {rank}. class={class_name:<20} "
                        f"index={class_idx:<3} confidence={conf * 100:6.2f}%"
                    )

            display_tested_image(
                image_path=p,
                detected_class=final_label,
                confidence=final_conf,
                winning_model=final_model,
            )

        print("------------------------------------------------------------")

    close_previous_display_window()

    dt = time.perf_counter() - t0

    print("\n============================================================")
    print("SUMMARY")
    print("============================================================")
    print(f"Total images processed : {total_images_processed}")
    print(f"Total time             : {dt:.2f} sec")

    if total_images_processed > 0:
        print(f"Avg time / image       : {dt / total_images_processed:.4f} sec")

    print("------------------------------------------------------------")
    print("Model win counts:")
    for k, v in per_model_wins.items():
        print(f"  {k:<20} : {v}")
    print("============================================================\n")


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main program:
      1) choose CPU or GPU
      2) list images
      3) load trained model checkpoints
      4) run inference
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_print("Using device:", device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if not MODELS:
        print("MODELS list is empty. Add at least one trained model config.")
        return

    image_paths = list_images_in_dir(TEST_IMAGE_DIR)

    run_directory_multi_model_classifier(
        image_paths=image_paths,
        models_cfg=MODELS,
        device=device
    )


if __name__ == "__main__":
    main()