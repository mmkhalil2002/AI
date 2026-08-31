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
    _ensure_import("dotenv", "python-dotenv")


ensure_deps_for_this_script()


# ============================================================
# NORMAL IMPORTS
# ============================================================

import os
import time
import socket
import threading
import struct
import json
import io
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
"""
DEBUG_FLAG = True

# ------------------------------------------------------------
# Directory containing unknown images to classify
# ------------------------------------------------------------
MODEL_BASE_DIR = "../../../../"

TEST_IMAGE_DIR = os.path.join(
    MODEL_BASE_DIR,
    "data",
    "CLASIFIER_TEST"
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
"""

import os
from dotenv import load_dotenv, find_dotenv

# Load .env
load_dotenv(find_dotenv())


# ============================================================
# HELPERS
# ============================================================
def get_str(name, default=""):
    return os.getenv(name, default)


def get_int(name, default=0):
    return int(os.getenv(name, default))


def get_float(name, default=0.0):
    return float(os.getenv(name, default))


def get_bool(name, default="False"):
    return os.getenv(name, default).lower() in ("true", "1", "yes", "on")


# ============================================================
# BASE PATH
# ============================================================
MODEL_PATH = get_str("MODEL_BASE", "../../../../")

MODEL_BASE_DIR =  os.path.join(MODEL_PATH,"model")

# ============================================================
# TEST IMAGE DIRECTORY (EXPANDED)
# ============================================================
# Default input mode reads images from:
#   <MODEL_PATH>/data/CLASIFIER_TEST
#
# You can override it from .env by setting:
#   TEST_IMAGE_DIR=D:/AI/AI/data/CLASIFIER_TEST
# ============================================================
TEST_IMAGE_DIR = os.path.expandvars(
    get_str(
        "TEST_IMAGE_DIR",
        os.path.join(MODEL_PATH, "data", "CLASIFIER_TEST")
    )
)


# ============================================================
# ALLOWED IMAGE TYPES (STATIC – keep in code)
# ============================================================
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# INFERENCE SETTINGS
# ============================================================
INFER_BATCH_SIZE = get_int("INFER_BATCH_SIZE", 64)


# ============================================================
# DISPLAY SETTINGS
# ============================================================
DISPLAY_TESTED_IMAGE = get_bool("DISPLAY_TESTED_IMAGE", "True")
DEBUG_FLAG = get_bool("DEBOG_FLAG", "False")
ENLARGE_FACTOR = get_int("ENLARGE_FACTOR", 6)
WAIT_FOR_ENTER_BETWEEN_IMAGES = get_bool("WAIT_FOR_ENTER_BETWEEN_IMAGES", "True")
DISPLAY_WINDOW_TITLE = get_str("DISPLAY_WINDOW_TITLE", "Tested Image Viewer")


# ============================================================
# CONFIDENCE THRESHOLD SETTINGS
# ============================================================
USE_LOW_CONFIDENCE_UNKNOWN_RULE = get_bool("USE_LOW_CONFIDENCE_UNKNOWN_RULE", "False")
LOW_CONFIDENCE_THRESHOLD = get_float("LOW_CONFIDENCE_THRESHOLD", 0.60)
UNKNOWN_LABEL_BELOW_THRESHOLD = get_str("UNKNOWN_LABEL_BELOW_THRESHOLD", "unknown")


# ============================================================
# TOP-K SETTINGS
# ============================================================
TOPK_TO_PRINT = get_int("TOPK_TO_PRINT", 3)


# ============================================================
# TCP SERVER SETTINGS
# ============================================================
# TCP mode allows another program/client to send one image to this
# classifier process. The classifier receives the image bytes, applies
# the same multi-model classifier routine, then returns the classification
# result to the TCP client as JSON.
#
# Protocol used by this script:
#   Client -> Server:
#       2 bytes image dimensions:
#           byte 0 = width
#           byte 1 = height
#       width * height * 3 bytes raw RGB image data
#
#   Server -> Client:
#       8 bytes unsigned big-endian json_size
#       json_size bytes containing UTF-8 JSON result
#
# IMPORTANT:
#   This protocol is for RAW RGB data, not JPG/PNG encoded files.
#   Example for a 32x32 RGB image:
#       client sends: bytes([32, 32]) + 3072 RGB bytes
#
# The default program mode is still directory classification from
# TEST_IMAGE_DIR / CLASIFIER_TEST. TCP mode is selected only when the
# user chooses it at startup.
# ============================================================
TCP_HOST = get_str("TCP_HOST", "0.0.0.0")
# TCP server port.
# Default assigned MED_PORT is 5055.
# You can override it in .env by adding: MED_PORT=5055
MED_PORT = get_int("MED_PORT", 5055)
TCP_PORT = MED_PORT
TCP_BACKLOG = get_int("TCP_BACKLOG", 5)
TCP_MAX_IMAGE_BYTES = get_int("TCP_MAX_IMAGE_BYTES", 25 * 1024 * 1024)
TCP_KEEP_RUNNING = get_bool("TCP_KEEP_RUNNING", "True")
TCP_SOCKET_TIMEOUT_SEC = get_float("TCP_SOCKET_TIMEOUT_SEC", 1.0)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================
CONV1_IN_CHANNELS = get_int("CONV1_IN_CHANNELS", 3)
CONV1_OUT_CHANNELS = get_int("CONV1_OUT_CHANNELS", 128)

CONV2_IN_CHANNELS = get_int("CONV2_IN_CHANNELS", 128)
CONV2_OUT_CHANNELS = get_int("CONV2_OUT_CHANNELS", 256)

CONV3_IN_CHANNELS = get_int("CONV3_IN_CHANNELS", 256)
CONV3_OUT_CHANNELS = get_int("CONV3_OUT_CHANNELS", 512)

CONV4_IN_CHANNELS = get_int("CONV4_IN_CHANNELS", 512)
CONV4_OUT_CHANNELS = get_int("CONV4_OUT_CHANNELS", 1024)


# ============================================================
# DEBUG PRINT (OPTIONAL)
# ============================================================
print("=" * 60)
print("[CONFIG] MODEL_PATH =", MODEL_PATH)
print("[CONFIG] TEST_IMAGE_DIR =", TEST_IMAGE_DIR)
print("[CONFIG] INFER_BATCH_SIZE =", INFER_BATCH_SIZE)
print("[CONFIG] DISPLAY_TESTED_IMAGE =", DISPLAY_TESTED_IMAGE)
print("[CONFIG] LOW_CONFIDENCE_THRESHOLD =", LOW_CONFIDENCE_THRESHOLD)
print("=" * 60)


# ------------------------------------------------------------
# Example group 1: first 10 classes
# ------------------------------------------------------------


MED_10_CLASSES_1  = [
    "actinic_keratoses_and_intraepithelial_carcinoma",   # 0
    "adipose",                                           # 1
    "background",                                        # 2
    "basal_cell_carcinoma",                              # 3
    "basophil",                                          # 4
    "benign_keratosis-like_lesions",                     # 5
    "bladder",                                           # 6
    "cancer-associated_stroma",                          # 7
    "colorectal_adenocarcinoma_epithelium",              # 8
    "unknownMED1"                                        # 
]
  
MED_10_CLASSES_2  = [
    "debris",                                                                 # 09
    "dermatofibroma",                                                         # 10
    "eosinophil",                                                             # 11
    "erythroblast",                                                           # 12
    "femur-left",                                                             # 13
    "femur-right",                                                            # 14
    "heart",                                                                  # 15
    "immature_granulocytes_myelocytes__metamyelocytes_and_promyelocytes_",    # 16
    "kidney-left",                                                            # 17
    "unknownMED2"                                                             # 9
    ]

MED_10_CLASSES_3  = [
    "kidney-right",         # 18
    "liver",                # 19
    "lung-left",            # 20
    "lung-right",           # 21
    "lymphocyte",           # 22
    "lymphocytes",          # 23
    "melanocytic_nevi",     # 24
    "melanoma",             # 25
    "monocyte",             # 26
    "unknownMED3"           # 
 ]

MED_10_CLASSES_4  = [

    "mucus",                # 27
    "neutrophil",           # 28
    "normal",               # 29
    "normal_colon_mucosa",  # 30
    "pancreas",             # 31
    "platelet",             # 32
    "pneumonia",            # 33
    "smooth_muscle",        # 34
    "spleen",               # 35
    "unknownMED4"
  ]




# ------------------------------------------------------------
# Optional safety checks for these example lists
# ------------------------------------------------------------



if len(MED_10_CLASSES_1) < 2:
    raise RuntimeError("MED_10_CLASSES_1 must contain at least 2 classes.")

if len(MED_10_CLASSES_2) < 2:
    raise RuntimeError("MED_10_CLASSES_2 must contain at least 2 classes.")

if len(MED_10_CLASSES_3) < 2:
    raise RuntimeError("MED_10_CLASSES_3 must contain at least 2 classes.")

if len(MED_10_CLASSES_4) < 2:
    raise RuntimeError("MED_10_CLASSES_4 must contain at least 2 classes.")


# ============================================================
# MODEL REGISTRY
# ============================================================

MODELS: List[Dict] = [
    
    {
        "name": "med00-08",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "med-00-08-unknown30-cnn-128-256-512-1024-1744s-L5205-A9999-T8766"
        ),
        "classes": MED_10_CLASSES_1,
        "temperature": 1.0,
    },
    {
        "name": "med09-17",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "med-09-17-unknown30-cnn-128-256-512-1024-1744s-L5205-A9999-T8766"
        ),
        "classes": MED_10_CLASSES_2,
        "temperature": 1.0,
    },

    {
        "name": "med18-26",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "med-18-26-unknown30-cnn-128-256-512-1024-1744s-L5205-A9999-T8766"
        ),
        "classes": MED_10_CLASSES_3,
        "temperature": 1.0,
    },

    {
        "name": "med27-35",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "med-27-35-unknown30-cnn-128-256-512-1024-1744s-L5205-A9999-T8766"
        ),
        "classes": MED_10_CLASSES_4,
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


def load_image_tensor_from_bytes(image_bytes: bytes) -> torch.Tensor:
    """
    Open one image received from TCP bytes, convert to RGB, apply
    inference transform, and return tensor [C,H,W].
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = INFER_TRANSFORM(img)
    return x

def load_image_tensor_from_raw_rgb(image_bytes: bytes, width: int, height: int) -> torch.Tensor:
    """
    Open one raw RGB image received from TCP bytes.

    Protocol assumption:
      - The client sends 2 bytes first:
            byte 0 = image width
            byte 1 = image height
      - Then the client sends exactly:
            width * height * 3 bytes
        where each pixel is RGB order:
            R, G, B, R, G, B, ...

    This function converts those raw RGB bytes into a PIL image,
    applies the same inference transform, and returns tensor [C,H,W].
    """
    expected = int(width) * int(height) * 3

    if len(image_bytes) != expected:
        raise ValueError(
            f"Raw RGB payload size mismatch. "
            f"Expected {expected} bytes for {width}x{height} RGB, "
            f"received {len(image_bytes)} bytes."
        )

    img = Image.frombytes("RGB", (int(width), int(height)), image_bytes)
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

                probs_cpu = probs.detach().cpu()

                classes = cfg["classes"]
                model_name = cfg["name"]

                for i in range(len(ok_paths)):
                    # ----------------------------------------------------
                    # PER-MODEL CLASS SELECTION RULE
                    # ----------------------------------------------------
                    # Each model outputs probabilities for its own class list
                    # (for example 9 real classes + 1 unknown class).
                    #
                    # We do NOT want to keep "unknown*" as the selected class
                    # if a real class is available in the same model output.
                    #
                    # So for this image:
                    #   1) rank all classes from highest confidence to lowest
                    #   2) ignore any class whose name starts with "unknown"
                    #   3) choose the next highest class that is not unknown
                    #
                    # Example:
                    #   ranked output:
                    #       1. unknown1   91.20%
                    #       2. apple       5.80%
                    #       3. bicycle     2.10%
                    #
                    #   selected class becomes:
                    #       apple, not unknown1
                    #
                    # Fallback:
                    #   If all ranked classes are unknown-prefixed, then we keep
                    #   the true top-1 class as fallback.
                    # ----------------------------------------------------
                    probs_row = probs_cpu[i]

                    # Build a full ranked list from highest confidence to lowest.
                    # Each item looks like:
                    #   (class_name, confidence, class_index)
                    full_ranked_preds = get_topk_predictions(
                        probs_row,
                        classes,
                        topk=len(classes)
                    )

                    selected_class_name = ""
                    selected_conf = -1.0
                    selected_pid = -1

                    # Ignore any class that starts with "unknown" and take the
                    # next highest class that does not include the unknown prefix.
                    for class_name, class_conf, class_idx in full_ranked_preds:
                        if not starts_with_unknown(class_name):
                            selected_class_name = class_name
                            selected_conf = float(class_conf)
                            selected_pid = int(class_idx)
                            break

                    # Safety fallback: if all classes are unknown-prefixed,
                    # keep the true top-1 result from this model.
                    if selected_pid < 0:
                        selected_class_name, selected_conf, selected_pid = full_ranked_preds[0]
                        selected_conf = float(selected_conf)
                        selected_pid = int(selected_pid)

                    # Keep only the configured Top-K items for display/printing.
                    topk_preds = full_ranked_preds[:TOPK_TO_PRINT]

                    conf = selected_conf
                    pid = selected_pid
                    cls_name = selected_class_name

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
# SHARED MODEL PREPARATION FOR TCP MODE
# ============================================================

def prepare_loaded_models(models_cfg, device):
    """
    Create each CNN architecture, load the trained checkpoint, move the
    model to the selected device, and switch it to evaluation mode.

    This helper is used by TCP mode so the trained models are loaded once
    when the server starts, not every time a client sends an image.
    """
    loaded_models = []

    for cfg in models_cfg:
        num_classes = len(cfg["classes"])

        if num_classes < 2:
            raise RuntimeError(f"Model {cfg['name']!r} must define at least 2 classes.")

        print("============================================================")
        print(f"[MODEL PREPARE] Creating architecture for model: {cfg['name']}")
        print(f"[MODEL PREPARE] Number of classes          : {num_classes}")

        model = StaticInitLearnableCNN(num_classes=num_classes)

        safe_load_state_dict(
            model=model,
            weights_path=cfg["weights"],
            device=device,
            expected_num_classes=num_classes
        )

        model.to(device)
        model.eval()

        print(f"[MODEL READY] {cfg['name']} is now in evaluation mode.\n")
        loaded_models.append((cfg, model))

    return loaded_models


# ============================================================
# SHARED SINGLE-IMAGE CLASSIFICATION FOR TCP MODE
# ============================================================

def classify_single_tensor_with_loaded_models(image_tensor, loaded_models, device, image_name="tcp_image"):
    """
    Classify one image tensor using already-loaded models.

    Returns a dictionary that can be printed locally or serialized to JSON
    and returned to a TCP client.
    """
    if not loaded_models:
        raise RuntimeError("No trained models were loaded.")

    pin = (device.type == "cuda")
    x = torch.stack([image_tensor], dim=0).to(device, non_blocking=pin)

    best_non_unknown_name = ""
    best_non_unknown_pred = -1
    best_non_unknown_conf = -1.0
    best_non_unknown_cls = ""

    best_overall_name = ""
    best_overall_pred = -1
    best_overall_conf = -1.0
    best_overall_cls = ""

    per_model_results = []

    with torch.no_grad():
        for cfg, model in loaded_models:
            logits = model(x)

            temp = float(cfg.get("temperature", 1.0) or 1.0)
            probs = torch.softmax(logits / temp, dim=1)
            probs_row = probs.detach().cpu()[0]

            classes = cfg["classes"]
            model_name = cfg["name"]

            # Build a full ranked list from highest confidence to lowest.
            # Each item looks like:
            #   (class_name, confidence, class_index)
            full_ranked_preds = get_topk_predictions(
                probs_row,
                classes,
                topk=len(classes)
            )

            selected_class_name = ""
            selected_conf = -1.0
            selected_pid = -1

            # Ignore any class that starts with "unknown" and take the
            # next highest class that does not include the unknown prefix.
            for class_name, class_conf, class_idx in full_ranked_preds:
                if not starts_with_unknown(class_name):
                    selected_class_name = class_name
                    selected_conf = float(class_conf)
                    selected_pid = int(class_idx)
                    break

            # Safety fallback: if all classes are unknown-prefixed,
            # keep the true top-1 result from this model.
            if selected_pid < 0:
                selected_class_name, selected_conf, selected_pid = full_ranked_preds[0]
                selected_conf = float(selected_conf)
                selected_pid = int(selected_pid)

            topk_preds = full_ranked_preds[:TOPK_TO_PRINT]

            per_model_results.append(
                {
                    "model_name": model_name,
                    "detected_class": selected_class_name,
                    "pred_id": selected_pid,
                    "confidence": selected_conf,
                    "is_unknown": starts_with_unknown(selected_class_name),
                    "topk": [
                        {
                            "rank": rank,
                            "class_name": class_name,
                            "class_index": class_idx,
                            "confidence": conf,
                        }
                        for rank, (class_name, conf, class_idx) in enumerate(topk_preds, start=1)
                    ],
                }
            )

            if selected_conf > best_overall_conf:
                best_overall_conf = selected_conf
                best_overall_pred = selected_pid
                best_overall_name = model_name
                best_overall_cls = selected_class_name

            if (not starts_with_unknown(selected_class_name)) and (selected_conf > best_non_unknown_conf):
                best_non_unknown_conf = selected_conf
                best_non_unknown_pred = selected_pid
                best_non_unknown_name = model_name
                best_non_unknown_cls = selected_class_name

    if best_non_unknown_conf >= 0.0:
        final_label = best_non_unknown_cls
        final_conf = best_non_unknown_conf
        final_model = best_non_unknown_name
        final_pred_id = best_non_unknown_pred
    else:
        final_label = best_overall_cls
        final_conf = best_overall_conf
        final_model = best_overall_name
        final_pred_id = best_overall_pred

    if USE_LOW_CONFIDENCE_UNKNOWN_RULE and final_conf < LOW_CONFIDENCE_THRESHOLD:
        final_label = UNKNOWN_LABEL_BELOW_THRESHOLD

    return {
        "ok": True,
        "image_name": image_name,
        "final_detection": final_label,
        "final_confidence": final_conf,
        "final_confidence_percent": final_conf * 100.0,
        "winning_model": final_model,
        "pred_id": final_pred_id,
        "model_by_model_results": per_model_results,
    }


# ============================================================
# TCP HELPERS
# ============================================================

def recv_exact(sock_obj, nbytes, stop_event=None):
    """
    Receive exactly nbytes from a TCP socket.
    Raises ConnectionError if the client disconnects early.

    Ctrl+C / shutdown note:
      - The socket must have a timeout so recv() does not block forever.
      - When Ctrl+C is pressed, the server sets stop_event.
      - This function checks stop_event between timeout wakeups.
    """
    chunks = []
    remaining = nbytes

    while remaining > 0:
        if stop_event is not None and stop_event.is_set():
            raise KeyboardInterrupt("TCP receive stopped by user.")

        try:
            chunk = sock_obj.recv(min(65536, remaining))
        except socket.timeout:
            # Timeout is intentional. It allows Ctrl+C/shutdown checks.
            continue

        if not chunk:
            raise ConnectionError("Client disconnected before all bytes were received.")

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)

def send_json_response(sock_obj, payload):
    """
    Send a JSON response using a 4-byte unsigned big-endian length field.

    Server -> Client:
        4 bytes JSON length
        JSON response bytes

    This matches the latest tcp_classifier_client_raw_rgb.py default:
        SERVER_RESPONSE_LENGTH_BYTES = 4
    """
    data = json.dumps(payload, indent=2).encode("utf-8")
    sock_obj.sendall(struct.pack("!I", len(data)))
    sock_obj.sendall(data)


def handle_tcp_client(conn, addr, loaded_models, device, stop_event=None):
    """
    Receive one raw RGB image from one TCP client, classify it, then return JSON.

    Client protocol:
      1) Send exactly 3 bytes:
            byte 0 = width
            byte 1 = height
            byte 2 = channels

      2) Send exactly width * height * channels bytes:
            raw image pixel data

    For this classifier, channels must be 3 because the model expects RGB.

    Example:
      For a 32x32 RGB image:
          header       = bytes([32, 32, 3])
          image_bytes  = 32 * 32 * 3 = 3072 bytes

    This replaces the old 8-byte image-size protocol. The error:
        Image too large: 2314854139998016353 bytes
    happened because the server was reading the first 8 bytes as a
    length field while the client was actually sending width/height/channel
    bytes first.
    """
    print("------------------------------------------------------------")
    print(f"[TCP] Client connected: {addr}")

    # Timeout is required so Ctrl+C can interrupt long recv() waits.
    try:
        conn.settimeout(float(TCP_SOCKET_TIMEOUT_SEC))
    except Exception:
        pass

    try:
        # ----------------------------------------------------
        # READ 3-BYTE RAW RGB HEADER
        # ----------------------------------------------------
        # The client sends:
        #   header[0] = width
        #   header[1] = height
        #   header[2] = channels
        #
        # For this classifier, channels must be 3 for RGB.
        # ----------------------------------------------------
        rgb_header = recv_exact(conn, 3, stop_event=stop_event)
        width = int(rgb_header[0])
        height = int(rgb_header[1])
        channels = int(rgb_header[2])

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions received: {width}x{height}")

        if channels != 3:
            raise ValueError(
                f"Invalid channel count received: {channels}. "
                f"This classifier expects RGB channels=3."
            )

        expected_image_size = width * height * channels

        if expected_image_size > TCP_MAX_IMAGE_BYTES:
            raise ValueError(
                f"Image too large: {expected_image_size} bytes for {width}x{height}x{channels}. "
                f"Limit is {TCP_MAX_IMAGE_BYTES} bytes."
            )

        print(f"[TCP] Image dimensions : {width}x{height}x{channels}")
        print(f"[TCP] Expected RGB size : {expected_image_size} bytes")

        # ----------------------------------------------------
        # READ RAW RGB IMAGE BYTES
        # ----------------------------------------------------
        # Because the width, height, and channels are known,
        # the server does not need an 8-byte image-size field.
        # ----------------------------------------------------
        image_bytes = recv_exact(conn, expected_image_size, stop_event=stop_event)

        image_tensor = load_image_tensor_from_raw_rgb(
            image_bytes=image_bytes,
            width=width,
            height=height
        )

        result = classify_single_tensor_with_loaded_models(
            image_tensor=image_tensor,
            loaded_models=loaded_models,
            device=device,
            image_name=f"tcp_raw_rgb_{width}x{height}_from_{addr[0]}_{addr[1]}"
        )

        # Include received dimensions in the JSON reply.
        result["received_width"] = width
        result["received_height"] = height
        result["received_channels"] = channels
        result["received_rgb_bytes"] = expected_image_size

        print(f"[TCP] FINAL DETECTION  : {result['final_detection']}")
        print(f"[TCP] FINAL CONFIDENCE : {result['final_confidence_percent']:.2f}%")
        print(f"[TCP] WINNING MODEL    : {result['winning_model']}")

        send_json_response(conn, result)

    except Exception as e:
        err = {
            "ok": False,
            "error": str(e),
        }
        print(f"[TCP-ERROR] {e}")
        try:
            send_json_response(conn, err)
        except Exception:
            pass

    finally:
        try:
            conn.close()
        except Exception:
            pass
        print(f"[TCP] Client disconnected: {addr}")


def run_tcp_classifier_server(models_cfg, device, host=TCP_HOST, port=TCP_PORT):
    """
    Start TCP classifier server.

    The server loads all trained models once, waits for a client image,
    classifies that image, and returns the result to the client.
    """
    loaded_models = prepare_loaded_models(models_cfg, device)

    print("\n============================================================")
    print("TCP MULTI-MODEL CLASSIFIER SERVER")
    print("============================================================")
    print(f"Device          : {device}")
    print(f"Listening host  : {host}")
    print(f"MED_PORT        : {port}")
    print(f"Max image bytes : {TCP_MAX_IMAGE_BYTES}")
    print(f"Keep running    : {TCP_KEEP_RUNNING}")
    print("Protocol        : 1-byte width + 1-byte height + 1-byte channels + raw RGB bytes")
    print("Response        : 4-byte JSON length + JSON bytes")
    print("============================================================\n")

    # --------------------------------------------------------
    # SERVER SOCKET
    # --------------------------------------------------------
    # Ctrl+C / KeyboardInterrupt handling is intentionally placed
    # around the accept loop so the user can terminate the TCP server
    # cleanly from the terminal without printing a Python stack trace.
    #
    # SO_REUSEADDR allows the server to restart quickly after shutdown
    # without waiting for the operating system to release the port.
    # --------------------------------------------------------
    stop_event = threading.Event()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, int(port)))
        server.listen(TCP_BACKLOG)

        # ----------------------------------------------------
        # IMPORTANT FOR CTRL+C ON WINDOWS
        # ----------------------------------------------------
        # Without a timeout, server.accept() can block and may not
        # return control to Python quickly when Ctrl+C is pressed.
        # A short timeout lets the loop wake up and check shutdown.
        # ----------------------------------------------------
        server.settimeout(float(TCP_SOCKET_TIMEOUT_SEC))

        print("[TCP] Server is listening. Press Ctrl+C to stop.")

        try:
            while not stop_event.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    # Timeout is intentional. It allows Ctrl+C checks.
                    continue

                handle_tcp_client(conn, addr, loaded_models, device, stop_event=stop_event)
        
                if not TCP_KEEP_RUNNING:
                    print("[TCP] TCP_KEEP_RUNNING is False. Server is stopping.")
                    break

        except KeyboardInterrupt:
            stop_event.set()
            print("\n[TCP] Ctrl+C detected. Stopping server...")

        finally:
            stop_event.set()
            # The socket is automatically closed by the with-statement.
            # This message confirms a clean shutdown path.
            print("[TCP] Server stopped cleanly.")


# ============================================================
# STARTUP MODE SELECTION
# ============================================================

def choose_classifier_input_mode():
    """
    Ask the user how to provide images.

    Default is directory mode from CLASIFIER_TEST / TEST_IMAGE_DIR.
    TCP mode is selected only if the user types T.
    """
    print("\n============================================================")
    print("SELECT CLASSIFIER INPUT MODE")
    print("============================================================")
    print("Press ENTER or D : classify images from CLASIFIER_TEST directory")
    print("Type T           : start TCP server and receive image from client")
    print("============================================================")

    choice = input("Select input mode [default=D]: ").strip().lower()

    if choice == "t":
        return "tcp"

    return "directory"

# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main program:
      1) choose CPU or GPU
      2) load trained model checkpoints
      3) start TCP server directly
      4) receive raw RGB images from TCP clients
      5) classify each received image
      6) return the classification result to the TCP client

    TCP-ONLY VERSION
    ------------------------------------------------------------
    This version removes all input-mode options and starts as a
    TCP server immediately.

    Port configuration:
      - The server reads MED_PORT from the .env file.
      - If MED_PORT is not defined, the default assigned port is 5055.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_print("Using device:", device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if not MODELS:
        print("MODELS list is empty. Add at least one trained model config.")
        return

    run_tcp_classifier_server(
        models_cfg=MODELS,
        device=device,
        host=TCP_HOST,
        port=MED_PORT
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # ----------------------------------------------------
        # TOP-LEVEL CTRL+C HANDLER
        # ----------------------------------------------------
        # If Ctrl+C happens outside the TCP accept loop, exit
        # cleanly without a long Python traceback.
        # ----------------------------------------------------
        print("\n[EXIT] Ctrl+C detected. Program terminated by user.")