# ============================================================
# multi_model_inference_directory.py
# ============================================================
# ✅ Inference-only classifier script (NO training)
#
# WHAT THIS SCRIPT DOES:
# ------------------------------------------------------------
# ✅ You define 1 or more trained models in a global list (MODELS)
# ✅ Each model has:
#     - weights file path
#     - its OWN class list (the classes it was trained on)
# ✅ You provide an INPUT directory that contains unknown images
# ✅ For each image:
#     - run ALL models
#     - compute raw confidence per model
#     - choose the model/class with MAX raw confidence
# ✅ Prints per-image result and a summary
# ✅ Optionally displays each tested image enlarged
# ✅ Before displaying a new image, the old displayed image
#    window is closed first
# ✅ Keyboard control while displaying images:
#     - ENTER -> move to next tested image
#     - E     -> exit the program immediately
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
    """
    Install one or more Python packages into the SAME interpreter
    running this script.

    Example:
        _pip_install(["torch", "torchvision"])
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *pkgs])


def _ensure_import(import_name, pip_name=None):
    """
    Try to import a module.
    If it fails, install it first, then import again.

    Parameters:
        import_name : actual module name used in Python import
        pip_name    : package name used by pip if different
    """
    try:
        importlib.import_module(import_name)
    except Exception:
        _pip_install([pip_name or import_name])
        importlib.import_module(import_name)


def ensure_deps_for_this_script():
    """
    Make sure the packages required by this script exist.

    Core ML stack:
        torch
        torchvision

    Utilities:
        numpy
        pillow
        tqdm
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


# 🔥 RUN AUTO-INSTALL NOW
ensure_deps_for_this_script()


# ============================================================
# NORMAL IMPORTS
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
# Directory containing UNKNOWN images to classify
# ------------------------------------------------------------
TEST_IMAGE_DIR = "../../../data/cifar10_clasifier_test"

# ------------------------------------------------------------
# Process only these extensions
# ------------------------------------------------------------
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ------------------------------------------------------------
# Inference batch size
# ------------------------------------------------------------
# This controls how many images are processed together per model.
#
# Larger values:
#   • usually faster on GPU
#   • require more GPU memory
#
# Smaller values:
#   • slower
#   • require less memory
# ------------------------------------------------------------
INFER_BATCH_SIZE = 64

# ------------------------------------------------------------
# OPTIONAL DISPLAY FEATURE
# ------------------------------------------------------------
# DISPLAY_TESTED_IMAGE:
#   • True  -> show each tested image enlarged
#   • False -> no display, only console output
#
# ENLARGE_FACTOR:
#   • image width  = original_width  × ENLARGE_FACTOR
#   • image height = original_height × ENLARGE_FACTOR
#
# Default requested value = 6
# ------------------------------------------------------------
DISPLAY_TESTED_IMAGE = True
ENLARGE_FACTOR = 6

# ------------------------------------------------------------
# OPTIONAL KEYBOARD CONTROL
# ------------------------------------------------------------
# If True:
#   after each displayed image:
#       ENTER -> continue
#       E     -> exit program
#
# If False:
#   script continues automatically without waiting
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
# MODEL ARCH CONSTANTS
# ============================================================
# Your CNN architecture:
#
#   conv1 :   3 -> 128
#   conv2 : 128 -> 256
#   conv3 : 256 -> 512
#   conv4 : 512 -> 1024
#
# After conv4:
#   GAP -> flatten -> FC
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
# GLOBAL CLASS GROUPS
# ============================================================
# IMPORTANT:
# ----------
# Each model MUST use the exact class order used during training.
#
# The model predicts an index:
#
#     0, 1, 2, ... , C-1
#
# This index is mapped into:
#
#     cfg["classes"][predicted_index]
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
# ============================================================

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
    "uknown1",        # 
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

# ------------------------------------------------------------
# Optional safety checks for these example lists
# ------------------------------------------------------------
if len(CIFAR_10_CLASSES_1) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_1 must contain at least 2 classes.")

if len(CIFAR_10_CLASSES_2) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_2 must contain at least 2 classes.")

if len(CIFAR_10_CLASSES_3) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_3 must contain at least 3 classes.")

if len(CIFAR_10_CLASSES_4) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_4 must contain at least 4 classes.")

if len(CIFAR_10_CLASSES_5) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_5 must contain at least 5 classes.")

if len(CIFAR_10_CLASSES_6) < 2:
    raise RuntimeError("CIFAR_10_CLASSES_6 must contain at least 6 classes.")




# ============================================================
# MODEL REGISTRY
# ============================================================
# Put the EXACT saved weight filename in "weights".
#
# IMPORTANT:
# ----------
# If your real files were saved with ".pth", include ".pth".
# If your real files were saved WITHOUT ".pth", keep them without
# ".pth".
#
# FLEXIBILITY:
# ------------
# Each model can define its own class-list size.
# The script automatically creates the correct FC output size.
# ============================================================

MODEL_BASE_DIR = "../../../"

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



]


# ============================================================
# DEBUG PRINT
# ============================================================

def debug_print(*args, **kwargs):
    """
    Debug print wrapper.
    If DEBUG_FLAG is True -> print
    If DEBUG_FLAG is False -> do nothing
    """
    if DEBUG_FLAG:
        print(*args, **kwargs)


# ============================================================
# DISPLAY WINDOW GLOBALS
# ============================================================
# We keep one window reference here.
# Before showing the next image, we destroy the old window.
# ============================================================

DISPLAY_ROOT = None
DISPLAY_LABEL = None
DISPLAY_PHOTO = None


# ============================================================
# CNN MODEL
# ============================================================

class StaticInitLearnableCNN(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()

        # --------------------------------------------------------
        # OVERALL CNN STRUCTURE
        # --------------------------------------------------------
        # Input image
        #   ↓
        # Conv1  : 3   → 128 channels
        # Pool
        # Conv2  : 128 → 256 channels
        # Conv3  : 256 → 512 channels
        # Conv4  : 512 → 1024 channels
        # GAP
        # Dropout
        # FC
        #
        # This is a classical convolutional neural network.
        #
        # Each convolution filter can be interpreted as a neuron
        # group that scans a local spatial region of the previous
        # layer.
        # --------------------------------------------------------


        # --------------------------------------------------------
        # INPUT IMAGE
        # --------------------------------------------------------
        # Expected input shape:
        #
        #     [B, 3, 32, 32]
        #
        # where:
        #   B = batch size
        #   3 = RGB channels
        #   32 = image height
        #   32 = image width
        #
        # Total raw input values per image:
        #
        #     3 × 32 × 32 = 3,072
        #
        # You can think of those as the initial input neurons.
        # --------------------------------------------------------


        # --------------------------------------------------------
        # LAYER 1 : CONVOLUTION
        # --------------------------------------------------------
        # Definition:
        #
        #     Conv2d(3 → 128, kernel=3×3, padding=1)
        #
        # Number of output feature maps / neuron groups:
        #
        #     128
        #
        # Since padding=1 and kernel=3, the spatial size remains:
        #
        #     32 × 32
        #
        # So total output neurons in this layer:
        #
        #     128 × 32 × 32 = 131,072 neurons
        #
        # Input seen by EACH neuron:
        #
        #     3 × 3 × 3 = 27 inputs
        #
        # because:
        #   kernel height  = 3
        #   kernel width   = 3
        #   input channels = 3
        #
        # Parameters per filter:
        #
        #     27 weights + 1 bias
        #
        # The 128 filters learn low-level patterns such as:
        #   • edges
        #   • corners
        #   • simple textures
        # --------------------------------------------------------
        self.conv1 = nn.Conv2d(CONV1_IN_CHANNELS, CONV1_OUT_CHANNELS, kernel_size=3, padding=1, bias=True)

        # --------------------------------------------------------
        # BATCH NORMALIZATION 1
        # --------------------------------------------------------
        # Normalizes the 128 output channels of conv1.
        #
        # Learnable parameters:
        #   • gamma for each channel
        #   • beta  for each channel
        #
        # This helps stabilize activations.
        # --------------------------------------------------------
        self.bn1 = nn.BatchNorm2d(CONV1_OUT_CHANNELS)

        # --------------------------------------------------------
        # MAX POOLING
        # --------------------------------------------------------
        # MaxPool2d(2,2)
        #
        # Reduces spatial size by half:
        #
        #     128 × 32 × 32
        #         ↓
        #     128 × 16 × 16
        #
        # Total neurons after pooling:
        #
        #     128 × 16 × 16 = 32,768
        #
        # Pooling has NO learnable parameters.
        # --------------------------------------------------------
        self.pool = nn.MaxPool2d(2, 2)


        # --------------------------------------------------------
        # LAYER 2 : CONVOLUTION
        # --------------------------------------------------------
        # Definition:
        #
        #     Conv2d(128 → 256, kernel=3×3, padding=1)
        #
        # Input shape:
        #
        #     128 × 16 × 16
        #
        # Output shape:
        #
        #     256 × 16 × 16
        #
        # Total output neurons:
        #
        #     256 × 16 × 16 = 65,536 neurons
        #
        # Input seen by EACH neuron:
        #
        #     3 × 3 × 128 = 1,152 inputs
        #
        # because each neuron sees a 3×3 patch across ALL 128
        # channels from the previous layer.
        #
        # Parameters per filter:
        #
        #     1,152 weights + 1 bias
        #
        # This layer learns richer mid-level features.
        # --------------------------------------------------------
        self.conv2 = nn.Conv2d(CONV2_IN_CHANNELS, CONV2_OUT_CHANNELS, kernel_size=3, padding=1, bias=True)

        # BatchNorm for 256 channels
        self.bn2 = nn.BatchNorm2d(CONV2_OUT_CHANNELS)


        # --------------------------------------------------------
        # LAYER 3 : CONVOLUTION
        # --------------------------------------------------------
        # Definition:
        #
        #     Conv2d(256 → 512, kernel=3×3, padding=1)
        #
        # Input shape:
        #
        #     256 × 16 × 16
        #
        # Output shape:
        #
        #     512 × 16 × 16
        #
        # Total output neurons:
        #
        #     512 × 16 × 16 = 131,072 neurons
        #
        # Input seen by EACH neuron:
        #
        #     3 × 3 × 256 = 2,304 inputs
        #
        # Parameters per filter:
        #
        #     2,304 weights + 1 bias
        #
        # This layer learns higher-level visual parts and patterns.
        # --------------------------------------------------------
        self.conv3 = nn.Conv2d(CONV3_IN_CHANNELS, CONV3_OUT_CHANNELS, kernel_size=3, padding=1, bias=True)

        # BatchNorm for 512 channels
        self.bn3 = nn.BatchNorm2d(CONV3_OUT_CHANNELS)


        # --------------------------------------------------------
        # LAYER 4 : CONVOLUTION
        # --------------------------------------------------------
        # Definition:
        #
        #     Conv2d(512 → 1024, kernel=3×3, padding=1)
        #
        # Input shape:
        #
        #     512 × 16 × 16
        #
        # Output shape:
        #
        #     1024 × 16 × 16
        #
        # Total output neurons:
        #
        #     1024 × 16 × 16 = 262,144 neurons
        #
        # Input seen by EACH neuron:
        #
        #     3 × 3 × 512 = 4,608 inputs
        #
        # Parameters per filter:
        #
        #     4,608 weights + 1 bias
        #
        # This is the deepest convolution layer and learns more
        # abstract combinations of visual features.
        # --------------------------------------------------------
        self.conv4 = nn.Conv2d(CONV4_IN_CHANNELS, CONV4_OUT_CHANNELS, kernel_size=3, padding=1, bias=True)

        # BatchNorm for 1024 channels
        self.bn4 = nn.BatchNorm2d(CONV4_OUT_CHANNELS)


        # --------------------------------------------------------
        # GLOBAL AVERAGE POOLING (GAP)
        # --------------------------------------------------------
        # Converts:
        #
        #     1024 × 16 × 16
        #
        # into:
        #
        #     1024 × 1 × 1
        #
        # Meaning:
        #   each of the 1024 feature maps is reduced to ONE value
        #   by averaging over spatial positions.
        #
        # Final neuron count after GAP:
        #
        #     1024 neurons
        #
        # This makes the model less dependent on spatial size at the
        # classifier stage.
        # --------------------------------------------------------
        self.gap = nn.AdaptiveAvgPool2d(1)


        # --------------------------------------------------------
        # DROPOUT
        # --------------------------------------------------------
        # Randomly disables 30% of the 1024 neurons during training.
        #
        # During inference, dropout is automatically disabled because
        # we call:
        #
        #     model.eval()
        # --------------------------------------------------------
        self.dropout = nn.Dropout(p=0.3)


        # --------------------------------------------------------
        # FINAL FULLY CONNECTED CLASSIFIER
        # --------------------------------------------------------
        # Linear(1024 → num_classes)
        #
        # IMPORTANT:
        # ----------
        # num_classes is VARIABLE and comes from:
        #
        #     len(cfg["classes"])
        #
        # So if one model has:
        #
        #     10 classes -> Linear(1024 → 10)
        #
        # and another model has:
        #
        #     15 classes -> Linear(1024 → 15)
        #
        # Each output neuron corresponds to ONE class in that model's
        # class list.
        #
        # Input size of EACH output neuron:
        #
        #     1024 inputs
        #
        # because each class neuron receives the 1024 values produced
        # after GAP and flatten.
        #
        # Total parameters:
        #
        #     1024 × num_classes weights
        #     num_classes biases
        # --------------------------------------------------------
        self.fc = nn.Linear(CONV4_OUT_CHANNELS, num_classes)

    def forward(self, x):
        """
        Forward pass of the network.

        Input:
            x : [B, 3, 32, 32]

        Flow:
            conv1 -> bn1 -> relu -> pool
            conv2 -> bn2 -> relu
            conv3 -> bn3 -> relu
            conv4 -> bn4 -> relu
            gap
            flatten
            dropout
            fc

        Output:
            logits : [B, num_classes]
        """
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
# This preprocessing must match training expectations.
#
# Resize((32,32)) means:
#   every input image is resized to 32×32 before inference.
# ============================================================

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def list_images_in_dir(root_dir: str) -> List[str]:
    """
    Return a sorted list of valid image file paths from a directory.
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


def load_image_tensor(image_path: str):
    """
    Load one image, convert to RGB, apply inference transform,
    and return a tensor [C,H,W].
    """
    img = Image.open(image_path).convert("RGB")
    x = INFER_TRANSFORM(img)
    return x


# ============================================================
# SAFE STATE LOAD
# ============================================================

def safe_load_state_dict(model, weights_path, device, expected_num_classes):
    """
    Load model weights safely and verify that the checkpoint FC layer
    matches the number of classes defined in the model config.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)

    state = torch.load(weights_path, map_location=device)

    # If checkpoint is wrapped inside a dict
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    # Verify class count from fc.weight
    if "fc.weight" not in state:
        raise RuntimeError(f"Checkpoint does not contain 'fc.weight': {weights_path}")

    checkpoint_num_classes = int(state["fc.weight"].shape[0])

    if checkpoint_num_classes != expected_num_classes:
        raise RuntimeError(
            f"Class count mismatch for weights file:\n"
            f"  {weights_path}\n"
            f"Checkpoint expects {checkpoint_num_classes} classes, "
            f"but your class list defines {expected_num_classes} classes."
        )

    model.load_state_dict(state)


# ============================================================
# KEYBOARD CONTROL
# ============================================================

def wait_for_input():
    """
    Keyboard control after displaying an image:

        ENTER -> continue to next image
        E     -> exit program immediately
    """
    print()
    key = input("Press ENTER for next image or 'E' to exit: ").strip().lower()

    if key == "e":
        print("Exiting program...")
        sys.exit(0)


# ============================================================
# CLOSE PREVIOUS DISPLAY WINDOW
# ============================================================

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


# ============================================================
# SAFE FILENAME HELPER
# ============================================================

def make_safe_filename(text: str) -> str:
    """
    Convert arbitrary text into a filename-safe string.
    """
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


# ============================================================
# DISPLAY IMAGE
# ============================================================

def display_tested_image(image_path, detected_class, confidence, winning_model):
    """
    Display the tested image enlarged.

    Before showing the new image, the old displayed image window
    is closed first.

    The displayed image includes:
        • detected class
        • confidence
        • winning model

    If WAIT_FOR_ENTER_BETWEEN_IMAGES is True:
        ENTER -> next image
        E     -> exit program
    """
    global DISPLAY_ROOT, DISPLAY_LABEL, DISPLAY_PHOTO

    if not DISPLAY_TESTED_IMAGE:
        return

    close_previous_display_window()

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[DISPLAY-SKIP] Could not open image for display: {image_path}  err={e}")
        return

    # --------------------------------------------------------
    # Enlarge image by ENLARGE_FACTOR
    # --------------------------------------------------------
    w, h = img.size
    img = img.resize((w * ENLARGE_FACTOR, h * ENLARGE_FACTOR), Image.NEAREST)

    # --------------------------------------------------------
    # Draw result text on top of the image
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Save the displayed version to disk (optional useful log)
    # --------------------------------------------------------
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
        img.save(out_path)
    except Exception as e:
        print(f"[DISPLAY-SKIP] Could not save displayed image: {out_path}  err={e}")

    # --------------------------------------------------------
    # Show in Tkinter window
    # --------------------------------------------------------
    try:
        DISPLAY_ROOT = tk.Tk()
        DISPLAY_ROOT.title(DISPLAY_WINDOW_TITLE)

        DISPLAY_PHOTO = ImageTk.PhotoImage(img)

        DISPLAY_LABEL = tk.Label(DISPLAY_ROOT, image=DISPLAY_PHOTO)
        DISPLAY_LABEL.pack()

        DISPLAY_ROOT.update_idletasks()
        DISPLAY_ROOT.update()

    except Exception as e:
        print(f"[DISPLAY-SKIP] Could not display image in window: {image_path}  err={e}")
        close_previous_display_window()
        return

    # --------------------------------------------------------
    # Optional keyboard wait
    # --------------------------------------------------------
    if WAIT_FOR_ENTER_BETWEEN_IMAGES:
        wait_for_input()

    close_previous_display_window()


# ============================================================
# MAIN MULTI-MODEL CLASSIFIER
# ============================================================

def run_directory_multi_model_classifier(image_paths, models_cfg, device):
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

    # --------------------------------------------------------
    # PRE-LOAD ALL MODELS
    # --------------------------------------------------------
    loaded_models = []

    for cfg in models_cfg:
        num_classes = len(cfg["classes"])

        # Flexible class count:
        # each model can have its own class list size
        if num_classes < 2:
            raise RuntimeError(
                f"Model {cfg['name']!r} must define at least 2 classes."
            )

        model = StaticInitLearnableCNN(num_classes=num_classes)

        safe_load_state_dict(
            model=model,
            weights_path=cfg["weights"],
            device=device,
            expected_num_classes=num_classes
        )

        model.to(device)
        model.eval()

        loaded_models.append((cfg, model))

        debug_print(f"[LOAD] model={cfg['name']!r}")
        debug_print(f"       weights={cfg['weights']!r}")
        debug_print(f"       num_classes={num_classes}")
        debug_print(f"       classes={cfg['classes']}")

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

    # --------------------------------------------------------
    # PROCESS IMAGES IN BATCHES
    # --------------------------------------------------------
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
        # For each image in the batch, keep the BEST overall result
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
                probs = torch.softmax(logits / temp, dim=1)

                confs, pred_ids = torch.max(probs, dim=1)

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
                    # WINNER SELECTION
                    # ----------------------------------------------------
                    # Choose the model with the largest RAW top1 confidence.
                    # ----------------------------------------------------
                    if conf > best_conf[i]:
                        best_conf[i] = conf
                        best_pred[i] = pid
                        best_name[i] = model_name
                        best_cls[i] = cls_name

        # --------------------------------------------------------
        # PRINT RESULTS FOR EACH IMAGE IN THE BATCH
        # --------------------------------------------------------
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
            )

        print("------------------------------------------------------------")

    # Close any remaining display window
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
    print("Model win counts (how many times each model had max confidence):")

    for k, v in per_model_wins.items():
        print(f"  {k:<20} : {v}")

    print("============================================================\n")


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main program:
      1) choose device
      2) list images from input directory
      3) run multi-model classifier
    """
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
        device=device
    )


if __name__ == "__main__":
    main()