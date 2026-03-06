# ============================================================
# multi_model_inference_directory.py
# ============================================================
# ✅ Inference-only classifier script (NO training)
#
# WHAT THIS SCRIPT DOES (your requested classifier):
# ------------------------------------------------------------
# ✅ You define 1+ trained models in a global list (MODELS)
# ✅ Each model has:
#     - weights file path (your long generated name)
#     - its OWN class list (the classes it was trained on)
# ✅ You provide an INPUT directory that contains images (unknown labels)
# ✅ For each image:
#     - run ALL models
#     - compute confidence per model (softmax max prob)
#     - choose the model/class with MAX confidence
# ✅ Prints per-image result and a summary
#
# NOTES:
# ------------------------------------------------------------
# • No ImageFolder dataset required (no ground-truth labels)
# • No train_loader / test_loader / train_model() needed
# • Works with ANY input directory that contains images
# • Uses your CNN architecture: 3→128→256→512→1024 + GAP + FC
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
import math
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# ============================================================
# GLOBAL CONFIG
# ============================================================

DEBUG_FLAG = True

# Directory containing unknown images (no subfolders required)
# Example:
#   INPUT_IMAGE_DIR = "../../../data/test_images"
INPUT_IMAGE_DIR = "../../../data/2D-dog10-datasets/test_images"

# Process only these extensions
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Inference batch size (directory inference)
INFER_BATCH_SIZE = 64

# If your GPU can handle larger, use 128/256
# (Images are resized, so memory is manageable)
# INFER_BATCH_SIZE = 128

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
# GLOBAL CLASSES (YOU SAID: classes defined globally)
# ============================================================
# IMPORTANT:
# ----------
# Each model must have the exact class order used in training.
# The model predicts an index [0..C-1] which maps to this list.
#
# Example for DOG 2D 10 classes (replace with your real names/order):
CIFAR_10_CLASSES_1 = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed"
    "bee",
    "beetle",
    "bicycle",
    "bottle"
   ]

CIFAR_10_CLASSES_2 = [
     "bowl",
     "boy",
     "bridge",
     "bus",
     "butterfly",
     "camel",
     "can",
     "castle",
     "caterpillar",
     "cattle"
   ]

# If you have CIFAR100 model, define list of 100 class names here:
# CIFAR100_CLASSES = [ ... 100 names in exact order ... ]


# ============================================================
# MODEL REGISTRY (1 OR MORE MODELS)
# ============================================================
# You asked:
# "If the trained model file is generated and called
#  cifar100-cnn-128-256-512-1024-3805s-L8785-A9998-T6818
#  how do we include this name?"
#
# Answer:
# Put it EXACTLY here in "weights". Extension is NOT implied.
# Use the exact filename you saved.
# ============================================================

MODEL_BASE_DIR = "../../../"  # folder where weight files exist

MODELS: List[Dict] = [
    {
        "name": "cifar0-9",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "dog2D-10-cnn-128-256-512-1024-158s-L5442-A9984-T9984.pth"  # <-- include .pth if it exists
        ),
        "classes": CIFAR_10_CLASSES_1,
        "temperature": 1.0,  # keep 1.0 unless you know calibration
    },

    # Example second model:
   {
        "name": "cifar10-19",
        "weights": os.path.join(
            MODEL_BASE_DIR,
            "dog2D-10-cnn-128-256-512-1024-158s-L5442-A9984-T9984.pth"  # <-- include .pth if it exists
        ),
        "classes": CIFAR_10_CLASSES_2,
        "temperature": 1.0,  # keep 1.0 unless you know calibration
    },
]


# ============================================================
# DEBUG PRINT
# ============================================================

def debug_print(*args, **kwargs):
    if DEBUG_FLAG:
        print(*args, **kwargs)


# ============================================================
# MODEL DEFINITION (INFERENCE-READY)
# ============================================================

class StaticInitLearnableCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(CONV1_IN_CHANNELS, CONV1_OUT_CHANNELS, kernel_size=3, padding=1, bias=True)
        self.bn1   = nn.BatchNorm2d(CONV1_OUT_CHANNELS)
        self.pool  = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(CONV2_IN_CHANNELS, CONV2_OUT_CHANNELS, kernel_size=3, padding=1, bias=True)
        self.bn2   = nn.BatchNorm2d(CONV2_OUT_CHANNELS)

        self.conv3 = nn.Conv2d(CONV3_IN_CHANNELS, CONV3_OUT_CHANNELS, kernel_size=3, padding=1, bias=True)
        self.bn3   = nn.BatchNorm2d(CONV3_OUT_CHANNELS)

        self.conv4 = nn.Conv2d(CONV4_IN_CHANNELS, CONV4_OUT_CHANNELS, kernel_size=3, padding=1, bias=True)
        self.bn4   = nn.BatchNorm2d(CONV4_OUT_CHANNELS)

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
# TRANSFORM (MUST MATCH TRAINING)
# ============================================================
# If you trained on 32x32, keep 32x32.
# If you trained on 64x64, change to (64,64).
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
    x = INFER_TRANSFORM(img)  # [C,H,W]
    return x


def softmax_confidence(logits: torch.Tensor, temperature: float = 1.0) -> Tuple[int, float]:
    """
    Returns (pred_id, conf) for ONE sample logits [C] or [1,C].
    conf is the max softmax probability.
    """
    if logits.ndim == 2:
        logits = logits[0]
    if temperature is None or temperature <= 0:
        temperature = 1.0
    probs = torch.softmax(logits / float(temperature), dim=0)
    pred_id = int(torch.argmax(probs).item())
    conf = float(probs[pred_id].item())
    return pred_id, conf


def safe_load_state_dict(model: nn.Module, weights_path: str, device: torch.device) -> None:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    state = torch.load(weights_path, map_location=device)

    # In case you saved a full checkpoint dict instead of pure state_dict:
    # e.g. {"model": state_dict, ...}
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    model.load_state_dict(state)


def run_directory_multi_model_classifier(
    image_paths: List[str],
    models_cfg: List[Dict],
    device: torch.device,
) -> None:
    """
    For each image:
      - run all models
      - compute max softmax confidence per model
      - pick overall best (max confidence)
    """

    if len(image_paths) == 0:
        print("❌ No images found in input directory.")
        return

    # Pre-load all models
    loaded_models = []
    for cfg in models_cfg:
        classes = cfg["classes"]
        num_classes = len(classes)

        model = StaticInitLearnableCNN(num_classes=num_classes)
        safe_load_state_dict(model, cfg["weights"], device=device)
        model.to(device)
        model.eval()

        loaded_models.append((cfg, model))

        debug_print(f"[LOAD] model={cfg['name']!r}")
        debug_print(f"       weights={cfg['weights']!r}")
        debug_print(f"       num_classes={num_classes}")

    # Inference loop (batched)
    pin = (device.type == "cuda")

    print("\n============================================================")
    print("MULTI-MODEL DIRECTORY CLASSIFIER (Inference Only)")
    print("============================================================")
    print(f"Device           : {device}")
    print(f"Input dir images  : {len(image_paths)}")
    print(f"Batch size        : {INFER_BATCH_SIZE}")
    print(f"Models loaded     : {len(loaded_models)}")
    print("============================================================\n")

    t0 = time.perf_counter()

    # Stats
    per_model_wins = {cfg["name"]: 0 for cfg, _ in loaded_models}

    for start in tqdm(range(0, len(image_paths), INFER_BATCH_SIZE), desc="Classifying"):
        batch_paths = image_paths[start:start + INFER_BATCH_SIZE]

        # Build batch tensor
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

        x = torch.stack(xs, dim=0).to(device, non_blocking=pin)  # [B,C,H,W]

        # For each model, compute predictions for this batch
        with torch.no_grad():
            # For each image in batch, keep the best overall (across models)
            best_name = [""] * x.size(0)
            best_pred = [-1] * x.size(0)
            best_conf = [-1.0] * x.size(0)
            best_cls  = [""] * x.size(0)

            for cfg, model in loaded_models:
                logits = model(x)  # [B, Cmodel]
                temp = float(cfg.get("temperature", 1.0) or 1.0)

                probs = torch.softmax(logits / temp, dim=1)  # [B, Cmodel]
                confs, pred_ids = torch.max(probs, dim=1)    # [B]

                pred_ids = pred_ids.detach().cpu().tolist()
                confs = confs.detach().cpu().tolist()

                classes = cfg["classes"]
                model_name = cfg["name"]

                for i in range(len(ok_paths)):
                    conf = float(confs[i])
                    pid  = int(pred_ids[i])

                    if conf > best_conf[i]:
                        best_conf[i] = conf
                        best_pred[i] = pid
                        best_name[i] = model_name
                        best_cls[i]  = classes[pid] if 0 <= pid < len(classes) else f"class_{pid}"

            # Print per-image result (compact)
            for i, p in enumerate(ok_paths):
                per_model_wins[best_name[i]] += 1
                print(
                    f"{os.path.basename(p):<40} | "
                    f"BEST_MODEL={best_name[i]:<15} | "
                    f"PRED={best_cls[i]:<20} | "
                    f"CONF={best_conf[i]*100:6.2f}%"
                )

    dt = time.perf_counter() - t0

    print("\n============================================================")
    print("SUMMARY")
    print("============================================================")
    print(f"Total images processed : {len(image_paths)}")
    print(f"Total time             : {dt:.2f} sec")
    if len(image_paths) > 0:
        print(f"Avg time / image       : {dt/len(image_paths):.4f} sec")
    print("------------------------------------------------------------")
    print("Model win counts (how many times each model had max confidence):")
    for k, v in per_model_wins.items():
        print(f"  {k:<20} : {v}")
    print("============================================================\n")


# ============================================================
# MAIN
# ============================================================

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_print("Using device:", device)

    # CUDA speed option (keep in main, not in model)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Check models config
    if not MODELS:
        print("❌ MODELS list is empty. Add at least one model in global MODELS.")
        return

    # List images
    image_paths = list_images_in_dir(INPUT_IMAGE_DIR)

    # Run classifier
    run_directory_multi_model_classifier(
        image_paths=image_paths,
        models_cfg=MODELS,
        device=device,
    )


if __name__ == "__main__":
    main()