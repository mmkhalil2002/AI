import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import msvcrt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


MODEL_PATH = "../../../"
MODEL_FILENAME = "cifar10-cnn-32-16-8"
DATA_PATH = "../../../data/mydata"
MG_WIDTH, IMG_HEIGHT = 32, 32  # Based on training dataset
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for valid detections
FILTER_WIDTH = 3
FILTER_HEIGHT = 3
BATCH_SIZE = 128
NUM_EPOCHS = 100
#LEARNING_RATE = 0.001


NUM_WORKERS = 0
STATIC_FILTERS = False
DEBUG_FLAG = True
# ============================================================
# EXPLANATION: HOW TRAINING WORKS IN THIS NETWORK
# ============================================================
#
# This network is a CLASSICAL CONVOLUTIONAL NEURAL NETWORK (CNN)
# with a progressive feature hierarchy:
#
#   • Layer 1 (conv1) extracts LOW-LEVEL features
#       (edges, color gradients, simple textures)
#
#   • Layer 2 (conv2) extracts MID-LEVEL features
#       (corners, repeated patterns, texture groupings)
#
#   • Layer 3 (conv3) extracts HIGHER-LEVEL features
#       (object parts, more complex shape compositions)
#
#   • TWO MAX POOLING operations are used:
#       - After conv1
#       - After conv2
#
#     This reduces spatial resolution gradually while
#     preserving important visual structure.
#
#   • GLOBAL AVERAGE POOLING (GAP) is used at the end
#       so the model works with ANY image size.
#
#   • Final layer (fc) is a fully connected classifier
#       that outputs class logits.
#
# ------------------------------------------------------------
# INPUT ASSUMPTION:
# ------------------------------------------------------------
#
# The network expects 3-channel RGB images:
#
#   • Shape: [B, 3, H, W]
#   • H, W can be ANY size:
#       - CIFAR-10 (32×32)
#       - resized datasets
#       - high-resolution images
#
# ------------------------------------------------------------
# IMPORTANT NOTE ABOUT "STATIC" FILTERS:
# ------------------------------------------------------------
#
# The word "static" (if enabled) means:
#
#   → Filters may be overwritten ONLY at INITIALIZATION TIME
#       (e.g., Sobel, edge, corner, blur kernels).
#
#   → After training starts, learning depends on requires_grad:
#
#       - If a layer is trainable (requires_grad=True),
#         it WILL learn normally via backpropagation.
#
#       - If you intentionally freeze a layer
#         (requires_grad=False),
#         it will NOT learn.
#
# This is NOT automatically a frozen-filter network.
#
# It is a CLASSICAL CNN that may START from known kernels
# and STILL learn end-to-end.
#
# ============================================================
# HOW LEARNING HAPPENS
# ============================================================
#
# During training, the following steps occur:
#
#   outputs = model(images)
#   loss    = criterion(outputs, labels)
#   loss.backward()
#   optimizer.step()
#
# PyTorch automatically computes gradients for ALL trainable
# parameters in the network.
#
# These include:
#
#   • conv1.weight, conv1.bias      (32-channel low-level filters)  ✅ UPDATED
#   • conv2.weight, conv2.bias      (16-channel mid-level filters)  ✅ UPDATED
#   • conv3.weight, conv3.bias      (8-channel higher-level filters)✅ UPDATED
#
#   • bn1.weight, bn1.bias          (BatchNorm scale/shift for conv1: 32) ✅ UPDATED
#   • bn2.weight, bn2.bias          (BatchNorm scale/shift for conv2: 16) ✅ UPDATED
#   • bn3.weight, bn3.bias          (BatchNorm scale/shift for conv3: 8)  ✅ UPDATED
#
#   • fc.weight,  fc.bias           (final classifier: 8 → num_classes)   ✅ UPDATED
#
# Pooling layers have NO learnable parameters:
#   • They perform fixed max operations only.
#
# GAP (AdaptiveAvgPool2d) also has NO learnable parameters:
#   • It averages spatial values:
#       [B, C, H, W] → [B, C, 1, 1]
#
# ============================================================
# WHAT optimizer.step() DOES
# ============================================================
#
# Because:
#   • No layers are frozen by default
#   • requires_grad = True for all parameters
#
# The optimizer updates:
#
#   ✔ conv1
#   ✔ conv2
#   ✔ conv3
#   ✔ all BatchNorm layers
#   ✔ the final fully connected layer
#
# ============================================================
# SO: WILL ALL LEARNABLE LAYERS LEARN?
# ============================================================
#
# YES (by default).
#
#   • conv1 learns (unless explicitly frozen)
#   • conv2 learns (unless explicitly frozen)
#   • conv3 learns (unless explicitly frozen)
#   • BatchNorm learns (gamma/beta + running statistics)
#   • fc learns
#
# Pooling and GAP NEVER learn — they are purely mathematical.
#
# ============================================================
# WHAT WOULD STOP LEARNING?
# ============================================================
#
# If you write:
#
#   param.requires_grad = False
#
# for any layer, that layer will STOP learning.
#
# Example: freezing conv1
#
#   for p in model.conv1.parameters():
#       p.requires_grad = False
#
# This is OPTIONAL and only done intentionally.
#
# ============================================================
# WHY THIS IS A CLASSICAL NEURAL NETWORK
# ============================================================
#
# Because:
#
#   • Filters are initialized
#   • Filters are trained (unless frozen)
#   • Weights change through backpropagation
#   • Learning is end-to-end
#   • Pooling reduces spatial size progressively
#   • GAP removes dependence on image resolution
#
# This is exactly how CNNs are trained in practice.
#
# The only difference here is OPTIONAL intelligent
# initialization instead of purely random filters.
#
# ============================================================
# NETWORK SHAPE (IMAGE-SIZE INDEPENDENT WITH GAP) ✅ UPDATED
# ============================================================
#
# Input image:                [3   x H   x W]
#
# After conv1:                [32  x H   x W]        ✅ UPDATED
# After pool1:                [32  x H/2 x W/2]      ✅ UPDATED
#
# After conv2:                [16  x H/2 x W/2]      ✅ UPDATED
# After pool2:                [16  x H/4 x W/4]      ✅ UPDATED
#
# After conv3:                [8   x H/4 x W/4]      ✅ UPDATED
#
# After GAP:                  [8   x 1   x 1]        ✅ UPDATED
# After flatten:              [8]                    ✅ UPDATED
# Output layer (fc):          [num_classes]
#
# ============================================================
# SUMMARY
# ============================================================
#
# ✅ Optional static initialization (conv layers may start from known kernels)
# ✅ Dynamic learning during training (unless layers are frozen)
# ✅ Progressive spatial reduction via pooling
# ✅ GAP enables ANY input image size
# ✅ Classical CNN trained end-to-end with backpropagation



class StaticInitLearnableCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # --------------------------------------------------------
        # cuDNN AUTOTUNER
        # --------------------------------------------------------
        torch.backends.cudnn.benchmark = True

        # ------------------------------------------------------
        # LAYER 1: 3 → 32 channels ✅ UPDATED
        # ------------------------------------------------------
        # 3 input channels (RGB) → 32 feature maps using 3x3 filters
        # Padding = 1 to keep spatial size
        #
        # Input shape assumption:
        #   [B, 3, H, W]   (ANY H, W)
        # ------------------------------------------------------
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,   # ✅ UPDATED
            kernel_size=3,
            padding=1,
            bias=True
        )

        # ------------------------------------------------------
        # BatchNorm for conv1 (normalizes 32 output channels) ✅ UPDATED
        # ------------------------------------------------------
        self.bn1 = nn.BatchNorm2d(32)  # ✅ UPDATED

        # ------------------------------------------------------
        # LAYER 2: 32 → 16 channels ✅ UPDATED
        # ------------------------------------------------------
        # 32 input feature maps → 16 output feature maps
        # using 3×3 filters, padding=1 keeps spatial size.
        #
        # After first pooling:
        #   input to conv2 : [B,  32, H/2, W/2]
        #   output of conv2: [B,  16, H/2, W/2]
        # ------------------------------------------------------
        self.conv2 = nn.Conv2d(
            in_channels=32,    # ✅ UPDATED
            out_channels=16,   # ✅ UPDATED
            kernel_size=3,
            padding=1,
            bias=True
        )

        # ------------------------------------------------------
        # BatchNorm for conv2 (normalizes 16 channels) ✅ UPDATED
        # ------------------------------------------------------
        self.bn2 = nn.BatchNorm2d(16)  # ✅ UPDATED

        # ------------------------------------------------------
        # POOLING LAYER: MaxPool2d(2, 2)
        # ------------------------------------------------------
        self.pool = nn.MaxPool2d(2, 2)

        # ------------------------------------------------------
        # ✅ LAYER 3: 16 → 8 channels ✅ UPDATED
        # ------------------------------------------------------
        # After second pooling:
        #   input to conv3 : [B,  16, H/4, W/4]
        #   output of conv3: [B,   8, H/4, W/4]
        # ------------------------------------------------------
        self.conv3 = nn.Conv2d(
            in_channels=16,   # ✅ UPDATED
            out_channels=8,   # ✅ UPDATED
            kernel_size=3,
            padding=1,
            bias=True
        )

        # ------------------------------------------------------
        # ✅ BatchNorm for conv3 (normalizes 8 channels) ✅ UPDATED
        # ------------------------------------------------------
        self.bn3 = nn.BatchNorm2d(8)  # ✅ UPDATED

        # ------------------------------------------------------
        # 🔑 GLOBAL AVERAGE POOLING (IMAGE-SIZE INDEPENDENT)
        # ------------------------------------------------------
        # In THIS model after conv3, C = 8 ✅ UPDATED:
        #   [B, 8, H', W'] → [B, 8, 1, 1]
        # ------------------------------------------------------
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ------------------------------------------------------
        # ✅ DROPOUT (GENERALIZATION BOOST)
        # ------------------------------------------------------
        self.dropout = nn.Dropout(p=0.3)

        # ------------------------------------------------------
        # FULLY CONNECTED CLASSIFIER (UPDATED)
        # ------------------------------------------------------
        # ✅ IMPORTANT UPDATE:
        # -------------------
        # Because conv3 now outputs 8 channels, GAP now outputs:
        #   [B, 8, 1, 1] → flatten → [B, 8]
        #
        # Therefore the classifier must be:
        #   nn.Linear(8, num_classes) ✅ UPDATED
        # ------------------------------------------------------
        self.fc = nn.Linear(8, num_classes)  # ✅ UPDATED

        # ------------------------------------------------------
        # STATIC FILTER INITIALIZATION (if enabled)
        # ------------------------------------------------------
        # IMPORTANT NOTE (accuracy-related):
        # ----------------------------------
        # If conv1 is overwritten with static filters, it may LIMIT
        # the benefit of increasing conv1 channels unless your
        # static filter bank actually fills/uses all 32 output maps ✅ UPDATED
        # ------------------------------------------------------
        if STATIC_FILTERS:
            self._init_conv1_static()
            # self._init_conv2_static()





    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # STATIC INITIALIZATION FOR LAYER 1
    # ----------------------------------------------------------
    def _init_conv1_static(self):
        with torch.no_grad():                                              # disable gradients during manual init
            w = self.conv1.weight                                          # conv1 weights → [out_channels, 3, 3, 3]
            out_channels, in_channels, kh, kw = w.shape                    # get conv1 shape

            # ✅ UPDATED FOR YOUR NEW MODEL (conv1=32, conv2=16, conv3=8):
            # ----------------------------------------------------------
            # Your conv1 is now:
            #   in_channels  = 3    (RGB)
            #   out_channels = 32   (32 feature maps / 32 filters) ✅ UPDATED
            #   kernel       = 3x3
            #
            # So conv1 produces:
            #   [B, 3, H, W] → [B, 32, H, W] ✅ UPDATED
            #
            # NOTE:
            # • This is still lightweight compared to huge models.
            # • Static init still works the same way; we just fill 32 filters now.
            assert in_channels == 3 and kh == 3 and kw == 3                # expect RGB input and 3x3 kernels

            # ------------------------------------------------------------------
            # BASIC FILTERS (IDENTITY, SHARPENING, SMOOTHING)
            # ------------------------------------------------------------------

            identity = torch.tensor([
                [0., 0., 0.],
                [0., 1., 0.],
                [0., 0., 0.],
            ])    # Identity → preserves pixel value; detects no feature (baseline)

            edge_detection = torch.tensor([
                [ 0., -1.,  0.],
                [-1.,  4., -1.],
                [ 0., -1.,  0.],
            ])     # Laplacian edge → detects edges from all directions equally

            sharpen = torch.tensor([
                [ 0., -1.,  0.],
                [-1.,  5., -1.],
                [ 0., -1.,  0.],
            ])      # Sharpens fine details and texture; enhances edges

            box_blur = (1/9) * torch.ones((3, 3))   # Uniform blur → reduces noise

            gaussian_blur = (1/16) * torch.tensor([
                [1., 2., 1.],
                [2., 4., 2.],
                [1., 2., 1.],
            ])  # Gaussian blur → smooths but preserves structure gracefully

            # ------------------------------------------------------------------
            # EDGE FILTERS (MULTIPLE ORIENTATIONS EVERY 45°)
            # ------------------------------------------------------------------

            edge_0 = torch.tensor([
                [ 1.,  1.,  1.],
                [-2., -2., -2.],
                [ 1.,  1.,  1.],
            ])          # horizontal edges (top vs bottom contrast)

            edge_45 = torch.tensor([
                [ 1., -2.,  1.],
                [ 1., -2.,  1.],
                [ 1., -2.,  1.],
            ])           # 45° edges (diagonal)

            edge_90 = torch.tensor([
                [-2.,  1.,  1.],
                [ 1., -2.,  1.],
                [ 1.,  1., -2.],
            ])            # vertical edges (left vs right contrast)

            edge_135 = torch.tensor([
                [ 1.,  1., -2.],
                [-2.,  1.,  1.],
                [ 1., -2., -2.],
            ])           # 135° diagonal edges

            edge_180 = torch.tensor([
                [-1., -1., -1.],
                [ 2.,  2.,  2.],
                [-1., -1., -1.],
            ])         # horizontal edges reversed orientation

            edge_225 = torch.tensor([
                [ 2., -1., -1.],
                [-1.,  2., -1.],
                [-1., -1.,  2.],
            ])         # diagonal 225° edge

            edge_270 = torch.tensor([
                [-1.,  2., -1.],
                [-1.,  2., -1.],
                [-1.,  2., -1.],
            ])         # vertical edges (reverse orientation)

            edge_315 = torch.tensor([
                [ 1., -1., -1.],
                [-1.,  1.,  1.],
                [-1., -1.,  1.],
            ])          # diagonal 315° edge

            # ------------------------------------------------------------------
            # CORNER DETECTION FILTERS
            # ------------------------------------------------------------------

            corner_0 = torch.tensor([
                [ 1.,  1.,  0.],
                [ 1.,  0., -1.],
                [ 0., -1., -1.],
            ])          # corner opening upward-right

            corner_45 = torch.tensor([
                [ 1.,  0.,  1.],
                [ 0., -1.,  1.],
                [-1., -1.,  0.],
            ])          # corner opening upward-left

            corner_90 = torch.tensor([
                [ 0.,  1.,  1.],
                [-1.,  0.,  1.],
                [-1., -1.,  0.],
            ])          # corner opening left-down

            corner_135 = torch.tensor([
                [ 1.,  0., -1.],
                [ 0.,  1.,  1.],
                [ 0., -1., -1.],
            ])          # corner opening right-down

            corner_180 = corner_0.clone()
            corner_225 = corner_45.clone()
            corner_270 = corner_90.clone()
            corner_315 = corner_135.clone()

            # ------------------------------------------------------------------
            # CURVE DETECTION FILTERS
            # ------------------------------------------------------------------

            curve_0 = torch.tensor([
                [ 0.,  1.,  0.],
                [-1.,  1., -1.],
                [ 0., -1.,  0.],
            ])          # curve bending upward

            curve_45 = torch.tensor([
                [ 1.,  0., -1.],
                [ 0.,  1.,  0.],
                [-1.,  0.,  1.],
            ])           # curve bending top-left to bottom-right

            curve_90 = torch.tensor([
                [ 0., -1.,  0.],
                [ 1.,  1.,  1.],
                [ 0., -1.,  0.],
            ])          # vertical "bulge"

            curve_135 = torch.tensor([
                [-1.,  0.,  1.],
                [ 0.,  1.,  0.],
                [ 1.,  0., -1.],
            ])          # curve bending bottom-left to top-right

            curve_180 = torch.tensor([
                [ 0., -1.,  0.],
                [-1.,  1., -1.],
                [ 0.,  1.,  0.],
            ])          # curve bending downward

            curve_225 = curve_135.clone()
            curve_270 = curve_90.clone()
            curve_315 = curve_45.clone()

            # ------------------------------------------------------------------
            # LINE DETECTION FILTERS
            # ------------------------------------------------------------------

            line_0 = torch.tensor([
                [ 1.,  1.,  1.],
                [-2., -2., -2.],
                [ 1.,  1.,  1.],
            ])            # horizontal line

            line_45 = torch.tensor([
                [-2.,  1.,  1.],
                [ 1., -2.,  1.],
                [ 1.,  1., -2.],
            ])            # 45° diagonal line

            line_90 = torch.tensor([
                [-1., -1., -1.],
                [ 2.,  2.,  2.],
                [-1., -1., -1.],
            ])         # vertical line

            line_135 = torch.tensor([
                [ 1., -2.,  1.],
                [ 1., -2.,  1.],
                [ 1., -2.,  1.],
            ])          # 135° diagonal line

            line_180 = torch.tensor([
                [-1., -1., -1.],
                [-2., -2., -2.],
                [-1., -1., -1.],
            ])      # reversed horizontal line

            line_225 = torch.tensor([
                [ 1.,  1., -2.],
                [ 1., -2.,  1.],
                [-2.,  1.,  1.],
            ])      # diagonal variant

            line_270 = torch.tensor([
                [-1.,  2., -1.],
                [-1.,  2., -1.],
                [-1.,  2., -1.],
            ])      # reversed vertical line

            line_315 = torch.tensor([
                [ 1., -1.,  1.],
                [-1., -2., -1.],
                [ 1., -1.,  1.],
            ])      # diagonal 315°

            # ------------------------------------------------------------------
            # SOBEL FILTERS — GRADIENT MAGNITUDE IN SPECIFIC DIRECTIONS
            # ------------------------------------------------------------------

            sobel_0 = torch.tensor([
                [-1.,  0.,  1.],
                [-2.,  0.,  2.],
                [-1.,  0.,  1.],
            ])           # gradient along x-axis

            sobel_45 = torch.tensor([
                [ 0., -1., -2.],
                [ 1.,  0., -1.],
                [ 2.,  1.,  0.],
            ])           # diagonal gradient

            sobel_90 = torch.tensor([
                [ 1.,  2.,  1.],
                [ 0.,  0.,  0.],
                [-1., -2., -1.],
            ])           # gradient along y-axis

            sobel_135 = torch.tensor([
                [ 2.,  1.,  0.],
                [ 1.,  0., -1.],
                [ 0., -1., -2.],
            ])          # 135° gradient

            sobel_180 = torch.tensor([
                [ 1.,  0., -1.],
                [ 2.,  0., -2.],
                [ 1.,  0., -1.],
            ])          # reverse x-gradient

            sobel_225 = torch.tensor([
                [ 0.,  1.,  2.],
                [-1.,  0.,  1.],
                [-2., -1.,  0.],
            ])          # diagonal gradient

            sobel_270 = torch.tensor([
                [-1., -2., -1.],
                [ 0.,  0.,  0.],
                [ 1.,  2.,  1.],
            ])          # reverse y-gradient

            sobel_315 = torch.tensor([
                [-2., -1.,  0.],
                [-1.,  0.,  1.],
                [ 0.,  1.,  2.],
            ])          # 315° gradient

            # ------------------------------------------------------------------
            # COLLECT ALL KERNELS
            # ------------------------------------------------------------------
            kernels = [
                identity, edge_detection, sharpen, box_blur, gaussian_blur,
                edge_0, edge_45, edge_90, edge_135, edge_180, edge_225, edge_270, edge_315,
                corner_0, corner_45, corner_90, corner_135, corner_180, corner_225, corner_270, corner_315,
                curve_0, curve_45, curve_90, curve_135, curve_180, curve_225, curve_270, curve_315,
                line_0, line_45, line_90, line_135, line_180, line_225, line_270, line_315,
                sobel_0, sobel_45, sobel_90, sobel_135, sobel_180, sobel_225, sobel_270, sobel_315,
            ]

            num_kernels = len(kernels)                                      # total number of base kernels (45)

            # ------------------------------------------------------------------
            # ASSIGN STATIC KERNELS → conv1 WEIGHTS (UPDATED FOR 32 CHANNELS)
            #
            # ✅ Your conv1 now has 32 output channels:
            #   out_channels = 32 ✅ UPDATED
            #
            # We currently have:
            #   num_kernels = 45 handcrafted 3×3 kernels
            #
            # Because 32 < 45:
            #   • We take ONLY the first 32 kernels (truncate) ✅ UPDATED
            #   • No wrap-around repetition is needed ✅ UPDATED
            #
            # This guarantees:
            #   • conv1 starts with strong low-level detectors
            #   • training converges easier (better inductive bias)
            #   • all 32 filters are UNIQUE handcrafted kernels
            # ------------------------------------------------------------------

            for i in range(out_channels):                                  # loop over each output filter (0..31) ✅ UPDATED
                k2d = kernels[i].to(w.dtype)                               # take first 32 kernels directly (no wrap)
                for c in range(in_channels):                               # copy same kernel into each RGB channel
                    w[i, c].copy_(k2d)                                     # write into conv1 weight tensor

            unused = max(0, num_kernels - out_channels)                    # how many handcrafted kernels were NOT used ✅ UPDATED
            print(
                f"[init_conv1_static] out_channels={out_channels}, num_kernels={num_kernels} → "
                f"{out_channels} used (unique) + {unused} unused (truncated)"
            )





    # ----------------------------------------------------------
    # STATIC INITIALIZATION FOR LAYER 2 (UPDATED FOR 16×32) ✅ UPDATED
    # ----------------------------------------------------------
    def _init_conv2_static(self):
        with torch.no_grad():                                                           # disable gradients (manual init)
            w = self.conv2.weight                                                       # conv2 weights → [out_channels, in_channels, 3, 3]
            out_channels, in_channels, kh, kw = w.shape

            # ✅ UPDATED FOR YOUR NEW MODEL (conv1=32, conv2=16, conv3=8):
            # ----------------------------------------------------------
            # Your conv2 is now:
            #   in_channels  = 32    (from conv1 out_channels) ✅ UPDATED
            #   out_channels = 16    (16 feature maps / filters) ✅ UPDATED
            #   kernel       = 3x3
            #
            # So conv2 produces:
            #   [B, 32, H/2, W/2] → [B, 16, H/2, W/2] ✅ UPDATED
            #
            # This is a structured mid-level feature builder after conv1.
            assert kh == 3 and kw == 3                                                  # ensure 3x3 kernel size

            # ---------------------------------------------------------------------
            # FILTER DEFINITIONS (EACH 3×3, WRITTEN IN THREE ROWS)
            #
            # conv2 receives 32 feature maps (NOT 16 anymore). ✅ UPDATED
            #
            # Meaning:
            #   • conv1 produced primitive detectors (edges/corners/etc.) across 32 maps ✅ UPDATED
            #   • conv2 combines those 32 maps into 16 mid-level features:
            #       - parts, textures, repeated patterns
            #       - stronger combinations of the static conv1 responses
            #
            # NOTE:
            #   Here out_channels is smaller than in_channels (16 < 32). ✅ UPDATED
            #   This is NOT wrong — it is a deliberate "feature compression" step:
            #     • reduces compute and memory
            #     • forces the network to learn useful mixtures of conv1 features
            #     • can generalize well with BN + ReLU
            # ---------------------------------------------------------------------

            # 1) Horizontal edge detector
            edge_h = torch.tensor([
                [-1., -1., -1.],
                [ 2.,  2.,  2.],
                [-1., -1., -1.],
            ])

            # 2) Vertical edge detector
            edge_v = torch.tensor([
                [-1.,  2., -1.],
                [-1.,  2., -1.],
                [-1.,  2., -1.],
            ])

            # 3) Emboss filter
            emboss = torch.tensor([
                [-2., -1.,  0.],
                [-1.,  1.,  1.],
                [ 0.,  1.,  2.],
            ])

            # 4) Average blur (3×3 mean filter)
            avg = (1/9) * torch.ones((3, 3))

            # 5) Sobel X (horizontal gradient)
            sobel_x = torch.tensor([
                [-1.,  0.,  1.],
                [-2.,  0.,  2.],
                [-1.,  0.,  1.],
            ])

            # 6) Sobel Y (vertical gradient)
            sobel_y = torch.tensor([
                [-1., -2., -1.],
                [ 0.,  0.,  0.],
                [ 1.,  2.,  1.],
            ])

            # Collect all filters into a kernel bank
            kernels = [
                edge_h,     # 0 horizontal edge
                edge_v,     # 1 vertical edge
                emboss,     # 2 emboss shading
                avg,        # 3 smoothing blur
                sobel_x,    # 4 gradient X
                sobel_y,    # 5 gradient Y
            ]
            num_kernels = len(kernels)

            # ---------------------------------------------------------------------
            # ASSIGN FILTERS TO ALL conv2 WEIGHTS (UPDATED FOR 16×32) ✅ UPDATED
            #
            # conv2 has:
            #   out_channels = 16   (filters) ✅ UPDATED
            #   in_channels  = 32   (input feature maps from conv1) ✅ UPDATED
            #
            # ✅ Why repetition is OK here:
            #   • conv2 has 16×32 = 512 small 3×3 kernels ✅ UPDATED (same count, different shape)
            #   • we only define 6 base kernels
            #   • repeating them still creates many different paths because each output
            #     channel mixes MANY input channels, and training learns useful combos
            #
            # This creates a structured conv2 initialization where:
            #   • some connections emphasize gradients (Sobel)
            #   • some emphasize edges (horizontal/vertical)
            #   • some smooth (avg)
            #   • some emboss (directional structure)
            # ---------------------------------------------------------------------
            for out_idx in range(out_channels):                                       # loop over all 16 output filters ✅ UPDATED
                for in_idx in range(in_channels):                                     # loop over all 32 input feature maps ✅ UPDATED

                    # Choose kernel pattern based on (out × in) mod #kernels
                    # NOTE:
                    #   We use max(1, in_idx) to avoid always selecting kernel[0] when in_idx=0.
                    k = kernels[(out_idx * max(1, in_idx)) % num_kernels].to(w.dtype)

                    # Copy kernel into weight tensor
                    w[out_idx, in_idx].copy_(k)

            print(f"[init_conv2_static] {out_channels}x{in_channels} 2D 3x3 kernels assigned")  # log ✅ UPDATED




    # ----------------------------------------------------------
    # STATIC INITIALIZATION FOR LAYER 3 (UPDATED FOR 8 FEATURES) ✅ UPDATED
    # ----------------------------------------------------------
    def _init_conv3_static(self):
        with torch.no_grad():                                                           # disable gradients (manual init)
            w = self.conv3.weight                                                       # conv3 weights → [out_channels, in_channels, 3, 3]
            out_channels, in_channels, kh, kw = w.shape

            # ✅ UPDATED FOR YOUR NEW MODEL (conv1=32, conv2=16, conv3=8):
            # ----------------------------------------------------------
            # Your conv3 is now:
            #   in_channels  = 16    (from conv2 out_channels) ✅ UPDATED
            #   out_channels = 8     (8 feature maps / filters) ✅ UPDATED
            #   kernel       = 3x3
            #
            # So conv3 produces:
            #   [B, 16, H/4, W/4] → [B, 8, H/4, W/4] ✅ UPDATED
            #
            # This layer is where the network starts forming higher-level patterns:
            #   • textures and repeated structures
            #   • object parts (wheels, wings, eyes, etc.)
            #   • shape composition from conv1+conv2 primitives
            assert kh == 3 and kw == 3                                                  # ensure 3x3 kernel size

            # ✅ EXTRA SAFETY CHECK (keeps your code robust if the architecture changes later)
            # -------------------------------------------------------------------------------
            # If someone accidentally changes conv2/conv3 channel counts later, this will fail fast
            # instead of silently initializing wrong shapes.
            assert in_channels == 16 and out_channels == 8                              # expect exact conv3 shape ✅ UPDATED

            # ---------------------------------------------------------------------
            # FILTER DEFINITIONS (EACH 3×3, WRITTEN IN THREE ROWS)
            #
            # conv3 receives 16 feature maps (NOT raw RGB anymore). ✅ UPDATED
            #
            # Meaning:
            #   • conv1: low-level primitives (edges/corners/lines) across 32 maps ✅ UPDATED
            #   • conv2: mid-level combinations (textures/parts) reduced to 16 maps ✅ UPDATED
            #   • conv3: stronger mid/high-level compositions (parts → object patterns) into 8 maps ✅ UPDATED
            #
            # IMPORTANT:
            # ----------
            # Even if we "start" with static kernels here, training can still learn
            # (unless you freeze parameters). This init just gives a helpful bias.
            # ---------------------------------------------------------------------

            # 1) Laplacian (all-direction edge / detail emphasis)
            laplacian = torch.tensor([
                [ 0., -1.,  0.],
                [-1.,  4., -1.],
                [ 0., -1.,  0.],
            ])

            # 2) Sharpen (stronger detail boost)
            sharpen = torch.tensor([
                [ 0., -1.,  0.],
                [-1.,  5., -1.],
                [ 0., -1.,  0.],
            ])

            # 3) High-pass (aggressive edge/detail extraction)
            high_pass = torch.tensor([
                [-1., -1., -1.],
                [-1.,  8., -1.],
                [-1., -1., -1.],
            ])

            # 4) Box blur (smooth noisy feature maps)
            box_blur = (1/9) * torch.ones((3, 3))

            # 5) Gaussian blur (smoother than box blur; preserves structure better)
            gaussian_blur = (1/16) * torch.tensor([
                [1., 2., 1.],
                [2., 4., 2.],
                [1., 2., 1.],
            ])

            # 6) Emboss (adds directional depth / relief)
            emboss = torch.tensor([
                [-2., -1.,  0.],
                [-1.,  1.,  1.],
                [ 0.,  1.,  2.],
            ])

            # 7) Sobel X (gradient along x: left↔right intensity changes)
            sobel_x = torch.tensor([
                [-1.,  0.,  1.],
                [-2.,  0.,  2.],
                [-1.,  0.,  1.],
            ])

            # 8) Sobel Y (gradient along y: top↔bottom intensity changes)
            sobel_y = torch.tensor([
                [ 1.,  2.,  1.],
                [ 0.,  0.,  0.],
                [-1., -2., -1.],
            ])

            # 9) Diagonal gradient (45°-ish emphasis)
            diag_45 = torch.tensor([
                [ 0., -1., -2.],
                [ 1.,  0., -1.],
                [ 2.,  1.,  0.],
            ])

            # 10) Diagonal gradient (135°-ish emphasis)
            diag_135 = torch.tensor([
                [ 2.,  1.,  0.],
                [ 1.,  0., -1.],
                [ 0., -1., -2.],
            ])

            # Collect all filters into a kernel bank
            kernels = [
                laplacian,        # 0
                sharpen,          # 1
                high_pass,        # 2
                box_blur,         # 3
                gaussian_blur,    # 4
                emboss,           # 5
                sobel_x,          # 6
                sobel_y,          # 7
                diag_45,          # 8
                diag_135,         # 9
            ]
            num_kernels = len(kernels)

            # ---------------------------------------------------------------------
            # ASSIGN FILTERS TO ALL conv3 WEIGHTS (UPDATED FOR 8×16) ✅ UPDATED
            #
            # conv3 has:
            #   out_channels = 8   (filters / output features) ✅ UPDATED
            #   in_channels  = 16  (input features from conv2) ✅ UPDATED
            #
            # Strategy:
            # ---------
            # We repeat a small bank of useful kernels across the 8×16 connections.
            #
            # Why this can help:
            #   • conv3 sees already-processed feature maps, not raw pixels
            #   • high-pass / sharpen emphasizes discriminative parts
            #   • blur kernels can stabilize noisy activations
            #   • sobel/diagonal gradients preserve directional structure
            #
            # NOTE:
            # -----
            # If conv3 is trainable (requires_grad=True), training will refine these
            # weights beyond the initial static patterns.
            # ---------------------------------------------------------------------
            for out_idx in range(out_channels):                                       # loop over all 8 output filters ✅ UPDATED
                for in_idx in range(in_channels):                                     # loop over all 16 input feature maps ✅ UPDATED

                    # Choose kernel pattern based on a mixed index to reduce repetition artifacts
                    # (still deterministic, but spreads kernels across channels more evenly)
                    k = kernels[(out_idx * 7 + in_idx * 3) % num_kernels].to(w.dtype)

                    # Copy kernel into weight tensor
                    w[out_idx, in_idx].copy_(k)

            print(f"[init_conv3_static] {out_channels}x{in_channels} 2D 3x3 kernels assigned")  # log ✅ UPDATED




    def forward(self, x):
        # At entry:
        #   x shape → [B, 3, H, W]
        #   (ANY image size: CIFAR-10, resized data, or original resolution)

        # -------------------
        # BLOCK 1: CONV1 → BN1 → ReLU → POOL
        # -------------------

        # Conv1: 3 → 32 channels, preserves H, W ✅ UPDATED
        #   [B, 3, H, W] → [B, 32, H, W]
        x = self.conv1(x)

        # BatchNorm on 32 channels (stabilizes activations) ✅ UPDATED
        x = self.bn1(x)

        # Non-linearity: ReLU
        x = F.relu(x)

        # MaxPool: H×W → H/2×W/2
        #   [B, 32, H, W] → [B, 32, H/2, W/2] ✅ UPDATED
        x = self.pool(x)

        # -------------------
        # BLOCK 2: CONV2 → BN2 → ReLU → POOL
        # -------------------

        # Conv2: 32 → 16 channels ✅ UPDATED
        #   [B, 32, H/2, W/2] → [B, 16, H/2, W/2]
        x = self.conv2(x)

        # BatchNorm on 16 channels ✅ UPDATED
        x = self.bn2(x)

        # ReLU
        x = F.relu(x)

        # ✅ SECOND POOLING STEP (matches your current __init__)
        # -----------------------------------------------------
        # Reduces spatial size again to build more compact features:
        #   [B, 16, H/2, W/2] → [B, 16, H/4, W/4] ✅ UPDATED
        x = self.pool(x)

        # -------------------
        # ✅ BLOCK 3: CONV3 → BN3 → ReLU
        # -------------------

        # Conv3: 16 → 8 channels ✅ UPDATED
        #   [B, 16, H/4, W/4] → [B, 8, H/4, W/4]
        x = self.conv3(x)

        # BatchNorm on 8 channels ✅ UPDATED
        x = self.bn3(x)

        # ReLU
        x = F.relu(x)

        # -------------------
        # GLOBAL AVERAGE POOLING (IMAGE-SIZE INDEPENDENT)
        # -------------------

        # Replaces hard-coded spatial flattening.
        #
        # Converts:
        #   [B, 8, H/4, W/4] → [B, 8, 1, 1] ✅ UPDATED
        #
        # This step removes dependence on image size.
        x = self.gap(x)

        # Flatten channel dimension only
        #   [B, 8, 1, 1] → [B, 8] ✅ UPDATED
        x = torch.flatten(x, 1)

        # -------------------
        # ✅ DROPOUT (GENERALIZATION BOOST)
        # -------------------
        x = self.dropout(x)

        # -------------------
        # LINEAR CLASSIFIER
        # -------------------

        # Fully connected layer:
        #   [B, 8] → [B, num_classes] ✅ UPDATED
        logits = self.fc(x)

        # logits are returned directly.
        # CrossEntropyLoss will apply softmax internally.
        return logits




# ------------------------------------------------------------------
# GLOBAL DEBUG FLAG + HELPER
# ------------------------------------------------------------------



def debug_print(*args, **kwargs):
    """
    Simple debug print wrapper.
    If DEBUG is True → behaves like print().
    If DEBUG is False → does nothing.
    """
    if DEBUG_FLAG:
        print(*args, **kwargs)

# ============================================================
# TRAINING FUNCTION (WORKS FOR STATIC AND DYNAMIC CNN MODELS)
# ============================================================
# Train the CNN model for a fixed number of epochs using the provided DataLoader.
#
# This training function works for ALL CNN configurations, including:
#   • CNNs with static (manually defined) filters in conv1 (e.g., Sobel, edges, corners).
#   • CNNs with randomly initialized and learnable filters.
#   • Networks with or without pooling layers.
#   • Standard datasets like CIFAR-10 or any custom dataset.
#   • Any input size supported by the model (e.g., 32×32 images).
#
# 💡 Why this works universally:
# Training depends on *backpropagation*, not on how filters are initialized.
# The optimizer updates only parameters that have requires_grad=True().
#
# -----------------------------------------------------------------------
# 🧠 Learning behavior by layer:
#
# conv1 — Low-level feature extraction:
#   • If FILTERS are STATIC:
#       → Kernels are pre-defined (Sobel, corners, edges).
#       → These filters DO NOT change during training.
#       → They behave as a fixed feature extractor.
#
#   • If FILTERS are TRAINABLE:
#       → Kernels are initialized randomly.
#       → Each weight is updated via gradient descent.
#       → Filters learn edges, patterns, and pixel textures directly from data.
#
# conv2 — Mid-level feature learning:
#   • Receives feature maps from conv1.
#   • Learns spatial combinations such as:
#       → corners
#       → shapes
#       → textures
#       → structural patterns.
#   • Weights adapt to match more meaningful patterns through backprop.
#
# filters (in ALL convolution layers):
#   • Every kernel is a matrix of learnable weights.
#
#   During training:
#     1. Forward pass:
#         image → conv → activation → output
#
#     2. Loss calculation:
#         prediction vs expected label → error signal
#
#     3. Backward pass:
#         Computes gradients for each filter weight:
#           dLoss / dWeight
#
#     4. Optimization:
#         optimizer.step() updates:
#           weight ← weight − learning_rate × gradient
#
#   Over many iterations:
#     → filters amplify useful structures
#     → suppress noise
#     → specialize for classification
#
# -----------------------------------------------------------------------
# ✅ Why a SINGLE training loop works for all CNNs:
#
#   • The optimizer automatically updates:
#         ONLY parameters where requires_grad == True
#
#   • Static filters have:
#         requires_grad=False → never updated
#
#   • Trainable filters have:
#         requires_grad=True → learned by backprop
#
# Therefore:
#   No conditional logic or special handling is needed in the training loop.
#   The same code trains both static and dynamic filter networks correctly.

def train_model(model, train_loader, device, num_epochs=2, lr=3e-3, test_loader=None):

    # --------------------------------------------------------
    # AUTOMATIC MIXED PRECISION (AMP)
    # --------------------------------------------------------
    # Uses float16 where safe and float32 where needed.
    #
    # Benefits:
    #   • Faster training on GPU
    #   • Lower GPU memory usage
    #
    # Automatically disabled on CPU
    # --------------------------------------------------------
    use_amp = (device.type == "cuda")

    # IMPORTANT VERSION FIX:
    # ----------------------
    # Some PyTorch builds do NOT support:
    #   torch.amp.GradScaler(device_type="cuda", ...)
    #
    # So we use the CUDA GradScaler (works across many versions on Windows):
    #   • Enabled only if device is CUDA
    #   • Disabled automatically on CPU
    #
    # IMPORTANT QUALITY FIX:
    # ----------------------
    # We also silence the "deprecated" warning by trying the newer API first:
    #   torch.amp.GradScaler("cuda", ...)
    # and falling back to:
    #   torch.cuda.amp.GradScaler(...)
    #
    # This keeps behavior identical, just more robust across versions.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ------------------------------------------------------------
    # SEND MODEL TO GPU (IF AVAILABLE) OR CPU
    # ------------------------------------------------------------
    model.to(device)  # move all model parameters + buffers to the selected device

    # ------------------------------------------------------------
    # ENABLE TRAINING MODE
    #   (activates dropout, batchnorm if they exist)
    # ------------------------------------------------------------
    model.train()

    # ------------------------------------------------------------
    # ✅ TRAINING VISIBILITY (DEVICE CONFIRMATION)
    # ------------------------------------------------------------
    # PURPOSE:
    # --------
    # In your logs you saw:
    #   images: ... cpu
    #
    # If device is CPU:
    #   • Training is MUCH slower
    #   • Getting very high CIFAR-10 accuracy (e.g., ~90%) is harder
    #
    # This print helps you confirm you are actually training on CUDA when available.
    # ------------------------------------------------------------
    debug_print(f"[TRAIN] device={device}  use_amp={use_amp}")

    # ------------------------------------------------------------
    # ✅ AUTO LR SCALING BASED ON BATCH SIZE (NEW)
    # ------------------------------------------------------------
    # PURPOSE:
    # --------
    # When you change batch_size, the gradient statistics change:
    #   • Small batch (32)  → noisier gradients → often needs smaller LR
    #   • Large batch (256+)→ smoother gradients → can use larger LR
    #
    # A common practical rule is "linear scaling":
    #   lr_scaled = lr * (batch_size / 64)
    #
    # We use a SAFE clamped version to avoid extreme values when batch=1024.
    #
    # SUPPORTED BATCHES YOU REQUESTED:
    #   32, 64, 128, 256, 1024
    #
    # NOTES:
    # • We treat 64 as the reference point.
    # • We clamp the scaling factor so OneCycleLR remains stable.
    # • You can still override lr manually by passing a different lr.
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # 📊 PRACTICAL LR SCALING EXAMPLES (REFERENCE TABLE)
    # ------------------------------------------------------------
    # Assuming:
    #   • base learning rate (lr) = 3e-3
    #   • reference batch size    = 64
    #
    # The linear scaling rule:
    #   lr_scaled = lr * (batch_size / 64)
    #
    # Produces approximately:
    #
    #   batch_size = 32   → scale = 0.5  → lr ≈ 0.0015
    #   batch_size = 64   → scale = 1.0  → lr ≈ 0.0030
    #   batch_size = 128  → scale = 2.0  → lr ≈ 0.0060
    #   batch_size = 256  → scale = 4.0  → lr ≈ 0.0120
    #   batch_size = 1024 → scale = 16.0 → lr ≈ 0.0480
    #
    # SAFETY NOTE:
    # ------------
    # For very large batches (e.g. 1024), we CLAMP the scale factor
    # to avoid unstable training with OneCycleLR:
    #
    #   scale = min(scale, 8.0)
    #
    # Which results in:
    #
    #   batch_size = 1024 → scale = 8.0 → lr ≈ 0.0240
    #
    # If you intentionally want FULL linear scaling (16×),
    # remove or relax the clamp.
    # ------------------------------------------------------------

    try:
        detected_batch_size = int(getattr(train_loader, "batch_size", 64) or 64)
    except Exception:
        detected_batch_size = 64

    # Reference batch size used for LR scaling
    ref_bs = 64

    # ------------------------------------------------------------
    # ✅ QUALITY FIX FOR VERY LARGE BATCH (GENERALIZATION)
    # ------------------------------------------------------------
    # PURPOSE:
    # --------
    # Extremely large batches (like 1024) can:
    #   • Reduce gradient noise too much
    #   • Hurt generalization (test accuracy)
    #   • Make training feel "saturated" early
    #
    # IMPORTANT:
    # ----------
    # We do NOT change the DataLoader batch size here (that is your choice).
    # We only limit how aggressive LR scaling becomes by using a "virtual batch"
    # for LR scaling purposes.
    #
    # Default behavior:
    #   • Use at most 256 as the LR-scaling reference for stability + accuracy.
    #
    # This helps you keep large-batch throughput while avoiding too-large LR behavior.
    # ------------------------------------------------------------
    virtual_bs_for_lr = min(detected_batch_size, 256)

    # Linear scaling factor
    lr_scale = virtual_bs_for_lr / ref_bs

    # Clamp scaling to prevent extremely large LR for giant batches (e.g., 1024)
    # You can adjust these clamps if you want more aggressive scaling.
    lr_scale = max(0.5, min(lr_scale, 8.0))

    # Compute scaled LR
    lr_scaled = lr * lr_scale

    # Print for visibility / debugging
    debug_print(
        f"[TRAIN] Detected batch_size={detected_batch_size} → "
        f"virtual_bs_for_lr={virtual_bs_for_lr} → "
        f"LR scaled from {lr:.6f} to {lr_scaled:.6f} (scale={lr_scale:.3f})"
    )

    # ------------------------------------------------------------
    # OPTIMIZER: UPDATES ALL LEARNABLE PARAMETERS
    # ------------------------------------------------------------
    # Create the optimizer that is responsible for *training the neural network*.
    #
    # Adam = Adaptive Moment Estimation:
    #   It is an advanced optimization algorithm that improves plain gradient descent.
    #
    # model.parameters():
    #   • Collects ALL trainable tensors in the model:
    #       - convolution filter weights
    #       - bias vectors
    #       - fully connected layers
    #       - batch normalization parameters
    #   • Only parameters with requires_grad = True are included.
    #   • Static / frozen layers are automatically ignored.
    #
    # lr (learning rate):
    #   • Controls how fast each weight changes.
    #   • Larger values = faster learning (but risk instability).
    #   • Smaller values = slower learning (but more stable training).
    #
    # Internally, Adam performs for EACH weight:
    #   1) Uses backpropagation to compute the gradient:
    #        gradient = ∂loss / ∂weight
    #
    #   2) Tracks moving average of gradients (momentum):
    #        m = β1 * previous_m + (1 − β1) * gradient
    #
    #   3) Tracks moving average of squared gradients (variance):
    #        v = β2 * previous_v + (1 − β2) * gradient²
    #
    #   4) Bias correction (makes early steps accurate):
    #        m_hat = m / (1 − β1^t)
    #        v_hat = v / (1 − β2^t)
    #
    #   5) Updates weights:
    #        weight = weight − lr × m_hat / (sqrt(v_hat) + ε)
    #
    # Outcome:
    #   • Each parameter learns at its own speed.
    #   • Large/noisy gradients are stabilized.
    #   • Convergence is faster and smoother than standard SGD.
    #
    # Without this optimizer:
    #   • loss.backward() computes gradients only.
    #   • optimizer.step() is required to APPLY updates.
    #
    # This single line controls learning for:
    #   • conv1 kernels
    #   • conv2 kernels
    #   • fully connected layers
    #   • bias terms
    #   • normalization layers
    #
    # ------------------------------------------------------------
    # OPTIMIZER: AdamW (Weight-decoupled Adam)
    # ------------------------------------------------------------
    # IMPORTANT FIX:
    # --------------
    # • OneCycleLR expects the optimizer LR to MATCH max_lr logic
    # • We therefore use `lr=lr` (the function argument)
    # • This avoids mismatches between base LR and OneCycle peak LR
    #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_scaled,               # ✅ AUTO-SCALED LR BASED ON BATCH SIZE
        weight_decay=1e-4
    )

    # ------------------------------------------------------------
    # LEARNING RATE SCHEDULER — OneCycleLR
    # ------------------------------------------------------------
    # PURPOSE:
    # --------
    # OneCycleLR is a *proactive* learning rate scheduler.
    # Instead of waiting for the loss to plateau (reactive),
    # it follows a predefined learning-rate curve that:
    #
    #   1️⃣ Gradually INCREASES the learning rate (warm-up phase)
    #   2️⃣ Reaches a MAXIMUM learning rate (exploration phase)
    #   3️⃣ Gradually DECREASES the learning rate (fine-tuning phase)
    #
    # This strategy helps the optimizer:
    #   • Escape poor local minima early
    #   • Converge faster
    #   • Achieve lower final loss
    #   • Improve generalization
    #
    # OneCycleLR is especially effective with:
    #   • Adam / AdamW optimizers
    #   • CNNs and vision models
    #   • Mixed Precision (AMP) training
    #
    # IMPORTANT DIFFERENCE vs ReduceLROnPlateau:
    # ------------------------------------------
    # • ReduceLROnPlateau → stepped ONCE per epoch using loss
    # • OneCycleLR        → stepped EVERY BATCH (iteration-based)
    #
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # IMPORTANT FIX (ROBUST TOTAL STEPS):
    # ------------------------------------------------------------
    # Using total_steps avoids subtle mismatches if:
    #   • train_loader length changes
    #   • you later change batch_size
    #   • you use a different sampler
    #
    # total_steps is the *true* number of scheduler.step() calls.
    # ------------------------------------------------------------
    total_steps = len(train_loader) * num_epochs

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,                      # ✅ The optimizer whose learning-rate we want to control (AdamW here).
                                        #    OneCycleLR will directly modify:
                                        #        optimizer.param_groups[i]["lr"]
                                        #    at EVERY mini-batch step.

        max_lr=lr_scaled,               # ✅ PEAK learning rate now auto-scales with batch size.
                                        #    This is the highest LR reached during training.

        total_steps=total_steps,        # ✅ Total number of LR updates across the whole run.

        pct_start=0.1,                  # ✅ Fraction of training used for the "warm-up" (LR INCREASE phase).

        anneal_strategy="cos",          # ✅ Cosine decay after warmup.

        div_factor=5.0,                 # ✅ Controls starting LR = max_lr / div_factor

        final_div_factor=1e3            # ✅ Controls final LR = start_lr / final_div_factor
    )

    # ============================================================
    # COMPLETE END-TO-END EXPLANATION:
    # IMAGE → CONVOLUTION → FEATURES → LOGITS → CrossEntropyLoss
    # ============================================================
    #
    # (KEEPING ALL YOUR COMMENTS EXACTLY AS-IS BELOW)
    #
    # ============================================================
    # STEP 1: INPUT IMAGE (4x4, 1 CHANNEL)
    # ============================================================
    #
    # Let the input image X be:
    #
    #   X =
    #   [
    #     [a11, a12, a13, a14],
    #     [a21, a22, a23, a24],
    #     [a31, a32, a33, a34],
    #     [a41, a42, a43, a44]
    #   ]
    #
    # Each a_ij is a pixel intensity.
    #
    # ============================================================
    # STEP 2: ZERO PADDING (padding = 1)
    # ============================================================
    #
    # Padding adds a border of zeros so convolution does NOT shrink
    # the image size.
    #
    # Padded image X_padded:
    #
    #   X_padded =
    #   [
    #     [  0,   0,   0,   0,   0,   0 ],
    #     [  0, a11, a12, a13, a14,   0 ],
    #     [  0, a21, a22, a23, a24,   0 ],
    #     [  0, a31, a32, a33, a34,   0 ],
    #     [  0, a41, a42, a43, a44,   0 ],
    #     [  0,   0,   0,   0,   0,   0 ]
    #   ]
    #
    # Output will still be 4x4.
    #
    # ============================================================
    # STEP 3: DEFINE A SINGLE CONVOLUTION FILTER (3x3)
    # ============================================================
    #
    # Filter F (learnable weights):
    #
    #   F =
    #   [
    #     [f11, f12, f13],
    #     [f21, f22, f23],
    #     [f31, f32, f33]
    #   ]
    #
    # These f_ij values are trainable parameters.
    #
    # ============================================================
    # STEP 4: APPLY CONVOLUTION → FEATURE MAP
    # ============================================================
    #
    # Example: compute FIRST output pixel y11:
    #
    # Extract 3x3 window from padded image:
    #
    #   [
    #     [  0,   0,   0 ],
    #     [  0, a11, a12 ],
    #     [  0, a21, a22 ]
    #   ]
    #
    # Compute dot product with filter:
    #
    #   y11 =
    #     (0  * f11) + (0  * f12) + (0  * f13)
    #   + (0  * f21) + (a11 * f22) + (a12 * f23)
    #   + (0  * f31) + (a21 * f32) + (a22 * f33)
    #
    # Slide across the image to compute the rest.
    #
    # Final feature map:
    #
    #   Y =
    #   [
    #     [y11, y12, y13, y14],
    #     [y21, y22, y23, y24],
    #     [y31, y32, y33, y34],
    #     [y41, y42, y43, y44]
    #   ]
    #
    # Each y_ij is a learned combination of nearby pixels.
    #
    # ============================================================
    # STEP 5: FLATTEN FEATURE MAP
    # ============================================================
    #
    # Convert Y into a feature vector:
    #
    #   feature_vector =
    #   [
    #     y11, y12, y13, y14,
    #     y21, y22, y23, y24,
    #     y31, y32, y33, y34,
    #     y41, y42, y43, y44
    #   ]
    #
    # This vector is the numeric "description" of the image.
    #
    # ============================================================
    # STEP 6: FULLY CONNECTED LAYER → LOGITS
    # ============================================================
    #
    # Suppose we have 2 classes:
    #
    #   Class 0 = CAT
    #   Class 1 = DOG
    #
    # FC layer:
    #
    #   W =
    #   [
    #     [w1, w2, ..., w16],   # CAT weights
    #     [v1, v2, ..., v16]    # DOG weights
    #   ]
    #
    # Bias:
    #
    #   b = [b_cat, b_dog]
    #
    # Logits computed as:
    #
    #   L_cat = Σ (wi * yi) + b_cat
    #   L_dog = Σ (vi * yi) + b_dog
    #
    # Output:
    #
    #   logits = [L_cat, L_dog]
    #
    # NOTE:
    #   logits are raw scores, NOT probabilities.
    #
    # ============================================================
    # STEP 7: nn.CrossEntropyLoss()
    # ============================================================
    #
    # In PyTorch:
    #
    #   criterion = nn.CrossEntropyLoss()
    #
    # expects:
    #
    #   logits shape → [batch_size, num_classes]
    #   labels shape → [batch_size]
    #
    # Example:
    #
    #   logits = [2.0, 0.5]
    #   label  = 0       # CAT
    #
    # ============================================================
    # STEP 8: SOFTMAX (DONE INTERNALLY)
    # ============================================================
    #
    # exp(2.0) = 7.389
    # exp(0.5) = 1.648
    #
    # Sum = 9.037
    #
    # Probability:
    #
    #   P(CAT) = 7.389 / 9.037 = 0.817
    #   P(DOG) = 1.648 / 9.037 = 0.183
    #
    # ============================================================
    # STEP 9: LOSS COMPUTATION
    # ============================================================
    #
    # CrossEntropyLoss takes ONLY probability of true class:
    #
    #   true = CAT → use P(CAT)
    #
    #   loss = -log(0.817) = 0.202
    #
    # Low loss → correct + confident
    #
    # ------------------------------------------------------------
    # Suppose logits were bad:
    #
    #   logits = [0.2, 3.0]
    #
    # Softmax:
    #
    #   P(CAT) = 0.057
    #   loss = -log(0.057) = 2.86   # BIG
    #
    # Model is wrong → large penalty.
    #
    # ============================================================
    # STEP 10: BACKPROPAGATION
    # ============================================================
    #
    # loss.backward() computes:
    #
    #   gradients for:
    #     • f_ij values (convolution filter)
    #     • w_i, v_i (classifier)
    #     • biases
    #
    # optimizer.step() updates:
    #
    #   filters
    #   weights
    #   biases
    #
    # to REDUCE loss in next iteration.
    #
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    #
    # Image → convolution → features → flatten → logits → softmax → loss
    #
    # CrossEntropyLoss:
    #
    #   ✅ converts logits → probabilities
    #   ✅ selects only correct class probability
    #   ✅ penalizes wrong predictions
    #   ✅ drives learning via gradients
    #
    # ============================================================

    # ------------------------------------------------------------
    # ✅ QUALITY IMPROVEMENT OPTION:
    # ------------------------------------------------------------
    # Label smoothing can significantly improve generalization on CIFAR-10:
    #   • reduces overconfidence
    #   • improves calibration
    #   • often yields higher test accuracy
    #
    # If you want label smoothing ON, uncomment the next line and comment the first.
    # ------------------------------------------------------------
    #criterion = nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # ------------------------------------------------------------
    # OPTIONAL: STORE EXECUTION TIME FOR EACH EPOCH
    # ------------------------------------------------------------
    # epoch_times will store:
    #   • how long EACH epoch took (seconds)
    #   • useful for:
    #       - profiling
    #       - ETA estimation
    #       - performance comparison (CPU vs GPU, AMP on/off)
    # ------------------------------------------------------------
    epoch_times = []

    # ------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------
    for ep in range(num_epochs):

        # ------------------------------------------------------------
        # IMPORTANT FIX:
        # --------------
        # Always re-enable training mode at the START of each epoch.
        #
        # Why:
        # • If you ran evaluation somewhere (model.eval()), BatchNorm/Dropout
        #   may remain in eval mode unless you explicitly restore train().
        #
        # This ensures consistent learning behavior every epoch.
        # ------------------------------------------------------------
        model.train()

        # --------------------------------------------------------
        # START TIMER FOR THIS EPOCH
        # --------------------------------------------------------
        # time.perf_counter():
        #   • high-resolution timer
        #   • ideal for performance measurement
        # --------------------------------------------------------
        epoch_start = time.perf_counter()

        # Track statistics over the epoch
        total = 0
        correct = 0
        running_loss = 0.0

        # --------------------------------------------
        # LOOP THROUGH MINI-BATCHES
        # --------------------------------------------
        for images, labels in train_loader:

            # Move batch to device (GPU/CPU)
            images = images.to(device)
            labels = labels.to(device)

            # ----------------------------------------
            # CLEAR OLD GRADIENTS
            # ----------------------------------------
            # optimizer.zero_grad():
            #   • Clears gradients stored in parameter.grad from the previous iteration.
            #   • Gradients accumulate by default in PyTorch, so we must reset them.
            #
            # set_to_none=True (optional speed optimization):
            #   • Sets grads to None instead of zeroing tensors, saving memory ops.
            optimizer.zero_grad(set_to_none=True)

            # ============================================================
            # WHAT THIS LINE DOES:
            #     outputs = model(images)
            # ============================================================
            # (keeping your full explanation block exactly as-is)
            # ============================================================

            # ------------------------------------------------------------
            # AMP FORWARD PASS (autocast)
            # ------------------------------------------------------------
            # If AMP is enabled (CUDA):
            #   • Runs many ops in float16 for speed (conv, matmul)
            #   • Keeps numerically sensitive ops in float32 (BatchNorm, reductions)
            #
            # If AMP is disabled (CPU):
            #   • This context becomes a no-op and everything runs in float32
            #
            # IMPORTANT FIX:
            # --------------
            # DO NOT hardcode device_type="cuda" when running on CPU.
            # We choose device_type dynamically based on the actual device.
            # ------------------------------------------------------------
            autocast_device_type = "cuda" if device.type == "cuda" else "cpu"

            with torch.amp.autocast(device_type=autocast_device_type, enabled=use_amp):

                outputs = model(images)

                # ------------------------------------------------------------
                # 🔎 DEBUG SHAPES (run once on first batch only)
                # ------------------------------------------------------------
                if ep == 0 and total == 0:
                    print("images:", images.shape, images.dtype, images.device)
                    print("labels:", labels.shape, labels.dtype, labels.min().item(), labels.max().item())
                    print("outputs:", outputs.shape, outputs.dtype, outputs.device)

                # ------------------------------------------------------------
                # ✅ HARD ASSERTS (will stop immediately if wrong)
                # ------------------------------------------------------------
                labels = labels.long()  # CrossEntropyLoss requires int64 class indices

                assert labels.ndim == 1, f"labels must be [N], got {labels.shape}"
                assert outputs.ndim == 2, f"outputs must be [N,C], got {outputs.shape}"
                assert outputs.size(0) == labels.size(0), f"batch mismatch: {outputs.size(0)} vs {labels.size(0)}"

                loss = criterion(outputs, labels)

            # ----------------------------------------
            # BACKWARD PASS (AMP)
            # ----------------------------------------
            # • Scales the loss to prevent float16 underflow
            # • Computes gradients in scaled space
            # • Builds the backward graph once
            #
            # IMPORTANT QUALITY FIX:
            # ----------------------
            # On CPU (use_amp=False), GradScaler is effectively disabled,
            # but the call sequence remains safe and consistent.
            scaler.scale(loss).backward()

            # ----------------------------------------
            # GRADIENT CLIPPING (STABILITY)
            # ----------------------------------------
            # Prevents rare large gradients early in training
            # from causing unstable updates and high loss.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ----------------------------------------
            # OPTIMIZER STEP (AMP SAFE)
            # ----------------------------------------
            # • Internally unscales gradients (brings them back to real magnitude)
            # • Checks for NaN/Inf gradients:
            #     - If found → SKIP the weight update to avoid corrupting weights
            #     - If clean  → run optimizer.step() to update parameters safely
            scaler.step(optimizer)

            # ----------------------------------------
            # UPDATE SCALER
            # ----------------------------------------
            # • Adjusts scaling factor dynamically
            # • Increases scale if training is stable
            # • Decreases scale if overflow is detected
            scaler.update()

            # ----------------------------------------
            # LEARNING RATE SCHEDULER STEP (OneCycleLR)
            # ----------------------------------------
            # • Updates learning rate EVERY ITERATION
            # • Controls warmup + cooldown automatically
            #   • Must be called AFTER optimizer.step()
            scheduler.step()

            # ----------------------------------------
            # STATISTICS
            # ----------------------------------------
            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # --------------------------------------------------------
        # END TIMER FOR THIS EPOCH
        # --------------------------------------------------------
        # Measure elapsed time for THIS epoch only
        # --------------------------------------------------------
        epoch_time = time.perf_counter() - epoch_start

        # Store it for later statistics
        epoch_times.append(epoch_time)

        # --------------------------------------------------------
        # COMPUTE AVERAGE LOSS & ACCURACY FOR THIS EPOCH
        # --------------------------------------------------------
        #
        # running_loss:
        #   • Accumulated: sum of (batch_loss * batch_size)
        # total:
        #   • Total number of samples seen in the epoch
        #
        # epoch_loss:
        #   • True average loss PER SAMPLE over the whole epoch
        #
        # epoch_acc:
        #   • Fraction of correctly classified samples
        #
        epoch_loss = running_loss / total
        epoch_acc  = correct / total

        # --------------------------------------------
        # PRINT EPOCH SUMMARY
        # --------------------------------------------
        debug_print(
            f"[TRAIN] Epoch {ep+1}/{num_epochs}  "
            f"Loss: {epoch_loss:.4f}  "
            f"Accuracy: {epoch_acc:.4f}  "
            f"Time: {epoch_time:.2f} sec"
        )

        # ------------------------------------------------------------
        # ✅ OPTIONAL TEST EVALUATION (PREDICTION QUALITY TRACKING)
        # ------------------------------------------------------------
        # PURPOSE:
        # --------
        # Training accuracy can look good even when test accuracy is not improving.
        # This optional block measures true "prediction quality" on test/validation data.
        #
        # IMPORTANT:
        # ----------
        # • This does NOT change training behavior.
        # • It only runs if you pass test_loader=train/test loader when calling train_model().
        # ------------------------------------------------------------
        if test_loader is not None:
            model.eval()
            correct_t = 0
            total_t = 0
            with torch.no_grad():
                for images_t, labels_t in test_loader:
                    images_t = images_t.to(device)
                    labels_t = labels_t.to(device)
                    outputs_t = model(images_t)
                    preds_t = outputs_t.argmax(1)
                    correct_t += (preds_t == labels_t).sum().item()
                    total_t += labels_t.size(0)
            test_acc = correct_t / max(1, total_t)
            debug_print(f"[TEST]  Epoch {ep+1}/{num_epochs}  Accuracy: {test_acc:.4f}")
            model.train()

        if scheduler is not None:

            # ------------------------------------------------------------
            # IMPORTANT OneCycleLR CORRECTION
            # ------------------------------------------------------------
            # OneCycleLR is a *BATCH-BASED* scheduler.
            #
            # That means:
            #   ❌ We must NOT call scheduler.step(epoch_loss)
            #   ❌ We must NOT step the scheduler per epoch
            #
            # The scheduler is ALREADY stepped once per mini-batch:
            #     scheduler.step()
            #
            # Calling it again here would:
            #   • Consume the LR schedule too fast
            #   • Break the cosine curve
            #   • Cause LR to stagnate or collapse
            #   • Make early loss look worse
            #
            # Therefore:
            #   ➜ We keep this block ONLY for LR logging
            # ------------------------------------------------------------

            # ❌ DISABLED — DO NOT USE WITH OneCycleLR
            # scheduler.step(epoch_loss)

            # Manual LR logging (safe — read-only)
            current_lr = optimizer.param_groups[0]['lr']
            debug_print(f"[LR Scheduler] End-of-epoch LR snapshot = {current_lr:.6f}")

    # ------------------------------------------------------------
    # OPTIONAL: PRINT TOTAL AND AVERAGE EXECUTION TIME
    # ------------------------------------------------------------
    # IMPORTANT FIX:
    # --------------
    # This MUST be OUTSIDE the epoch loop, so it prints once at the end.
    # ------------------------------------------------------------
    if epoch_times:
        total_time = sum(epoch_times)
        avg_time = total_time / len(epoch_times)
        print(
            f"[TRAIN] Finished {num_epochs} epochs "
            f"in {total_time:.2f} sec "
            f"(avg {avg_time:.2f} sec/epoch)"
        )

    # ------------------------------------------------------------
    # RETURN TRAINED MODEL
    # ------------------------------------------------------------
    return model










# ============================================================
# DETECTION / SINGLE-IMAGE INFERENCE FUNCTION
# ============================================================
def detect_single_image(model, test_dataset, device, index=None):
    """
    Loads ONE RANDOM image from the test dataset (unless index is provided),
    runs the model, and prints:

        • True label ID & name
        • Predicted label ID & name

    Args:
        model ........ the trained PyTorch CNN
        test_dataset . a torchvision dataset (CIFAR-10, ImageFolder, etc.)
        device ....... "cuda" or "cpu"
        index ........ optional fixed index; if None → choose random image

    Returns:
        img_tensor, true_label_id, predicted_label_id
    """

    # --------------------------------------------------------
    # MOVE MODEL TO DEVICE AND SWITCH TO EVAL MODE
    # --------------------------------------------------------
    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # AUTO-DETECT CLASS NAMES (works for CIFAR-10 + ImageFolder)
    # --------------------------------------------------------
    class_names = getattr(test_dataset, "classes", None)
    if class_names is None:
        # Fallback: no classes attribute found
        class_names = [str(i) for i in range(10)]  # generic labels 0..9

    # --------------------------------------------------------
    # NORMALIZE & VALIDATE INDEX
    #   • If index is None → choose random
    #   • If index is string → convert to int
    #   • Clamp / reject out-of-range indices
    # --------------------------------------------------------
    if index is None:
        # Select random sample if no index is given
        index = random.randint(0, len(test_dataset) - 1)
    else:
        # If index is passed as a string (e.g. from input()), convert it
        if isinstance(index, str):
            try:
                index = int(index)
            except ValueError:
                print(f"[detect_single_image] Invalid index value '{index}', using 0 instead.")
                index = 0

        # Range check
        if index < 0 or index >= len(test_dataset):
            print(f"[detect_single_image] Index {index} is out of range 0–{len(test_dataset) - 1}, using 0 instead.")
            index = 0

    # --------------------------------------------------------
    # LOAD IMAGE + TRUE LABEL
    # --------------------------------------------------------
    img, true_label = test_dataset[index]

    # Some datasets may return label as tensor, normalize to Python int
    try:
        true_label_id = int(true_label)
    except Exception:
        true_label_id = true_label  # keep as-is if already int-like

    # ✅ NEW (SAFE): Print actual image tensor size so you see H,W at runtime
    # img is [C, H, W]
    c, h, w = img.shape
    # Note: This confirms the pipeline is NOT constrained to 32x32 anymore.

    # Add batch dimension → shape becomes [1, C, H, W]
    img_input = img.unsqueeze(0).to(device)

    # --------------------------------------------------------
    # FORWARD PASS (NO GRADIENT TRACKING)
    # --------------------------------------------------------
    with torch.no_grad():
        logits = model(img_input)
        pred_label = logits.argmax(1).item()

        # ✅ NEW (OPTIONAL): confidence score for the predicted class
        # Softmax converts logits → probabilities
        probs = torch.softmax(logits, dim=1)
        pred_conf = probs[0, pred_label].item()  # probability for predicted class

    # --------------------------------------------------------
    # CONVERT LABEL IDS → HUMAN-READABLE NAMES
    # --------------------------------------------------------
    # Defensive check in case labels are outside class_names length
    if 0 <= true_label_id < len(class_names):
        true_name = class_names[true_label_id]
    else:
        true_name = f"class_{true_label_id}"

    if 0 <= pred_label < len(class_names):
        pred_name = class_names[pred_label]
    else:
        pred_name = f"class_{pred_label}"

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------
    print("--------------------------------------------------")
    print(f"DETECTION RESULT FOR TEST IMAGE INDEX: {index}")
    print(f"Input image shape : [C={c}, H={h}, W={w}]")  # ✅ NEW: shows actual size used
    print(f"True label index  : {true_label_id} → {true_name}")
    print(f"Pred label index  : {pred_label} → {pred_name}")
    print(f"Confidence        : {pred_conf*100:.2f}%")   # ✅ NEW: optional but useful
    print("--------------------------------------------------")

    return img, true_label_id, pred_label




# ============================================================
# MAIN PROGRAM
# ============================================================
# assume these are defined globally somewhere above:
# DATA_PATH = "../../../data/mydata"
# MODEL_PATH = "../../../"
# MODEL_FILENAME = "cifar10_model_custom_file"
# NUM_EPOCHS = 2
# from your_module import StaticInitLearnableCNN, train_model, detect_single_image

# ------------------------------------------------------------------
# Simple debug print helper (example)
# ------------------------------------------------------------------
def main():

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_print("Using device:", device)

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

    debug_print("Training images from:", train_path)
    debug_print("Testing  images from:", test_path)

    # --------------------------------------------------------
    # DATA TRANSFORMS FOR YOUR DATA (ALL IMAGES SAME SIZE)
    # --------------------------------------------------------
    # ASSUMPTION (YOUR REQUEST):
    # • All images in train/ and test/ have the SAME size (same H and W).
    #
    # WHAT THIS MEANS:
    # • DataLoader can stack images into batches safely.
    # • No Resize() is required.
    #
    # IMPORTANT:
    # • If even ONE image has a different size, DataLoader will fail with:
    #     RuntimeError: stack expects each tensor to be equal size
    # --------------------------------------------------------
    # ============================================================
    # TRAIN TRANSFORM (USED DURING TRAINING)
    # ============================================================
    train_transform = transforms.Compose([

        # --------------------------------------------------------
        # RANDOM HORIZONTAL FLIP (DATA AUGMENTATION)
        # --------------------------------------------------------
        # With probability p=0.5:
        #   • The image is flipped left ↔ right.
        #
        # WHY WE USE THIS:
        # • Many objects look the same when mirrored (cars, animals, people).
        # • Doubles the effective dataset size.
        # • Helps prevent overfitting.
        #
        # WHY IT IS SAFE:
        # • Does NOT change image size (H, W remain the same).
        # • Does NOT distort pixel values.
        #
        # Input  : PIL Image (H x W x C)
        # Output : PIL Image (H x W x C)
        # --------------------------------------------------------
        transforms.RandomHorizontalFlip(p=0.5),

        # --------------------------------------------------------
        # CONVERT IMAGE TO PYTORCH TENSOR
        # --------------------------------------------------------
        # Converts:
        #   • PIL Image or NumPy array
        # into:
        #   • PyTorch Tensor
        #
        # Pixel value conversion:
        #   Original pixels:  [0, 255]   (uint8)
        #   Tensor pixels:    [0.0, 1.0] (float32)
        #
        # Shape conversion:
        #   PIL format : [H, W, C]
        #   Tensor     : [C, H, W]
        #
        # WHY THIS IS REQUIRED:
        # • PyTorch models ONLY accept tensors.
        # • Floating point values are required for gradient computation.
        #
        # IMPORTANT:
        # • Does NOT resize or crop the image.
        # • Works because ALL images already share the same H and W.
        # --------------------------------------------------------
        transforms.ToTensor(),

        # --------------------------------------------------------
        # NORMALIZATION (IMAGENET STATISTICS)
        # --------------------------------------------------------
        # This line performs:
        #
        #   normalized_pixel = (pixel - mean) / std
        #
        # Per-channel normalization:
        #   Channel 0 (Red)   → mean=0.485, std=0.229
        #   Channel 1 (Green) → mean=0.456, std=0.224
        #   Channel 2 (Blue)  → mean=0.406, std=0.225
        #
        # WHY WE USE IMAGENET STATS:
        # • Most CNNs are trained assuming these statistics.
        # • Helps stabilize gradients.
        # • Reduces initial loss.
        # • Faster convergence.
        #
        # WHAT RANGE DO PIXELS END UP IN?
        # • Roughly: [-2.5, +2.5]
        #
        # WHY NORMALIZATION HELPS:
        # • Prevents one color channel from dominating.
        # • Makes learning scale-consistent.
        #
        # SAFE FOR ANY IMAGE SIZE:
        # • Applied PER PIXEL.
        # • Independent of H and W.
        # --------------------------------------------------------
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet channel means (RGB)
            std=[0.229, 0.224, 0.225]     # ImageNet channel stds  (RGB)
        ),
    ])

    # ============================================================
    # TEST TRANSFORM (USED DURING VALIDATION / INFERENCE)
    # ============================================================
    test_transform = transforms.Compose([

        # --------------------------------------------------------
        # CONVERT IMAGE TO PYTORCH TENSOR
        # --------------------------------------------------------
        # Same behavior as training:
        #   • Converts to float tensor
        #   • Scales pixels to [0.0, 1.0]
        #   • Converts shape to [C, H, W]
        #
        # IMPORTANT DIFFERENCE FROM TRAIN:
        # • NO data augmentation here.
        # • We want deterministic, repeatable results.
        # --------------------------------------------------------
        transforms.ToTensor(),

        # --------------------------------------------------------
        # NORMALIZATION (MATCH TRAINING EXACTLY)
        # --------------------------------------------------------
        # IMPORTANT NOTE:
        # ⚠️ TRAIN and TEST normalization SHOULD MATCH.
        #
        # Your original code used mean/std = 0.5 in test.
        # Since we want correct evaluation, we match training here.
        # --------------------------------------------------------
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # ------------------------------------------------------------------
    # LOAD DATASETS USING ImageFolder
    # ------------------------------------------------------------------
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=train_transform      # ✅ use training transform with augmentation
    )

    test_dataset = datasets.ImageFolder(
        root=test_path,
        transform=test_transform       # ✅ use test transform (no augmentation)
    )

    debug_print(f"[main] Loaded train_dataset with {len(train_dataset)} images")
    debug_print(f"[main] Loaded test_dataset  with {len(test_dataset)} images")

    # --------------------------------------------------------
    # FULL RANDOMIZATION OF TRAIN AND TEST DATASETS
    # --------------------------------------------------------
    # By default, ImageFolder builds its internal 'samples' list in
    # alphabetical class folder order, e.g.:
    #   airplane/, automobile/, bird/, ...
    #
    # That means that BEFORE shuffling, indices 0..N may all come from
    # the first class (e.g., airplane). To achieve COMPLETE randomization:
    #
    #   ✅ We random.shuffle(train_dataset.samples)
    #   ✅ We random.shuffle(test_dataset.samples)
    #
    # This permutes the underlying (path, label) list itself so the
    # dataset no longer starts with a long block of one class.
    #
    # Combined with DataLoader(shuffle=True), this gives full randomness:
    #   • dataset level  (samples list)
    #   • batch order    (DataLoader index sampling)
    # --------------------------------------------------------
    #random.shuffle(train_dataset.samples)
    #random.shuffle(test_dataset.samples)
    #debug_print("[main] Shuffled train_dataset.samples for full randomization")
    #debug_print("[main] Shuffled test_dataset.samples  for full randomization")

    # Show class mapping as seen by ImageFolder
    debug_print("[main] Class index → name mapping (from train_dataset.classes):")
    for idx, name in enumerate(train_dataset.classes):
        debug_print(f"   {idx}: {name}")

    # Optionally show first few training samples to verify labels AFTER shuffle
    max_show = min(5, len(train_dataset))
    for i in range(max_show):
        _, lbl = train_dataset[i]                  # (image_tensor, label_index)
        cls_name = train_dataset.classes[lbl]
        debug_print(f"[main] Sample train index {i} (after shuffle) → label {lbl} ('{cls_name}')")

    # And also show a few test samples AFTER shuffle
    max_show_test = min(5, len(test_dataset))
    for i in range(max_show_test):
        _, lbl = test_dataset[i]
        cls_name = test_dataset.classes[lbl]
        debug_print(f"[main] Sample test  index {i} (after shuffle) → label {lbl} ('{cls_name}')")

    # ============================================================
    # DATALOADERS
    # ============================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,    # ✅ still keep this True for per-epoch randomization
        num_workers=NUM_WORKERS
    )

    # For *complete* randomization in testing as requested,
    # we also use shuffle=True here. Note:
    #   • For strict benchmark evaluation, usually shuffle=False,
    #     but since your focus is interactive detection / exploration,
    #     we enable full randomization as you requested.
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=True,    # ✅ full randomization for test as well
        num_workers=2
    )

    # --------------------------------------------------------
    # DETERMINE NUMBER OF CLASSES FROM DATASET
    # --------------------------------------------------------
    num_classes = len(train_dataset.classes)
    debug_print("Number of classes detected in train:", num_classes)
    debug_print("Class names:", train_dataset.classes)

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
        debug_print(f"Loading trained weights from: {model_filename}")
        state_dict = torch.load(model_filename, map_location=device)
        model.load_state_dict(state_dict)
    else:
        debug_print("No saved model found. Training a new model...")
        model = train_model(model, train_loader, device, num_epochs=NUM_EPOCHS, lr=1e-3)
        debug_print(f"Saving trained model to: {model_filename}")
        torch.save(model.state_dict(), model_filename)

    # ------------------------------------------------------------
    # INTERACTIVE LOOP FOR USER-DRIVEN DETECTION
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # HELPER: READ A POSITIVE INTEGER USING msvcrt (DIGITS UNTIL ENTER)
    # ------------------------------------------------------------
    # This helper lets us reuse the SAME "type digits, press ENTER" logic
    # for BOTH:
    #   • image index input
    #   • N-random-images evaluation input
    # ------------------------------------------------------------
    def _read_int_from_keyboard_msvcrt(prompt: str):
        """
        Reads digits until ENTER and returns an int.
        Returns:
          • int value on success
          • None if invalid
          • "EXIT" if user pressed 'e'
        """

        print(prompt, end="", flush=True)

        # READ FIRST CHARACTER WITHOUT PRESSING ENTER
        first = msvcrt.getch().decode(errors="ignore").lower()

        # IF USER PRESSES 'e' → EXIT IMMEDIATELY
        if first == "e":
            print("e")
            return "EXIT"

        # If first key is NOT a digit → invalid
        if not first.isdigit():
            print(first)
            return None

        # Echo the first digit
        print(first, end="", flush=True)

        # READ REMAINING DIGITS UNTIL ENTER
        s = first
        while True:
            ch = msvcrt.getch()
            if ch in [b"\r", b"\n"]:  # ENTER pressed
                print()              # move to next line
                break

            try:
                c = ch.decode(errors="ignore")
            except Exception:
                continue

            # Allow EXIT inside typing
            if c.lower() == "e":
                print("e")
                return "EXIT"

            # Accept only digits
            if c.isdigit():
                s += c
                print(c, end="", flush=True)
            else:
                # Ignore non-digit keys
                continue

        if not s.isdigit():
            return None

        return int(s)

    # ------------------------------------------------------------
    # RUN N RANDOM TEST IMAGES AND REPORT HIT/MISS RATIO
    # ------------------------------------------------------------
    # This routine:
    #   • randomly selects N test images
    #   • runs inference
    #   • prints predicted vs true
    #   • computes hit ratio and miss ratio
    # ------------------------------------------------------------
    def run_n_random_images(model, test_dataset, device, n: int):
        """
        Runs the model on N random images from test_dataset and reports hit/miss ratio.
        """

        # --------------------------------------------------------
        # SAFETY CHECKS
        # --------------------------------------------------------
        if n <= 0:
            print("❌ N must be >= 1.")
            return

        if len(test_dataset) <= 0:
            print("❌ test_dataset is empty.")
            return

        # --------------------------------------------------------
        # PUT MODEL IN EVAL MODE (DISABLE DROPOUT, FIX BATCHNORM)
        # --------------------------------------------------------
        model.eval()

        # --------------------------------------------------------
        # MOVE MODEL TO DEVICE (IF NOT ALREADY)
        # --------------------------------------------------------
        model.to(device)

        # --------------------------------------------------------
        # PICK N RANDOM INDICES
        # --------------------------------------------------------
        # If N <= dataset size → sample without replacement (unique indices)
        # Else → allow repeats (with replacement) so the user can request big N
        # --------------------------------------------------------
        if n <= len(test_dataset):
            indices = random.sample(range(len(test_dataset)), k=n)
        else:
            indices = [random.randrange(len(test_dataset)) for _ in range(n)]

        # --------------------------------------------------------
        # INFERENCE LOOP
        # --------------------------------------------------------
        correct = 0
        total = 0

        print("\n--------------------------------------------------")
        print(f"Running N-Random Evaluation (N={n})")
        print("--------------------------------------------------\n")

        with torch.no_grad():

            for k, idx in enumerate(indices, start=1):

                # ------------------------------------------------
                # LOAD ONE TEST SAMPLE
                #   test_dataset[idx] → (image_tensor [C,H,W], label_index)
                # ------------------------------------------------
                image_tensor, true_label = test_dataset[idx]

                # ------------------------------------------------
                # ADD BATCH DIMENSION
                #   [C,H,W] → [1,C,H,W]
                # ------------------------------------------------
                image_tensor = image_tensor.unsqueeze(0).to(device)

                # ------------------------------------------------
                # FORWARD PASS (PREDICT)
                # ------------------------------------------------
                outputs = model(image_tensor)

                # ------------------------------------------------
                # ARGMAX → PREDICTED CLASS INDEX
                # ------------------------------------------------
                pred_label = int(outputs.argmax(1).item())

                # ------------------------------------------------
                # UPDATE METRICS
                # ------------------------------------------------
                is_hit = (pred_label == int(true_label))
                correct += 1 if is_hit else 0
                total += 1

                # ------------------------------------------------
                # PRINT PER-IMAGE RESULT
                # ------------------------------------------------
                true_name = test_dataset.classes[int(true_label)]
                pred_name = test_dataset.classes[int(pred_label)]
                status = "✅ HIT" if is_hit else "❌ MISS"

                print(f"[{k:>3}/{n}] idx={idx:>6} | true={true_name:<20} pred={pred_name:<20} → {status}")

        # --------------------------------------------------------
        # FINAL HIT / MISS RATIOS
        # --------------------------------------------------------
        hit_ratio = (correct / total) if total > 0 else 0.0
        miss_ratio = 1.0 - hit_ratio

        print("\n--------------------------------------------------")
        print("N-Random Evaluation Summary")
        print("--------------------------------------------------")
        print(f"Total images : {total}")
        print(f"Hits         : {correct}")
        print(f"Misses       : {total - correct}")
        print(f"Hit ratio    : {hit_ratio:.4f}  ({hit_ratio*100:.2f}%)")
        print(f"Miss ratio   : {miss_ratio:.4f} ({miss_ratio*100:.2f}%)")
        print("--------------------------------------------------\n")

    print("\n--------------------------------------------------")
    print("Interactive Image Detection Mode")
    print("You are now ALWAYS in detection mode.")
    print("Just type an image index and press ENTER.")
    print("Type 'n' to run N random test images and print hit/miss ratio.")
    print("Press 'e' at any time to exit.")
    print("--------------------------------------------------\n")

    while True:

        print(f"Enter image index (0 – {len(test_dataset)-1}), or 'n' for N-random, or 'e' to exit: ",
              end="", flush=True)

        # READ ONE CHARACTER WITHOUT PRESSING ENTER
        key = msvcrt.getch().decode(errors="ignore").lower()

        # IF USER PRESSES 'e' → EXIT IMMEDIATELY
        if key == 'e':
            print("e")
            print("Exiting program. Goodbye!")
            break

        # ------------------------------------------------
        # OPTION: USER PRESSES 'n' → RUN N RANDOM IMAGES
        # ------------------------------------------------
        if key == "n":
            print("n")  # echo the key

            # ------------------------------------------------
            # ASK USER FOR N USING THE SAME DIGIT-UNTIL-ENTER LOGIC
            # ------------------------------------------------
            n_val = _read_int_from_keyboard_msvcrt(
                "Enter N (number of random test images) and press ENTER (or 'e' to exit): "
            )

            if n_val == "EXIT":
                print("Exiting program. Goodbye!")
                break

            if n_val is None:
                print("❌ Invalid input. N must be a number.")
                continue

            # ------------------------------------------------
            # RUN N-RANDOM EVALUATION + HIT/MISS RATIO
            # ------------------------------------------------
            run_n_random_images(model, test_dataset, device, n=int(n_val))
            continue

        # If first key is NOT a digit → invalid
        if not key.isdigit():
            print(key)
            print("❌ Invalid input. Enter a number, 'n', or 'e' to exit.")
            continue

        # Echo the first digit
        print(key, end="", flush=True)

        # READ REMAINING DIGITS UNTIL ENTER
        idx_str = key
        while True:
            ch = msvcrt.getch()
            if ch in [b'\r', b'\n']:   # ENTER pressed
                print()               # move to next line
                break
            try:
                c = ch.decode(errors="ignore")
            except Exception:
                continue

            # Allow EXIT inside typing
            if c.lower() == 'e':
                print("e")
                print("Exiting program. Goodbye!")
                return

            # Accept only digits
            if c.isdigit():
                idx_str += c
                print(c, end="", flush=True)
            else:
                # Ignore non-digit keys
                continue

        # ------------------------------------------------
        # VALIDATE INDEX
        # ------------------------------------------------
        if not idx_str.isdigit():
            print("❌ Invalid index. Must be a number.")
            continue

        idx = int(idx_str)

        if idx < 0 or idx >= len(test_dataset):
            print("❌ Index out of range. Try again.")
            continue

        # ------------------------------------------------
        # RUN DETECTION
        # ------------------------------------------------
        print(f"\nRunning detection on test image index {idx} ...")
        detect_single_image(model, test_dataset, device, index=idx)



# ------------------------------------------------------------
# RUN PROGRAM
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
