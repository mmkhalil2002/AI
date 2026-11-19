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
MODEL_FILENAME = "cifar10-cnn-128-256-512-762s-L6172-A9957"
DATA_PATH = "../../../data/mydata"
BATCH_SIZE = 128
NUM_EPOCHS = 100
#LEARNING_RATE = 0.001


NUM_WORKERS = 0
STATIC_FILTERS = False
DEBUG_FLAG = True

# ============================================================
# GLOBAL CHANNEL CONFIG (SINGLE SOURCE OF TRUTH)
# ============================================================
# These constants define the architecture in ONE place.
# Then we reuse them in:
#   • __init__ (Conv2d / BatchNorm2d / Linear)
#   • _init_conv*_static() (assertions + logs)
#
# ✅ Benefits:
#   • no magic numbers scattered in the code
#   • if you change channels later, you change them here once
#   • your static init checks automatically stay correct
# ============================================================

CONV1_IN_CHANNEL  = 3
CONV1_OUT_CHANNEL = 128

CONV2_IN_CHANNEL  = CONV1_OUT_CHANNEL
CONV2_OUT_CHANNEL = 256

CONV3_IN_CHANNEL  = CONV2_OUT_CHANNEL
CONV3_OUT_CHANNEL = 512
# ============================================================
# EXPLANATION: HOW TRAINING WORKS IN THIS NETWORK
# ============================================================
#
# This network is a CLASSICAL CONVOLUTIONAL NEURAL NETWORK (CNN)
# where:
#
#   • Layer 1 (conv1) extracts low-level features
#       (edges, color gradients, small textures)
#
#   • Layer 2 (conv2) extracts mid-level features
#       (corners, repeated patterns, texture groupings)
#
#   • Layer 3 (conv3) extracts higher-level features
#       (object parts, more complex shape compositions)
#
#   • Only ONE early MAX POOLING step is used (after conv1)
#       to reduce spatial size while preserving detail
#
#   • After feature extraction, we use GLOBAL AVERAGE POOLING (GAP)
#       so the model works with ANY image size
#
#   • Final layer (fc) is a standard fully connected classifier
#       that outputs class logits
#
# ------------------------------------------------------------
# INPUT ASSUMPTION:
# ------------------------------------------------------------
#
# The network expects 3-channel RGB images:
#
#   • Shape: [B, 3, H, W]
#   • H, W can be ANY size
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
# So this is NOT automatically a "frozen-filter" network.
#
# It is a CLASSICAL CNN that can start from known kernels
# and still learn end-to-end.
#
# ============================================================
# HOW LEARNING HAPPENS
# ============================================================
#
# During training, the following steps occur:
#
#   logits = model(images)
#   loss    = criterion(logits, labels)
#   loss.backward()
#   optimizer.step()
#
# PyTorch automatically computes gradients for ALL trainable
# parameters in the network.
#
# These include:
#
#   • conv1.weight, conv1.bias      (128-channel low-level filters) ✅ UPDATED
#   • conv2.weight, conv2.bias      (256-channel mid-level filters) ✅ UPDATED
#   • conv3.weight, conv3.bias      (512-channel higher-level filters) ✅ UPDATED
#
#   • bn1.weight, bn1.bias          (BatchNorm scale/shift for conv1)
#   • bn2.weight, bn2.bias          (BatchNorm scale/shift for conv2)
#   • bn3.weight, bn3.bias          (BatchNorm scale/shift for conv3)
#
#   • fc.weight,  fc.bias           (final classifier)
#
# Pooling layers have NO learnable parameters:
#   • They only perform fixed max operations
#
# GAP (AdaptiveAvgPool2d) also has NO learnable parameters:
#   • It simply averages spatial values:
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
#   • Pooling reduces spatial size while keeping strong features
#   • GAP removes dependence on image resolution
#
# This is exactly how CNNs are trained in practice,
# except many models start with RANDOM initialization.
#
# Here, you optionally start with INTELLIGENT initialization,
# while still training normally.
#
# ============================================================
# NETWORK SHAPE (IMAGE-SIZE INDEPENDENT WITH GAP)
# ============================================================
#
# Input image:                [3    x H   x W]
#
# After conv1:                [128  x H   x W]       ✅ UPDATED
# After pool1:                [128  x H/2 x W/2]     ✅ UPDATED
#
# After conv2:                [256  x H/2 x W/2]     ✅ UPDATED
#   (NO pooling here to preserve spatial detail)
#
# After conv3:                [512  x H/2 x W/2]     ✅ UPDATED
#
# After GAP:                  [512  x 1   x 1]       ✅ UPDATED
# After flatten:              [512]                  ✅ UPDATED
# Output layer (fc):          [num_classes]
#
# ============================================================
# SUMMARY
# ============================================================
#
# ✅ Optional static initialization (conv layers may start from known kernels)
# ✅ Dynamic learning during training (unless layers are frozen)
# ✅ Single early pooling preserves important spatial information
# ✅ GAP makes the model work with ANY image size
# ✅ Classical CNN trained end-to-end with backpropagation




    def __init__(self, num_classes: int = 10):
        super().__init__()

        # --------------------------------------------------------
        # cuDNN AUTOTUNER
        # --------------------------------------------------------
        # Enables cuDNN to find the fastest convolution algorithms
        # for your hardware and input sizes.
        #
        # Works best when:
        #   • You run on GPU (CUDA) with cuDNN available
        #   • Input image sizes are constant (e.g. always 32x32)
        #   • You train for many iterations
        #
        # WARNING:
        #   • Slightly slower first iteration (benchmarking)
        #   • Faster training afterwards
        #
        # NOTE:
        #   This model is IMAGE-SIZE INDEPENDENT because we
        #   no longer hard-code spatial dimensions in the classifier.
        # --------------------------------------------------------
        torch.backends.cudnn.benchmark = True

        # ------------------------------------------------------
        # ✅ LAYER 1: 3 → 128 channels ✅ UPDATED
        # ------------------------------------------------------
        # 3 input channels (RGB) → 128 feature maps using 3x3 filters
        # Padding = 1 to keep spatial size
        #
        # Input shape assumption:
        #   [B, 3, H, W]   (ANY H, W)
        #
        # Either native CIFAR-10 images, OR any custom dataset
        # without resizing constraints.
        # ------------------------------------------------------
        self.conv1 = nn.Conv2d(
            in_channels=CONV1_IN_CHANNEL,
            out_channels=CONV1_OUT_CHANNEL,
            kernel_size=3,
            padding=1,
            bias=True
        )

        # ------------------------------------------------------
        # ✅ BatchNorm for conv1 (normalizes 128 output channels) ✅ UPDATED
        # ------------------------------------------------------
        # WHY WE USE BATCHNORM2d(128):
        # ----------------------------
        # • It normalizes each of the 128 feature maps across the batch
        # • Keeps mean ≈ 0 and variance ≈ 1
        # • Reduces internal covariate shift
        # • Allows faster and more stable training
        # • Acts as light regularization (reduces overfitting)
        #
        # Training flow with BatchNorm:
        #   conv1 → bn1 → ReLU → pool
        #
        # Result:
        #   ➤ Faster convergence
        #   ➤ Smoother gradients
        #   ➤ Sometimes significantly higher accuracy
        # ------------------------------------------------------
        self.bn1 = nn.BatchNorm2d(CONV1_OUT_CHANNEL)

        # ------------------------------------------------------
        # POOLING LAYER: MaxPool2d(2, 2)
        # ------------------------------------------------------
        # Max pooling:
        #     kernel_size = 2
        #     stride      = 2
        #
        # Effect on spatial dimensions (ONLY ONE POOL IN THIS MODEL):
        #   H×W → H/2×W/2     (after the single pool after conv1)
        #
        # This works for ANY input resolution.
        # ------------------------------------------------------
        self.pool = nn.MaxPool2d(2, 2)

        # ------------------------------------------------------
        # ✅ LAYER 2: 128 → 256 channels ✅ UPDATED
        # ------------------------------------------------------
        # 128 input feature maps → 256 output feature maps
        # using 3×3 filters, padding=1 keeps spatial size.
        #
        # After the single pooling:
        #   input to conv2 : [B, 128, H/2, W/2]
        #   output of conv2: [B, 256, H/2, W/2]
        # ------------------------------------------------------
        self.conv2 = nn.Conv2d(
            in_channels=CONV2_IN_CHANNEL,
            out_channels=CONV2_OUT_CHANNEL,
            kernel_size=3,
            padding=1,
            bias=True
        )

        # ------------------------------------------------------
        # ✅ BatchNorm for conv2 (normalizes 256 channels) ✅ UPDATED
        # ------------------------------------------------------
        # Why BatchNorm2d(256)?
        # ---------------------
        # • Conv2 outputs 256 feature maps
        # • BatchNorm stabilizes all 256 channels
        #
        # Overall:
        #   conv2 → bn2 → ReLU
        #
        # BatchNorm especially helps deeper layers where
        # activations become more chaotic.
        # ------------------------------------------------------
        self.bn2 = nn.BatchNorm2d(CONV2_OUT_CHANNEL)

        # ------------------------------------------------------
        # ✅ LAYER 3: 256 → 512 channels ✅ UPDATED
        # ------------------------------------------------------
        # WHY THIS IMPROVES PREDICTION QUALITY:
        # ------------------------------------
        # This deeper conv layer learns higher-level combinations of features:
        #   • textures
        #   • shapes
        #   • object parts
        #
        # IMPORTANT:
        # • padding=1 keeps spatial size
        # • Works for ANY input H, W
        #
        # NOTE (UPDATED FOR ONE-POOL DESIGN):
        # ----------------------------------
        # There is NO second pooling in this model.
        # So conv3 operates on the SAME spatial size as conv2:
        #
        #   input to conv3 : [B, 256, H/2, W/2]
        #   output of conv3: [B, 512, H/2, W/2]
        # ------------------------------------------------------
        self.conv3 = nn.Conv2d(
            in_channels=CONV3_IN_CHANNEL,
            out_channels=CONV3_OUT_CHANNEL,
            kernel_size=3,
            padding=1,
            bias=True
        )

        # ------------------------------------------------------
        # ✅ BatchNorm for conv3 (normalizes 512 channels) ✅ UPDATED
        # ------------------------------------------------------
        # Why BatchNorm2d(512)?
        # ---------------------
        # • Conv3 outputs 512 feature maps
        # • BN helps stabilize deeper activations
        # • Makes training smoother and improves generalization
        # ------------------------------------------------------
        self.bn3 = nn.BatchNorm2d(CONV3_OUT_CHANNEL)

        # ------------------------------------------------------
        # 🔑 GLOBAL AVERAGE POOLING (IMAGE-SIZE INDEPENDENT)
        # ------------------------------------------------------
        # Replaces hard-coded flattening of spatial dimensions.
        #
        # Converts:
        #   [B, C, H, W] → [B, C, 1, 1]
        #
        # In THIS model after conv3, C = 512:
        #   [B, 512, H', W'] → [B, 512, 1, 1]
        #
        # This makes the model work with ANY image size.
        # ------------------------------------------------------
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ------------------------------------------------------
        # ✅ DROPOUT (GENERALIZATION BOOST)
        # ------------------------------------------------------
        # WHY DROPOUT HELPS:
        # ------------------
        # • Prevents the classifier from over-relying on a few features
        # • Reduces overfitting and improves test-time accuracy
        #
        # Dropout is applied after feature extraction (after GAP + flatten)
        # and before the final classifier.
        # ------------------------------------------------------
        self.dropout = nn.Dropout(p=0.3)

        # ------------------------------------------------------
        # FULLY CONNECTED CLASSIFIER (UPDATED)
        # ------------------------------------------------------
        # OLD (size-dependent):
        #   nn.Linear(C * H * W, num_classes)
        #
        # NEW (size-independent):
        #   nn.Linear(C, num_classes)
        #
        # Depends ONLY on channel count, not spatial size.
        # ------------------------------------------------------
        # ✅ IMPORTANT UPDATE:
        # -------------------
        # Because conv3 now outputs 512 channels, GAP now outputs:
        #   [B, 512, 1, 1] → flatten → [B, 512]
        #
        # Therefore the classifier must be:
        #   nn.Linear(512, num_classes)
        # ------------------------------------------------------
        self.fc = nn.Linear(CONV3_OUT_CHANNEL, num_classes)

        # ------------------------------------------------------
        # STATIC FILTER INITIALIZATION (if enabled)
        # ------------------------------------------------------
        # These functions overwrite the conv1 and/or conv2 weights
        # with your custom 3×3 static kernels:
        #   • Edges (0°–315°)
        #   • Corners (0°–315°)
        #   • Curves (0°–315°)
        #   • Lines (0°–315°)
        #   • Sobel filters
        #   • Sharpen, blur, Gaussian, etc.
        #
        # These filters act like hand-crafted feature detectors,
        # while deeper layers learn freely.
        #
        # IMPORTANT NOTE (accuracy-related):
        # ----------------------------------
        # If conv1 is overwritten with static filters, it may LIMIT
        # the benefit of increasing conv1 channels unless your
        # static filter bank actually fills/uses all 128 output maps.
        # ------------------------------------------------------
        if STATIC_FILTERS:
            self._init_conv1_static()
            self._init_conv2_static()
            self._init_conv3_static()



    # ----------------------------------------------------------
    # STATIC INITIALIZATION FOR LAYER 1
    # ----------------------------------------------------------
    def _init_conv1_static(self):
        with torch.no_grad():                                              # disable gradients during manual init
            w = self.conv1.weight                                          # conv1 weights → [out_channels, 3, 3, 3]
            out_channels, in_channels, kh, kw = w.shape                    # get conv1 shape

            # ✅ UPDATED FOR YOUR NEW MODEL:
            # -----------------------------
            # Your conv1 is now:
            #   in_channels  = 3    (RGB)
            #   out_channels = 128  (128 feature maps / 128 filters) ✅ UPDATED
            #   kernel       = 3x3
            #
            # So conv1 produces:
            #   [B, 3, H, W] → [B, 128, H, W] ✅ UPDATED
            #
            # This gives the network MUCH higher early capacity than 16 channels.
            #
            # ✅ FIX:
            # ------
            # Instead of hard-coding 3 and 128, we validate against the GLOBAL constants:
            #   CONV1_IN_CHANNEL, CONV1_OUT_CHANNEL
            #
            # This ensures you can change architecture in ONE place safely.
            assert in_channels == CONV1_IN_CHANNEL and kh == 3 and kw == 3  # expect RGB input and 3x3 kernels
            assert out_channels == CONV1_OUT_CHANNEL                        # expect correct conv1 output channels ✅ FIXED

            # ------------------------------------------------------------------
            # BASIC FILTERS (IDENTITY, SHARPENING, SMOOTHING)
            #
            # These filters detect extremely simple local patterns:
            #   • identity: keeps pixels unchanged (baseline response)
            #   • edge_detection (Laplacian): strong center-edge contrast extractor
            #   • sharpen: highlights fine details and texture
            #   • box_blur: smooths noise uniformly
            #   • gaussian_blur: smoother blur preserving structure better
            #
            # These are fundamental low-level feature detectors.
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

            num_kernels = len(kernels)                                      # total number of base kernels

            # ------------------------------------------------------------------
            # ASSIGN STATIC KERNELS → conv1 WEIGHTS (UPDATED FOR 512 CHANNELS)
            #
            # ✅ Your conv1 now has 512 output channels:
            #   out_channels = 512 ✅ UPDATED
            #
            # We currently have:
            #   num_kernels = 45 handcrafted 3×3 kernels
            #
            # Because 512 > 45:
            #   • The first 45 conv1 filters get UNIQUE handcrafted kernels
            #   • The remaining 467 filters repeat from the beginning (wrap-around) ✅ UPDATED
            #
            # This guarantees:
            #   • conv1 starts with strong edge/corner/curve/line/gradient detectors
            #   • training converges easier (better inductive bias)
            #   • extra channels still exist (via repetition) so the network has 512 feature maps ✅ UPDATED
            # ------------------------------------------------------------------

            for i in range(out_channels):                                  # loop over each output filter (0..511) ✅ UPDATED
                k2d = kernels[i % num_kernels].to(device=w.device, dtype=w.dtype)  # pick kernel (wrap-around) + move/cast ✅ FIXED
                for c in range(in_channels):                               # copy same kernel into each RGB channel
                    w[i, c].copy_(k2d)                                     # write into conv1 weight tensor

            repeats = max(0, out_channels - num_kernels)                   # number of repeated filters due to wrap-around
            print(
                f"[init_conv1_static] out_channels={out_channels}, num_kernels={num_kernels} → "
                f"{min(out_channels, num_kernels)} unique + {repeats} repeated (wrap-around)"
            )





  
    # ----------------------------------------------------------
    # STATIC INITIALIZATION FOR LAYER 3 (UPDATED FOR 128 FEATURES) ✅ UPDATED
    # ----------------------------------------------------------
    def _init_conv3_static(self):
        with torch.no_grad():                                                           # disable gradients (manual init)
            w = self.conv3.weight                                                       # conv3 weights → [out_channels, in_channels, 3, 3]
            out_channels, in_channels, kh, kw = w.shape

            # ✅ UPDATED FOR YOUR MODEL CONFIGURATION:
            # --------------------------------------
            # Your conv3 is now:
            #   in_channels  = 256    (from conv2 out_channels) ✅ UPDATED
            #   out_channels = 512    (512 feature maps / filters) ✅ UPDATED
            #   kernel       = 3x3
            #
            # So conv3 produces:
            #   [B, 256, H/2, W/2] → [B, 512, H/2, W/2] ✅ UPDATED
            #
            # This layer is where the network starts forming higher-level patterns:
            #   • textures and repeated structures
            #   • object parts (wheels, wings, eyes, etc.)
            #   • shape composition from conv1+conv2 primitives
            assert kh == 3 and kw == 3                                                  # ensure 3x3 kernel size

            # ✅ EXTRA SAFETY CHECK (ARCHITECTURE-SAFE)
            # ----------------------------------------
            # Validate against your GLOBAL conv3 constants (no hard-coded numbers).
            #
            # These should be defined once near the top of your file:
            #   CONV3_IN_CHANNEL  = 256
            #   CONV3_OUT_CHANNEL = 128
            assert in_channels == CONV3_IN_CHANNEL and out_channels == CONV3_OUT_CHANNEL  # expect exact conv3 shape ✅ UPDATED

            # ---------------------------------------------------------------------
            # FILTER DEFINITIONS (EACH 3×3, WRITTEN IN THREE ROWS)
            #
            # conv3 receives 256 feature maps (NOT raw RGB anymore). ✅ UPDATED
            #
            # Meaning:
            #   • conv1: low-level primitives (edges/corners/lines)
            #   • conv2: mid-level combinations (textures/parts)
            #   • conv3: stronger mid/high-level compositions (parts → object patterns)
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
            # ASSIGN FILTERS TO ALL conv3 WEIGHTS (UPDATED FOR 128×256) ✅ UPDATED
            #
            # conv3 has:
            #   out_channels = 128  (filters / output features) ✅ UPDATED
            #   in_channels  = 256  (input features from conv2) ✅ UPDATED
            #
            # Strategy:
            # ---------
            # We repeat a small bank of useful kernels across the 128×256 connections.
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
            for out_idx in range(out_channels):                                       # loop over all 128 output filters ✅ UPDATED
                for in_idx in range(in_channels):                                     # loop over all 256 input feature maps ✅ UPDATED

                    # Choose kernel pattern based on a mixed index to reduce repetition artifacts
                    # (still deterministic, but spreads kernels across channels more evenly)
                    k = kernels[(out_idx * 7 + in_idx * 3) % num_kernels].to(
                        device=w.device, dtype=w.dtype
                    )                                                                # ensure device + dtype match conv3 weights

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

        # Conv1: 3 → 128 channels, preserves H, W   ✅ UPDATED
        #   [B, 3, H, W] → [B, 128, H, W]
        x = self.conv1(x)

        # BatchNorm on 128 channels (stabilizes activations) ✅ UPDATED
        x = self.bn1(x)

        # Non-linearity: ReLU
        x = F.relu(x)

        # MaxPool: H×W → H/2×W/2
        #   [B, 128, H, W] → [B, 128, H/2, W/2] ✅ UPDATED
        x = self.pool(x)

        # -------------------
        # BLOCK 2: CONV2 → BN2 → ReLU
        # -------------------

        # Conv2: 128 → 256 channels ✅ UPDATED
        #   [B, 128, H/2, W/2] → [B, 256, H/2, W/2]
        x = self.conv2(x)

        # BatchNorm on 256 channels ✅ UPDATED
        x = self.bn2(x)

        # ReLU
        x = F.relu(x)

        # ✅ IMPORTANT QUALITY FIX:
        # ------------------------
        # We REMOVE the second pooling step here.
        #
        # WHY?
        # • Pooling too early destroys spatial detail (edges, parts, textures)
        # • CIFAR-10 benefits a lot from keeping more resolution longer
        #
        # So we do NOT do:
        #   x = self.pool(x)

        # -------------------
        # BLOCK 3: CONV3 → BN3 → ReLU
        # -------------------

        # Conv3: 256 → 512 channels ✅ UPDATED
        #   [B, 256, H/2, W/2] → [B, 512, H/2, W/2]
        x = self.conv3(x)

        # BatchNorm on 512 channels ✅ UPDATED
        x = self.bn3(x)

        # ReLU
        x = F.relu(x)

        # -------------------
        # GLOBAL AVERAGE POOLING (IMAGE-SIZE INDEPENDENT)
        # -------------------

        # Replaces hard-coded spatial flattening.
        #
        # Converts:
        #   [B, 512, H/2, W/2] → [B, 512, 1, 1] ✅ UPDATED
        #
        # This step removes dependence on image size.
        x = self.gap(x)

        # Flatten channel dimension only
        #   [B, 512, 1, 1] → [B, 512] ✅ UPDATED
        x = torch.flatten(x, 1)

        # -------------------
        # DROPOUT (GENERALIZATION BOOST)
        # -------------------
        x = self.dropout(x)

        # -------------------
        # LINEAR CLASSIFIER
        # -------------------

        # Fully connected layer:
        #   [B, 512] → [B, num_classes] ✅ UPDATED
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
#   • CNNs with static-INITIALIZED filters in conv layers (e.g., Sobel, edges, corners).
#   • CNNs with randomly initialized and learnable filters.
#   • Networks with or without pooling layers.
#   • Standard datasets like CIFAR-10 or any custom dataset.
#   • Any input size supported by the model (e.g., 32×32 images).
#
# 💡 Why this works universally:
# Training depends on *backpropagation*, not on how filters are initialized.
# The optimizer updates only parameters that have requires_grad=True.
#
# -----------------------------------------------------------------------
# 🧠 Learning behavior by layer (UPDATED FOR STATIC INIT vs TRUE FREEZE):
#
# conv1 — Low-level feature extraction:
#   • If FILTERS are STATIC-INITIALIZED:
#       → Kernels are pre-defined (Sobel, corners, edges) ONLY AT INIT TIME.
#       → After that, they will STILL LEARN normally if requires_grad=True (default).
#       → They become a "smart starting point", not automatically frozen filters.
#
#   • If FILTERS are TRULY STATIC / FROZEN:
#       → You explicitly set requires_grad=False for conv1.
#       → Then conv1 kernels DO NOT change during training.
#       → conv1 behaves as a fixed feature extractor.
#
#   • If FILTERS are RANDOM/TRAINABLE:
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
#   • If static-initialized, it starts from a helpful bias but still learns unless frozen.
#
# conv3 — Higher-level feature learning:
#   • Combines conv1+conv2 features into stronger compositions:
#       → repeated structures, parts, object patterns.
#   • If static-initialized, it starts from useful kernels but still learns unless frozen.
#
# filters (in ALL convolution layers):
#   • Every kernel is a matrix of weights.
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
#   • Frozen layers have:
#         requires_grad=False → never updated
#
#   • Trainable layers have:
#         requires_grad=True → learned by backprop
#
# Therefore:
#   No conditional logic or special handling is needed in the training loop.
#   The same code trains both static-init and fully dynamic networks correctly.
#
# ============================================================
# IMPORTANT ARCHITECTURE NOTE (UPDATED FOR YOUR MODEL)
# ============================================================
#
# Your current CNN uses:
#   • conv1: 3   → 512
#   • pool: ONLY ONCE (after conv1)
#   • conv2: 512 → 256
#   • conv3: 256 → 128
#   • GAP:  [B, 128, H/2, W/2] → [B, 128, 1, 1]
#   • FC:   128 → num_classes
#
# This training loop DOES NOT need to change with channel sizes.
# It will train whatever parameters exist in your model.
# ============================================================
def train_model(
    model,
    train_loader,
    device,
    num_epochs=2,
    lr=3e-3,
    test_loader=None,
    # ------------------------------------------------------------
    # ✅ GENERALIZATION IMPROVEMENTS (NEW)
    # ------------------------------------------------------------
    # These options improve TEST accuracy (generalization), not just TRAIN accuracy.
    #
    # early_stop_patience:
    #   • If test_loader is provided, we monitor test accuracy.
    #   • If test accuracy does not improve for this many epochs → stop early.
    #
    # restore_best_weights:
    #   • If True, we load the best-performing (highest test-accuracy) weights at the end.
    #
    # weight_decay:
    #   • Regularization strength (AdamW). Higher can reduce overfitting.
    #   • Your previous fixed value was 1e-4; you can try 5e-4 or 1e-3.
    #
    # use_ema / ema_decay:
    #   • EMA (Exponential Moving Average) of weights often improves test accuracy slightly.
    #   • During evaluation, we temporarily swap to EMA weights (if enabled).
    # ------------------------------------------------------------
    early_stop_patience=15,
    restore_best_weights=True,
    weight_decay=1e-4,
    use_ema=False,
    ema_decay=0.999,
):

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
    # ✅ FIX: EARLY SAFETY CHECKS (PREVENT OneCycleLR total_steps=0)
    # ------------------------------------------------------------
    # If train_loader is empty, len(train_loader)=0 which breaks OneCycleLR.
    # Also, num_epochs must be >= 1 to do meaningful training.
    # ------------------------------------------------------------
    if num_epochs <= 0:
        print("❌ num_epochs must be >= 1.")
        return model

    if len(train_loader) <= 0:
        print("❌ train_loader is empty (len(train_loader)=0). Cannot train.")
        return model

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
    # AdamW = Adam with weight-decoupled regularization:
    #   • Similar to Adam, but weight_decay behaves more predictably for deep nets.
    #
    # model.parameters():
    #   • Collects ALL trainable tensors in the model:
    #       - convolution filter weights
    #       - bias vectors
    #       - fully connected layers
    #       - batch normalization parameters
    #   • Only parameters with requires_grad = True are included.
    #   • Frozen layers are automatically ignored.
    #
    # lr (learning rate):
    #   • Controls how fast each weight changes.
    #   • Larger values = faster learning (but risk instability).
    #   • Smaller values = slower learning (but more stable training).
    #
    # Without this optimizer:
    #   • loss.backward() computes gradients only.
    #   • optimizer.step() is required to APPLY updates.
    #
    # This single line controls learning for:
    #   • conv1 kernels
    #   • conv2 kernels
    #   • conv3 kernels
    #   • batchnorm layers
    #   • fc layer
    # ------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_scaled,               # ✅ AUTO-SCALED LR BASED ON BATCH SIZE
        weight_decay=weight_decay   # ✅ NOW CONFIGURABLE (GENERALIZATION)
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
    # IMPORTANT DIFFERENCE vs ReduceLROnPlateau:
    # ------------------------------------------
    # • ReduceLROnPlateau → stepped ONCE per epoch using loss
    # • OneCycleLR        → stepped EVERY BATCH (iteration-based)
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
        optimizer,                      # ✅ Optimizer whose LR we control (AdamW)
        max_lr=lr_scaled,               # ✅ Peak LR auto-scales with batch size
        total_steps=total_steps,        # ✅ Total LR updates across the whole run
        pct_start=0.1,                  # ✅ Warm-up fraction
        anneal_strategy="cos",          # ✅ Cosine decay after warmup
        div_factor=5.0,                 # ✅ start_lr = max_lr / div_factor
        final_div_factor=1e3            # ✅ end_lr = start_lr / final_div_factor
    )

    # ============================================================
    # COMPLETE END-TO-END EXPLANATION (UPDATED FOR GAP ARCHITECTURE)
    # IMAGE → CONVOLUTION → FEATURES → GAP → LOGITS → CrossEntropyLoss
    # ============================================================
    #
    # (keeping your full step-by-step style, but UPDATED so it matches
    #  your current network that uses GAP instead of full spatial flattening)
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
    #     [y31, y32, y33, y44],
    #     [y41, y42, y43, y44]
    #   ]
    #
    # Each y_ij is a learned combination of nearby pixels.
    #
    # ============================================================
    # STEP 5 (UPDATED): GLOBAL AVERAGE POOLING (GAP) + FLATTEN
    # ============================================================
    #
    # In THIS model, we do NOT flatten the entire H×W feature map.
    # Instead, we use Global Average Pooling (AdaptiveAvgPool2d(1)):
    #
    #   [B, C, H', W'] → [B, C, 1, 1]
    #
    # This makes the network IMAGE-SIZE INDEPENDENT.
    #
    # Then we flatten channels only:
    #
    #   [B, C, 1, 1] → [B, C]
    #
    # In your current model:
    #   C = 128 (conv3 out_channels)
    #
    # So:
    #   [B, 128, H/2, W/2] → GAP → [B, 128, 1, 1] → flatten → [B, 128]
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
    #     [w1, w2, ..., wC],   # CAT weights
    #     [v1, v2, ..., vC]    # DOG weights
    #   ]
    #
    # Bias:
    #
    #   b = [b_cat, b_dog]
    #
    # Logits computed as:
    #
    #   L_cat = Σ (wi * fi) + b_cat
    #   L_dog = Σ (vi * fi) + b_dog
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
    # loss.backward() computes gradients for trainable parameters:
    #   • convolution filters (unless frozen)
    #   • batchnorm parameters
    #   • classifier weights/bias
    #
    # optimizer.step() updates parameters to REDUCE loss next iteration.
    #
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    #
    # Image → convolution → features → GAP → logits → softmax → loss
    #
    # CrossEntropyLoss:
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
    # criterion = nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # ------------------------------------------------------------
    # OPTIONAL: STORE EXECUTION TIME FOR EACH EPOCH
    # ------------------------------------------------------------
    epoch_times = []

    # ------------------------------------------------------------
    # ✅ NEW: BEST-CHECKPOINT + EARLY STOPPING STATE
    # ------------------------------------------------------------
    # PURPOSE:
    # --------
    # Your logs show:
    #   TRAIN accuracy → ~0.995
    #   TEST  accuracy → ~0.883
    #
    # That indicates OVERFITTING:
    #   • Model memorizes train set very well
    #   • But does not generalize to unseen test images as well
    #
    # Fix:
    #   • Track the BEST test accuracy
    #   • Save best weights
    #   • Optionally STOP early if no improvement for many epochs
    # ------------------------------------------------------------
    best_test_acc = -1.0
    best_epoch = -1
    best_state_dict = None
    epochs_since_improve = 0

    # ------------------------------------------------------------
    # ✅ OPTIONAL: EMA (Exponential Moving Average) WEIGHTS (NEW)
    # ------------------------------------------------------------
    # EMA can improve test accuracy slightly by smoothing noisy updates.
    #
    # Implementation:
    #   • Maintain a copy of weights: ema_state
    #   • After each optimizer step: ema = decay*ema + (1-decay)*weight
    #   • For evaluation, we temporarily swap model weights with EMA weights
    # ------------------------------------------------------------
    ema_state = None
    if use_ema:
        ema_state = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                ema_state[name] = p.detach().clone()

    def _ema_update(model, ema_state, decay: float):
        """Updates EMA weights in-place. Only parameters that require_grad are tracked."""
        if ema_state is None:
            return
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name not in ema_state:
                    ema_state[name] = p.detach().clone()
                else:
                    ema_state[name].mul_(decay).add_(p.detach(), alpha=(1.0 - decay))

    def _ema_swap_in(model, ema_state):
        """Swap EMA weights into the model, returning a backup of current weights."""
        if ema_state is None:
            return None
        backup = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                backup[name] = p.detach().clone()
                p.copy_(ema_state[name])
        return backup

    def _ema_swap_out(model, backup_state):
        """Restore original weights after EMA evaluation."""
        if backup_state is None:
            return
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in backup_state:
                    p.copy_(backup_state[name])

    # ------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------
    for ep in range(num_epochs):

        # ------------------------------------------------------------
        # IMPORTANT FIX:
        # --------------
        # Always re-enable training mode at the START of each epoch.
        # ------------------------------------------------------------
        model.train()

        # --------------------------------------------------------
        # START TIMER FOR THIS EPOCH
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
            optimizer.zero_grad(set_to_none=True)

            # ------------------------------------------------------------
            # AMP FORWARD PASS (autocast)
            # ------------------------------------------------------------
            autocast_device_type = "cuda" if device.type == "cuda" else "cpu"

            with torch.amp.autocast(device_type=autocast_device_type, enabled=use_amp):

                logits = model(images)

                # ------------------------------------------------------------
                # 🔎 DEBUG SHAPES (run once on first batch only)
                # ------------------------------------------------------------
                if ep == 0 and total == 0:
                    print("images:", images.shape, images.dtype, images.device)
                    print("labels:", labels.shape, labels.dtype, labels.min().item(), labels.max().item())
                    print("logits:", logits.shape, logits.dtype, logits.device)

                # ------------------------------------------------------------
                # ✅ HARD ASSERTS (will stop immediately if wrong)
                # ------------------------------------------------------------
                labels = labels.long()  # CrossEntropyLoss requires int64 class indices
                assert labels.ndim == 1, f"labels must be [N], got {labels.shape}"
                assert logits.ndim == 2, f"logits must be [N,C], got {logits.shape}"
                assert logits.size(0) == labels.size(0), f"batch mismatch: {logits.size(0)} vs {labels.size(0)}"

                loss = criterion(logits, labels)

            # ----------------------------------------
            # BACKWARD PASS (AMP)
            # ----------------------------------------
            scaler.scale(loss).backward()

            # ----------------------------------------
            # GRADIENT CLIPPING (STABILITY)
            # ----------------------------------------
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ----------------------------------------
            # OPTIMIZER STEP (AMP SAFE)
            # ----------------------------------------
            scaler.step(optimizer)

            # ----------------------------------------
            # UPDATE SCALER
            # ----------------------------------------
            scaler.update()

            # ----------------------------------------
            # LEARNING RATE SCHEDULER STEP (OneCycleLR)
            # ----------------------------------------
            scheduler.step()

            # ------------------------------------------------------------
            # ✅ EMA UPDATE (NEW)
            # ------------------------------------------------------------
            # We update EMA AFTER optimizer.step() because we want EMA to reflect
            # the latest parameters (post-update).
            # ------------------------------------------------------------
            if use_ema:
                _ema_update(model, ema_state, decay=float(ema_decay))

            # ----------------------------------------
            # STATISTICS
            # ----------------------------------------
            # NOTE:
            # -----
            # This is TRAIN accuracy (on the train_loader), not test accuracy.
            # Very high values here can happen even when test accuracy is lower.
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # --------------------------------------------------------
        # END TIMER FOR THIS EPOCH
        # --------------------------------------------------------
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        # --------------------------------------------------------
        # COMPUTE AVERAGE LOSS & ACCURACY FOR THIS EPOCH
        # --------------------------------------------------------
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
        if test_loader is not None:
            model.eval()
            correct_t = 0
            total_t = 0

            # ------------------------------------------------------------
            # ✅ EMA EVAL SWAP-IN (NEW)
            # ------------------------------------------------------------
            # If EMA is enabled, we evaluate using EMA weights because they often
            # generalize better than the raw last-step weights.
            # ------------------------------------------------------------
            ema_backup = None
            if use_ema:
                ema_backup = _ema_swap_in(model, ema_state)

            with torch.no_grad():
                for images_t, labels_t in test_loader:
                    images_t = images_t.to(device)
                    labels_t = labels_t.to(device)
                    logits_t = model(images_t)
                    preds_t = logits_t.argmax(1)
                    correct_t += (preds_t == labels_t).sum().item()
                    total_t += labels_t.size(0)

            # Restore original weights after EMA evaluation
            if use_ema:
                _ema_swap_out(model, ema_backup)

            test_acc = correct_t / max(1, total_t)
            debug_print(f"[TEST]  Epoch {ep+1}/{num_epochs}  Accuracy: {test_acc:.4f}")
            model.train()

            # ------------------------------------------------------------
            # ✅ BEST CHECKPOINT TRACKING (NEW)
            # ------------------------------------------------------------
            # We track BEST test accuracy and store weights.
            # This prevents ending training with worse generalization.
            # ------------------------------------------------------------
            improved = (test_acc > best_test_acc + 1e-12)
            if improved:
                best_test_acc = float(test_acc)
                best_epoch = int(ep)
                epochs_since_improve = 0

                # Save best weights (CPU copy to reduce GPU memory pressure)
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                debug_print(
                    f"[BEST] New best TEST accuracy = {best_test_acc:.4f} "
                    f"at epoch {best_epoch+1}/{num_epochs} (saving best weights)"
                )
            else:
                epochs_since_improve += 1

            # ------------------------------------------------------------
            # ✅ EARLY STOPPING (NEW)
            # ------------------------------------------------------------
            # If test accuracy does not improve for `early_stop_patience` epochs,
            # we stop training early to reduce overfitting.
            # ------------------------------------------------------------
            if early_stop_patience is not None and early_stop_patience > 0:
                if epochs_since_improve >= int(early_stop_patience):
                    debug_print(
                        f"[EARLY STOP] No TEST improvement for {epochs_since_improve} epochs "
                        f"(patience={early_stop_patience}). Stopping early at epoch {ep+1}."
                    )
                    break

        if scheduler is not None:

            # ------------------------------------------------------------
            # IMPORTANT OneCycleLR CORRECTION
            # ------------------------------------------------------------
            # OneCycleLR is a *BATCH-BASED* scheduler.
            # ------------------------------------------------------------

            # Manual LR logging (safe — read-only)
            current_lr = optimizer.param_groups[0]['lr']
            debug_print(f"[LR Scheduler] End-of-epoch LR snapshot = {current_lr:.6f}")

    # ------------------------------------------------------------
    # ✅ RESTORE BEST WEIGHTS AT END (NEW)
    # ------------------------------------------------------------
    # If you provide test_loader, we restore the best-generalizing weights.
    # This usually improves your final saved model performance.
    # ------------------------------------------------------------
    if test_loader is not None and restore_best_weights and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        debug_print(
            f"[RESTORE] Restored BEST weights from epoch {best_epoch+1} "
            f"with best TEST accuracy = {best_test_acc:.4f}"
        )

    # ------------------------------------------------------------
    # OPTIONAL: PRINT TOTAL AND AVERAGE EXECUTION TIME
    # ------------------------------------------------------------
    if epoch_times:
        total_time = sum(epoch_times)
        avg_time = total_time / len(epoch_times)
        print(
            f"[TRAIN] Finished {len(epoch_times)} epochs "
            f"in {total_time:.2f} sec "
            f"(avg {avg_time:.2f} sec/epoch)"
        )

    # ------------------------------------------------------------
    # RETURN TRAINED MODEL
    # ------------------------------------------------------------
    return model





# ============================================================
# COMPLETE PROCEDURE (INCLUDING main)
# ============================================================
# ✅ What you requested (DONE):
#   1) When user presses 'n' and enters N:
#        ❌ DO NOT print per-image lines like:
#           [746/1000] idx=... true=... pred=... HIT/MISS
#
#        ✅ Instead print ONLY:
#           • For EACH CLASS (based on TRUE class):
#               - # success (hits for that true class)
#               - hit ratio for that true class
#               - # fail (misses for that true class)
#               - miss ratio for that true class
#
#           • At the end:
#               - total hit rate
#               - total miss rate
#
#   2) Provide complete procedure including main() ✅
#
# ✅ NEW (ADDED NOW):
#   3) When user presses 'a':
#        ✅ Run FULL evaluation on the ENTIRE test set (test_loader)
#        ✅ Print per-class success/fail + total accuracy
#        ✅ This matches the logic of your [TEST] accuracy calculation
#
# NOTE:
#   • This counts "per-class success/fail" by TRUE label (like per-class accuracy/recall):
#       total_true_of_class = number of samples whose true label == class
#       success_of_class    = among those, predicted correctly
#       fail_of_class       = among those, predicted incorrectly
#       hit_ratio_class     = success_of_class / total_true_of_class
#       miss_ratio_class    = fail_of_class / total_true_of_class
#
#   • Interactive index mode still prints per-image (that’s OK, you asked to remove messages for N mode)
# ============================================================


# ============================================================
# DETECTION / SINGLE-IMAGE INFERENCE FUNCTION
# ============================================================
def detect_single_image(model, test_dataset, device, index=None):
    """
    Loads ONE RANDOM image from the test dataset (unless index is provided),
    runs the model, and prints:
        • True label ID & name
        • Predicted label ID & name
        • Confidence (softmax probability for predicted class)
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
        class_names = [str(i) for i in range(10)]

    # --------------------------------------------------------
    # NORMALIZE & VALIDATE INDEX
    # --------------------------------------------------------
    if index is None:
        index = random.randint(0, len(test_dataset) - 1)
    else:
        if isinstance(index, str):
            try:
                index = int(index)
            except ValueError:
                print(f"[detect_single_image] Invalid index value '{index}', using 0 instead.")
                index = 0

        if index < 0 or index >= len(test_dataset):
            print(f"[detect_single_image] Index {index} is out of range 0–{len(test_dataset)-1}, using 0 instead.")
            index = 0

    # --------------------------------------------------------
    # LOAD IMAGE + TRUE LABEL
    # --------------------------------------------------------
    img, true_label = test_dataset[index]
    true_label_id = int(true_label)

    # img is [C, H, W]
    c, h, w = img.shape

    # Add batch dimension → [1, C, H, W]
    # ------------------------------------------------------------
    # ✅ SPEED FIX:
    # If CUDA is used and the DataLoader has pin_memory=True,
    # then non_blocking=True makes host→GPU copy faster.
    # ------------------------------------------------------------
    img_input = img.unsqueeze(0).to(device, non_blocking=(device.type == "cuda"))

    # --------------------------------------------------------
    # FORWARD PASS (NO GRADIENT TRACKING)
    # --------------------------------------------------------
    with torch.no_grad():
        logits = model(img_input)
        pred_label = int(logits.argmax(1).item())

        probs = torch.softmax(logits, dim=1)
        pred_conf = float(probs[0, pred_label].item())

    # --------------------------------------------------------
    # LABEL NAMES
    # --------------------------------------------------------
    true_name = class_names[true_label_id] if 0 <= true_label_id < len(class_names) else f"class_{true_label_id}"
    pred_name = class_names[pred_label] if 0 <= pred_label < len(class_names) else f"class_{pred_label}"

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------
    print("--------------------------------------------------")
    print(f"DETECTION RESULT FOR TEST IMAGE INDEX: {index}")
    print(f"Input image shape : [C={c}, H={h}, W={w}]")
    print(f"True label index  : {true_label_id} → {true_name}")
    print(f"Pred label index  : {pred_label} → {pred_name}")
    print(f"Confidence        : {pred_conf*100:.2f}%")
    print("--------------------------------------------------")

    return img, true_label_id, pred_label


# ============================================================
# MAIN PROGRAM
# ============================================================
def main():

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_print("Using device:", device)

    # --------------------------------------------------------
    # ✅ SPEED IMPROVEMENT (CUDA)
    # --------------------------------------------------------
    # If you are on CUDA and image sizes are consistent, this can speed up convs.
    # Safe to keep on for typical CNN training/inference.
    # --------------------------------------------------------
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # --------------------------------------------------------
    # ASSUME GLOBAL PATHS ARE ALREADY DEFINED
    # --------------------------------------------------------
    # DATA_PATH = "../../../data/mydata"
    # MODEL_PATH = "../../../"
    # MODEL_FILENAME = "cifar10_model_custom_file"
    # NUM_EPOCHS = 2
    # BATCH_SIZE = 64
    # NUM_WORKERS = 2
    # --------------------------------------------------------
    debug_print(f"[main] Global DATA_PATH = {DATA_PATH!r}")

    train_path = os.path.join(DATA_PATH, "train")
    test_path  = os.path.join(DATA_PATH, "test")

    debug_print(f"[main] Computed train_path = {train_path}")
    debug_print(f"[main] Computed test_path  = {test_path}")

    debug_print("Training images from:", train_path)
    debug_print("Testing  images from:", test_path)

    # ============================================================
    # TRAIN TRANSFORM (USED DURING TRAINING)
    # ============================================================
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # ============================================================
    # TEST TRANSFORM (USED DURING VALIDATION / INFERENCE)
    # ============================================================
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # ------------------------------------------------------------
    # LOAD DATASETS USING ImageFolder
    # ------------------------------------------------------------
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_path,
        transform=test_transform
    )

    debug_print(f"[main] Loaded train_dataset with {len(train_dataset)} images")
    debug_print(f"[main] Loaded test_dataset  with {len(test_dataset)} images")

    # Show class mapping as seen by ImageFolder
    debug_print("[main] Class index → name mapping (from train_dataset.classes):")
    for idx, name in enumerate(train_dataset.classes):
        debug_print(f"   {idx}: {name}")

    # ============================================================
    # DATALOADERS
    # ============================================================
    # ------------------------------------------------------------
    # ✅ SPEED FIX (CUDA):
    # pin_memory=True allows faster CPU→GPU transfer.
    # persistent_workers=True keeps worker processes alive across epochs (faster).
    # ------------------------------------------------------------
    pin = (device.type == "cuda")
    pers = (NUM_WORKERS > 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=pers
    )

    test_loader = DataLoader(
        test_dataset,
        # ------------------------------------------------------------
        # ✅ CORRECTNESS FIX:
        # Evaluation should NOT shuffle for stable metrics.
        # ------------------------------------------------------------
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=pers
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
    # ✅ Your updated architecture note:
    #   conv1: 3   -> 128
    #   conv2: 128 -> 256
    #   conv3: 256 -> 512
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
        model = train_model(model, train_loader, device, num_epochs=NUM_EPOCHS, lr=1e-3, test_loader=test_loader)
        debug_print(f"Saving trained model to: {model_filename}")
        torch.save(model.state_dict(), model_filename)

    # ------------------------------------------------------------
    # HELPER: READ A POSITIVE INTEGER USING msvcrt (DIGITS UNTIL ENTER)
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

        first = msvcrt.getch().decode(errors="ignore").lower()

        if first == "e":
            print("e")
            return "EXIT"

        if not first.isdigit():
            print(first)
            return None

        print(first, end="", flush=True)

        s = first
        while True:
            ch = msvcrt.getch()
            if ch in [b"\r", b"\n"]:
                print()
                break

            try:
                c = ch.decode(errors="ignore")
            except Exception:
                continue

            if c.lower() == "e":
                print("e")
                return "EXIT"

            if c.isdigit():
                s += c
                print(c, end="", flush=True)

        if not s.isdigit():
            return None

        return int(s)

    # ------------------------------------------------------------
    # RUN N RANDOM TEST IMAGES AND PRINT PER-CLASS SUCCESS/FAIL ONLY
    # ------------------------------------------------------------
    def run_n_random_images(model, test_dataset, device, n: int):
        """
        Runs the model on N random images and prints ONLY:
          • For EACH TRUE CLASS:
              - # success (hits)
              - hit ratio
              - # fail (misses)
              - miss ratio
          • Total hit/miss rates
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
        model.to(device)

        # --------------------------------------------------------
        # AUTO-DETECT CLASS NAMES
        # --------------------------------------------------------
        class_names = getattr(test_dataset, "classes", None)
        if class_names is None:
            class_names = [str(i) for i in range(10)]

        num_classes = len(class_names)

        # --------------------------------------------------------
        # PICK N RANDOM INDICES
        # --------------------------------------------------------
        if n <= len(test_dataset):
            indices = random.sample(range(len(test_dataset)), k=n)
        else:
            indices = [random.randrange(len(test_dataset)) for _ in range(n)]

        # --------------------------------------------------------
        # PER-CLASS METRICS (BASED ON TRUE LABEL)
        # --------------------------------------------------------
        # total_true[c]   = how many samples had true label == c
        # hit_true[c]     = among those, predicted correctly
        # miss_true[c]    = among those, predicted incorrectly
        total_true = [0] * num_classes
        hit_true   = [0] * num_classes
        miss_true  = [0] * num_classes

        # Overall metrics
        total = 0
        hits  = 0

        # --------------------------------------------------------
        # ✅ SPEED FIX (MAJOR):
        # --------------------------------------------------------
        # Instead of running the model 1 image at a time,
        # we batch images together and do fewer forward passes.
        #
        # Behavior is IDENTICAL (same logits/argmax), only faster.
        # --------------------------------------------------------
        eval_bs = 128  # you can tune this (128/256) depending on GPU memory

        with torch.no_grad():

            # Process indices in chunks of eval_bs
            for start in range(0, len(indices), eval_bs):

                batch_indices = indices[start:start + eval_bs]

                # Build batch tensors
                images_list = []
                true_ids = []

                for idx in batch_indices:
                    image_tensor, true_label = test_dataset[idx]
                    true_id = int(true_label)

                    images_list.append(image_tensor)
                    true_ids.append(true_id)

                # Stack into [B, C, H, W]
                images_batch = torch.stack(images_list, dim=0).to(
                    device,
                    non_blocking=(device.type == "cuda")
                )

                logits = model(images_batch)
                pred_ids = logits.argmax(1).detach().cpu().tolist()

                # Update counts
                for true_id, pred_id in zip(true_ids, pred_ids):

                    total += 1

                    # Count this sample into its TRUE class bucket
                    if 0 <= true_id < num_classes:
                        total_true[true_id] += 1

                        if pred_id == true_id:
                            hit_true[true_id] += 1
                            hits += 1
                        else:
                            miss_true[true_id] += 1

        misses = total - hits
        total_hit_rate  = (hits / total) if total > 0 else 0.0
        total_miss_rate = 1.0 - total_hit_rate

        # --------------------------------------------------------
        # PRINT SUMMARY ONLY (NO PER-IMAGE LINES)
        # --------------------------------------------------------
        print("\n--------------------------------------------------")
        print(f"N-Random Evaluation Summary (N={n})")
        print("Per-class results (based on TRUE class)")
        print("--------------------------------------------------")

        for c in range(num_classes):
            t = total_true[c]
            h = hit_true[c]
            m = miss_true[c]

            # If class didn't appear in the random sample, ratios are 0
            hit_ratio  = (h / t) if t > 0 else 0.0
            miss_ratio = (m / t) if t > 0 else 0.0

            print(
                f"Class {c:>2} ({class_names[c]:<20}) | "
                f"success={h:>6}  hit_ratio={hit_ratio:>7.4f} ({hit_ratio*100:>6.2f}%) | "
                f"fail={m:>6}     miss_ratio={miss_ratio:>7.4f} ({miss_ratio*100:>6.2f}%) | "
                f"total={t:>6}"
            )

        print("--------------------------------------------------")
        print("TOTAL (ALL CLASSES)")
        print("--------------------------------------------------")
        print(f"Total images : {total}")
        print(f"Total hits   : {hits}")
        print(f"Total misses : {misses}")
        print(f"Hit rate     : {total_hit_rate:.4f}  ({total_hit_rate*100:.2f}%)")
        print(f"Miss rate    : {total_miss_rate:.4f} ({total_miss_rate*100:.2f}%)")
        print("--------------------------------------------------\n")

    # ------------------------------------------------------------
    # RUN FULL TEST SET EVALUATION (MATCHES [TEST] ACCURACY)
    # ------------------------------------------------------------
    def run_full_test_evaluation(model, test_loader, test_dataset, device):
        """
        Runs the model on the ENTIRE test_loader and prints ONLY:
          • For EACH TRUE CLASS:
              - # success (hits)
              - hit ratio
              - # fail (misses)
              - miss ratio
          • Total hit/miss rates

        IMPORTANT:
        ----------
        This is the same style as your per-epoch test evaluation:
          preds = logits.argmax(1)
          correct += (preds == labels).sum().item()
          total += labels.size(0)

        So the final "Hit rate" here should match your [TEST] Accuracy.
        """

        # --------------------------------------------------------
        # PUT MODEL IN EVAL MODE (DISABLE DROPOUT, FIX BATCHNORM)
        # --------------------------------------------------------
        model.eval()
        model.to(device)

        # --------------------------------------------------------
        # AUTO-DETECT CLASS NAMES
        # --------------------------------------------------------
        class_names = getattr(test_dataset, "classes", None)
        if class_names is None:
            class_names = [str(i) for i in range(10)]

        num_classes = len(class_names)

        # --------------------------------------------------------
        # PER-CLASS METRICS (BASED ON TRUE LABEL)
        # --------------------------------------------------------
        # total_true[c]   = how many test samples had true label == c
        # hit_true[c]     = among those, predicted correctly
        # miss_true[c]    = among those, predicted incorrectly
        total_true = [0] * num_classes
        hit_true   = [0] * num_classes
        miss_true  = [0] * num_classes

        # Overall metrics
        total = 0
        hits  = 0

        # --------------------------------------------------------
        # FULL-DATASET PASS (BATCHED BY test_loader)
        # --------------------------------------------------------
        with torch.no_grad():
            for images_t, labels_t in test_loader:

                images_t = images_t.to(device, non_blocking=(device.type == "cuda"))
                labels_t = labels_t.to(device, non_blocking=(device.type == "cuda")).long()

                logits_t = model(images_t)
                preds_t = logits_t.argmax(1)

                # Update global totals
                total += labels_t.size(0)
                hits  += (preds_t == labels_t).sum().item()

                # Update per-class totals (based on TRUE class)
                labels_cpu = labels_t.detach().cpu().tolist()
                preds_cpu  = preds_t.detach().cpu().tolist()

                for true_id, pred_id in zip(labels_cpu, preds_cpu):
                    if 0 <= true_id < num_classes:
                        total_true[true_id] += 1
                        if pred_id == true_id:
                            hit_true[true_id] += 1
                        else:
                            miss_true[true_id] += 1

        misses = total - hits
        total_hit_rate  = (hits / total) if total > 0 else 0.0
        total_miss_rate = 1.0 - total_hit_rate

        # --------------------------------------------------------
        # PRINT SUMMARY ONLY (NO PER-IMAGE LINES)
        # --------------------------------------------------------
        print("\n--------------------------------------------------")
        print("FULL Test Evaluation Summary (ENTIRE test_loader)")
        print("Per-class results (based on TRUE class)")
        print("--------------------------------------------------")

        for c in range(num_classes):
            t = total_true[c]
            h = hit_true[c]
            m = miss_true[c]

            hit_ratio  = (h / t) if t > 0 else 0.0
            miss_ratio = (m / t) if t > 0 else 0.0

            print(
                f"Class {c:>2} ({class_names[c]:<20}) | "
                f"success={h:>6}  hit_ratio={hit_ratio:>7.4f} ({hit_ratio*100:>6.2f}%) | "
                f"fail={m:>6}     miss_ratio={miss_ratio:>7.4f} ({miss_ratio*100:>6.2f}%) | "
                f"total={t:>6}"
            )

        print("--------------------------------------------------")
        print("TOTAL (ALL CLASSES)")
        print("--------------------------------------------------")
        print(f"Total images : {total}")
        print(f"Total hits   : {hits}")
        print(f"Total misses : {misses}")
        print(f"Hit rate     : {total_hit_rate:.4f}  ({total_hit_rate*100:.2f}%)")
        print(f"Miss rate    : {total_miss_rate:.4f} ({total_miss_rate*100:.2f}%)")
        print("--------------------------------------------------\n")

    # ------------------------------------------------------------
    # INTERACTIVE LOOP FOR USER-DRIVEN DETECTION
    # ------------------------------------------------------------
    print("\n--------------------------------------------------")
    print("Interactive Image Detection Mode")
    print("Type an image index and press ENTER (prints that single result).")
    print("Type 'n' then enter N to run N random images (prints ONLY summary).")
    print("Type 'a' to evaluate the ENTIRE test set (prints ONLY summary).")
    print("Press 'e' at any time to exit.")
    print("--------------------------------------------------\n")

    while True:

        print(
            f"Enter image index (0 – {len(test_dataset)-1}), or 'n' for N-random, or 'a' for full test, or 'e' to exit: ",
            end="",
            flush=True
        )

        key = msvcrt.getch().decode(errors="ignore").lower()

        if key == 'e':
            print("e")
            print("Exiting program. Goodbye!")
            break

        # ------------------------------------------------
        # OPTION: USER PRESSES 'a' → RUN FULL TEST EVALUATION
        # ------------------------------------------------
        if key == "a":
            print("a")
            run_full_test_evaluation(model, test_loader, test_dataset, device)
            continue

        # ------------------------------------------------
        # OPTION: USER PRESSES 'n' → RUN N RANDOM IMAGES
        # ------------------------------------------------
        if key == "n":
            print("n")

            n_val = _read_int_from_keyboard_msvcrt(
                "Enter N (number of random test images) and press ENTER (or 'e' to exit): "
            )

            if n_val == "EXIT":
                print("Exiting program. Goodbye!")
                break

            if n_val is None:
                print("❌ Invalid input. N must be a number.")
                continue

            run_n_random_images(model, test_dataset, device, n=int(n_val))
            continue

        # If first key is NOT a digit → invalid
        if not key.isdigit():
            print(key)
            print("❌ Invalid input. Enter a number, 'n', 'a', or 'e' to exit.")
            continue

        # Echo first digit
        print(key, end="", flush=True)

        # Read remaining digits until ENTER
        idx_str = key
        while True:
            ch = msvcrt.getch()
            if ch in [b'\r', b'\n']:
                print()
                break

            try:
                c = ch.decode(errors="ignore")
            except Exception:
                continue

            if c.lower() == 'e':
                print("e")
                print("Exiting program. Goodbye!")
                return

            if c.isdigit():
                idx_str += c
                print(c, end="", flush=True)

        if not idx_str.isdigit():
            print("❌ Invalid index. Must be a number.")
            continue

        idx = int(idx_str)

        if idx < 0 or idx >= len(test_dataset):
            print("❌ Index out of range. Try again.")
            continue

        print(f"\nRunning detection on test image index {idx} ...")
        detect_single_image(model, test_dataset, device, index=idx)


# ------------------------------------------------------------
# RUN PROGRAM
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
