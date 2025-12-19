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
MODEL_FILENAME = "cifar10_model_custom_file"
DATA_PATH = "../../../data/mydata"
MG_WIDTH, IMG_HEIGHT = 32, 32  # Based on training dataset
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for valid detections
FILTER_WIDTH = 3
FILTER_HEIGHT = 3
BATCH_SIZE = 64
NUM_EPOCHS = 700
LEARNING_RATE = 0.001


NUM_WORKERS = 0
STATIC_FILTERS = False
DEBUG_FLAG = True
# ============================================================
# EXPLANATION: HOW TRAINING WORKS IN THIS NETWORK
# ============================================================
#
# This network is a CLASSICAL CONVOLUTIONAL NEURAL NETWORK (CNN)
# where:
#
#   • Layer 1 (conv1) STARTS with static handcrafted filters
#   • Layer 2 (conv2) STARTS with static handcrafted filters
#   • Between conv layers, we apply MAX POOLING to shrink feature maps
#   • Layer 3 (fc)    is a standard fully connected classifier
#
# INPUT ASSUMPTION:
# -----------------
# The network expects 3-channel images with spatial size 32x32:
#   • Directly from datasets like CIFAR-10, OR
#   • From custom images (e.g., 128x128) that are resized to 32x32
#     using transforms.Resize((32, 32)) in the input pipeline.
#
# IMPORTANT:
# ----------
# The word "static" here means:
#   → The filters are STATIC ONLY AT INITIALIZATION TIME.
#   → After training starts, ALL layers are learnable.
#
# So this is NOT a "frozen-filter" network.
# It is a CLASSICAL neural network that starts from known kernels
# and then learns normally from data.
#
# ============================================================
# HOW LEARNING HAPPENS
# ============================================================
#
# When this code runs:
#
#   outputs = model(images)
#   loss    = criterion(outputs, labels)
#   loss.backward()
#   optimizer.step()
#
# PyTorch computes derivatives (gradients) for:
#
#   • conv1.weight
#   • conv1.bias
#   • conv2.weight
#   • conv2.bias
#   • fc.weight
#   • fc.bias
#
# (Pooling layers have NO learnable parameters, so they do not
#  have weights or biases, and nothing is trained inside pooling.)
#
# Because:
#   - we did NOT freeze any layer
#   - requires_grad = True for all parameters
#
# Then:
#
#   optimizer.step()
#
# updates ALL learnable layers (conv1, conv2, fc).
#
# ============================================================
# SO: WILL ALL 3 LEARNABLE LAYERS LEARN?
# ============================================================
#
# YES.
#
#   • Layer 1 learns
#   • Layer 2 learns
#   • Layer 3 learns
#
# Pooling only performs a fixed mathematical operation
# (max over windows) and does not learn.
#
# ============================================================
# WHAT WOULD STOP LEARNING?
# ============================================================
#
# If you write:
#
#   param.requires_grad = False
#
# on any layer, that layer will STOP learning.
#
# We DO NOT do this here.
#
# ============================================================
# WHY THIS IS A CLASSICAL NEURAL NETWORK
# ============================================================
#
# Because:
#
#   • Filters are initialized
#   • Filters are trained
#   • Weights change through backpropagation
#   • Learning is end-to-end
#   • Pooling is used to reduce spatial resolution and keep
#     the most important features.
#
# This is exactly how CNNs are trained in practice, except
# most networks start with RANDOM initialization.
#
# Here you start with INTELLIGENT initialization.
#
# ============================================================
# NETWORK SHAPE (32x32 RGB INPUT WITH POOLING)
# ============================================================
#
# Input image:                [3  x 32 x 32]
#   (e.g., CIFAR-10, or custom images resized to 32x32)
#
# After conv1:                [16 x 32 x 32]
# After max-pool1 (2x2):      [16 x 16 x 16]
# After conv2:                [32 x 16 x 16]
# After max-pool2 (2x2):      [32 x  8 x  8]
# After flattening:           [32 * 8 * 8] = [2048]
# Output layer (fc):          [C classes]
#
# ============================================================
# SUMMARY
# ============================================================
#
# ✅ Static at start (conv1, conv2 initialization)
# ✅ Dynamic during training (all learnable layers)
# ✅ Pooling reduces spatial size and keeps strong features
# ✅ Works with CIFAR-10 OR any 3x32x32 images
# ✅ Classical CNN
#


class StaticInitLearnableCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # --------------------------------------------------------
        # cuDNN AUTOTUNER
        # --------------------------------------------------------
        # Enables cuDNN to find the fastest convolution algorithms
        # for your hardware and input sizes.
        #
        # Works best when:
        #   • Input image sizes are constant (e.g. always 32x32)
        #   • You train for many iterations
        #
        # WARNING:
        #   • Slightly slower first iteration (benchmarking)
        #   • Faster training afterwards
        # --------------------------------------------------------
        torch.backends.cudnn.benchmark = True
        # ------------------------------------------------------
        # LAYER 1: 3 → 16 channels
        # 3 input channels (RGB) → 16 feature maps using 3x3 filters
        # Padding = 1 to keep spatial size 32x32
        #
        # Input shape assumption:
        #   [B, 3, 32, 32]
        #
        # Either native CIFAR-10 images, OR any custom dataset
        # resized to 32×32 in transforms.
        # ------------------------------------------------------
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1,
            bias=True
        )

        # ------------------------------------------------------
        # BatchNorm for conv1 (normalizes 16 output channels)
        # ------------------------------------------------------
        # WHY WE USE BATCHNORM2d(16):
        # ---------------------------
        # • It normalizes each of the 16 feature maps across the batch
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
        self.bn1 = nn.BatchNorm2d(16)

        # ------------------------------------------------------
        # LAYER 2: 16 → 32 channels
        #
        # 16 input feature maps → 32 output feature maps
        # using 3×3 filters, padding=1 keeps spatial size.
        #
        # Before Pool:
        #   input to conv2 : [B, 16, 16, 16]
        #   output of conv2: [B, 32, 16, 16]
        # ------------------------------------------------------
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
            bias=True
        )

        # ------------------------------------------------------
        # BatchNorm for conv2 (normalizes 32 channels)
        # ------------------------------------------------------
        # Why BatchNorm2d(32)?
        # --------------------
        # • Conv2 outputs 32 feature maps
        # • BatchNorm stabilizes all 32 channels
        #
        # Overall:
        #   conv2 → bn2 → ReLU → pool
        #
        # BatchNorm especially helps deeper layers where
        # activations become more chaotic.
        # ------------------------------------------------------
        self.bn2 = nn.BatchNorm2d(32)

        # ------------------------------------------------------
        # POOLING LAYER: MaxPool2d(2, 2)
        #
        # Max pooling:
        #     kernel_size = 2
        #     stride      = 2
        #
        # Effect on spatial dimensions:
        #   32×32 → 16×16   (after first pool)
        #   16×16 →  8×8    (after second pool)
        #
        # Both conv1 and conv2 use the SAME pooling layer.
        # ------------------------------------------------------
        self.pool = nn.MaxPool2d(2, 2)

        # ------------------------------------------------------
        # FULLY CONNECTED CLASSIFIER
        #
        # After both conv+pool blocks:
        #   conv1 + pool → [B, 16, 16, 16]
        #   conv2 + pool → [B, 32,  8,  8]
        #
        # Flattened dimension:
        #   32 * 8 * 8 = 2048 features
        #
        # num_classes:
        #   • 10 for CIFAR-10
        #   • OR dynamic based on len(train_dataset.classes)
        # ------------------------------------------------------
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

        # ------------------------------------------------------
        # STATIC FILTER INITIALIZATION (if enabled)
        #
        # These functions overwrite the conv1 and conv2 weights
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
        # ------------------------------------------------------
        if STATIC_FILTERS:
            self._init_conv1_static()
            self._init_conv2_static()

    # ----------------------------------------------------------
    # STATIC INITIALIZATION FOR LAYER 1
    # ----------------------------------------------------------
    def _init_conv1_static(self):
        with torch.no_grad():                                              # disable gradients during manual init
            w = self.conv1.weight                                          # conv1 weights → [out_channels, 3, 3, 3]
            out_channels, in_channels, kh, kw = w.shape                    # get conv1 shape
            assert in_channels == 3 and kh == 3 and kw == 3                # expect RGB input and 3x3 kernels

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
            #
            # These detect straight edges in specific directions.
            # Early CNN layers rely heavily on directional edges.
            #
            # edge_0:     horizontal edges
            # edge_45:    diagonal edge (45°)
            # edge_90:    vertical edges
            # edge_135:   diagonal (135°)
            # ...
            #
            # Having 8 orientations helps the CNN capture global geometry.
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
            #
            # Corners are critical primitives for shape recognition.
            # A corner is "two edges meeting", so these filters detect L-shapes.
            #
            # Rotated versions detect corners in all orientations.
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
            #
            # Detect small curved structures, arcs, rounded shapes.
            # Useful for detecting object silhouettes, digits, animals, etc.
            #
            # Each rotation detects curves bending in a different direction.
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
            #
            # These detect long straight lines of different orientations.
            # Lines = stronger than edges because they extend across the kernel.
            #
            # Helps capture object boundaries & global geometry structure.
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
            #
            # Sobel filters compute approximate derivatives.
            #
            # sobel_0:   detect vertical edges (dx)
            # sobel_90:  detect horizontal edges (dy)
            #
            # Rotated Sobels capture gradients at 45° increments.
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

            num_kernels = len(kernels)                                      # count number of base kernels

            # ------------------------------------------------------------------
            # ASSIGN STATIC KERNELS → conv1 WEIGHTS
            #
            # conv1 has out_channels filters (e.g., 16 or 32).
            # If out_channels > number of kernels, we repeat them in order.
            #
            # This guarantees:
            #   • conv1 sees edges, corners, lines, curves, gradients instantly
            #   • training becomes easier (better inductive bias)
            #   • the CNN behaves like a hybrid handcrafted + learned feature extractor
            # ------------------------------------------------------------------

            for i in range(out_channels):                                  # loop over each output filter
                k2d = kernels[i % num_kernels].to(w.dtype)                 # pick 2D kernel and cast dtype
                for c in range(in_channels):                               # assign same 3x3 kernel to each RGB channel
                    w[i, c].copy_(k2d)                                     # write into conv1 weight tensor

            print(f"[init_conv1_static] {out_channels} filters initialized with 2D 3x3 kernels")

    # ----------------------------------------------------------
    # STATIC INITIALIZATION FOR LAYER 2
    # ----------------------------------------------------------
    def _init_conv2_static(self):
        with torch.no_grad():                                                           # disable gradients (manual init)
            w = self.conv2.weight                                                       # conv2 weights → [32,16,3,3]
            out_channels, in_channels, kh, kw = w.shape                                 # expected [32,16,3,3]
            assert kh == 3 and kw == 3                                                  # ensure 3x3 kernel size

            # ---------------------------------------------------------------------
            #  FILTER DEFINITIONS (EACH 3×3, WRITTEN IN THREE ROWS)
            #  conv2 receives 16 feature maps → deeper filters detect stronger edges,
            #  transitions, gradients, shape composition, and embossed structure.
            # ---------------------------------------------------------------------

            # 1) Horizontal edge detector
            #    Strong response to horizontal lines and transitions.
            edge_h = torch.tensor([
                [-1., -1., -1.],
                [ 2.,  2.,  2.],
                [-1., -1., -1.],
            ])

            # 2) Vertical edge detector
            #    Strong response to vertical edges or vertical texture discontinuities.
            edge_v = torch.tensor([
                [-1.,  2., -1.],
                [-1.,  2., -1.],
                [-1.,  2., -1.],
            ])

            # 3) Emboss filter
            #    Creates a shaded 3D-like emboss effect; highlights directional depth.
            emboss = torch.tensor([
                [-2., -1.,  0.],
                [-1.,  1.,  1.],
                [ 0.,  1.,  2.],
            ])

            # 4) Average blur (3×3 mean filter)
            #    Smooths noise and merges nearby features.
            avg = (1/9) * torch.ones((3, 3))

            # 5) Sobel X (horizontal gradient)
            #    Detects left–right intensity changes (vertical edges).
            sobel_x = torch.tensor([
                [-1.,  0.,  1.],
                [-2.,  0.,  2.],
                [-1.,  0.,  1.],
            ])

            # 6) Sobel Y (vertical gradient)
            #    Detects top–bottom intensity transitions (horizontal edges).
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
            #  ASSIGN FILTERS TO ALL conv2 WEIGHTS
            #
            #  conv2 has: out_channels = 32   (filters)
            #             in_channels  = 16   (input maps from conv1)
            #
            #  For each output filter and each input channel, we choose a kernel
            #  using modulo indexing so the filters repeat periodically.
            #
            #  This creates a structured 32×16 kernel matrix where:
            #     • Some paths detect gradients
            #     • Some detect edges
            #     • Some emboss or smooth
            #
            #  This provides conv2 with rich static feature extraction.
            # ---------------------------------------------------------------------
            for out_idx in range(out_channels):                       # loop over all 32 output filters
                for in_idx in range(in_channels):                     # loop over all 16 input feature maps

                    # Choose kernel pattern based on (out × in) mod #kernels
                    k = kernels[(out_idx * in_idx) % num_kernels].to(w.dtype)

                    # Copy kernel into weight tensor
                    w[out_idx, in_idx].copy_(k)

            print(f"[init_conv2_static] {out_channels}x{in_channels} 2D 3x3 kernels assigned")  # log

    # ----------------------------------------------------------
    # FORWARD PASS
    # ----------------------------------------------------------
    # FORWARD PROPAGATION THROUGH THE NETWORK
    #  ---------------------------------------
    # This method defines how input images flow through the network.
    #
    # INPUT:
    #     x : Tensor of shape [B, 3, H, W]
    #         B = batch size
    #         3 = RGB channels
    #         H, W = image dimensions (ideally 32×32)
    #
    # OUTPUT:
    #     logits : Tensor of shape [B, num_classes]
    #         Raw class scores (logits) before softmax.
    #         These are passed into CrossEntropyLoss during training.
    def forward(self, x):
        # At entry:
        #   x shape → [B, 3, 32, 32]
        #   (CIFAR-10 or resized custom data)

        # -------------------
        # BLOCK 1: CONV1 → BN1 → ReLU → POOL
        # -------------------

        # Conv1: 3 → 16 channels, preserves H, W
        #   [B, 3, 32, 32] → [B, 16, 32, 32]
        x = self.conv1(x)

        # BatchNorm on 16 channels (stabilizes activations)
        x = self.bn1(x)

        # Non-linearity: ReLU
        x = F.relu(x)

        # MaxPool: 32×32 → 16×16
        #   [B, 16, 32, 32] → [B, 16, 16, 16]
        x = self.pool(x)

        # -------------------
        # BLOCK 2: CONV2 → BN2 → ReLU → POOL
        # -------------------

        # Conv2: 16 → 32 channels
        #   [B, 16, 16, 16] → [B, 32, 16, 16]
        x = self.conv2(x)

        # BatchNorm on 32 channels
        x = self.bn2(x)

        # ReLU
        x = F.relu(x)

        # MaxPool: 16×16 → 8×8
        #   [B, 32, 16, 16] → [B, 32, 8, 8]
        x = self.pool(x)

        # -------------------
        # FLATTEN + LINEAR CLASSIFIER
        # -------------------

        # Flatten all channels + spatial dims:
        #   [B, 32, 8, 8] → [B, 32*8*8] = [B, 2048]
        x = torch.flatten(x, 1)

        # Fully connected layer:
        #   [B, 2048] → [B, num_classes]
        logits = self.fc(x)

        # logits are returned directly.
        # CrossEntropyLoss will apply softmax internally.
        return logits

# ------------------------------------------------------------------
# GLOBAL DEBUG FLAG + HELPER
# ------------------------------------------


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



def train_model(model, train_loader, device, num_epochs=2, lr=1e-3):
    


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
    scaler = torch.amp.GradScaler(
        device_type="cuda",
        enabled=use_amp
    )
        # ------------------------------------------------------------
        # SEND MODEL TO GPU (IF AVAILABLE) OR CPU
        # ------------------------------------------------------------
    model.to(device)# ------------------------------------------------------------
    # ENABLE TRAINING MODE
    #   (activates dropout, batchnorm if they exist)
    # ------------------------------------------------------------
    model.train()

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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
   # ------------------------------------------------------------
    # LEARNING RATE SCHEDULER — ReduceLROnPlateau
    # ------------------------------------------------------------
    #
    # PURPOSE:
    # --------
    # During training, the optimizer may stop improving because the
    # learning rate (LR) is TOO HIGH for fine adjustments.
    #
    # Example scenario:
    #   Epoch 70 → loss = 0.43
    #   Epoch 71 → loss = 0.43
    #   Epoch 72 → loss = 0.43
    #   Epoch 73 → loss = 0.43
    #
    # Loss is "plateauing" — the model is stuck.
    #
    # ReduceLROnPlateau monitors the loss and:
    #   • If the loss does NOT improve for N epochs,
    #     it REDUCES the learning rate automatically.
    #
    # This allows:
    #   • Big steps early in training (fast learning)
    #   • Small steps later (fine tuning)
    #
    # RESULT:
    #   → smoother convergence
    #   → lower final loss
    #   → better accuracy
    # ------------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,

        mode='min',         # Scheduler monitors a quantity and tries to make it MINIMAL.
                            # Here we monitor epoch_loss (lower is better).

        factor=0.5,         # When LR needs reduction:
                            #     new_lr = old_lr * factor
                            #
                            # If old_lr = 0.001:
                            #     new_lr = 0.001 * 0.5 = 0.0005
                            #
                            # If LR plateaus again, scheduler reduces again:
                            #     0.0005 → 0.00025 → 0.000125 → …

        patience=5          # Number of epochs to wait with NO improvement before reducing LR.
                            #
                            # Example:
                            #   Epoch 40 → loss = 0.42
                            #   Epoch 41 → loss = 0.42
                            #   Epoch 42 → loss = 0.43
                            #   Epoch 43 → loss = 0.422
                            #   Epoch 44 → loss = 0.422
                            #   Epoch 45 → loss = 0.423
                            #
                            # If no improvement for 5 epochs → lower LR.

        # NOTE:
        # PyTorch 2.x REMOVED support for verbose=True.
        # We will print LR manually after scheduler.step().
    )

    
    # ============================================================
    # COMPLETE END-TO-END EXPLANATION:
    # IMAGE → CONVOLUTION → FEATURES → LOGITS → CrossEntropyLoss
    # ============================================================
    #
    # This is a FULL PIPELINE explanation showing exactly what:
    #
    #   criterion = nn.CrossEntropyLoss()
    #
    # means, from the image level up to the loss computation.
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

   
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------
    # OPTIONAL: STORE EXECUTION TIME FOR EACH EPOCH
    #   • epoch_times will hold duration (seconds) for every epoch.
    #   • Useful for performance diagnostics and ETA estimation.
    # ------------------------------------------------------------
    epoch_times = []

    # ------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------
    for ep in range(num_epochs):

        # --------------------------------------------------------
        # START TIMER FOR THIS EPOCH
        #   • Capture high-resolution start time BEFORE any work.
        #   • At the end of the epoch we subtract to get duration.
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
            optimizer.zero_grad()

            
            # ============================================================
            # WHAT THIS LINE DOES:
            #     outputs = model(images)
            # ============================================================
            # FINAL SUMMARY:
            # ============================================================
            #
            # outputs = model(images) does:
            #
            #   • Runs convolution layers (feature extraction)
            #   • Runs pooling layers (dimensionality reduction)
            #   • Flattens features into a vector
            #   • Applies classifier weights
            #   • Produces raw class scores (logits)
            #
            # These logits are then interpreted by:
            #   criterion = nn.CrossEntropyLoss()
            #
            # to compute how wrong the prediction is.
            #
            # ============================================================
            # ============================================================
            #
            # This runs the ENTIRE neural network forward pass:
            #
            #     Image → Convolution → Pooling → Flatten → Classifier → Logits
            #
            # Below is a COMPLETE STEP-BY-STEP example using:
            #
            #   • One 4x4 image
            #   • One 3x3 filter
            #   • Padding = 1
            #   • Max Pooling
            #   • Fully connected classifier
            #   • 3 output classes: CAT, DOG, MAN
            #
            # ============================================================
            # INPUT IMAGE (4×4, 1 channel)
            # ============================================================
            #
            # X =
            # [
            #   [a11, a12, a13, a14],
            #   [a21, a22, a23, a24],
            #   [a31, a32, a33, a34],
            #   [a41, a42, a43, a44]
            # ]
            #
            # ============================================================
            # STAGE 1 — CONVOLUTION (3x3 FILTER, padding = 1)
            # ============================================================
            #
            # Filter F =
            # [
            #   [f11, f12, f13],
            #   [f21, f22, f23],
            #   [f31, f32, f33]
            # ]
            #
            # First pad image with zeros:
            #
            # X_padded =
            # [
            #   [0, 0, 0, 0, 0, 0],
            #   [0, a11, a12, a13, a14, 0],
            #   [0, a21, a22, a23, a24, 0],
            #   [0, a31, a32, a33, a34, 0],
            #   [0, a41, a42, a43, a44, 0],
            #   [0, 0, 0, 0, 0, 0]
            # ]
            #
            # Output stays 4x4
            #
            # Example: top-left output pixel:
            #
            # y11 =
            #   a11*f22 + a12*f23
            # + a21*f32 + a22*f33
            #
            # Compute rest the same way → Feature Map Y:
            #
            # Y =
            # [
            #   [y11, y12, y13, y14],
            #   [y21, y22, y23, y24],
            #   [y31, y32, y33, y34],
            #   [y41, y42, y43, y44]
            # ]
            #
            # ============================================================
            # STAGE 2 — MAX POOLING (2x2)
            # ============================================================
            #
            # Pooling keeps the maximum in each 2×2 block:
            #
            # Example:
            #
            # Input block:
            # [
            #   y11, y12
            #   y21, y22
            # ]
            #
            # Output:
            #   max(y11, y12, y21, y22)
            #
            # Resulting pooled map:
            #
            # P =
            # [
            #   [p11, p12],
            #   [p21, p22]
            # ]
            #
            # Size reduces: 4x4 → 2x2
            #
            # ============================================================
            # STAGE 3 — FLATTEN FEATURES
            # ============================================================
            #
            # Convert P into a vector:
            #
            # feature_vector =
            # [
            #   p11, p12,
            #   p21, p22
            # ]
            #
            # Shape: [4]
            #
            # ============================================================
            # STAGE 4 — CLASSIFIER (FULLY CONNECTED LAYER)
            # ============================================================
            #
            # Suppose we classify:
            #
            #   0 → CAT
            #   1 → DOG
            #   2 → MAN
            #
            # The FC layer has weights:
            #
            # W =
            # [
            #   [w1, w2, w3, w4],   # CAT
            #   [v1, v2, v3, v4],   # DOG
            #   [u1, u2, u3, u4]    # MAN
            # ]
            #
            # Bias:
            #
            # b = [b_cat, b_dog, b_man]
            #
            # Compute logits:
            #
            # L_cat = w1*p11 + w2*p12 + w3*p21 + w4*p22 + b_cat
            # L_dog = v1*p11 + v2*p12 + v3*p21 + v4*p22 + b_dog
            # L_man = u1*p11 + u2*p12 + u3*p21 + u4*p22 + b_man
            #
            # Model output:
            #
            # outputs = [L_cat, L_dog, L_man]
            #
          
            outputs = model(images)

            # ============================================================
            # WHAT THIS LINE DOES:
            #     loss = criterion(outputs, labels)
            # ============================================================
            #
            # Here:
            #   • criterion = nn.CrossEntropyLoss()
            #   • outputs  = model(images) → raw class scores (logits)
            #   • labels   = true class indices for each image in the batch
            #
            # ------------------------------------------------------------
            # SHAPES (SINGLE IMAGE EXAMPLE)
            # ------------------------------------------------------------
            #
            # Suppose:
            #
            #   • We pass ONE image through the model (batch_size = 1)
            #   • We have 3 classes: 0 = CAT, 1 = DOG, 2 = MAN
            #
            # Then:
            #
            #   outputs.shape = [1, 3]
            #   labels.shape  = [1]
            #
            # ------------------------------------------------------------
            # SHAPE EXPLANATION
            # ------------------------------------------------------------
            #
            # outputs.shape = [1, 3]
            #
            # means:
            #
            #   • 1 = number of images in this batch (batch_size = 1)
            #   • 3 = number of classes (for example: CAT, DOG, MAN)
            #
            # So outputs contains:
            #
            #   one row of predictions,
            #   and each row has ONE score for each class.
            #
            # Example:
            #
            #   outputs = [[2.4, 0.3, -1.2]]
            #
            # Interpretation:
            #
            #   outputs[0][0] → score for class CAT
            #   outputs[0][1] → score for class DOG
            #   outputs[0][2] → score for class MAN
            #
            # ------------------------------------------------------------
            # labels.shape = [1]
            #
            # means:
            #
            #   • There is ONE correct class label
            #     because there is ONE image in this batch.
            #
            # Example:
            #
            #   labels = [0]
            #
            # Meaning:
            #
            #   The true class for this image is:
            #       index 0 → CAT
            #
            # ------------------------------------------------------------
            # WHY THESE SHAPES MATCH
            # ------------------------------------------------------------
            #
            # For every row in outputs (one image),
            # there MUST be exactly ONE label.
            #
            # ------------------------------------------------------------
            # GENERAL RULE
            # ------------------------------------------------------------
            #
            # If batch_size = N and number_of_classes = C:
            #
            #   outputs.shape = [N, C]
            #   labels.shape  = [N]
            #
            # Example for batch_size = 5 and 3 classes:
            #
            #   outputs.shape = [5, 3]
            #   labels.shape  = [5]
            #
            # Means:
            #
            #   5 images → 5 prediction rows
            #   each row has 3 class scores
            # ------------------------------------------------------------

            # ------------------------------------------------------------
            # STEP 1: CrossEntropyLoss TAKES "outputs" AND "labels"
            # ------------------------------------------------------------
            #
            # When we call:
            #
            #   loss = criterion(outputs, labels)
            #
            # PyTorch does the following internally for EACH sample:
            #
            #   1) Applies softmax to the logits (outputs) to convert them
            #      into probabilities.
            #
            #   2) Selects the probability corresponding to the TRUE label.
            #
            #   3) Computes the negative log of that probability.
            #
            #   4) Averages over the batch (if batch_size > 1).
            #
            # ------------------------------------------------------------
            # STEP 2: SOFTMAX ON OUR EXAMPLE LOGITS
            # ------------------------------------------------------------
            #
            # outputs = [[2.4, 0.3, -1.2]]
            #
            # First, compute exponentials:
            #
            #   exp(2.4)  ≈ 11.02
            #   exp(0.3)  ≈  1.35
            #   exp(-1.2) ≈  0.30
            #
            # Sum them:
            #
            #   total = 11.02 + 1.35 + 0.30 = 12.67
            #
            # Probabilities:
            #
            #   P(CAT) = 11.02 / 12.67 ≈ 0.87
            #   P(DOG) =  1.35 / 12.67 ≈ 0.11
            #   P(MAN) =  0.30 / 12.67 ≈ 0.02
            #
            # ------------------------------------------------------------
            # STEP 3: USE THE TRUE LABEL (labels = [0])
            # ------------------------------------------------------------
            #
            # labels = [0] means:
            #   • The correct class for this image is index 0 → CAT.
            #
            # CrossEntropyLoss picks the probability of the true class:
            #
            #   P_true = P(CAT) = 0.87
            #
            # ------------------------------------------------------------
            # STEP 4: COMPUTE THE LOSS VALUE
            # ------------------------------------------------------------
            #
            # CrossEntropyLoss for this sample:
            #
            #   loss = -log(P_true)
            #   loss = -log(0.87)  ≈ 0.139
            #
            # Small loss → model is confident and correct.
            #
            # If the model was wrong / unsure (e.g., P_true ≈ 0.1),
            # then:
            #
            #   loss = -log(0.1) = 2.302  (much larger)
            #
            # Large loss → strong error signal for learning.
            #
            # ------------------------------------------------------------
            # BATCH EXAMPLE (TWO IMAGES)
            # ------------------------------------------------------------
            #
            # Suppose batch_size = 2, still 3 classes:
            #
            #   outputs =
            #     [
            #       [ 2.4,  0.3, -1.2],   # image 0: logits
            #       [-0.5,  1.7,  0.0]    # image 1: logits
            #     ]
            #
            #   labels = [0, 1]
            #
            # Meaning:
            #
            #   • image 0 → true class = CAT (0)
            #   • image 1 → true class = DOG (1)
            #
            # CrossEntropyLoss will:
            #
            #   • compute loss_0 from outputs[0] and label 0
            #   • compute loss_1 from outputs[1] and label 1
            #   • final loss = (loss_0 + loss_1) / 2
            #
            # ------------------------------------------------------------
            # HOW THIS RELATES TO THE IMAGE AND FILTERS
            # ------------------------------------------------------------
            #
            # For each training step:
            #
            #   1) 4x4 image → conv + padding 1 + 3x3 filters
            #   2) feature map → pooling → flattened feature vector
            #   3) fully connected layer → logits (outputs)
            #   4) loss = criterion(outputs, labels)
            #   5) loss.backward() → computes gradients
            #   6) optimizer.step() → updates filters + weights to reduce loss
            #
            # So THIS LINE:
            #
            #   loss = criterion(outputs, labels)
            #
            # is where we measure:
            #
            #   "How wrong were the predictions for this batch,
            #    given the true labels?"
            #
            # ============================================================
            # ================================================================
            # WHAT THIS LINE COMPUTES:
            # ================================================================
            #
            #    loss = criterion(outputs, labels)
            #
            # Where:
            #
            #   outputs → model logits (raw class scores)
            #   labels  → true class indices
            #
            # ================================================================
            # TENSOR SHAPES
            # ================================================================
            #
            #   outputs.shape = [N, C]
            #       N = batch size (number of images)
            #       C = number of classes
            #
            #   labels.shape  = [N]
            #       Each value = correct class index for each image
            #
            # ================================================================
            # MATHEMATICAL FORMULA (CrossEntropyLoss)
            # ================================================================
            #
            # For each image i (from 0 to N-1):
            #
            #   Step 1: Apply Softmax
            #
            #     i → index of the image in the batch
            #     j → index of the CLASS  
            #     P[i, j] = exp(outputs[i, j]) / Σ exp(outputs[i, k])  j is a class
            #                                       k = 0..C-1
            #
            #     → Converts raw logits into probabilities
            #
            # ------------------------------------------------
            # Step 2: Pick probability of the TRUE class
            #
            #     P_true = P[i, labels[i]]
            #
            # ------------------------------------------------
            # Step 3: Compute negative log-likelihood
            #
            #     Loss per image:
            #
            #         L[i] = -log(P_true)
            #
            # ------------------------------------------------
            # Step 4: Average across batch
            #
            #     Final loss:
            #
            #         loss = (1 / N) * Σ L[i]
            #                         i = 0..N-1
            #
            # ================================================================
            # COMPACT FORM:
            # ================================================================
            #
            #   loss = -(1/N) × Σ log( exp(Z[i, y[i]]) / Σ exp(Z[i, k]) )
            #                  i                 k
            #
            # Where:
            #
            #   Z = outputs logits
            #   y = labels ground-truth
            #   C = number of classes
            #   N = batch size
            #
            # ================================================================
            # INTERPRETATION:
            # ================================================================
            #
            # ✔ Large probability for correct class → LOW loss
            # ✔ Small probability for correct class → HIGH loss
            # ✔ Model is punished when it's wrong
            # ✔ Model is rewarded when it's confident and right
            #
            # ================================================================
            # IMPORTANT:
            # ================================================================
            #
            # PyTorch's CrossEntropyLoss automatically:
            #
            #   • Applies softmax
            #   • Computes -log
            #   • Computes batch mean
            #
            # So you MUST provide RAW logits (not probabilities)
            #
            # ================================================================

           

            loss = criterion(outputs, labels)


            # ----------------------------------------
            # BACKPROPAGATION
            #   Compute gradients for:
            #     • conv1 weights
            #     • conv2 weights
            #     • fully connected weights
            # ----------------------------------------
           # Perform BACKPROPAGATION.
            #
            # This line computes gradients for ALL trainable parameters in the network.
            #
            # What exactly is happening:
            #
            #   1) PyTorch walks backwards through the computation graph.
            #      This graph was created during the forward pass when:
            #          outputs = model(images)
            #          loss = criterion(outputs, labels)
            #      
            #
            #   2) Using the chain rule, it computes:
            #         ∂loss / ∂parameter
            #      for every weight and bias where requires_grad=True.
            #
            #   3) Each parameter tensor receives its gradient:
            #         parameter.grad   ←  gradient value
            #
            #   4) These gradients indicate:
            #         • direction to move the parameter
            #         • how large the update should be
            #
            #   5) No weights are updated yet.
            #       This ONLY computes gradients.
            #
            # How gradients flow:
            #   loss
            #     ↓
            #   classifier
            #     ↓
            #   conv layers
            #     ↓
            #   feature extraction filters
            #     ↓
            #   earliest layers (conv1)
            #
            # Each layer contributes using the chain rule:
            #   ∂loss/∂w = ∂loss/∂output × ∂output/∂w
            #
            # Importance:
            #   Without this line:
            #     optimizer.step()
            #   would have NOTHING to update.
            #
            # Memory note:
            #   • PyTorch frees the computation graph after backward() by default.
            #   • To reuse the graph, you would call:
            #         loss.backward(retain_graph=True)
            #
            # After this call:
            #   • parameter.grad contains new gradient values.
            #   • optimizer.step() can now APPLY updates.
            #
            # Errors you may see here:
            #   • RuntimeError: Trying to backward twice → graph already freed.
            #   • NaN gradients → exploding gradients or bad data.
            #   • No gradients appear → requires_grad=False issue.
            #
            # ------------------------------------------------------------
            # NUMERICAL EXAMPLE FOR loss.backward()
            # ------------------------------------------------------------
            #
            # PROBLEM:
            # --------
            # Classify images into 3 classes:
            #
            #   0 → CAT
            #   1 → DOG
            #   2 → MAN
            #
            # Batch size = 2   (two images in one training step)
            #
            # ------------------------------------------------------------
            # MODEL OUTPUTS (LOGITS)
            # ------------------------------------------------------------
            #
            # When you run:
            #
            #    outputs = model(images)
            #
            # Example numeric output:
            #
            #    outputs =
            #      [
            #        [ 2.0,  0.5, -1.0 ],    # image 0 (CAT score highest)
            #        [ 0.2,  1.8,  0.3 ]     # image 1 (DOG score highest)
            #      ]
            #
            # Shape:
            #
            #    outputs.shape = [2, 3]
            #
            #   2 = batch size
            #   3 = number of classes
            #
            # ------------------------------------------------------------
            # TRUE LABELS
            # ------------------------------------------------------------
            #
            # Suppose the correct answers:
            #
            #    labels = [0, 1]
            #
            # Meaning:
            #
            #   image 0 → CAT
            #   image 1 → DOG
            #
            # Shape:
            #
            #   labels.shape = [2]
            #
            # ------------------------------------------------------------
            # GRADIENT DERIVATION USING CHAIN RULE (SYMBOLIC FORM)
            # ------------------------------------------------------------
            #
            # We want:
            #
            #     ∂L / ∂W
            #
            # Where:
            #   L = loss
            #   W = any weight in the network (conv or fully connected)
            #
            # ------------------------------------------------------------
            # STEP 1 — Loss depends on LOGITS
            # ------------------------------------------------------------
            #
            # The loss L does NOT directly depend on W.
            # It depends on the logits Z.
            #
            # So we apply chain rule:
            #
            #     ∂L / ∂W = ∂L / ∂Z · ∂Z / ∂W
            #
            # ------------------------------------------------------------
            # STEP 2 — Logits are a function of weights
            # ------------------------------------------------------------
            #
            # For one neuron (or output unit):
            #
            #     Z = W·X + b
            #
            # Where:
            #   W = weight
            #   X = input feature vector
            #   b = bias
            #
            # Then:
            #
            #     ∂Z / ∂W = X
            #
            # ------------------------------------------------------------
            # STEP 3 — LOSS depends on probabilities (SOFTMAX)
            # ------------------------------------------------------------
            #
            # Softmax definition:
            #
            # Z[j] = logit (raw, unnormalized score) for class j
            #
            # Example with 3 classes:
            #
            #   Z[0] → Score for CAT
            #   Z[1] → Score for DOG
            #   Z[2] → Score for MAN
            #
            # These values come from:
            #
            #   Z = W · X + b
            #
            # They are NOT probabilities.
            # They are RAW confidence scores and will be converted
            # into probabilities by softmax inside CrossEntropyLoss。

            #     P[i] = exp(Z[i]) / Σ exp(Z[j])  i is the image index and j is the class index
            #
            # Loss definition:
            #
            #     L = - Σ y[i] · log(P[i])
            #
            # Where:
            #   y[j] = true class distribution
            #   (1 for correct class, 0 otherwise)
            #
            # ------------------------------------------------------------
            # STEP 4 — APPLY CHAIN RULE
            # ------------------------------------------------------------
            #
            # Expanded chain rule:
            #
            #     ∂L / ∂W
            #       = ∂L / ∂P · ∂P / ∂Z · ∂Z / ∂W
            #
            # ------------------------------------------------------------
            # EXPANDING THIS TERM USING THE CHAIN RULE:
            #
            #     ∂L / ∂P  ·  ∂P / ∂Z
            #
            # ------------------------------------------------------------
            # STEP 1 — DEFINE VARIABLES
            # ------------------------------------------------------------
            #
            # Let:
            #
            #   Z[j] = logit for class j
            #   P[j] = softmax probability for class j
            #   y[j] = true one-hot label:
            #          y[j] = 1 if j is correct class
            #          y[j] = 0 otherwise
            #
            #   Softmax:
            #
            #       P[j] = exp(Z[j]) / Σ exp(Z[k])
            #                              k
            #
            #   Loss function:
            #
            #       L = - Σ y[j] · log(P[j])
            #              j
            #
            # ------------------------------------------------------------
            # STEP 2 — PARTIAL DERIVATIVE: ∂L / ∂P
            # ------------------------------------------------------------
            #
            # Differentiate loss with respect to probability P[j]:
            #
            #       ∂L / ∂P[j] = - y[j] / P[j]
            #
            # Explanation:
            #
            # • If class j is correct → y[j] = 1 → loss depends on P[j]
            # • If class j is not correct → y[j] = 0 → no contribution
            #
            # ------------------------------------------------------------
            # STEP 3 — PARTIAL DERIVATIVE: ∂P / ∂Z (Softmax Jacobian)
            # ------------------------------------------------------------
            #
            # Softmax is a VECTOR FUNCTION, not scalar.
            # So its derivative is a MATRIX called the JACOBIAN:
            #
            #       ∂P[i] / ∂Z[j]
            #
            # Two cases:
            #
            # ------------------------------------------------
            # Case 1: i == j (diagonal term)
            #
            #     ∂P[j] / ∂Z[j]
            #       = P[j] · (1 - P[j])
            ## ============================================================
            # DERIVING THE SOFTMAX GRADIENT (STEP BY STEP)
            # ============================================================
            #
            # Softmax definition:
            #
            #   P[i] = exp(Z[i]) / Σ exp(Z[k])
            #                         k
            #
            # Meaning of symbols:
            #
            #   i = class index we are computing the probability for NOW
            #   k = class index we are SUMMING over
            # Let:
            #
            #   S = Σ exp(Z[k])
            #
            # so:
            #
            #   P[i] = exp(Z[i]) / S
            #
            # ============================================================
            # CASE 1: i == j   (DIAGONAL TERM)
            # ============================================================
            #
            # We differentiate P[j] with respect to Z[j]:
            #
            #       P[j] = exp(Z[j]) / S
            #
            # Use QUOTIENT RULE:
            #
            #       d/dx (f / g) = (g f' - f g') / g²
            #
            # Here:
            #
            #   f = exp(Z[j])
            #   g = S = Σ exp(Z[k])
            #
            # Derivatives:
            #
            #   df/dZ[j] = exp(Z[j])
            #
            #   dg/dZ[j] = exp(Z[j])    # only one term in sum depends on Z[j]
            #
            # Apply quotient rule:
            #
            #   ∂P[j] / ∂Z[j]
            #     = ( S·exp(Z[j]) - exp(Z[j])·exp(Z[j]) ) / S²
            #
            # Factor:
            #
            #     = exp(Z[j]) / S - exp(Z[j]) / S.exp(Z[j]) / S)
            #
            # Recognize:
            #
            #   exp(Z[j]) / S = P[j]
            #
            # So:
            #   ∂P[j] / ∂Z[j] = P[j] -P[J].P[j]
            #   ∂P[j] / ∂Z[j] = P[j] · (1 - P[j])
            #
            # ============================================================
            # CASE 2: i ≠ j   (OFF-DIAGONAL TERM)
            # ============================================================
            #
            # Now differentiate P[i] where i ≠ j:
            #
            #       P[i] = exp(Z[i]) / S
            #
            # Now numerator does NOT depend on Z[j]:
            #
            #   df/dZ[j] = 0
            #
            #   dg/dZ[j] = exp(Z[j])
            #
            # Apply quotient rule:
            #
            #   ∂P[i] / ∂Z[j]
            #     = (0·S - exp(Z[i])·exp(Z[j])) / S²
            #
            # Simplify:
            #
            #     = - (exp(Z[i]) / S) · (exp(Z[j]) / S)
            #
            # Recognize:
            #
            #   exp(Z[i]) / S = P[i]
            #   exp(Z[j]) / S = P[j]
            #
            # So:
            #
            #   ∂P[i] / ∂Z[j] = - P[i] · P[j]
            #
            # ============================================================
            # FINAL RESULT (JACOBIAN)
            # ============================================================
            #
            # Diagonal:
            #
            #     ∂P[j] / ∂Z[j] = P[j] · (1 - P[j])
            #
            # Off-diagonal:
            #
            #     ∂P[i] / ∂Z[j] = - P[i] · P[j]   for i ≠ j
            #
            # ============================================================
            # ------------------------------------------------
            # Case 2: i ≠ j (off-diagonal terms)
            #
            #     ∂P[i] / ∂Z[j]
            #       = - P[i] · P[j]
            #
            # ------------------------------------------------------------
            # STEP 4 — MULTIPLY VECTORS AND MATRICES (CHAIN RULE)
            # ------------------------------------------------------------
            # ============================================================
            # WHY DIAGONAL + OFF-DIAGONAL TERMS SIMPLIFY TO:
            #
            #     ∂L / ∂Z[j] = P[j] - y[j]
            #
            # ============================================================
            #
            # We start from the chain rule:
            #
            #     L = - Σ y[j] · log(P[i])
            #              i
            #     ∂L / ∂Z[j] = Σ ( ∂L / ∂P[i] ) · ( ∂P[i] / ∂Z[j] )
            #                   i
            #
            # We already know:
            #
            #   ∂L / ∂P[i] = - y[i] / P[i]
            #
            # Softmax derivatives:
            #
            #   If i == j (DIAGONAL):
            #
            #       ∂P[j] / ∂Z[j] = P[j] · (1 - P[j])
            #
            #   If i ≠ j (OFF-DIAGONAL):
            #
            #       ∂P[i] / ∂Z[j] = - P[i] · P[j]
            #
            # ============================================================
            # SPLIT SUM INTO TWO PARTS:
            # ============================================================
            #
            # One term where i == j
            # One sum over all i ≠ j
            #
            # So:
            #
            #   ∂L / ∂Z[j]
            #     = ( ∂L / ∂P[j] ) · ( ∂P[j] / ∂Z[j] )
            #       + Σ_{i ≠ j} ( ∂L / ∂P[i] ) · ( ∂P[i] / ∂Z[j] )
            #
            # ============================================================
            # SUBSTITUTE FORMULAS
            # ============================================================
            #
            # DIAGONAL TERM:
            #
            #   ( - y[j] / P[j] ) · ( P[j](1 - P[j]) )
            #
            # OFF-DIAGONAL SUM:
            #
            #   Σ_{i ≠ j} ( - y[i] / P[i] ) · ( - P[i] P[j] )
            #
            # ============================================================
            # SIMPLIFY EACH PART
            # ============================================================
            #
            # ---------- DIAGONAL ----------
            #
            #   ( - y[j] / P[j] ) · P[j](1 - P[j])
            #
            # Cancel P[j]:
            #
            #   = - y[j] (1 - P[j])
            #
            # ---------- OFF-DIAGONAL ----------
            #
            #   ( - y[i] / P[i] ) · ( - P[i] P[j] )
            #
            # Cancel minus signs and P[i]:
            #
            #   = y[i] · P[j]
            #
            # Now sum over i ≠ j:
            #
            #   Σ y[i] · P[j]
            #   = P[j] · Σ_{i ≠ j} y[i]
            #
            # ============================================================
            # USE ONE-HOT PROPERTY
            # ============================================================
            #
            # Because y is one-hot:
            #
            #   Σ y[i] = 1
            #
            # Therefore:
            #
            #   Σ_{i ≠ j} y[i] = 1 - y[j]
            #
            # ============================================================
            # PUT BOTH PARTS TOGETHER
            # ============================================================
            #
            #   ∂L / ∂Z[j]
            #     = -y[j] (1 - P[j])   +   P[j] (1 - y[j])
            #
            # Expand:
            #
            #   = -y[j] + y[j]P[j] + P[j] - y[j]P[j]
            #
            # Cancel middle terms:
            #
            #   = P[j] - y[j]
            #
            # ============================================================
            # FINAL RESULT
            # ============================================================
            #
            #   ∂L / ∂Z[j] = P[j] - y[j]
            #
            # ============================================================
            # INTERPRETATION
            # ============================================================
            #
            # If model probability is TOO BIG:
            #   P[j] > y[j]  → gradient is positive → push logit DOWN
            #
            # If model probability is TOO SMALL:
            #   P[j] < y[j]  → gradient is negative → push logit UP
            #
            # This is why cross-entropy + softmax is perfectly matched.
            # ============================================================
            #
            # ------------------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------------------
            #
            # This result happens because:
            #
            # • Softmax and cross-entropy are a matched pair
            # • Their gradients simplify beautifully
            # • The log cancels exp during differentiation
            #
            # ------------------------------------------------------------
            # FINAL CHAIN RULE CONNECTION:
            # ------------------------------------------------------------
            #
            #   ∂L / ∂W
            #       = ∂L / ∂Z · ∂Z / ∂W
            #       = (P - y) · Xᵀ
            #
            # ------------------------------------------------------------
               # ------------------------------------------------------------
            # FULL DERIVATIVE EQUATION
            # ------------------------------------------------------------
            #
            # Let:
            #   Z = W·X
            #   P = softmax(Z)
            #   y = ground-truth distribution
            #
            # Then:
            #
            #     ∂L / ∂W = (P - y) · Xᵀ
            #
            # ------------------------------------------------------------
            # FOR MULTIPLE SAMPLES (BATCH)
            # ------------------------------------------------------------
            #
            # For batch size N:
            #
            #     ∂L / ∂W
            #       = (1/N) · Σ (P[i] - y[i]) · X[i]ᵀ
            #               i
            #
            # ------------------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------------------
            #
            # • (P - y)  = error signal
            # • X        = input that caused the error
            # • The gradient W is pushed in a direction that:
            #     → decreases wrong scores
            #     → increases correct class score
            #
            # ------------------------------------------------------------
            # ============================================================
            # FULL SOFTMAX JACOBIAN + CHAIN RULE WITH CROSS ENTROPY
            # ============================================================
            #
            # GOAL:
            #   Understand how we go from:
            #
            #       L = CrossEntropy(softmax(Z), y)
            #
            #   to the very simple gradient:
            #
            #       ∂L / ∂Z = P - y
            #
            #   where:
            #       Z = logits (raw scores)   → shape [C]
            #       P = softmax probabilities → shape [C]
            #       y = one-hot true labels   → shape [C]
            #
            # ============================================================
            # 1) DEFINITIONS
            # ============================================================
            #
            # Let number of classes = C.
            #
            # Logits (vector):
            #
            #   Z = [Z[0], Z[1], ..., Z[C-1]]
            #
            # Softmax:
            #
            #   P[j] = exp(Z[j]) / Σ exp(Z[k])
            #                          k
            #
            # True label (one-hot):
            #
            #   y[j] = 1 if j is correct class
            #        = 0 otherwise
            #
            # Cross-entropy loss:
            #
            #   L = - Σ y[j] · log(P[j])
            #           j
            #
            # For a single sample (no batch).
            #
            # ============================================================
            # 2) JACOBIAN OF SOFTMAX: ∂P / ∂Z
            # ============================================================
            #
            # We consider P as function of Z:
            #
            #   P : R^C → R^C
            #
            # Its derivative is a C×C matrix (Jacobian):
            #
            #     J_softmax[i, j] = ∂P[i] / ∂Z[j]
            #
            # We derived:
            #
            #   For i == j (diagonal terms):
            #
            #       ∂P[j] / ∂Z[j] = P[j] · (1 - P[j])
            #
            #   For i ≠ j (off-diagonal terms):
            #
            #       ∂P[i] / ∂Z[j] = - P[i] · P[j]
            #
            # Matrix form (for C = 3, just as a picture):
            #
            #   [ ∂P[0]/∂Z[0]   ∂P[0]/∂Z[1]   ∂P[0]/∂Z[2] ]   [  P[0](1-P[0])   -P[0]P[1]      -P[0]P[2]   ]
            #   [ ∂P[1]/∂Z[0]   ∂P[1]/∂Z[1]   ∂P[1]/∂Z[2] ] = [ -P[1]P[0]       P[1](1-P[1])   -P[1]P[2]   ]
            #   [ ∂P[2]/∂Z[0]   ∂P[2]/∂Z[1]   ∂P[2]/∂Z[2] ]   [ -P[2]P[0]      -P[2]P[1]       P[2](1-P[2])]
            #
            # ------------------------------------------------------------
            # WHAT AMP (AUTOMATIC MIXED PRECISION) IS DOING INTERNALLY
            # ------------------------------------------------------------

            # 1️⃣ FORWARD PASS (autocast enabled)
            #
            # • Uses float16 where it is numerically safe (convolutions, matmul)
            # • Uses float32 where precision is required (BatchNorm, reductions)
            # • Leverages GPU Tensor Cores for much faster computation
            #

            # 2️⃣ scaler.scale(loss)
            #
            # • Multiplies the loss by a large scaling factor (e.g., 2^16)
            # • Prevents very small float16 gradients from underflowing to zero
            # • Ensures meaningful gradient values during backpropagation
            #

            # 3️⃣ scaler.step(optimizer)
            #
            # • Unscales gradients back to their true magnitude
            # • Checks gradients for NaN or Inf values
            # • If NaN/Inf detected → skips optimizer update (protects weights)
            # • If gradients are valid → applies optimizer.step() safely
            #

            # 4️⃣ scaler.update()
            #
            # • Automatically adjusts the scaling factor over time
            # • Increases scale when training is stable
            # • Decreases scale when numerical overflow is detected
            # • No manual tuning of scaling factor is required
            #
            # ------------------------------------------------------------

            # ----------------------------------------
            # BACKWARD PASS (AMP)
            # ----------------------------------------
            # • Scales the loss to prevent float16 underflow
            # • Computes gradients in scaled space
            # • Builds the backward graph once
            scaler.scale(loss).backward()

            # ----------------------------------------
            # OPTIMIZER STEP (AMP SAFE)
            # ----------------------------------------
            # • Unscales gradients
            # • Skips update if NaN/Inf detected
            # • Applies optimizer.step() internally
            scaler.step(optimizer)

            # ----------------------------------------
            # UPDATE SCALER
            # ----------------------------------------
            # • Adjusts scaling factor dynamically
            # • Increases scale if training is stable
            # • Decreases scale if overflow is detected
            scaler.update()

           
            # ----------------------------------------
            # STATISTICS
            # ----------------------------------------
            # Accumulate the total loss for the epoch (scaled by batch size).
            #
            # loss.item():
            #   • Extracts the numerical value from the loss tensor.
            #   • Removes it from the computation graph (no gradients attached).
            #
            # images.size(0):
            #   • Batch size (number of samples in this batch).
            #
            # Why multiply?
            #   • Because loss is averaged per-sample by default.
            #   • Multiplying converts it back to TOTAL loss for this batch.
            #   • Ensures correct averaging across batches of different sizes.
            #
            # running_loss:
            #   • Tracks TOTAL loss across all batches in the epoch.
            #
            # ------------------------------------------------------------
            # SIMPLE NUMERICAL EXAMPLE:
            # ------------------------------------------------------------
            #
            # Suppose:
            #
            #   batch_size = 4 images
            #   loss.item() = 0.5   (this is the average loss per image)
            #
            # Then:
            #
            #   total_batch_loss = loss.item() * batch_size
            #                    = 0.5 * 4
            #                    = 2.0
            #
            # So:
            #
            #   running_loss += 2.0
            #
            # ------------------------------------------------------------
            # MULTIPLE BATCHES EXAMPLE:
            # ------------------------------------------------------------
            #
            # Assume dataset has 10 images with batch_size = 4:
            #
            #   Batch 1: 4 images, loss.item() = 0.6 → 0.6 * 4 = 2.4
            #   Batch 2: 4 images, loss.item() = 0.8 → 0.8 * 4 = 3.2
            #   Batch 3: 2 images, loss.item() = 0.5 → 0.5 * 2 = 1.0
            #
            # Total running_loss = 2.4 + 3.2 + 1.0 = 6.6
            #
            # ------------------------------------------------------------
            # FINAL EPOCH LOSS (AVERAGE):
            # ------------------------------------------------------------
            #
            # To compute average loss per image:
                       #
            #   average_epoch_loss = running_loss / total_images
            #                      = 6.6 / 10
            #                      = 0.66
            #
            running_loss += loss.item() * images.size(0)


            # Compute predicted class labels from model outputs.
            #
            # outputs:
            #   • Tensor shape: (batch_size, num_classes)
            #   • Contains raw logits for each class.
            #
            # argmax(1):
            #   • Selects the class with the highest score for each sample.
            #   • Dimension "1" means across class scores.
            #
            # preds:
            #   • Tensor shape: (batch_size)
            #   • Each value is the predicted class index.
            #
            # ------------------------------------------------------------
            # NUMERICAL EXAMPLE:
            # ------------------------------------------------------------
            #
            # Assume 3 classes:
            #
            #   Class 0 → CAT
            #   Class 1 → DOG
            #   Class 2 → MAN
            #
            # Batch size = 2 images
            #
            # outputs =
            # [
            #   [ 2.5,  1.0, -0.5 ],   # Image 0 scores (logits)
            #   [ 0.2,  3.1,  0.8 ]    # Image 1 scores (logits)
            # ]
            #
            # ------------------------------------------------------------
            # APPLY ARGMAX over DIMENSION=1 (classes):
            # ------------------------------------------------------------
            #
            # For Image 0:
            #   CAT = 2.5
            #   DOG = 1.0
            #   MAN = -0.5
            #
            #   Highest value = 2.5 (CAT)
            #
            #   Prediction = class 0
            #
            # For Image 1:
            #   CAT = 0.2
            #   DOG = 3.1
            #   MAN = 0.8
            #
            #   Highest value = 3.1 (DOG)
            #
            #   Prediction = class 1
            #
            # ------------------------------------------------------------
            # RESULT:
            # ------------------------------------------------------------
            #
            # preds = [0, 1]
            #
            # ------------------------------------------------------------
            # INTERPRETATION:
            # ------------------------------------------------------------
            #
            # preds[i] = predicted class index for image i
            #
            # Image 0 → CAT
            # Image 1 → DOG
            #
            preds = outputs.argmax(1)


            # Count how many predictions are correct in this batch.
            #
            # preds == labels:
            #   • Performs element-wise comparison.
            #   • Result is a Boolean tensor:
            #       True  → correct prediction
            #       False → wrong prediction
            #
            # .sum():
            #   • Counts how many True values are present.
            #
            # .item():
            #   • Converts the count tensor into a Python integer.
            #
            # correct:
            #   • Accumulates number of correct predictions across all batches.
            #
            # ------------------------------------------------------------
            # LINE EXPLAINED:
            # ------------------------------------------------------------
            #
            #   correct += (preds == labels).sum().item()
            #
            # ------------------------------------------------------------
            # ASSUME BATCH SIZE = 5
            # ------------------------------------------------------------
            #
            # preds  → predicted classes from the model (argmax result)
            # labels → true class labels from dataset
            #
            # Example:
            #
            # preds  = [0, 2, 1, 1, 0]   # model predictions
            # labels = [0, 1, 1, 2, 0]   # true labels
            #
            # ------------------------------------------------------------
            # STEP 1: ELEMENTWISE COMPARISON
            # ------------------------------------------------------------
            #
            # preds == labels  →
            #
            # [ True, False, True, False, True ]
            #
            # Explanation:
            #
            #   Image 0 → 0 == 0 ✅
            #   Image 1 → 2 != 1 ❌
            #   Image 2 → 1 == 1 ✅
            #   Image 3 → 1 != 2 ❌
            #   Image 4 → 0 == 0 ✅
            #
            # ------------------------------------------------------------
            # STEP 2: CONVERT TRUE/FALSE TO NUMBERS
            # ------------------------------------------------------------
            #
            # PyTorch treats:
            #
            #   True  → 1
            #   False → 0
            #
            # So tensor becomes:
            #
            # [ 1, 0, 1, 0, 1 ]
            #
            # ------------------------------------------------------------
            # STEP 3: SUM THE VALUES
            # ------------------------------------------------------------
            #
            # (preds == labels).sum() =
            #
            # 1 + 0 + 1 + 0 + 1 = 3
            #
            # ------------------------------------------------------------
            # STEP 4: CONVERT TO PYTHON NUMBER
            # ------------------------------------------------------------
            #
            # .item() converts tensor → Python integer:
            #
            # 3
            #
            # ------------------------------------------------------------
            # STEP 5: ADD TO TOTAL CORRECT COUNTER
            # ------------------------------------------------------------
            #
            # If correct was previously:
            #
            # correct = 10
            #
            # After:
            #
            # correct += 3
            #
            # New value:
            #
            # correct = 13
            #
            # ------------------------------------------------------------
            # FINAL MEANING:
            # ------------------------------------------------------------
            #
            # This line:
            #
            # ✅ counts how many predictions were correct in the current batch
            # ✅ adds them to the total correct across all batches
            #
            # ------------------------------------------------------------
            # USED LATER FOR ACCURACY:
            # ------------------------------------------------------------
            #
            # Accuracy = correct / total
            #
            # Example:
            #
            # correct = 130
            # total   = 200
            #
            # accuracy = 130 / 200 = 0.65 = 65%

            correct += (preds == labels).sum().item()


            # Count how many total samples have been evaluated.
            #
            # labels.size(0):
            #   • Number of samples in this batch.
            #
            # total:
            #   • Accumulates TOTAL number of samples processed in the epoch.
            #
            total += labels.size(0)

        # --------------------------------------------------------
        # END TIMER FOR THIS EPOCH
        #   • Compute how long this epoch took in seconds.
        #   • Store duration so we can summarize later.
        # --------------------------------------------------------
        epoch_time = time.perf_counter() - epoch_start
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
        # HOW THE LEARNING RATE (LR) IMPROVES TRAINING
        # ------------------------------------------------------------
        #
        # The optimizer (Adam, SGD, etc.) updates the model weights using:
        #
        #       new_weight = old_weight − LR * gradient
        #
        # The **learning rate (LR)** controls how BIG each update step is.
        #
        # ------------------------------------------------------------
        # PHASE 1 — EARLY TRAINING (LR is HIGH)
        # ------------------------------------------------------------
        # • At the beginning of training, we WANT large updates.
        # • The loss surface is rough and gradients are strong.
        # • A higher LR helps the model quickly move toward good regions.
        #
        # Example:
        #   LR = 0.001  → fast improvement during first 20–30 epochs
        #
        # ------------------------------------------------------------
        # PHASE 2 — MID TRAINING (LR TOO HIGH TO IMPROVE)
        # ------------------------------------------------------------
        # Eventually the model reaches a “plateau”:
        #
        #   Epoch 70 → loss = 0.43
        #   Epoch 71 → loss = 0.43
        #   Epoch 72 → loss = 0.43
        #
        # Loss stops improving because:
        #   → LR is now TOO LARGE to make fine updates.
        #
        # The optimizer jumps OVER the small valleys where the true minimum is.
        #
        # ------------------------------------------------------------
        # PHASE 3 — LR Scheduler Reduces LR for FINE TUNING
        # ------------------------------------------------------------
        # ReduceLROnPlateau detects this plateau.
        #
        # If `epoch_loss` does NOT improve for `patience` epochs:
        #
        #       new_lr = old_lr * factor
        #
        # With factor=0.5:
        #
        #       0.001   → 0.0005   → 0.00025   → 0.000125 → ...
        #
        # When LR becomes smaller:
        #   • Weight updates become more precise.
        #   • The optimizer no longer overshoots minima.
        #   • Loss begins to decrease again (fine convergence).
        #
        # RESULT:
        #   → Lower final loss
        #   → Higher accuracy
        #   → More stable training
        #
        # ------------------------------------------------------------
        # HOW scheduler.step(epoch_loss) WORKS:
        # ------------------------------------------------------------
        # When called every epoch:
        #
        #   scheduler.step(epoch_loss)
        #
        # The scheduler:
        #   • Monitors the value of epoch_loss
        #   • Remembers the BEST (lowest) loss so far
        #   • If no improvement for `patience` epochs → reduce LR
        #
        # ------------------------------------------------------------
        # WHY MANUAL LR LOGGING?
        # ------------------------------------------------------------
        # PyTorch ≥ 2.0 removed verbose=True.
        # We print LR manually to track scheduler actions:
        #
        #   current_lr = optimizer.param_groups[0]['lr']
        #   debug_print(f"[LR Scheduler] Current Learning Rate = {current_lr:.6f}")
        #
        # This lets you SEE when LR drops, which helps with debugging and tuning.
        # ------------------------------------------------------------

        if scheduler is not None:
            scheduler.step(epoch_loss)

            # Manual LR logging
            current_lr = optimizer.param_groups[0]['lr']
            debug_print(f"[LR Scheduler] Current Learning Rate = {current_lr:.6f}")

    # ------------------------------------------------------------
    # OPTIONAL: PRINT TOTAL AND AVERAGE EXECUTION TIME
    #   • total_time: sum of all epoch durations.
    #   • avg_time:   mean seconds per epoch.
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

    # Add batch dimension → shape becomes [1, C, H, W]
    img_input = img.unsqueeze(0).to(device)

    # --------------------------------------------------------
    # FORWARD PASS (NO GRADIENT TRACKING)
    # --------------------------------------------------------
    with torch.no_grad():
        logits = model(img_input)
        pred_label = logits.argmax(1).item()

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
    print(f"True label index : {true_label_id} → {true_name}")
    print(f"Pred label index : {pred_label} → {pred_name}")
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


def debug_print(msg: str):
    """Print debug messages only when DEBUG_FLAG is True."""
    if DEBUG_FLAG:
        print(msg)

def main():

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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

    print("Training images from:", train_path)
    print("Testing  images from:", test_path)

    # --------------------------------------------------------
    # DATA TRANSFORMS FOR YOUR DATA
    # --------------------------------------------------------
    # We now keep the ORIGINAL image size (no Resize here).
    # For training:
    #   • RandomHorizontalFlip → data augmentation (mirroring)
    #   • RandomCrop(32, padding=4) → CIFAR-style jitter (if images >= 32x32)
    #   • ToTensor + Normalize → standard scaling to [-1, 1]
    #
    # For testing:
    #   • No augmentation (only ToTensor + Normalize)
    #
    # If your images are exactly 32x32 (CIFAR), this behaves like
    # standard augmentation. If they are larger, crop will take 32x32
    # patches. If you want to keep full resolution, remove RandomCrop.
    # --------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),      # mirror images randomly
        transforms.RandomCrop(32, padding=4),        # CIFAR-style random crop
        transforms.ToTensor(),                       # convert to [C, H, W] in [0, 1]
        transforms.Normalize(                        # normalize to [-1, 1]
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),                       # convert to [C, H, W] in [0, 1]
        transforms.Normalize(                        # normalize to [-1, 1]
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
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
    print("Number of classes detected in train:", num_classes)
    print("Class names:", train_dataset.classes)

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
        print(f"Loading trained weights from: {model_filename}")
        state_dict = torch.load(model_filename, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("No saved model found. Training a new model...")
        model = train_model(model, train_loader, device, num_epochs=NUM_EPOCHS, lr=1e-3)
        print(f"Saving trained model to: {model_filename}")
        torch.save(model.state_dict(), model_filename)

    # ------------------------------------------------------------
    # INTERACTIVE LOOP FOR USER-DRIVEN DETECTION
    # ------------------------------------------------------------
    #import msvcrt

    print("\n--------------------------------------------------")
    print("Interactive Image Detection Mode")
    print("You are now ALWAYS in detection mode.")
    print("Just type an image index and press ENTER.")
    print("Press 'e' at any time to exit.")
    print("--------------------------------------------------\n")

    while True:

        print(f"Enter image index (0 – {len(test_dataset)-1}) or 'e' to exit: ", end="", flush=True)

        # READ ONE CHARACTER WITHOUT PRESSING ENTER
        key = msvcrt.getch().decode().lower()

        # IF USER PRESSES 'e' → EXIT IMMEDIATELY
        if key == 'e':
            print("e")
            print("Exiting program. Goodbye!")
            break

        # If first key is NOT a digit → invalid
        if not key.isdigit():
            print(key)
            print("❌ Invalid input. Enter a number or 'e' to exit.")
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
                c = ch.decode()
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
