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
NUM_EPOCHS = 150
#LEARNING_RATE = 0.001


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
   def __init__(self, num_classes: int = 10, num_anchors: int = 3):
    super().__init__()

    # --------------------------------------------------------
    # cuDNN AUTOTUNER
    # --------------------------------------------------------
    # cuDNN benchmark is fastest when input sizes are CONSTANT.
    # For object detection, input images may have varying H,W.
    #
    # If your pipeline resizes all images to a fixed size
    # (e.g., 640x640), you can set this to True for speed.
    #
    # If your images vary in size frequently, keep False to avoid
    # repeated benchmarking overhead.
    # --------------------------------------------------------
    torch.backends.cudnn.benchmark = False  # safer for variable-size detection inputs

    # ------------------------------------------------------
    # LAYER 1: 3 → 16 channels
    # ------------------------------------------------------
    # IMPORTANT CHANGE FOR DETECTION:
    #   We do NOT assume [B, 3, 32, 32] anymore.
    #   We assume: [B, 3, H, W]  where H and W can be any size.
    #
    # Padding=1 keeps spatial size at this stage:
    #   [B, 3, H, W] → [B, 16, H, W]
    # ------------------------------------------------------
    self.conv1 = nn.Conv2d(
        in_channels=3,          # RGB input
        out_channels=16,        # feature maps produced
        kernel_size=3,          # 3x3 convolution kernel
        padding=1,              # keep H,W unchanged for this layer
        bias=False              # bias not needed when BatchNorm is used
    )

    # ------------------------------------------------------
    # BatchNorm for conv1 (normalizes 16 output channels)
    # ------------------------------------------------------
    self.bn1 = nn.BatchNorm2d(16)

    # ------------------------------------------------------
    # LAYER 2: 16 → 32 channels
    # ------------------------------------------------------
    # Still keeps spatial size at this conv stage:
    #   [B, 16, H/2, W/2] → [B, 32, H/2, W/2] (after pooling)
    # ------------------------------------------------------
    self.conv2 = nn.Conv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        padding=1,
        bias=False
    )

    # ------------------------------------------------------
    # BatchNorm for conv2 (normalizes 32 channels)
    # ------------------------------------------------------
    self.bn2 = nn.BatchNorm2d(32)

    # ------------------------------------------------------
    # OPTIONAL (RECOMMENDED) EXTRA DEPTH FOR DETECTION
    # ------------------------------------------------------
    # Detection usually needs more capacity than classification.
    # Adding a 3rd conv block improves feature richness, especially
    # for localization and objectness prediction.
    # ------------------------------------------------------
    self.conv3 = nn.Conv2d(
        in_channels=32,
        out_channels=64,        # deeper features for detection head
        kernel_size=3,
        padding=1,
        bias=False
    )
    self.bn3 = nn.BatchNorm2d(64)

    # ------------------------------------------------------
    # POOLING LAYER: MaxPool2d(2, 2)
    # ------------------------------------------------------
    # Each pooling halves spatial resolution:
    #   [B, C, H, W] → [B, C, H/2, W/2]
    #
    # After conv1 + pool: [B, 16, H/2, W/2]
    # After conv2 + pool: [B, 32, H/4, W/4]
    # After conv3 + pool: [B, 64, H/8, W/8]
    #
    # This final (H/8, W/8) grid becomes the "prediction grid".
    # ------------------------------------------------------
    self.pool = nn.MaxPool2d(2, 2)

    # ------------------------------------------------------
    # ACTIVATION
    # ------------------------------------------------------
    # ReLU is fine. Many detection models also use LeakyReLU or SiLU.
    # ------------------------------------------------------
    self.act = nn.ReLU(inplace=True)

    # ======================================================
    # DETECTION HEAD (YOLO-STYLE GRID PREDICTION)
    # ======================================================
    # Instead of a fully-connected classifier, we output predictions
    # at each spatial grid location.
    #
    # For each grid cell and each anchor, predict:
    #   • 4 box values:     (tx, ty, tw, th)
    #   • 1 objectness:     (to)  → probability an object exists here
    #   • num_classes scores: class logits
    #
    # Per-anchor prediction dimension:
    #   pred_dim = 5 + num_classes
    #
    # Total output channels for the head:
    #   out_channels = num_anchors * (5 + num_classes)
    #
    # Because this is Conv2d, it works for ANY H,W.
    # Output feature map size becomes:
    #   [B, num_anchors*(5+num_classes), H/8, W/8]
    # ======================================================
    self.num_classes = num_classes                 # number of classes in dataset
    self.num_anchors = num_anchors                 # anchors per grid cell (YOLO style)
    self.pred_dim = 5 + num_classes                # (box4 + obj1 + class_scores)

    self.detect_head = nn.Conv2d(
        in_channels=64,                            # comes from conv3 feature maps
        out_channels=num_anchors * self.pred_dim,  # predictions per grid cell
        kernel_size=1,                             # 1x1 conv = per-location predictor
        padding=0,                                 # no padding needed for 1x1 conv
        bias=True                                  # bias OK for output head
    )

    # ------------------------------------------------------
    # STATIC FILTER INITIALIZATION (if enabled)
    # ------------------------------------------------------
    # NOTE FOR DETECTION:
    #   Static filters can help early edges, but they can also reduce
    #   the detector’s ability to learn task-specific localization.
    #
    # Recommended:
    #   • Apply static filters ONLY to conv1
    #   • Let conv2/conv3/detect_head learn freely
    # ------------------------------------------------------
    if STATIC_FILTERS:
        self._init_conv1_static()
        # self._init_conv2_static()  # optional; often better to leave learnable for detection


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

    def forward(self, x):
        # At entry:
        #   x shape → [B, 3, H, W]  (variable H, W, e.g., 256x256, 512x512, etc.)
        #   B = batch size, 3 = RGB channels

        # -------------------
        # BLOCK 1: CONV1 → BN1 → ReLU → POOL
        # -------------------
        # Conv1: 3 → 16 channels, preserves H, W (padding keeps size)
        #   [B, 3, H, W] → [B, 16, H, W]
        x = self.conv1(x)

        # BatchNorm for Conv1 (stabilizes activations)
        x = self.bn1(x)

        # Non-linearity: ReLU activation
        x = F.relu(x)

        # MaxPool: [B, 16, H, W] → [B, 16, H/2, W/2]
        x = self.pool(x)

        # -------------------
        # BLOCK 2: CONV2 → BN2 → ReLU → POOL
        # -------------------
        # Conv2: 16 → 32 channels
        #   [B, 16, H/2, W/2] → [B, 32, H/2, W/2]
        x = self.conv2(x)

        # BatchNorm for Conv2
        x = self.bn2(x)

        # ReLU activation
        x = F.relu(x)

        # MaxPool: [B, 32, H/2, W/2] → [B, 32, H/4, W/4]
        x = self.pool(x)

        # -------------------
        # BLOCK 3: CONV3 → BN3 → ReLU → POOL (optional depth for detection)
        # -------------------
        # Conv3: 32 → 64 channels (helps with feature richness)
        #   [B, 32, H/4, W/4] → [B, 64, H/4, W/4]
        x = self.conv3(x)

        # BatchNorm for Conv3
        x = self.bn3(x)

        # ReLU activation
        x = F.relu(x)

        # MaxPool: [B, 64, H/4, W/4] → [B, 64, H/8, W/8]
        x = self.pool(x)

        # -------------------
        # DETECTION HEAD
        # -------------------
        # Predict bounding boxes (tx, ty, tw, th), objectness (to), and class scores
        #   Output shape: [B, num_anchors * (5 + num_classes), H/8, W/8]
        # Where:
        #   5 = (tx, ty, tw, th, objectness)
        #   num_classes = number of classes (num_class in dataset)
        #   num_anchors = how many bounding boxes to predict per grid cell
        x = self.detect_head(x)

        # ------------------------------------------------------
        # OUTPUT:
        #   - Predicted grid (size H/8, W/8) with multiple anchors
        #   - Each grid cell predicts (tx, ty, tw, th, to) + class scores
        #
        #   Grid prediction is returned in this format:
        #     [B, num_anchors * (5 + num_classes), H/8, W/8]
        return x



def debug_print(*args, **kwargs):
    """
    Simple debug print wrapper.
    If DEBUG is True → behaves like print().
    If DEBUG is False → does nothing.
    """
    if DEBUG_FLAG:
        print(*args, **kwargs)

def train_model(model, train_loader, device, num_epochs=2, lr=3e-3):

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
    # OPTIMIZER: UPDATES ALL LEARNABLE PARAMETERS
    # ------------------------------------------------------------
    # (keeping your optimizer explanation exactly as-is)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------
    # LEARNING RATE SCHEDULER — ReduceLROnPlateau
    # ------------------------------------------------------------
    # (keeping your scheduler explanation exactly as-is)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,              # ✅ The optimizer whose learning rate we want to change (Adam/SGD/etc.).
                                #    The scheduler will edit: optimizer.param_groups[i]["lr"]

        mode="min",             # ✅ What direction is “better” for the monitored metric?
                                #    • "min"  → lower metric is better (typical for loss)
                                #    • "max"  → higher metric is better (typical for accuracy)

        factor=0.5,             # ✅ How much to reduce LR when plateau is detected:
                                #    new_lr = old_lr * factor

        patience=2,             # ✅ How many epochs to wait WITHOUT meaningful improvement
                                #    before reducing LR.

        threshold=1e-3,         # ✅ Minimum change required to count as an “improvement”.

        threshold_mode="rel",   # ✅ How to interpret the threshold:
                                #    • "rel" (relative)
                                #    • "abs" (absolute)

        min_lr=1e-6             # ✅ Lower bound (floor) for LR.
    )

    # ============================================================
    # COMPLETE END-TO-END EXPLANATION:
    # IMAGE → CONVOLUTION → FEATURES → LOGITS → CrossEntropyLoss
    # ============================================================
    #
    # ✅ KEEPING YOUR FULL EXPLANATION BLOCK AS-IS.
    #
    # NOTE FOR OBJECT DETECTION:
    # --------------------------
    # In object detection, the model output is NOT a single [B, num_classes] vector.
    # Instead, it outputs a grid/anchors of:
    #   • bounding box parameters
    #   • objectness score
    #   • class logits per predicted box
    #
    # CrossEntropyLoss is still commonly used INSIDE the detection loss
    # for the classification component (loss_cls), but it is NOT the only loss.
    #
    # Your explanation remains valid for the "classification part" of detection.
    # ============================================================

    criterion = nn.CrossEntropyLoss()  # ✅ still useful if your detection_loss uses CE for class loss

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
        # --------------------------------------------------------
        epoch_start = time.perf_counter()

        # Track statistics over the epoch
        #
        # IMPORTANT FOR OBJECT DETECTION:
        # -------------------------------
        # We DO NOT compute "classification accuracy" (preds == labels) anymore,
        # because each image can contain multiple objects and multiple boxes.
        #
        # Instead, we track:
        #   • total detection loss
        #   • box regression loss
        #   • objectness loss
        #   • class loss
        #
        running_loss_total = 0.0
        running_loss_box   = 0.0
        running_loss_obj   = 0.0
        running_loss_cls   = 0.0

        total_images = 0  # number of images processed in this epoch (for averaging losses)

        # --------------------------------------------
        # LOOP THROUGH MINI-BATCHES
        # --------------------------------------------
        #
        # CHANGE FOR OBJECT DETECTION:
        # ----------------------------
        # train_loader MUST return:
        #
        #   images  : Tensor [B, 3, H, W]
        #   targets : list length B, each element is a dict like:
        #       {
        #         "boxes":  Tensor [N, 4]  (x1,y1,x2,y2 OR xc,yc,w,h)
        #         "labels": Tensor [N]
        #       }
        #
        # Because N (number of objects) is different per image,
        # targets is typically a list (not a stacked tensor).
        #
        for images, targets in train_loader:

            # Move batch to device (GPU/CPU)
            images = images.to(device)

            # Move target tensors to device
            #
            # targets is a list of dicts, so we move each dict's tensors:
            targets = [
                {
                    "boxes":  t["boxes"].to(device),
                    "labels": t["labels"].to(device)
                }
                for t in targets
            ]

            # Track how many images we processed (used for correct averaging)
            batch_size = images.size(0)
            total_images += batch_size

            # ----------------------------------------
            # CLEAR OLD GRADIENTS
            # ----------------------------------------
            optimizer.zero_grad()

            # ------------------------------------------------------------
            # AMP FORWARD PASS (autocast)
            # ------------------------------------------------------------
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):

                # ============================================================
                # WHAT THIS LINE DOES:
                #     outputs = model(images)
                # ============================================================
                #
                # OBJECT DETECTION NOTE:
                # ----------------------
                # For detection, outputs is NOT [B, num_classes].
                #
                # Typical output might be:
                #   [B, A*(5+C), Hgrid, Wgrid]
                #
                # Where:
                #   A = anchors per grid cell
                #   5 = (tx,ty,tw,th,obj)
                #   C = num_classes
                #
                outputs = model(images)

                # ============================================================
                # OBJECT DETECTION LOSS (REPLACES: loss = criterion(outputs, labels))
                # ============================================================
                #
                # For detection, we compute a COMPOSITE loss:
                #
                #   loss_total = loss_box + loss_obj + loss_cls
                #
                # Where:
                #   loss_box : bounding box regression loss (IoU/GIoU/DIoU/CIoU/L1)
                #   loss_obj : objectness loss (BCEWithLogits)
                #   loss_cls : classification loss for boxes (CE or BCE depending on setup)
                #
                # IMPORTANT:
                # ----------
                # You must implement detection_loss() to:
                #   • match GT boxes to grid/anchors
                #   • build training targets
                #   • compute losses from outputs vs targets
                #
                # It should return 3 tensors (or 4 if you want extra terms).
                #
                loss_box, loss_obj, loss_cls = detection_loss(
                    outputs=outputs,
                    targets=targets,
                    num_classes=getattr(model, "num_classes", None),
                    criterion_cls=criterion
                )

                # Total detection loss (this is what we backprop)
                loss = loss_box + loss_obj + loss_cls

            # ----------------------------------------
            # BACKWARD PASS (AMP)
            # ----------------------------------------
            scaler.scale(loss).backward()

            # ----------------------------------------
            # OPTIMIZER STEP (AMP SAFE)
            # ----------------------------------------
            scaler.step(optimizer)

            # ----------------------------------------
            # UPDATE SCALER
            # ----------------------------------------
            scaler.update()

            # ----------------------------------------
            # STATISTICS
            # ----------------------------------------
            #
            # KEEPING YOUR "loss.item() * images.size(0)" IDEA:
            # -------------------------------------------------
            # We do the same weighting by batch size so epoch averages are correct.
            #
            running_loss_total += loss.item() * batch_size
            running_loss_box   += loss_box.item() * batch_size
            running_loss_obj   += loss_obj.item() * batch_size
            running_loss_cls   += loss_cls.item() * batch_size

        # --------------------------------------------------------
        # END TIMER FOR THIS EPOCH
        # --------------------------------------------------------
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        # --------------------------------------------------------
        # COMPUTE AVERAGE LOSSES FOR THIS EPOCH
        # --------------------------------------------------------
        #
        # For detection, we report the average loss per image:
        #
        #   epoch_loss_total = running_loss_total / total_images
        #
        epoch_loss_total = running_loss_total / max(1, total_images)
        epoch_loss_box   = running_loss_box   / max(1, total_images)
        epoch_loss_obj   = running_loss_obj   / max(1, total_images)
        epoch_loss_cls   = running_loss_cls   / max(1, total_images)

        # --------------------------------------------
        # PRINT EPOCH SUMMARY
        # --------------------------------------------
        debug_print(
            f"[TRAIN-DETECT] Epoch {ep+1}/{num_epochs}  "
            f"Loss: {epoch_loss_total:.4f}  "
            f"(box={epoch_loss_box:.4f}, obj={epoch_loss_obj:.4f}, cls={epoch_loss_cls:.4f})  "
            f"Time: {epoch_time:.2f} sec"
        )

        # ------------------------------------------------------------
        # HOW scheduler.step(epoch_loss) WORKS:
        # ------------------------------------------------------------
        # For detection, we step using TOTAL detection loss.
        # (Your LR comments remain valid.)
        # ------------------------------------------------------------
        if scheduler is not None:
            scheduler.step(epoch_loss_total)

            # Manual LR logging
            current_lr = optimizer.param_groups[0]['lr']
            debug_print(f"[LR Scheduler] Current Learning Rate = {current_lr:.6f}")

    # ------------------------------------------------------------
    # OPTIONAL: PRINT TOTAL AND AVERAGE EXECUTION TIME
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
# DETECTION / SINGLE-IMAGE INFERENCE FUNCTION (OBJECT DETECTION)
# ============================================================
def detect_single_image(model, test_dataset, device, index=None, score_thresh=0.25, draw=True, max_dets=50):
    """
    Loads ONE RANDOM image from the test dataset (unless index is provided),
    runs the DETECTION model, and prints / returns:

        • Ground-truth boxes + labels (if dataset provides them)
        • Predicted boxes + scores + labels

    Args:
        model ........ the trained PyTorch DETECTOR (not classifier)
        test_dataset . detection dataset
                      Expected __getitem__ returns either:
                        (img, target_dict)  where target_dict has:
                            • "boxes": Tensor [N, 4]
                            • "labels": Tensor [N]
                      OR (img, label) for classification (we handle gracefully).
        device ....... "cuda" or "cpu"
        index ........ optional fixed index; if None → choose random image
        score_thresh . filter predictions below this confidence
        draw ......... if True, draw predicted boxes on the image (PIL)
        max_dets ..... limit number of drawn/printed detections

    Returns:
        img, target, predictions
        where predictions is a dict:
            {
              "boxes":  Tensor [M, 4],
              "scores": Tensor [M],
              "labels": Tensor [M]
            }
    """

    import random
    import torch

    # --------------------------------------------------------
    # MOVE MODEL TO DEVICE AND SWITCH TO EVAL MODE
    # --------------------------------------------------------
    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # AUTO-DETECT CLASS NAMES (works for many datasets)
    # --------------------------------------------------------
    class_names = getattr(test_dataset, "classes", None)
    if class_names is None:
        # Fallback: no classes attribute found
        # NOTE: detection datasets often have custom label maps; adjust as needed.
        class_names = [str(i) for i in range(1000)]

    # --------------------------------------------------------
    # NORMALIZE & VALIDATE INDEX
    #   • If index is None → choose random
    #   • If index is string → convert to int
    #   • Clamp / reject out-of-range indices
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
            print(f"[detect_single_image] Index {index} is out of range 0–{len(test_dataset) - 1}, using 0 instead.")
            index = 0

    # --------------------------------------------------------
    # LOAD IMAGE + TARGET
    # --------------------------------------------------------
    sample = test_dataset[index]

    # Many DETECTION datasets return: (img, target_dict)
    # CIFAR-style classification returns: (img, label)
    if isinstance(sample, (tuple, list)) and len(sample) == 2:
        img, target = sample
    else:
        # Defensive fallback for unusual datasets
        img = sample
        target = None

    # Ensure image is a Tensor [C, H, W]
    # (Most torchvision transforms return Tensor already.)
    if not torch.is_tensor(img):
        raise TypeError("Expected img to be a Tensor [C,H,W]. Ensure your dataset transform includes ToTensor().")

    # Add batch dimension → [1, C, H, W]
    img_input = img.unsqueeze(0).to(device)

    # --------------------------------------------------------
    # FORWARD PASS (NO GRADIENT TRACKING)
    # --------------------------------------------------------
    # DETECTION MODEL OUTPUT EXPECTATIONS:
    # -----------------------------------
    # Common detector outputs:
    #
    # (A) Torchvision detectors:
    #     outputs = model(images)  -> list length B
    #     outputs[0] = {"boxes": [M,4], "scores":[M], "labels":[M]}
    #
    # (B) Custom detector:
    #     outputs could be a dict, or raw tensors you must decode.
    #
    with torch.no_grad():
        outputs = model(img_input)

    # --------------------------------------------------------
    # NORMALIZE OUTPUT FORMAT → predictions dict
    # --------------------------------------------------------
    # We will convert whatever comes out into:
    #   preds = {"boxes": ..., "scores": ..., "labels": ...}
    #
    if isinstance(outputs, (list, tuple)) and len(outputs) > 0 and isinstance(outputs[0], dict):
        # Torchvision style: list of dicts
        out0 = outputs[0]
        preds = {
            "boxes":  out0.get("boxes",  torch.empty((0, 4), device=device)).detach().cpu(),
            "scores": out0.get("scores", torch.empty((0,),   device=device)).detach().cpu(),
            "labels": out0.get("labels", torch.empty((0,),   device=device)).detach().cpu(),
        }
    elif isinstance(outputs, dict):
        # Custom dict style
        preds = {
            "boxes":  outputs.get("boxes",  torch.empty((0, 4), device=device)).detach().cpu(),
            "scores": outputs.get("scores", torch.empty((0,),   device=device)).detach().cpu(),
            "labels": outputs.get("labels", torch.empty((0,),   device=device)).detach().cpu(),
        }
    else:
        # Raw tensor style (YOLO-like) requires decoding:
        # You MUST implement your own decoder here.
        raise TypeError(
            "Model output is not in a recognized detection format. "
            "If this is YOLO/custom output tensor, add a decode step to convert to boxes/scores/labels."
        )

    # --------------------------------------------------------
    # APPLY SCORE THRESHOLD + LIMIT MAX DETECTIONS
    # --------------------------------------------------------
    if preds["scores"].numel() > 0:
        keep = preds["scores"] >= float(score_thresh)
        preds["boxes"]  = preds["boxes"][keep]
        preds["scores"] = preds["scores"][keep]
        preds["labels"] = preds["labels"][keep]

        # Optional: keep only top-K detections by score
        if preds["scores"].numel() > max_dets:
            topk = torch.topk(preds["scores"], k=max_dets).indices
            preds["boxes"]  = preds["boxes"][topk]
            preds["scores"] = preds["scores"][topk]
            preds["labels"] = preds["labels"][topk]

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------
    print("--------------------------------------------------")
    print(f"DETECTION RESULT FOR TEST IMAGE INDEX: {index}")

    # Print GT if it exists and is detection-style
    if isinstance(target, dict) and "boxes" in target and "labels" in target:
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        try:
            gt_boxes = gt_boxes.detach().cpu()
        except Exception:
            pass
        try:
            gt_labels = gt_labels.detach().cpu()
        except Exception:
            pass
        print(f"[GT] Num objects: {len(gt_labels)}")
    else:
        print("[GT] No detection target found (dataset might be classification-style).")

    # Print predictions
    num_preds = preds["labels"].numel()
    print(f"[PRED] Num detections (score>={score_thresh}): {num_preds}")

    for i in range(min(int(num_preds), int(max_dets))):
        lbl = int(preds["labels"][i])
        score = float(preds["scores"][i]) if preds["scores"].numel() > 0 else -1.0
        box = preds["boxes"][i].tolist() if preds["boxes"].numel() > 0 else None

        name = class_names[lbl] if 0 <= lbl < len(class_names) else f"class_{lbl}"
        print(f"  #{i:02d}  label={lbl}({name})  score={score:.3f}  box={box}")

    print("--------------------------------------------------")

    # --------------------------------------------------------
    # OPTIONAL: DRAW BOXES ON IMAGE (PIL)
    # --------------------------------------------------------
    if draw:
        try:
            from PIL import Image, ImageDraw, ImageFont
            import torchvision.transforms.functional as TF

            # Convert tensor [C,H,W] -> PIL
            pil_img = TF.to_pil_image(img.cpu())

            draw_ctx = ImageDraw.Draw(pil_img)

            # Draw each predicted box
            for i in range(min(int(num_preds), int(max_dets))):
                x1, y1, x2, y2 = preds["boxes"][i].tolist()
                lbl = int(preds["labels"][i])
                score = float(preds["scores"][i])

                name = class_names[lbl] if 0 <= lbl < len(class_names) else f"class_{lbl}"
                caption = f"{name} {score:.2f}"

                # Rectangle + label
                draw_ctx.rectangle([x1, y1, x2, y2], width=2)
                draw_ctx.text((x1, max(0, y1 - 12)), caption)

            # Show image (works in notebooks; in scripts you can save to file)
            # pil_img.show()  # optional
            return img, target, preds, pil_img

        except Exception as e:
            print(f"[detect_single_image] Draw skipped (PIL not available or error): {e}")

    return img, target, preds



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
    debug_print(f"[main] Global DATA_PATH = {DATA_PATH!r}")

    # Build train and test directories from the global DATA_PATH
    train_path = os.path.join(DATA_PATH, "train")
    test_path  = os.path.join(DATA_PATH, "test")

    debug_print(f"[main] Computed train_path = {train_path}")
    debug_print(f"[main] Computed test_path  = {test_path}")

    print("Training images from:", train_path)
    print("Testing  images from:", test_path)

    # --------------------------------------------------------
    # ⚠️ OBJECT DETECTION TRANSFORMS (BBOX-AWARE)
    # --------------------------------------------------------
    # For detection, you cannot use RandomCrop(32) unless you also
    # update bounding boxes accordingly.
    #
    # SAFE, simple start:
    #   • Convert to tensor
    #   • Normalize
    #
    # If you want augmentation (flip/resize/crop), you MUST use a
    # bbox-aware pipeline (e.g., Albumentations) or implement box transforms.
    # --------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.ToTensor(),  # image -> Tensor [C,H,W] in [0,1]
        transforms.Normalize(   # normalize to roughly [-1,1] if mean=0.5 std=0.5
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    # ------------------------------------------------------------------
    # LOAD DETECTION DATASETS (NOT ImageFolder)
    # ------------------------------------------------------------------
    # ImageFolder returns (img, class_id) which is classification only.
    #
    # Detection dataset must return:
    #   img: Tensor [C,H,W]
    #   target: dict with:
    #       "boxes":  FloatTensor [N,4]  (x1,y1,x2,y2)
    #       "labels": LongTensor  [N]
    #
    # Example options:
    #   • torchvision.datasets.VOCDetection   (needs parsing)
    #   • torchvision.datasets.CocoDetection  (needs parsing)
    #   • Custom dataset reading your annotations (recommended)
    # ------------------------------------------------------------------
    train_dataset = MyDetectionDataset(
        images_dir=train_path,
        ann_file=os.path.join(train_path, "annotations.json"),  # <-- example placeholder
        transform=train_transform
    )

    test_dataset = MyDetectionDataset(
        images_dir=test_path,
        ann_file=os.path.join(test_path, "annotations.json"),   # <-- example placeholder
        transform=test_transform
    )

    debug_print(f"[main] Loaded train_dataset with {len(train_dataset)} images")
    debug_print(f"[main] Loaded test_dataset  with {len(test_dataset)} images")

    # --------------------------------------------------------
    # DATALOADER COLLATE FUNCTION (REQUIRED FOR DETECTION)
    # --------------------------------------------------------
    # Detection targets have variable number of boxes per image.
    # Default collate will fail; we must return lists instead.
    # --------------------------------------------------------
    def detection_collate_fn(batch):
        # batch = [(img0, target0), (img1, target1), ...]
        images, targets = zip(*batch)
        return list(images), list(targets)

    # ============================================================
    # DATALOADERS
    # ============================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=detection_collate_fn,  # ✅ critical for detection
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,                     # ✅ detection inference is often easiest 1-by-1
        shuffle=True,
        num_workers=2,
        collate_fn=detection_collate_fn,  # ✅ critical for detection
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------------
    # DETERMINE NUMBER OF CLASSES
    # --------------------------------------------------------
    # For detection you typically include background internally (depends on model).
    # Your dataset should define a label map.
    # --------------------------------------------------------
    num_classes = getattr(train_dataset, "num_classes", None)
    if num_classes is None:
        # Fallback: derive from dataset label map if available
        label_map = getattr(train_dataset, "label_map", None)
        if isinstance(label_map, dict):
            num_classes = len(label_map)
        else:
            raise ValueError("num_classes not found. Define train_dataset.num_classes or label_map.")

    print("Number of classes detected:", num_classes)

    # --------------------------------------------------------
    # CREATE DETECTION MODEL (EXAMPLE: torchvision Faster R-CNN)
    # --------------------------------------------------------
    # This model accepts ANY image size: [B,3,H,W]
    # --------------------------------------------------------
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the box predictor head with your number of classes
    # NOTE: FasterRCNN expects num_classes INCLUDING background.
    # Usually: num_classes = (your_classes + 1 background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

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

        # ⚠️ Training routine must be detection-style:
        # model(images, targets) returns a dict of losses
        # (classification + box regression + RPN losses)
        model = train_detector(model, train_loader, device, num_epochs=NUM_EPOCHS, lr=1e-3)

        print(f"Saving trained model to: {model_filename}")
        torch.save(model.state_dict(), model_filename)

    # ------------------------------------------------------------
    # INTERACTIVE LOOP FOR USER-DRIVEN DETECTION
    # ------------------------------------------------------------
    import msvcrt

    print("\n--------------------------------------------------")
    print("Interactive Image Detection Mode")
    print("Type an image index and it will run detection.")
    print("Press 'e' at any time to exit.")
    print("--------------------------------------------------\n")

    while True:

        print(f"Enter image index (0 – {len(test_dataset)-1}) or 'e' to exit: ", end="", flush=True)

        key = msvcrt.getch().decode(errors="ignore").lower()

        if key == 'e':
            print("e")
            print("Exiting program. Goodbye!")
            break

        if not key.isdigit():
            print(key)
            print("❌ Invalid input. Enter a number or 'e' to exit.")
            continue

        print(key, end="", flush=True)

        idx_str = key
        while True:
            ch = msvcrt.getch()
            if ch in [b'\r', b'\n']:
                print()
                break

            try:
                c = ch.decode()
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

        # ------------------------------------------------
        # RUN DETECTION
        # ------------------------------------------------
        print(f"\nRunning detection on test image index {idx} ...")

        # If you used the updated detect_single_image that can draw boxes:
        img, target, preds, pil_img = detect_single_image(
            model,
            test_dataset,
            device,
            index=idx,
            score_thresh=0.25,
            draw=True
        )

        # Optional: show/save result
        try:
            pil_img.show()
        except Exception:
            pass



# ------------------------------------------------------------
# RUN PROGRAM
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
