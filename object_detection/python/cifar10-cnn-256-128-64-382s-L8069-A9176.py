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
MODEL_FILENAME = "cifar10-cnn-256-128-64-382s-L8069-A9176"
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
# This network is a STANDARD CONVOLUTIONAL NEURAL NETWORK (CNN)
# with the following characteristics:
#
#   • Layer 1 (conv1) may START with static handcrafted filters
#   • Layer 2 (conv2) may START with static handcrafted filters
#   • Layer 3 (conv3) may START with static handcrafted filters
#   • A single early MaxPool layer reduces spatial resolution
#   • Global Average Pooling (GAP) converts feature maps into a fixed-length vector
#   • A fully connected layer produces final class scores (logits)
#
# ------------------------------------------------------------
# INPUT ASSUMPTION
# ------------------------------------------------------------
# The network expects 3-channel RGB images with shape:
#
#     [B, 3, H, W]
#
# where the spatial resolution (H, W) is provided by the dataset.
# The model processes the input resolution directly as given.
#
# ------------------------------------------------------------
# MEANING OF "STATIC FILTERS"
# ------------------------------------------------------------
# The term "static" refers ONLY to how certain convolution layers
# are initialized.
#
#   • Static filters are assigned at initialization time
#   • After initialization, ALL layers remain trainable
#   • Gradients flow through every convolution and linear layer
#
# This is a standard CNN that starts from meaningful kernels
# and then learns normally via backpropagation.
#
# ------------------------------------------------------------
# HOW LEARNING HAPPENS
# ------------------------------------------------------------
#
# During training:
#
#   outputs = model(images)
#   loss    = criterion(outputs, labels)
#   loss.backward()
#   optimizer.step()
#
# PyTorch computes gradients for ALL learnable parameters:
#
#   • conv1.weight, conv1.bias
#   • conv2.weight, conv2.bias
#   • conv3.weight, conv3.bias
#   • fc.weight,   fc.bias
#
# Pooling layers and GAP contain no parameters and therefore
# do not participate in learning.
#
# Because all parameters have requires_grad = True,
# the optimizer updates every convolutional and linear layer.
#
# ------------------------------------------------------------
# WILL ALL LEARNABLE LAYERS LEARN?
# ------------------------------------------------------------
#
# YES.
#
#   • conv1 learns
#   • conv2 learns
#   • conv3 learns
#   • fc learns
#
# Pooling and GAP perform fixed mathematical operations only.
#
# ------------------------------------------------------------
# WHAT WOULD PREVENT A LAYER FROM LEARNING?
# ------------------------------------------------------------
#
# A layer would stop learning only if:
#
#   param.requires_grad = False
#
# is explicitly set.
#
# ------------------------------------------------------------
# WHY THIS IS A STANDARD CNN
# ------------------------------------------------------------
#
# Because:
#
#   • Filters are initialized
#   • Filters are optimized via backpropagation
#   • Weights change every training step
#   • Learning is end-to-end
#   • Pooling reduces spatial resolution
#   • GAP ensures resolution independence
#
# The only difference from many CNNs is the option to use
# meaningful (handcrafted) initialization in early layers.
#
# ------------------------------------------------------------
# NETWORK SHAPE (SYMBOLIC)
# ------------------------------------------------------------
#
# Input image:            [3    x H   x W]
# After conv1 + pool:     [256  x H/2 x W/2]
# After conv2:            [128  x H/2 x W/2]
# After conv3:            [64   x H/2 x W/2]
# After GAP:              [64]
# Output layer (fc):      [num_classes]
#
# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
#
# ✅ Optional static initialization (conv1 / conv2 / conv3)
# ✅ Fully learnable during training (all layers)
# ✅ Single early pooling preserves spatial detail
# ✅ GAP removes dependency on spatial resolution
# ✅ End-to-end classical CNN training




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
        # LAYER 1: 3 → 256 channels  ✅ UPDATED
        # ------------------------------------------------------
        # 3 input channels (RGB) → 256 feature maps using 3x3 filters
        # Padding = 1 to keep spatial size
        #
        # Input shape assumption:
        #   [B, 3, H, W]   (ANY H, W)
        #
        # Either native CIFAR-10 images, OR any custom dataset
        # without resizing constraints.
        # ------------------------------------------------------
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=256,
            kernel_size=3,
            padding=1,
            bias=True
        )

        # ------------------------------------------------------
        # BatchNorm for conv1 (normalizes 256 output channels) ✅ UPDATED
        # ------------------------------------------------------
        # WHY WE USE BATCHNORM2d(256):
        # ----------------------------
        # • It normalizes each of the 256 feature maps across the batch
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
        self.bn1 = nn.BatchNorm2d(256)

        # ------------------------------------------------------
        # LAYER 2: 256 → 128 channels ✅ UPDATED
        # ------------------------------------------------------
        # 256 input feature maps → 128 output feature maps
        # using 3×3 filters, padding=1 keeps spatial size.
        #
        # Before Pool (after first pooling):
        #   input to conv2 : [B, 256, H/2, W/2]
        #   output of conv2: [B, 128, H/2, W/2]
        # ------------------------------------------------------
        self.conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
            bias=True
        )

        # ------------------------------------------------------
        # BatchNorm for conv2 (normalizes 128 channels) ✅ UPDATED
        # ------------------------------------------------------
        # Why BatchNorm2d(128)?
        # ---------------------
        # • Conv2 outputs 128 feature maps
        # • BatchNorm stabilizes all 128 channels
        #
        # Overall:
        #   conv2 → bn2 → ReLU → pool
        #
        # BatchNorm especially helps deeper layers where
        # activations become more chaotic.
        # ------------------------------------------------------
        self.bn2 = nn.BatchNorm2d(128)

        # ------------------------------------------------------
        # POOLING LAYER: MaxPool2d(2, 2)
        # ------------------------------------------------------
        # Max pooling:
        #     kernel_size = 2
        #     stride      = 2
        #
        # Effect on spatial dimensions:
        #   H×W → H/2×W/2     (after first pool)
        #   H/2×W/2 → H/4×W/4 (after second pool)
        #
        # This works for ANY input resolution.
        # ------------------------------------------------------
        self.pool = nn.MaxPool2d(2, 2)

        # ------------------------------------------------------
        # ✅ NEW: LAYER 3 (CAPACITY BOOST): 128 → 64 channels ✅ UPDATED
        # ------------------------------------------------------
        # WHY THIS IMPROVES PREDICTION QUALITY:
        # ------------------------------------
        # Your previous network ended early (low channel capacity) before GAP.
        # That is a major feature bottleneck for CIFAR-10 and most real images.
        #
        # Adding conv3 allows the network to learn richer, higher-level patterns:
        #   • textures
        #   • shapes
        #   • object parts
        #
        # IMPORTANT:
        # • padding=1 keeps spatial size
        # • Works for ANY input H, W
        # ------------------------------------------------------
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=True
        )

        # ------------------------------------------------------
        # ✅ NEW: BatchNorm for conv3 (normalizes 64 channels) ✅ UPDATED
        # ------------------------------------------------------
        # Why BatchNorm2d(64)?
        # --------------------
        # • Conv3 outputs 64 feature maps
        # • BN helps stabilize deeper activations
        # • Makes training smoother and improves generalization
        # ------------------------------------------------------
        self.bn3 = nn.BatchNorm2d(64)

        # ------------------------------------------------------
        # 🔑 GLOBAL AVERAGE POOLING (IMAGE-SIZE INDEPENDENT)
        # ------------------------------------------------------
        # Replaces hard-coded flattening of spatial dimensions.
        #
        # Converts:
        #   [B, C, H, W] → [B, C, 1, 1]
        #
        # In THIS model after conv3, C = 64:
        #   [B, 64, H', W'] → [B, 64, 1, 1]
        #
        # This makes the model work with ANY image size.
        # ------------------------------------------------------
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ------------------------------------------------------
        # ✅ NEW: DROPOUT (GENERALIZATION BOOST)
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
        # Because conv3 now outputs 64 channels, GAP now outputs:
        #   [B, 64, 1, 1] → flatten → [B, 64]
        #
        # Therefore the classifier must be:
        #   nn.Linear(64, num_classes)
        # ------------------------------------------------------
        self.fc = nn.Linear(64, num_classes)

        # ------------------------------------------------------
        # STATIC FILTER INITIALIZATION (if enabled)
        # ------------------------------------------------------
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
        #
        # IMPORTANT NOTE (accuracy-related):
        # ----------------------------------
        # If conv1 is overwritten with static filters, it may LIMIT
        # the benefit of increasing conv1 channels to 256 unless your
        # static filter bank actually fills/uses those 256 output maps.
        # ------------------------------------------------------
        if STATIC_FILTERS:
            self._init_conv1_static()
            # self._init_conv2_static()



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
            # conv1 has out_channels filters (256 in this model).
            # If out_channels > number of kernels, we repeat them in order.
            #
            # This guarantees:
            #   • conv1 starts with edges, corners, lines, curves, gradients immediately
            #   • training begins from meaningful low-level detectors (better inductive bias)
            #   • the CNN behaves like a hybrid handcrafted + learned feature extractor
            # ------------------------------------------------------------------

            for i in range(out_channels):                                  # loop over each output filter
                k2d = kernels[i % num_kernels].to(w.dtype)                 # pick 2D kernel and cast dtype
                for c in range(in_channels):                               # assign same 3x3 kernel to each RGB channel
                    w[i, c].copy_(k2d)                                     # write into conv1 weight tensor

            print(f"[init_conv1_static] {out_channels} filters initialized with 2D 3x3 kernels")

   # ----------------------------------------------------------

# ----------------------------------------------------------
# STATIC INITIALIZATION FOR LAYER 2
# ----------------------------------------------------------
    def _init_conv2_static(self):
        with torch.no_grad():                                                           # disable gradients (manual init)
            w = self.conv2.weight                                                       # conv2 weights → [128, 256, 3, 3]
            out_channels, in_channels, kh, kw = w.shape                                 # expected [128, 256, 3, 3]
            assert kh == 3 and kw == 3                                                  # ensure 3x3 kernel size

            # ---------------------------------------------------------------------
            # FILTER DEFINITIONS (EACH 3×3)
            #
            # conv2 receives 256 feature maps from conv1 and produces 128 output maps.
            #
            # These deeper filters operate on already-detected low-level primitives
            # (edges, corners, curves) and emphasize:
            #   • stronger transitions
            #   • gradient composition
            #   • embossed structure
            #   • feature smoothing
            #
            # This stage begins building mid-level visual representations.
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

            # ---------------------------------------------------------------------
            # COLLECT FILTER BANK
            # ---------------------------------------------------------------------
            kernels = [
                edge_h,     # horizontal edge
                edge_v,     # vertical edge
                emboss,     # emboss shading
                avg,        # smoothing blur
                sobel_x,    # gradient X
                sobel_y,    # gradient Y
            ]

            num_kernels = len(kernels)

            # ---------------------------------------------------------------------
            # ASSIGN STATIC FILTERS → conv2 WEIGHTS
            #
            # conv2 configuration:
            #   • out_channels = 128
            #   • in_channels  = 256
            #
            # For each output channel and each input channel, a kernel is selected
            # using modulo indexing so the kernel bank repeats deterministically.
            #
            # This creates a structured 128 × 256 kernel tensor where different
            # channel interactions emphasize edges, gradients, embossing, and blur.
            # ---------------------------------------------------------------------
            for out_idx in range(out_channels):                       # loop over all 128 output filters
                for in_idx in range(in_channels):                     # loop over all 256 input feature maps

                    # Select kernel pattern deterministically
                    k = kernels[(out_idx * in_idx) % num_kernels].to(w.dtype)

                    # Copy kernel into conv2 weight tensor
                    w[out_idx, in_idx].copy_(k)

            print(
                f"[init_conv2_static] {out_channels}x{in_channels} "
                f"static 3x3 kernels assigned"
            )


    # ----------------------------------------------------------
    # STATIC INITIALIZATION FOR LAYER 3
    # ----------------------------------------------------------
    def _init_conv3_static(self):
        with torch.no_grad():                                                           # disable gradients (manual init)
            w = self.conv3.weight                                                       # conv3 weights → [64, 128, 3, 3]
            out_channels, in_channels, kh, kw = w.shape                                 # expected [64, 128, 3, 3]
            assert kh == 3 and kw == 3                                                  # ensure 3x3 kernel size

            # ---------------------------------------------------------------------
            # FILTER DEFINITIONS (EACH 3×3)
            #
            # conv3 receives 128 feature maps from conv2 and produces 64 output maps.
            #
            # At this depth, features are more abstract (parts, textures, blobs).
            # Static kernels here can:
            #   • enhance mid-level patterns
            #   • detect stronger directional structure
            #   • emphasize center-surround contrast (DoG/LoG-like)
            #   • keep training stable at the beginning (strong inductive bias)
            #
            # NOTE:
            # This is still a learnable network — these kernels are only the
            # starting point. Backprop will update conv3 normally.
            # ---------------------------------------------------------------------

            # 1) Center-surround (Laplacian-like)
            #    Highlights local contrast and blob-like structures.
            laplacian = torch.tensor([
                [ 0., -1.,  0.],
                [-1.,  4., -1.],
                [ 0., -1.,  0.],
            ])

            # 2) Stronger center-surround
            #    Higher emphasis on the center pixel response.
            laplacian_strong = torch.tensor([
                [-1., -1., -1.],
                [-1.,  8., -1.],
                [-1., -1., -1.],
            ])

            # 3) High-pass sharpening
            #    Boosts fine detail and edges after earlier feature extraction.
            high_pass = torch.tensor([
                [-1., -1., -1.],
                [-1.,  9., -1.],
                [-1., -1., -1.],
            ])

            # 4) Mild smoothing (mean blur)
            #    Stabilizes noisy activations at deeper depth.
            mean_blur = (1/9) * torch.ones((3, 3))

            # 5) Gaussian smoothing
            #    Gentler smoothing than mean blur.
            gaussian_blur = (1/16) * torch.tensor([
                [1., 2., 1.],
                [2., 4., 2.],
                [1., 2., 1.],
            ])

            # 6) Directional edge: horizontal
            edge_h = torch.tensor([
                [-1., -1., -1.],
                [ 2.,  2.,  2.],
                [-1., -1., -1.],
            ])

            # 7) Directional edge: vertical
            edge_v = torch.tensor([
                [-1.,  2., -1.],
                [-1.,  2., -1.],
                [-1.,  2., -1.],
            ])

            # 8) Sobel X (gradient X)
            sobel_x = torch.tensor([
                [-1.,  0.,  1.],
                [-2.,  0.,  2.],
                [-1.,  0.,  1.],
            ])

            # 9) Sobel Y (gradient Y)
            sobel_y = torch.tensor([
                [-1., -2., -1.],
                [ 0.,  0.,  0.],
                [ 1.,  2.,  1.],
            ])

            # ---------------------------------------------------------------------
            # COLLECT FILTER BANK
            # ---------------------------------------------------------------------
            kernels = [
                laplacian,          # 0 center-surround
                laplacian_strong,   # 1 stronger center-surround
                high_pass,          # 2 high-pass sharpen
                mean_blur,          # 3 mean blur
                gaussian_blur,      # 4 gaussian blur
                edge_h,             # 5 horizontal edge
                edge_v,             # 6 vertical edge
                sobel_x,            # 7 sobel x
                sobel_y,            # 8 sobel y
            ]

            num_kernels = len(kernels)

            # ---------------------------------------------------------------------
            # ASSIGN STATIC FILTERS → conv3 WEIGHTS
            #
            # conv3 configuration:
            #   • out_channels = 64
            #   • in_channels  = 128
            #
            # For each output channel and each input channel, a kernel is selected
            # deterministically so the bank repeats across the full tensor.
            #
            # This creates a structured 64 × 128 kernel tensor where different
            # channel interactions emphasize contrast, edges, gradients, and smoothing.
            # ---------------------------------------------------------------------
            for out_idx in range(out_channels):                                   # loop over all 64 output filters
                for in_idx in range(in_channels):                                 # loop over all 128 input feature maps

                    # Select kernel pattern deterministically
                    k = kernels[(out_idx + in_idx) % num_kernels].to(w.dtype)

                    # Copy kernel into conv3 weight tensor
                    w[out_idx, in_idx].copy_(k)

            print(
                f"[init_conv3_static] {out_channels}x{in_channels} "
                f"static 3x3 kernels assigned"
            )


   # ----------------------------------------------------------
    # FORWARD PASS
    # ----------------------------------------------------------
    # FORWARD PROPAGATION THROUGH THE NETWORK
    # --------------------------------------
    # This method defines EXACTLY how input images flow
    # through the convolutional neural network from pixels
    # to final class predictions.
    #
    # DESIGN GOALS:
    # -------------
    # ✔ Operate on arbitrary spatial resolutions provided by the dataset
    # ✔ Preserve spatial detail early (better accuracy)
    # ✔ Extract progressively higher-level features
    # ✔ Produce stable logits for CrossEntropyLoss
    #
    # ----------------------------------------------------------
    # INPUT:
    # ----------------------------------------------------------
    #     x : Tensor of shape [B, 3, H, W]
    #
    #     Where:
    #       B = batch size
    #       3 = RGB color channels
    #       H = image height
    #       W = image width
    #
    #     The spatial resolution (H, W) is determined by the dataset
    #     and is passed through the network without modification.
    #
    #     This network is IMAGE-SIZE AGNOSTIC because:
    #       • Spatial dimensions are never flattened explicitly
    #       • Global Average Pooling (GAP) converts feature maps
    #         into a fixed-length channel vector
    #
    # ----------------------------------------------------------
    # NETWORK FLOW (HIGH LEVEL):
    # ----------------------------------------------------------
    #
    #     Input Image
    #         ↓
    #     Conv1 (256) → BatchNorm → ReLU → MaxPool
    #         ↓
    #     Conv2 (128) → BatchNorm → ReLU
    #         ↓
    #     Conv3 (64)  → BatchNorm → ReLU
    #         ↓
    #     Global Average Pooling (GAP)
    #         ↓
    #     Dropout
    #         ↓
    #     Fully Connected Layer
    #         ↓
    #     Logits (raw class scores)
    #
    # ----------------------------------------------------------
    # OUTPUT:
    # ----------------------------------------------------------
    #     logits : Tensor of shape [B, num_classes]
    #
    #     Where:
    #       B = batch size
    #       num_classes = number of target classes (e.g. 10 for CIFAR-10)
    #
    #     Meaning:
    #     --------
    #     • Each row corresponds to one input image
    #     • Each column corresponds to one class
    #     • Values are raw, unnormalized scores (logits)
    #
    # ----------------------------------------------------------
    # WHY LOGITS (NOT SOFTMAX OUTPUT):
    # ----------------------------------------------------------
    # • Softmax is applied internally by nn.CrossEntropyLoss
    # • Passing logits directly improves numerical stability
    # • This is the standard and recommended PyTorch practice
    #
    # During training:
    #     loss = CrossEntropyLoss(logits, labels)
    #
    # During inference:
    #     predictions = argmax(logits, dim=1)
    #
    # ----------------------------------------------------------
    # KEY QUALITY CHARACTERISTICS OF THIS FORWARD PASS:
    # ----------------------------------------------------------
    # ✔ Single early pooling layer preserves spatial structure
    # ✔ Progressive channel compression (256 → 128 → 64)
    # ✔ GAP removes dependence on input resolution
    # ✔ Dropout improves generalization
    #
    # ----------------------------------------------------------

    def forward(self, x):
        # Entry shape:
        #   x → [B, 3, H, W]

        # -------------------
        # BLOCK 1: CONV1 → BN1 → ReLU → POOL
        # -------------------

        # Conv1: 3 → 256 channels
        #   [B, 3, H, W] → [B, 256, H, W]
        x = self.conv1(x)

        # BatchNorm on 256 channels
        x = self.bn1(x)

        # Non-linearity
        x = F.relu(x)

        # Spatial downsampling
        #   [B, 256, H, W] → [B, 256, H/2, W/2]
        x = self.pool(x)

        # -------------------
        # BLOCK 2: CONV2 → BN2 → ReLU
        # -------------------

        # Conv2: 256 → 128 channels
        #   [B, 256, H/2, W/2] → [B, 128, H/2, W/2]
        x = self.conv2(x)

        # BatchNorm on 128 channels
        x = self.bn2(x)

        # Non-linearity
        x = F.relu(x)

        # -------------------
        # BLOCK 3: CONV3 → BN3 → ReLU
        # -------------------

        # Conv3: 128 → 64 channels
        #   [B, 128, H/2, W/2] → [B, 64, H/2, W/2]
        x = self.conv3(x)

        # BatchNorm on 64 channels
        x = self.bn3(x)

        # Non-linearity
        x = F.relu(x)

        # -------------------
        # GLOBAL AVERAGE POOLING
        # -------------------

        #   [B, 64, H/2, W/2] → [B, 64, 1, 1]
        x = self.gap(x)

        #   [B, 64, 1, 1] → [B, 64]
        x = torch.flatten(x, 1)

        # -------------------
        # DROPOUT
        # -------------------
        x = self.dropout(x)

        # -------------------
        # CLASSIFIER
        # -------------------

        #   [B, 64] → [B, num_classes]
        logits = self.fc(x)

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
