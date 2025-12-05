import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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

        # ------------------------------------------------------
        # LAYER 1: 3 → 16 channels
        # 3 input channels (RGB) → 16 feature maps using 3x3 filters
        # Padding = 1 to keep spatial size 32x32
        # This assumes the input has shape [B, 3, 32, 32]
        # (either native 32x32, or resized to 32x32 in transforms).
        # ------------------------------------------------------
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)

        # ------------------------------------------------------
        # LAYER 2: 16 → 32 channels
        # 16 input feature maps → 32 feature maps using 3x3 filters
        # Padding = 1 to keep spatial size before pooling:
        #   input to conv2: [B, 16, 16, 16]
        #   output of conv2: [B, 32, 16, 16]
        # ------------------------------------------------------
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)

        # ------------------------------------------------------
        # POOLING LAYER: MaxPool2d(2, 2)
        #
        # Max pooling with:
        #   kernel_size = 2
        #   stride      = 2
        #
        # Effect on spatial size:
        #   32x32 → 16x16
        #   16x16 →  8x8
        #
        # We reuse the SAME pool layer twice (after conv1 and conv2).
        # ------------------------------------------------------
        self.pool = nn.MaxPool2d(2, 2)

        # ------------------------------------------------------
        # FULLY CONNECTED CLASSIFIER
        #
        # After:
        #   conv1 + pool → [B, 16, 16, 16]
        #   conv2 + pool → [B, 32,  8,  8]
        #
        # Flattened feature vector size:
        #   32 * 8 * 8 = 2048
        #
        # So fc in_features = 2048, out_features = num_classes.
        # num_classes should match:
        #   • 10 for CIFAR-10
        #   • or len(train_dataset.classes) for custom ImageFolder
        # ------------------------------------------------------
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

        # Static initialization (starting point only)
        self._init_conv1_static()
        self._init_conv2_static()

    # ----------------------------------------------------------
    # STATIC INITIALIZATION FOR LAYER 1
    # ----------------------------------------------------------
    def _init_conv1_static(self):

        # Disable gradient tracking during manual weight initialization.
        # We are NOT training here, only assigning initial filter values.
        with torch.no_grad():

            # self.conv1.weight has shape: [16, 3, 3, 3]
            #   16 = number of output filters
            #   3  = input channels (RGB)
            #   3x3 = kernel size
            w = self.conv1.weight

            # Sobel X filter → detects vertical edges
            sobel_x = torch.tensor(
                [[-1,  0,  1],
                 [-2,  0,  2],
                 [-1,  0,  1]],
                dtype=torch.float32
            )

            # Sobel Y filter → detects horizontal edges
            sobel_y = torch.tensor(
                [[-1, -2, -1],
                 [ 0,  0,  0],
                 [ 1,  2,  1]],
                dtype=torch.float32
            )

            # Laplacian filter → detects corners and strong edges
            laplacian = torch.tensor(
                [[ 0, -1,  0],
                 [-1,  4, -1],
                 [ 0, -1,  0]],
                dtype=torch.float32
            )

            # Sharpen filter → enhances edges and fine details
            sharpen = torch.tensor(
                [[ 0, -1,  0],
                 [-1,  5, -1],
                 [ 0, -1,  0]],
                dtype=torch.float32
            )

            # Average filter (blur) → smooths image and removes noise
            avg = (1 / 9) * torch.ones((3, 3), dtype=torch.float32)

            # Identity filter → copies the original pixel values unchanged
            identity = torch.zeros((3, 3), dtype=torch.float32)
            identity[1, 1] = 1.0   # middle pixel passes through unchanged

            # We have 6 base kernels but 16 conv filters,
            # so we will cycle through them repeatedly.
            kernels = [sobel_x, sobel_y, laplacian, sharpen, avg, identity]

            # Convert a single-channel 3x3 kernel into an RGB 3x3x3 kernel
            # by repeating the same 3x3 kernel for R, G, and B channels.
            def rgb(k):
                return k.repeat(3, 1, 1)  # [3, 3, 3]

            # Assign static filters to conv1 weights for all 16 output channels
            for i in range(16):
                base_kernel = kernels[i % len(kernels)]
                rgb_kernel = rgb(base_kernel)
                w[i].copy_(rgb_kernel)

    # ----------------------------------------------------------
    # STATIC INITIALIZATION FOR LAYER 2
    # ----------------------------------------------------------
    def _init_conv2_static(self):

        # Disable gradients while setting initial weights for conv2
        with torch.no_grad():
            # conv2.weight shape:
            #   [32, 16, 3, 3]
            # 32 output filters, 16 input channels, 3x3 kernels
            w = self.conv2.weight

            # Simple 3x3 horizontal edge filter
            edge_h = torch.tensor(
                [[-1, -1, -1],
                 [ 2,  2,  2],
                 [-1, -1, -1]],
                dtype=torch.float32
            )

            # Simple 3x3 vertical edge filter
            edge_v = torch.tensor(
                [[-1,  2, -1],
                 [-1,  2, -1],
                 [-1,  2, -1]],
                dtype=torch.float32
            )

            # Emboss filter → gives a raised/embossed appearance
            emboss = torch.tensor(
                [[-2, -1,  0],
                 [-1,  1,  1],
                 [  0,  1,  2]],
                dtype=torch.float32
            )

            # Average filter → slight smoothing
            avg = (1 / 9) * torch.ones((3, 3), dtype=torch.float32)

            kernels = [edge_h, edge_v, emboss, avg]

            # Helper to expand a single 3x3 kernel across ALL 16 input channels
            # Input:  k → [3, 3]
            # Output: [16, 3, 3]
            def full(k):
                return k.repeat(16, 1, 1)

            # Assign kernels to 32 output filters in conv2
            for i in range(32):
                base_kernel = kernels[i % len(kernels)]
                w[i].copy_(full(base_kernel))

    # ----------------------------------------------------------
    # FORWARD PASS
    # ----------------------------------------------------------
    def forward(self, x):
        # x shape at input:
        #   [B, 3, 32, 32]  where:
        #      B = batch size
        #      3 = RGB channels
        #     32x32 = spatial size
        #   (either CIFAR-10 or any custom dataset resized to 32x32)
        x = F.relu(self.conv1(x))      # After conv1: [B, 16, 32, 32]
        x = self.pool(x)               # After pool1: [B, 16, 16, 16]

        x = F.relu(self.conv2(x))      # After conv2: [B, 32, 16, 16]
        x = self.pool(x)               # After pool2: [B, 32,  8,  8]

        # Flatten all channels and spatial dimensions into a single vector
        # Current shape: [B, 32, 8, 8]
        # Flattened:      [B, 32*8*8] = [B, 2048]
        x = torch.flatten(x, 1)

        # Fully connected classifier:
        # Input:  [B, 2048]
        # Output: [B, num_classes]
        x = self.fc(x)

        return x


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
    
    
    # ------------------------------------------------------------
    # SEND MODEL TO GPU (IF AVAILABLE) OR CPU
    # ------------------------------------------------------------
    model.to(device)

    # ------------------------------------------------------------
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


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
    # TRAINING LOOP
    # ------------------------------------------------------------
    for ep in range(num_epochs):

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
            # ============================================================
            # STAGE 5 — OUTPUT = LOGITS
            # ============================================================
            #
            # Example numeric logits:
            #
            # outputs = [2.4, 0.3, -1.2]
            #
            # Meaning:
            #   CAT → strong
            #   DOG → weak
            #   MAN → very weak
            #
            # ============================================================
            # STAGE 6 — SOFTMAX (INSIDE CrossEntropyLoss)
            # ============================================================
            #
            # Convert logits to probabilities:
            #
            # exp(2.4) = 11.02
            # exp(0.3) =  1.35
            # exp(-1.2)=  0.30
            #
            # Sum = 12.67
            #
            # P(CAT) = 11.02 / 12.67 = 0.86
            # P(DOG) =  1.35 / 12.67 = 0.11
            # P(MAN) =  0.30 / 12.67 = 0.03
            #
            # ============================================================
            # STAGE 7 — CROSSENTROPY LOSS
            # ============================================================
            #
            # If true label is:
            #
            #   label = 0 (CAT)
            #
            # Then loss:
            #
            #   loss = -log(0.86) = LOW → GOOD
            #
            # If model guessed wrongly:
            #
            #   loss becomes HIGH → BAD
            #
            # ============================================================


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
            # Example values:
            #
            #   outputs = [[ 2.4,  0.3, -1.2 ]]   # logits for [CAT, DOG, MAN]
            #   labels  = [0]                     # true class = CAT
            #
            # So:
            #   outputs[0][0] = 2.4  → score for CAT
            #   outputs[0][1] = 0.3  → score for DOG
            #   outputs[0][2] = -1.2 → score for MAN
            #
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
            #      (This graph was created during the forward pass when:
            #          outputs = model(images)
            #       and:
            #          loss = criterion(outputs, labels)
            #      )
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
            loss.backward()


            # ----------------------------------------
            # PARAMETER UPDATE
            #   optimizer changes all learnable weights
            # ----------------------------------------
            # Apply the optimizer update step.
            #
            # This is the moment where the neural network actually LEARNS.
            #
            # What this line does:
            #   • Reads all gradients computed by loss.backward().
            #   • Uses the optimization algorithm (Adam here) to update each parameter.
            #
            # Sequence context:
            #   loss.backward()   → computes gradients
            #   optimizer.step()  → applies updates
            #
            # Internally, for EACH trainable parameter:
            #
            #   1) The optimizer reads:
            #        param.grad  (computed gradient).
            #
            #   2) Adam updates its internal states:
            #        m  (first moment / momentum)
            #        v  (second moment / variance)
            #
            #   3) Bias correction is applied:
            #        m̂ = m / (1 − β1^t)
            #        v̂ = v / (1 − β2^t)
            #
            #   4) Weight update is computed:
            #        param ← param − lr × (m̂ / (sqrt(v̂) + ε))
            #
            # Effects:
            #   • Large gradients are dampened.
            #   • Small gradients are amplified.
            #   • Each parameter gets its own adaptive step size.
            #
            # Results:
            #   • Feature detectors in conv layers improve.
            #   • Fully-connected layers become better decision makers.
            #   • Biases and normalization layers self-adjust.
            #
            # Important:
            #   • This updates ONLY parameters that have requires_grad=True.
            #   • Frozen layers remain unchanged.
            #
            # What happens if you skip this line:
            #   ❌ No learning occurs.
            #   ❌ Model weights never change.
            #   ❌ Loss stays constant across epochs.
            #
            # Debug tip:
            #   • Check param.grad before step() to verify gradients exist.
            #   • Print weight values before/after step() to confirm learning is happening.
            #
            # Note:
            #   • Parameters are updated in-place.
            #   • The computational graph is NOT rebuilt here.
            #
            optimizer.step()


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


        # --------------------------------------------
        # PRINT EPOCH SUMMARY
        # --------------------------------------------
        print(f"[TRAIN] Epoch {ep+1}/{num_epochs}  "
              f"Loss: {running_loss / total:.4f}  "
              f"Accuracy: {correct / total:.4f}")

    # ------------------------------------------------------------
    # RETURN TRAINED MODEL
    # ------------------------------------------------------------
    return model




# ============================================================
# DETECTION / SINGLE-IMAGE INFERENCE FUNCTION
# ============================================================
def detect_single_image(model, test_dataset, device, index=0):
    """
    Loads one image from the test set (by index),
    runs the model in eval mode, and prints:

        - True label id
        - Predicted label id
        - True class name
        - Predicted class name

    Works for:
        • CIFAR-10
        • ImageFolder datasets
        • Any custom dataset (classes detected dynamically)

    Returns:
        (image_tensor, true_label, predicted_label)
    """

    # --------------------------------------------------------
    # MOVE MODEL TO DEVICE AND SWITCH TO EVAL MODE
    # --------------------------------------------------------
    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # GET CLASS NAMES FROM DATASET (AUTOMATIC)
    # --------------------------------------------------------
    # Works for ImageFolder, CIFAR-10, and torchvision datasets
    class_names = test_dataset.classes

    # --------------------------------------------------------
    # EXTRACT ONE IMAGE FROM DATASET
    # --------------------------------------------------------
    img, true_label = test_dataset[index]

    # Add batch dimension → [1, C, H, W]
    img_input = img.unsqueeze(0).to(device)

    # --------------------------------------------------------
    # MODEL INFERENCE (NO GRADIENTS)
    # --------------------------------------------------------
    with torch.no_grad():
        logits = model(img_input)
        pred_label = logits.argmax(1).item()

    # --------------------------------------------------------
    # MAP LABEL ID → CLASS NAME
    # --------------------------------------------------------
    true_name = class_names[true_label]
    pred_name = class_names[pred_label]

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------
    print("--------------------------------------------------")
    print(f"DETECTION RESULT FOR TEST IMAGE INDEX: {index}")
    print(f"True label index : {true_label} → {true_name}")
    print(f"Pred label index : {pred_label} → {pred_name}")
    print("--------------------------------------------------")

    return img, true_label, pred_label


# ============================================================
# MAIN PROGRAM
# ============================================================
def main():

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------------------------------------
    # PATH TO SAVE / LOAD MODEL WEIGHTS
    # --------------------------------------------------------
    # UPDATED: use a different file name so you do not overwrite CIFAR-10 model
    MODEL_PATH = "dynamic_cnn_mydata.pth"

    # --------------------------------------------------------
    # DATA TRANSFORMS FOR YOUR 128x128 DATA
    # --------------------------------------------------------
    # UPDATED:
    #   • Add Resize((32, 32)) so 128x128 images become 32x32
    #   • Keep ToTensor + Normalize like CIFAR-10
    #   • This way the network input shape is the same as CIFAR-10
    # --------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((32, 32)),        # UPDATED: 128x128 → 32x32
        transforms.ToTensor(),              # convert to [C, H, W] in [0, 1]
        transforms.Normalize(               # normalize to [-1, 1]
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # --------------------------------------------------------
    # LOAD YOUR CUSTOM TRAIN AND TEST SETS USING ImageFolder
    # --------------------------------------------------------
    # EXPECTED FOLDER STRUCTURE:
    #   mydata/
    #       train/
    #           class0/
    #           class1/
    #           ...
    #       test/
    #           class0/
    #           class1/
    #           ...
    #
    # Each "classX" folder contains images for that class.
    # ImageFolder will automatically assign class indices:
    #   0, 1, 2, ... in alphabetical order of folder names.
    # --------------------------------------------------------
    train_dataset = datasets.ImageFolder(
        root="./mydata/train",   # UPDATED: your train path
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root="./mydata/test",    # UPDATED: your test path
        transform=transform
    )

   # ============================================================
    # FUNCTIONAL PURPOSE:
    # ============================================================
    # This DataLoader is responsible for feeding training data
    # into the neural network in SMALL GROUPS (mini-batches)
    # instead of sending all images at once.
    #
    # The DataLoader:
    #   • Loads images from the dataset
    #   • Applies transformations (resize, normalize, etc.)
    #   • Groups images into batches
    #   • Shuffles order every epoch (if enabled)
    #   • Uses parallel workers to load data faster
    #   • Feeds batches into the training loop
    #
    # During training:
    #   → The model never sees ALL images at once.
    #   → It sees small batches repeatedly.
    #   → Loss is computed PER BATCH.
    #   → Gradients are computed PER BATCH.
    #   → Weights are updated PER BATCH.
    #
    # ============================================================

    train_loader = DataLoader(

        # --------------------------------------------------------
        # train_dataset
        # --------------------------------------------------------
        # This is the DATA SOURCE.
        #
        # It may be:
        #   • CIFAR10 dataset
        #   • ImageFolder dataset
        #   • Any custom PyTorch Dataset class
        #
        # When DataLoader needs data, it calls:
        #
        #   image, label = train_dataset[index]
        #
        # Which returns:
        #   image → Tensor [C, H, W]
        #   label → Integer class index
        # --------------------------------------------------------
        train_dataset,

        # --------------------------------------------------------
        # batch_size = 10
        # --------------------------------------------------------
        # This controls HOW MANY samples are processed before:
        #   • computing loss
        #   • computing gradients
        #   • performing optimizer step
        #
        # Meaning:
        #   → 10 images form ONE training step
        #   → Loss = average over 10 images
        #   → Weight updates use group statistics
        #
        # Example (1000 images total):
        #
        #   batch_size = 10
        #   → 100 batches per epoch
        # --------------------------------------------------------
        batch_size=10,

        # --------------------------------------------------------
        # shuffle = True
        # --------------------------------------------------------
        # Means:
        #
        #   → BEFORE every epoch:
        #       • All image indices are randomly reshuffled.
        #
        #   → Batch composition changes every epoch.
        #   → No fixed grouping of classes.
        #
        # Prevents:
        #   • bias from dataset ordering
        #   • memorization due to fixed sequence
        #
        # Improves:
        #   • generalization
        #   • convergence stability
        # --------------------------------------------------------
        shuffle=True,

        # --------------------------------------------------------
        # num_workers = 2
        # --------------------------------------------------------
        # Controls HOW MANY CPU PROCESSES load data simultaneously.
        #
        # Instead of loading images sequentially:
        #
        #   worker-1 → loads batch 1
        #   worker-2 → loads batch 2
        #
        # While:
        #   GPU is training on batch 1
        #
        # Benefits:
        #   ✅ Reduced waiting time
        #   ✅ Faster throughput
        #   ✅ Efficient CPU usage
        #
        # Notes:
        #   On Windows:
        #       Use num_workers = 0 or 1
        #   On Linux:
        #       2–8 is common
        # --------------------------------------------------------
        num_workers=2
        )

    
    # Optional: test_loader if you want evaluation later
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2
    )

    # --------------------------------------------------------
    # DETERMINE NUMBER OF CLASSES FROM DATASET
    # --------------------------------------------------------
    # UPDATED:
    #   Instead of hard-coding num_classes=10,
    #   we read the number of classes from the train dataset.
    #   ImageFolder exposes:
    #       train_dataset.classes → list of class names
    # --------------------------------------------------------
    num_classes = len(train_dataset.classes)
    print("Number of classes detected in mydata/train:", num_classes)
    print("Class names:", train_dataset.classes)

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------
    # UPDATED: pass num_classes detected from your data
    # The rest of the model (conv/pool/etc.) stays the same.
    # --------------------------------------------------------
    model =StaticInitLearnableCNN(num_classes=num_classes)

    # --------------------------------------------------------
    # LOAD OR TRAIN MODEL
    # --------------------------------------------------------
    if os.path.exists(MODEL_PATH):
        print(f"Loading trained dynamic model from: {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("No saved dynamic model found. Training a new model...")
        model = train_model(
            model,
            train_loader,
            device,
            num_epochs=2,
            lr=1e-3
        )
        print(f"Saving trained dynamic model to: {MODEL_PATH}")
        torch.save(model.state_dict(), MODEL_PATH)

    # --------------------------------------------------------
    # DETECTION: RUN MODEL ON A SINGLE TEST IMAGE
    # --------------------------------------------------------
    # detect_single_image is assumed to take:
    #   (model, dataset, device, index)
    # This still works the same with ImageFolder:
    #   test_dataset[index] → (PIL-transformed image tensor, label)
    # --------------------------------------------------------
    detect_single_image(model, test_dataset, device, index=0)


# ------------------------------------------------------------
# RUN PROGRAM
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
