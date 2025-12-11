import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
MODEL_PATH = "../../../"
MODEL_FILENAME = "cifar10-cnn-filters-model-file"
DATA_PATH = "../../../cifar10_data"
IMG_WIDTH, IMG_HEIGHT = 32, 32  # Based on training dataset
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for valid detections
FILTER_WIDTH = 3
FILTER_HEIGHT = 3
BATCH_SIZE = 32
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
DYNAMIC_FILTERS = False


# ======================================================================
# GLOBAL DEBUG SWITCH
# ======================================================================
DEBUG_FLAG = True   # ✅ Change to False to disable debug logs


# ======================================================================
# DEBUG PRINT HELPER
# ======================================================================
def debug_print(*args, **kwargs):
    """
    Prints only when DEBUG_FLAG is enabled.
    Works exactly like print().
    """
    if DEBUG_FLAG:
        print(*args, **kwargs)


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
# NETWORK SHAPE (CIFAR-10 EXAMPLE WITH POOLING)
# ============================================================
#
# Input image:                [3  x 32 x 32]
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
# ✅ Full learning
# ✅ Classical CNN
#


class StaticInitLearnableCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ------------------------------------------------------
        # LAYER 1: 3 → 16 channels
        # 3 input channels (RGB) → 16 feature maps using 3x3 filters
        # Padding = 1 to keep spatial size 32x32
        # ------------------------------------------------------
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)

        # ------------------------------------------------------
        # LAYER 2: 16 → 32 channels
        # 16 input feature maps → 32 feature maps using 3x3 filters
        # Padding = 1 to keep spatial size (before pooling)
        # ------------------------------------------------------
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)

        # ------------------------------------------------------
        # POOLING LAYER: MaxPool2d(2, 2)
        #
        # Max pooling with:
        #   kernel_size = 2
        #   stride      = 2
        #
        # Effect:
        #   spatial size is halved in each dimension:
        #       32x32 → 16x16
        #       16x16 →  8x8
        #
        # We will reuse the SAME pool layer twice (after conv1 and conv2).
        # ------------------------------------------------------
        self.pool = nn.MaxPool2d(2, 2)

        # ------------------------------------------------------
        # FULLY CONNECTED CLASSIFIER
        #
        # After:
        #   conv1 + pool → [16 x 16 x 16]
        #   conv2 + pool → [32 x  8 x  8]
        #
        # Flattened feature vector size:
        #   32 * 8 * 8 = 2048
        #
        # So fc in_features = 2048, out_features = num_classes (CIFAR-10 → 10).
        # ------------------------------------------------------
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

        # Static initialization (starting point only)
        if DYNAMIC_FILTERS:
          self._init_conv1_static()
          self._init_conv2_static()

    # ----------------------------------------------------------
    # STATIC INITIALIZATION FOR LAYER 1
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # STATIC INITIALIZATION FOR LAYER 1
    # ----------------------------------------------------------
    def _init_conv1_static(self):
        with torch.no_grad():                                              # disable gradients during manual init
            w = self.conv1.weight                                          # conv1 weights → [out_channels, 3, 3, 3]
            out_channels, in_channels, kh, kw = w.shape                    # get conv1 shape
            assert in_channels == 3 and kh == 3 and kw == 3                # expect RGB input and 3x3 kernels

            identity = torch.tensor([[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]])    # identity filter 2D
            edge_detection = torch.tensor([[0.,-1.,0.],[-1.,4.,-1.],[0.,-1.,0.]])   # laplacian edge 2D
            sharpen = torch.tensor([[0.,-1.,0.],[-1.,5.,-1.],[0.,-1.,0.]]) # sharpen 2D
            box_blur = (1/9) * torch.ones((3,3))                           # box blur 2D
            gaussian_blur = (1/16) * torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])  # gaussian 2D

            edge_0   = torch.tensor([[ 1.,1.,1.],[-2.,-2.,-2.],[ 1.,1.,1.]])          # edge 0°
            edge_45  = torch.tensor([[ 1.,-2.,1.],[1.,-2.,1.],[1.,-2.,1.]])           # edge 45°
            edge_90  = torch.tensor([[-2.,1.,1.],[1.,-2.,1.],[1.,1.,-2.]])            # edge 90°
            edge_135 = torch.tensor([[1.,1.,-2.],[-2.,1.,1.],[1.,-2.,-2.]])           # edge 135°
            edge_180 = torch.tensor([[-1.,-1.,-1.],[2.,2.,2.],[-1.,-1.,-1.]])         # edge 180°
            edge_225 = torch.tensor([[2.,-1.,-1.],[-1.,2.,-1.],[-1.,-1.,2.]])         # edge 225°
            edge_270 = torch.tensor([[-1.,2.,-1.],[-1.,2.,-1.],[-1.,2.,-1.]])         # edge 270°
            edge_315 = torch.tensor([[1.,-1.,-1.],[-1.,1.,1.],[-1.,-1.,1.]])          # edge 315°

            corner_0   = torch.tensor([[1.,1.,0.],[1.,0.,-1.],[0.,-1.,-1.]])          # corner 0°
            corner_45  = torch.tensor([[1.,0.,1.],[0.,-1.,1.],[-1.,-1.,0.]])          # corner 45°
            corner_90  = torch.tensor([[0.,1.,1.],[-1.,0.,1.],[-1.,-1.,0.]])          # corner 90°
            corner_135 = torch.tensor([[1.,0.,-1.],[0.,1.,1.],[0.,-1.,-1.]])          # corner 135°
            corner_180 = corner_0.clone()                                             # corner 180°
            corner_225 = corner_45.clone()                                            # corner 225°
            corner_270 = corner_90.clone()                                            # corner 270°
            corner_315 = corner_135.clone()                                           # corner 315°

            curve_0   = torch.tensor([[ 0.,1.,0.],[-1.,1.,-1.],[0.,-1.,0.]])          # curve 0°
            curve_45  = torch.tensor([[ 1.,0.,-1.],[0.,1.,0.],[-1.,0.,1.]])           # curve 45°
            curve_90  = torch.tensor([[ 0.,-1.,0.],[1.,1.,1.],[0.,-1.,0.]])           # curve 90°
            curve_135 = torch.tensor([[-1.,0.,1.],[0.,1.,0.],[1.,0.,-1.]])            # curve 135°
            curve_180 = torch.tensor([[ 0.,-1.,0.],[-1.,1.,-1.],[0.,1.,0.]])          # curve 180°
            curve_225 = curve_135.clone()                                             # curve 225°
            curve_270 = curve_90.clone()                                              # curve 270°
            curve_315 = curve_45.clone()                                              # curve 315°

            line_0   = torch.tensor([[1.,1.,1.],[-2.,-2.,-2.],[1.,1.,1.]])            # line 0°
            line_45  = torch.tensor([[-2.,1.,1.],[1.,-2.,1.],[1.,1.,-2.]])            # line 45°
            line_90  = torch.tensor([[-1.,-1.,-1.],[2.,2.,2.],[-1.,-1.,-1.]])         # line 90°
            line_135 = torch.tensor([[1.,-2.,1.],[1.,-2.,1.],[1.,-2.,1.]])            # line 135°
            line_180 = torch.tensor([[-1.,-1.,-1.],[-2.,-2.,-2.],[-1.,-1.,-1.]])      # line 180°
            line_225 = torch.tensor([[1.,1.,-2.],[1.,-2.,1.],[-2.,1.,1.]])            # line 225°
            line_270 = torch.tensor([[-1.,2.,-1.],[-1.,2.,-1.],[-1.,2.,-1.]])         # line 270°
            line_315 = torch.tensor([[1.,-1.,1.],[-1.,-2.,-1.],[1.,-1.,1.]])          # line 315°

            sobel_0   = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])           # sobel 0°
            sobel_45  = torch.tensor([[0.,-1.,-2.],[1.,0.,-1.],[2.,1.,0.]])           # sobel 45°
            sobel_90  = torch.tensor([[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]])           # sobel 90°
            sobel_135 = torch.tensor([[2.,1.,0.],[1.,0.,-1.],[0.,-1.,-2.]])           # sobel 135°
            sobel_180 = torch.tensor([[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]])           # sobel 180°
            sobel_225 = torch.tensor([[0.,1.,2.],[-1.,0.,1.],[-2.,-1.,0.]])           # sobel 225°
            sobel_270 = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]])           # sobel 270°
            sobel_315 = torch.tensor([[-2.,-1.,0.],[-1.,0.,1.],[0.,1.,2.]])           # sobel 315°

            kernels = [identity, edge_detection, sharpen, box_blur, gaussian_blur,
                    edge_0, edge_45, edge_90, edge_135, edge_180, edge_225, edge_270, edge_315,
                    corner_0, corner_45, corner_90, corner_135, corner_180, corner_225, corner_270, corner_315,
                    curve_0, curve_45, curve_90, curve_135, curve_180, curve_225, curve_270, curve_315,
                    line_0, line_45, line_90, line_135, line_180, line_225, line_270, line_315,
                    sobel_0, sobel_45, sobel_90, sobel_135, sobel_180, sobel_225, sobel_270, sobel_315]

            num_kernels = len(kernels)                                      # count number of base kernels

            for i in range(out_channels):                                  # loop over each output filter
                k2d = kernels[i % num_kernels].to(w.dtype)                 # pick 2D kernel and cast dtype
                for c in range(in_channels):                               # assign same 2D kernel to each input channel
                    w[i, c].copy_(k2d)                                     # copy 3x3 into w[out, in, :, :]

            print(f"[init_conv1_static] {out_channels} filters initialized with 2D 3x3 kernels")  # log








    # ----------------------------------------------------------
    # STATIC INITIALIZATION FOR LAYER 2
    # ----------------------------------------------------------
    def _init_conv2_static(self):
        with torch.no_grad():                                                           # disable gradients
            w = self.conv2.weight                                                       # conv2 weights → [32,16,3,3]
            out_channels, in_channels, kh, kw = w.shape                                 # get shape
            assert kh == 3 and kw == 3                                                  # expect 3x3 kernels

            edge_h = torch.tensor([[-1.,-1.,-1.],[2.,2.,2.],[-1.,-1.,-1.]])             # horizontal edge 2D
            edge_v = torch.tensor([[-1.,2.,-1.],[-1.,2.,-1.],[-1.,2.,-1.]])             # vertical edge 2D
            emboss = torch.tensor([[-2.,-1.,0.],[-1.,1.,1.],[0.,1.,2.]])                # emboss 2D
            avg = (1/9) * torch.ones((3,3))                                             # average blur 2D
            sobel_x = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])               # sobel x 2D
            sobel_y = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]])               # sobel y 2D

            kernels = [edge_h, edge_v, emboss, avg, sobel_x, sobel_y]                   # kernel bank 2D
            num_kernels = len(kernels)                                                  # count kernels

            for out_idx in range(out_channels):                                         # loop over 32 output filters
                for in_idx in range(in_channels):                                       # loop over 16 input channels
                    k = kernels[(out_idx * in_idx) % num_kernels].to(w.dtype)           # choose 2D kernel pattern
                    w[out_idx, in_idx].copy_(k)                                         # assign 3x3 into w[out,in,:,:]

            print(f"[init_conv2_static] {out_channels}x{in_channels} 2D 3x3 kernels assigned")  # log





    # ----------------------------------------------------------
    # FORWARD PASS
    # ----------------------------------------------------------
    def forward(self, x):
        # x shape at input:
        #   [B, 3, 32, 32]  where:
        #      B = batch size
        #      3 = RGB channels
        #     32x32 = spatial size (CIFAR-10)
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

def train_model(model, train_loader, device, num_epochs=2, lr=LEARNING_RATE):
    
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    # ------------------------------------------------------------
    # LOSS FUNCTION FOR MULTI-CLASS CLASSIFICATION
    # ------------------------------------------------------------
    # Define the loss function used to TRAIN the model.
    #
    # nn.CrossEntropyLoss() is used for *multi-class classification* problems,
    # such as CIFAR-10, ImageNet, digit recognition, or object-category classification.
    #
    # The loss function measures:
    #   → How wrong the model’s predictions are compared to the correct label.
    #
    # The model produces RAW SCORES (called "logits"), not probabilities.
    # Example output from the network:
    #     [2.1, -0.9, 0.5, 1.2]
    # These values do NOT have to sum to 1.
    #
    # Internally, CrossEntropyLoss does TWO operations automatically:
    #
    #   1) LogSoftmax:
    #       • Converts logits into probabilities.
    #       • Ensures outputs sum to 1.
    #       • Pushes confident predictions higher.
    #
    #   2) Negative Log Likelihood (NLLLoss):
    #       • Penalizes the model based on how unlikely the correct class is.
    #
    # Mathematically:
    #   loss = -log( predicted_probability_of_correct_class )
    #
    # Example:
    #   true label = 0
    #   predicted probabilities = [0.05, 0.02, 0.80, 0.13]
    #   loss = -log(0.05) = large penalty (bad prediction)
    #
    #   predicted probabilities = [0.90, 0.03, 0.02, 0.05]
    #   loss = -log(0.90) = small penalty (good prediction)
    #
    # Expected input:
    #   • model output: tensor of shape (batch_size, num_classes)
    #   • target labels: tensor of shape (batch_size)
    #
    # Important rules:
    #   • DO NOT apply Softmax in the model before this loss.
    #   • Feed RAW logits directly into CrossEntropyLoss.
    #
    # The output:
    #   • A single scalar number (average loss for the batch).
    #   • Lower value → better prediction.
    #   • Higher value → worse prediction.
    #
    # This loss function drives learning:
    #   → It creates gradients that flow backward through:
    #         classifier
    #         conv layers
    #         filters
    #         feature extractors
    #   → Tells every trainable parameter how to change to reduce error.
    #
    # CrossEntropyLoss gracefully handles:
    #   • Wrong confident predictions (large penalty)
    #   • Slight mistakes (small penalty)
    #   • Class competition (only the true class is rewarded)
    #
    # This is why CrossEntropyLoss is ideal for:
    #   • Image classification
    #   • NLP classification
    #   • Speech recognition
    #   • Any task with mutually-exclusive classes
    #
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------
    # OPTIONAL: STORE EPOCH TIMES FOR ANALYSIS
    #   • epoch_times will hold the duration (in seconds) of each epoch.
    #   • Useful for performance debugging and estimating total training time.
    # ------------------------------------------------------------
    epoch_times = []

    # ------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------
    for ep in range(num_epochs):

        # --------------------------------------------------------
        # START TIMER FOR THIS EPOCH
        #   • time.perf_counter() gives a high-resolution timestamp.
        #   • We subtract later to get the duration of this epoch.
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

            # ----------------------------------------
            # FORWARD PASS
            #   Images → Conv layers → Pooling → FC
            # ----------------------------------------
           # Run the neural network on the input images (this EXECUTES model.forward()).
            #
            # When you write:
            #     outputs = model(images)
            # PyTorch automatically calls:
            #     model.forward(images)
            #
            # The __call__() method wraps forward() and also:
            #   • sets up Autograd graph tracking
            #   • handles hooks (if any)
            #   • manages training/evaluation mode behaviors (Dropout, BatchNorm)
            #
            # What "images" contains:
            #   A batch tensor of shape:
            #       (batch_size, channels, height, width)
            #   Example:
            #       (64, 3, 32, 32) for CIFAR-10
            #
            # What happens internally:
            #
            #   1) images are passed into conv1:
            #       - Extracts low-level features like edges and textures.
            #
            #   2) Output goes through activation function (e.g., ReLU):
            #       - Adds non-linearity so the network can model complex patterns.
            #
            #   3) Results propagate into conv2 / deeper layers:
            #       - These layers learn shapes, objects, and class patterns.
            #
            #   4) Feature maps are flattened:
            #       - Converts spatial tensors into vectors for classification layers.
            #
            #   5) Fully connected layers map features → class scores.
            #
            #   6) The final output is a logits tensor:
            #       - One value per class.
            #       - Raw scores (not probabilities).
            #
            # What "outputs" represents:
            #   A tensor with shape:
            #       (batch_size, num_classes)
            #
            #   Each row:
            #       → Score for each class.
            #
            # Development notes:
            #   • forward() is called EXACTLY ONCE by this line.
            #   • The execution order is defined in the model's forward method body.
            #   • You do NOT call forward() manually.
            #
            # During training:
            #   • Autograd tracks this entire computation chain.
            #   • Every math operation builds the computational graph.
            #   • This is required for loss.backward() to work.
            #
            # During inference:
            #   • forward() still runs.
            #   • Gradients may be disabled with torch.no_grad().
            #
            # IMPORTANT:
            #   • DO NOT put Softmax at the end of the model if you use CrossEntropyLoss.
            #   • That loss expects raw logits, not normalized probabilities.
            #
            # Summary:
            #   This ONE LINE runs your CNN, extracts features, produces class scores,
            #   and creates the graph used for gradient computation.
            #
            outputs = model(images)



            # ----------------------------------------
            # LOSS COMPUTATION
            # ----------------------------------------

            # Compute the training LOSS between model predictions and true labels.
            #
            # When you write:
            #     loss = criterion(outputs, labels)
            # PyTorch executes the forward logic of CrossEntropyLoss (or whichever loss is assigned to 'criterion').
            #
            # What "outputs" contains:
            #   • Raw logits from the model (NOT probabilities).
            #   • Tensor shape:
            #         (batch_size, num_classes)
            #
            # What "labels" contains:
            #   • Ground truth class indices.
            #   • Tensor shape:
            #         (batch_size)
            #   • Each value is the correct class index for an input sample.
            #     Example:
            #       outputs = [[2.1, -0.4, 1.0]]
            #       labels  = [2]
            #
            # What this function does INTERNALLY:
            #
            #   1) Applies LogSoftmax to each output row:
            #         logits → log(probabilities)
            #
            #   2) Extracts the log-probability of the correct class for each sample.
            #
            #   3) Applies Negative Log Likelihood loss:
            #         loss_i = -log(p_true_class)
            #
            #   4) Averages across the batch to produce one scalar value:
            #         final_loss = mean(loss_i)
            #
            # Mathematical form:
            #   loss = - (1 / N) × Σ log( softmax(outputs)[i][labels[i]] )
            #
            # Functional meaning:
            #   • Small loss → model is confident AND correct.
            #   • Large loss → model is wrong or uncertain.
            #
            # Differentiability:
            #   • This produces a scalar TENSOR with grad_fn attached.
            #   • PyTorch tracks this value inside the computation graph.
            #
            # After this line:
            #   • loss.backward() can compute ∂loss / ∂weights automatically.
            #   • Each trainable parameter receives its gradient.
            #
            # Common mistakes to avoid:
            #   ❌ Applying Softmax to outputs BEFORE this line.
            #   ❌ Giving one-hot encoded labels instead of class indices.
            #   ❌ Passing float labels instead of LongTensor class IDs.
            #
            # Debug tip:
            #   • If loss == NaN or very large, inspect:
            #         - labels range
            #         - output magnitude
            #         - batch normalization stability
            #
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


        # --------------------------------------------------------
        # END TIMER FOR THIS EPOCH
        #   • Compute how long this epoch took in seconds.
        #   • Store it for later summary or plotting.
        # --------------------------------------------------------
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        # --------------------------------------------
        # PRINT EPOCH SUMMARY
        #   • Includes loss, accuracy, and elapsed time.
        # --------------------------------------------
        print(
            f"[TRAIN] Epoch {ep+1}/{num_epochs}  "
            f"Loss: {running_loss / total:.4f}  "
            f"Accuracy: {correct / total:.4f}  "
            f"Time: {epoch_time:.2f} sec"
        )

    # ------------------------------------------------------------
    # OPTIONAL: PRINT TOTAL AND AVERAGE TRAINING TIME
    # ------------------------------------------------------------
    total_time = sum(epoch_times)
    avg_time = total_time / len(epoch_times) if epoch_times else 0.0
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
    class_names = test_dataset.classes

    # --------------------------------------------------------
    # SELECT RANDOM INDEX IF NONE PROVIDED
    # --------------------------------------------------------
    if index is None:
        index = random.randint(0, len(test_dataset) - 1)

    # --------------------------------------------------------
    # LOAD IMAGE + TRUE LABEL
    # --------------------------------------------------------
    img, true_label = test_dataset[index]

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
    true_name = class_names[true_label]
    pred_name = class_names[pred_label]

    # --------------------------------------------------------
    # PRINT RESULTS
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
    #MODEL_PATH = "static_init_cnn_cifar10.pth"

    # --------------------------------------------------------
    # CIFAR-10 DATA TRANSFORMS
    # --------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])


    
    model_path = os.path.join(MODEL_PATH, "data")
    print(model_path)   # → /home/user/myproject/data
    # --------------------------------------------------------
    # LOAD CIFAR-10 TRAIN AND TEST SETS
    # --------------------------------------------------------
    train_dataset = datasets.CIFAR10(
        root=model_path,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=model_path,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------
    model = StaticInitLearnableCNN(num_classes=10)

    # --------------------------------------------------------
    # LOAD OR TRAIN MODEL
    # --------------------------------------------------------
    model_filename = os.path.join(MODEL_PATH, MODEL_FILENAME)

   
    # ------------------------------------------------------------
    # LOAD MODEL IF IT EXISTS
    # ------------------------------------------------------------
    if os.path.exists(model_filename):
        print(f"Loading trained weights from: {model_filename}")
        state_dict = torch.load(model_filename, map_location=device)   # load weights
        model.load_state_dict(state_dict)                              # restore model
    else:
        print("No saved model found. Training a new model...")
        model = train_model(model, train_loader, device, num_epochs=NUM_EPOCHS, lr=1e-3)
        print(f"Saving trained model to: {model_filename}")
        torch.save(model.state_dict(), model_filename)

    # ------------------------------------------------------------
    # INTERACTIVE LOOP FOR USER-DRIVEN DETECTION
    # ------------------------------------------------------------
    print("\n--------------------------------------------------")
    print("Interactive Image Detection Mode")
    print("Press:")
    print("   d  → detect on an image index")
    print("   e  → exit program")
    print("--------------------------------------------------\n")

    while True:
        user_input = input("Enter command (d = detect, e = exit): ").strip().lower()

        if user_input == 'e':
            print("Exiting program. Goodbye!")
            break

        elif user_input == 'd':
            # Ask user for the test image index
            idx_str = input(f"Enter image index (0 – {len(test_dataset)-1}): ").strip()

            # Validate the index
            if not idx_str.isdigit():
                print("❌ Invalid index. Must be a number.")
                continue

            idx = int(idx_str)

            if idx < 0 or idx >= len(test_dataset):
                print("❌ Index out of range. Try again.")
                continue

            # Run detection
            print(f"\nRunning detection on test image index {idx} ...")
            detect_single_image(model, test_dataset, device, index=idx)

        else:
            print("❌ Unknown command. Use 'd' for detect or 'e' to exit.")



# ------------------------------------------------------------
# RUN PROGRAM
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
