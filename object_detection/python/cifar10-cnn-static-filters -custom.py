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
