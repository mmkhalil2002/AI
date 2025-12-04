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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------
    # LOSS FUNCTION FOR MULTI-CLASS CLASSIFICATION
    # ------------------------------------------------------------
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
            outputs = model(images)

            # ----------------------------------------
            # LOSS COMPUTATION
            # ----------------------------------------
            loss = criterion(outputs, labels)

            # ----------------------------------------
            # BACKPROPAGATION
            #   Compute gradients for:
            #     • conv1 weights
            #     • conv2 weights
            #     • fully connected weights
            # ----------------------------------------
            loss.backward()

            # ----------------------------------------
            # PARAMETER UPDATE
            #   optimizer changes all learnable weights
            # ----------------------------------------
            optimizer.step()

            # ----------------------------------------
            # STATISTICS
            # ----------------------------------------
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
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

    # --------------------------------------------------------
    # DATA LOADERS (BATCHING)
    # --------------------------------------------------------
    # No change needed in logic — only datasets changed.
    # batch_size controls how many images per training step.
    # --------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    # Optional: test_loader if you want evaluation later
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
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
    model = DynamicLearnableCNN(num_classes=num_classes)

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
