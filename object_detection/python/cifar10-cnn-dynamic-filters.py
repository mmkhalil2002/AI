import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# EXPLANATION: DYNAMIC FILTER CNN (FULLY LEARNABLE)
# ============================================================
#
# This network is a CLASSICAL CONVOLUTIONAL NEURAL NETWORK (CNN)
# where:
#
#   • Layer 1 (conv1) uses dynamic 3x3 filters (random init)
#   • Layer 2 (conv2) uses dynamic 3x3 filters (random init)
#   • Between conv layers, we apply MAX POOLING to shrink feature maps
#   • Layer 3 (fc)    is a standard fully connected classifier
#
# IMPORTANT:
# ----------
# Here we do NOT manually define any "static" filters.
# The convolution filters are initialized by PyTorch (randomly)
# and are trained END-TO-END from the data.
#
# Pooling layers (MaxPool2d) have NO learnable parameters.
# They only perform a fixed mathematical operation that reduces
# spatial size and keeps the strongest responses.
#
# This is the standard way CNNs are usually trained.
#
# ============================================================
# HOW LEARNING HAPPENS
# ============================================================
#
# In the training loop:
#
#   outputs = model(images)
#   loss    = criterion(outputs, labels)
#   loss.backward()
#   optimizer.step()
#
# PyTorch computes gradients for:
#
#   • conv1.weight
#   • conv1.bias
#   • conv2.weight
#   • conv2.bias
#   • fc.weight
#   • fc.bias
#
# (Pooling has no weights/biases, so there is nothing to train there.)
#
# Because:
#   - All parameters have requires_grad = True (default)
#   - We pass model.parameters() to the optimizer
#
# Then optimizer.step() updates ALL of them:
#
#   param := param - lr * grad
#
# So in this model:
#
#   • Layer 1 learns filters
#   • Layer 2 learns filters
#   • Layer 3 learns classifier weights
#
# ============================================================
# WHAT WOULD STOP LEARNING?
# ============================================================
#
# A layer would stop learning if you:
#
#   1) Set requires_grad = False on its parameters, OR
#   2) Do not include its parameters in the optimizer.
#
# We do NEITHER here, so EVERY learnable layer is fully trainable.
#
# ============================================================
# WHY THIS IS A CLASSICAL NEURAL NETWORK
# ============================================================
#
# Because:
#
#   • Filters start from random initialization
#   • Filters are updated by backpropagation
#   • Pooling is used to reduce spatial resolution (2x2 windows)
#   • The entire network (conv + fc) is trained on data
#
# This is the typical CNN used in most literature.
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
# After flattening:           [32*8*8] = 2048
# Output layer (fc):          [C classes]
#
# ============================================================
# SUMMARY
# ============================================================
#
# ✅ No manual/static filters
# ✅ Dynamic (random) initialization
# ✅ Pooling reduces spatial size and keeps strong activations
# ✅ Full learning in all layers (conv + fc)
# ✅ Classic CNN as used in most practice
#


class DynamicLearnableCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ------------------------------------------------------
        # LAYER 1: 3 → 16 channels with 3x3 dynamic filters
        #
        # in_channels  = 3  (RGB image)
        # out_channels = 16 (number of learned feature maps)
        # kernel_size  = 3x3
        # padding      = 1 to keep spatial size at 32x32
        # ------------------------------------------------------
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1,   # keep 32x32 for CIFAR-10
            bias=True
        )

        # ------------------------------------------------------
        # LAYER 2: 16 → 32 channels with 3x3 dynamic filters
        #
        # in_channels  = 16 (output of conv1)
        # out_channels = 32 (more feature maps)
        # kernel_size  = 3x3
        # padding      = 1 to keep spatial size before pooling
        # ------------------------------------------------------
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,   # keep 16x16 before pooling
            bias=True
        )

        # ------------------------------------------------------
        # POOLING LAYER: MaxPool2d(2, 2)
        #
        # kernel_size = 2
        # stride      = 2
        #
        # Effect:
        #   Spatial size is divided by 2 in each dimension:
        #     32x32 → 16x16
        #     16x16 →  8x8
        #
        # There are NO weights here. Pooling is a fixed operation.
        # We will reuse this same pool after conv1 and after conv2.
        # ------------------------------------------------------
        self.pool = nn.MaxPool2d(2, 2)

        # ------------------------------------------------------
        # FULLY CONNECTED LAYER
        #
        # After:
        #   conv1 + pool → [16 x 16 x 16]
        #   conv2 + pool → [32 x  8 x  8]
        #
        # Flattened feature vector size:
        #   32 * 8 * 8 = 2048
        #
        # So fc in_features = 2048, out_features = num_classes.
        # ------------------------------------------------------
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

        # NOTE:
        # We do NOT override weights with static kernels here.
        # PyTorch's default initialization is used.
        # All parameters (conv + fc) are learnable by default.





    # ----------------------------------------------------------
    # FORWARD PASS
    # ----------------------------------------------------------
    def forward(self, x):
        # x: input batch of images
        # shape: [B, 3, 32, 32]
        #   B = batch size
        #   3 = RGB channels
        #   32x32 = CIFAR-10 spatial size

        x = F.relu(self.conv1(x))   # After conv1: [B, 16, 32, 32]
        x = self.pool(x)            # After pool1: [B, 16, 16, 16]

        x = F.relu(self.conv2(x))   # After conv2: [B, 32, 16, 16]
        x = self.pool(x)            # After pool2: [B, 32,  8,  8]

        # Flatten all spatial dimensions into a single feature vector
        # Before flatten: [B, 32, 8, 8]
        # After flatten:  [B, 32*8*8] = [B, 2048]
        x = torch.flatten(x, 1)

        # Fully connected classifier:
        # Input:  [B, 2048]
        # Output: [B, num_classes]
        x = self.fc(x)

        return x



# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(model, train_loader, device, num_epochs=2, lr=1e-3):
    """
    Trains the model on the given train_loader for num_epochs.
    Returns the trained model.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(num_epochs):
        total = 0
        correct = 0
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"[TRAIN Dynamic] Epoch {ep+1}/{num_epochs}  "
              f"Loss: {running_loss / total:.4f}  "
              f"Accuracy: {correct / total:.4f}")

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

    Works with:
        • CIFAR-10
        • ImageFolder
        • Custom datasets (labels auto-detected)

    Returns:
        (image_tensor, true_label, predicted_label)
    """
    # --------------------------------------------------------
    # MOVE MODEL TO DEVICE AND SWITCH TO EVAL MODE
    # --------------------------------------------------------
    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # READ CLASS NAMES FROM THE DATASET (AUTOMATIC)
    # --------------------------------------------------------
    # Torchvision exposes class names via dataset.classes
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
    # LABEL ID → CLASS NAME
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
    MODEL_PATH = "dynamic_cnn_cifar10.pth"

    # --------------------------------------------------------
    # CIFAR-10 DATA TRANSFORMS
    # --------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # --------------------------------------------------------
    # LOAD CIFAR-10 TRAIN AND TEST SETS
    # --------------------------------------------------------
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------
    model = DynamicLearnableCNN(num_classes=10)

    # --------------------------------------------------------
    # LOAD OR TRAIN MODEL
    # --------------------------------------------------------
    if os.path.exists(MODEL_PATH):
        print(f"Loading trained dynamic model from: {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("No saved dynamic model found. Training a new model...")
        model = train_model(model, train_loader, device,
                            num_epochs=2, lr=1e-3)
        print(f"Saving trained dynamic model to: {MODEL_PATH}")
        torch.save(model.state_dict(), MODEL_PATH)

    # --------------------------------------------------------
    # DETECTION: RUN MODEL ON A SINGLE TEST IMAGE
    # --------------------------------------------------------
    detect_single_image(model, test_dataset, device, index=0)


# ------------------------------------------------------------
# RUN PROGRAM
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
