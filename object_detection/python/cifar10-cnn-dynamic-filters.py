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
#   • Layer 3 (fc)    is a standard fully connected classifier
#
# IMPORTANT:
# ----------
# Here we do NOT manually define any "static" filters.
# The convolution filters are initialized by PyTorch (randomly)
# and are trained END-TO-END from the data.
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
# We do NEITHER here, so EVERY layer is fully trainable.
#
# ============================================================
# WHY THIS IS A CLASSICAL NEURAL NETWORK
# ============================================================
#
# Because:
#
#   • Filters start from random initialization
#   • Filters are updated by backpropagation
#   • The entire network (conv + fc) is trained on data
#
# This is the typical CNN used in most literature.
#
# ============================================================
# NETWORK SHAPE (CIFAR-10 EXAMPLE)
# ============================================================
#
# Input image:        [3 x 32 x 32]
# After conv1:        [16 x 32 x 32]
# After conv2:        [32 x 32 x 32]
# After flattening:   [32*32*32]
# Output layer:       [C classes]
#
# ============================================================
# SUMMARY
# ============================================================
#
# ✅ No manual/static filters
# ✅ Dynamic (random) initialization
# ✅ Full learning in all layers
# ✅ Classic CNN as used in most practice
#


class DynamicLearnableCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ------------------------------------------------------
        # LAYER 1: 3 → 16 channels with 3x3 dynamic filters
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
        # ------------------------------------------------------
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,   # keep 32x32
            bias=True
        )

        # ------------------------------------------------------
        # FULLY CONNECTED LAYER
        # After conv layers (no pooling): [32 x 32 x 32]
        # Flattened feature vector = 32 * 32 * 32
        # ------------------------------------------------------
        self.fc = nn.Linear(32 * 32 * 32, num_classes)

        # NOTE:
        # We do NOT override weights with static kernels here.
        # PyTorch's default initialization is used.
        # All parameters are learnable by default.

    # ----------------------------------------------------------
    # FORWARD PASS
    # ----------------------------------------------------------
    def forward(self, x):
        # x: [B, 3, 32, 32]
        x = F.relu(self.conv1(x))   # [B, 16, 32, 32]
        x = F.relu(self.conv2(x))   # [B, 32, 32, 32]
        x = torch.flatten(x, 1)     # [B, 32*32*32]
        x = self.fc(x)              # [B, num_classes]
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

        - True label
        - Predicted label

    Also returns (image_tensor, true_label, predicted_label).
    """
    model.to(device)
    model.eval()

    # CIFAR-10 label names for readability
    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    # Get one sample from the test dataset
    img, label = test_dataset[index]   # img: [3, 32, 32] (already transformed)
    img_input = img.unsqueeze(0).to(device)  # add batch dimension → [1, 3, 32, 32]

    with torch.no_grad():
        logits = model(img_input)
        pred_class = logits.argmax(1).item()

    true_name = cifar10_classes[label]
    pred_name = cifar10_classes[pred_class]

    print("--------------------------------------------------")
    print(f"[Dynamic] DETECTION RESULT FOR TEST IMAGE INDEX: {index}")
    print(f"True label index: {label}  →  {true_name}")
    print(f"Pred label index: {pred_class}  →  {pred_name}")
    print("--------------------------------------------------")

    return img, label, pred_class


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
