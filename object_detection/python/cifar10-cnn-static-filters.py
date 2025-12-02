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
# Because:
#   - we did NOT freeze any layer
#   - requires_grad = True for all parameters
#
# Then:
#
#   optimizer.step()
#
# updates ALL three layers.
#
# ============================================================
# SO: WILL ALL 3 LAYERS LEARN?
# ============================================================
#
# YES.
#
#   • Layer 1 learns
#   • Layer 2 learns
#   • Layer 3 learns
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
#
# This is exactly how CNNs are trained in practice, except
# most networks start with RANDOM initialization.
#
# Here you start with INTELLIGENT initialization.
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
# ✅ Static at start
# ✅ Dynamic during training
# ✅ Full learning
# ✅ Classical CNN
#


class StaticInitLearnableCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ------------------------------------------------------
        # LAYER 1: 3 → 16 channels
        # ------------------------------------------------------
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)

        # ------------------------------------------------------
        # LAYER 2: 16 → 32 channels
        # ------------------------------------------------------
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)

        # ------------------------------------------------------
        # FULLY CONNECTED CLASSIFIER
        # ------------------------------------------------------
        self.fc = nn.Linear(32 * 32 * 32, num_classes)

        # Static initialization (starting point only)
        self._init_conv1_static()
        self._init_conv2_static()

    # ----------------------------------------------------------
    # STATIC INITIALIZATION FOR LAYER 1
    # ----------------------------------------------------------
    def _init_conv1_static(self):
        with torch.no_grad():
            w = self.conv1.weight  # shape: [16, 3, 3, 3]

            sobel_x = torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 2]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]], dtype=torch.float32)
            laplacian = torch.tensor([[ 0, -1,  0],
                                      [-1,  4, -1],
                                      [ 0, -1,  0]], dtype=torch.float32)
            sharpen = torch.tensor([[ 0, -1,  0],
                                    [-1,  5, -1],
                                    [ 0, -1,  0]], dtype=torch.float32)
            avg = (1/9) * torch.ones((3, 3), dtype=torch.float32)
            identity = torch.zeros((3, 3), dtype=torch.float32)
            identity[1, 1] = 1.0

            kernels = [sobel_x, sobel_y, laplacian, sharpen, avg, identity]

            def rgb(k):
                # replicate single-channel 3x3 kernel to 3 input channels (RGB)
                return k.repeat(3, 1, 1)  # [3, 3, 3]

            for i in range(16):
                w[i].copy_(rgb(kernels[i % len(kernels)]))

    # ----------------------------------------------------------
    # STATIC INITIALIZATION FOR LAYER 2
    # ----------------------------------------------------------
    def _init_conv2_static(self):
        with torch.no_grad():
            w = self.conv2.weight  # shape: [32, 16, 3, 3]

            edge_h = torch.tensor([[-1, -1, -1],
                                   [ 2,  2,  2],
                                   [-1, -1, -1]], dtype=torch.float32)
            edge_v = torch.tensor([[-1,  2, -1],
                                   [-1,  2, -1],
                                   [-1,  2, -1]], dtype=torch.float32)
            emboss = torch.tensor([[-2, -1, 0],
                                   [-1,  1, 1],
                                   [ 0,  1, 2]], dtype=torch.float32)
            avg = (1/9) * torch.ones((3, 3), dtype=torch.float32)

            kernels = [edge_h, edge_v, emboss, avg]

            def full(k):
                # repeat same kernel for all 16 input channels
                return k.repeat(16, 1, 1)  # [16, 3, 3]

            for i in range(32):
                w[i].copy_(full(kernels[i % len(kernels)]))

    # ----------------------------------------------------------
    # FORWARD PASS
    # ----------------------------------------------------------
    def forward(self, x):
        x = F.relu(self.conv1(x))     # Layer 1 learns
        x = F.relu(self.conv2(x))     # Layer 2 learns
        x = torch.flatten(x, 1)       # [B, 32*32*32]
        x = self.fc(x)                # Classifier learns
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

        print(f"[TRAIN] Epoch {ep+1}/{num_epochs}  "
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
    print(f"DETECTION RESULT FOR TEST IMAGE INDEX: {index}")
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
    MODEL_PATH = "static_init_cnn_cifar10.pth"

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
    model = StaticInitLearnableCNN(num_classes=10)

    # --------------------------------------------------------
    # LOAD OR TRAIN MODEL
    # --------------------------------------------------------
    if os.path.exists(MODEL_PATH):
        print(f"Loading trained weights from: {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("No saved model found. Training a new model...")
        model = train_model(model, train_loader, device, num_epochs=2, lr=1e-3)
        print(f"Saving trained model to: {MODEL_PATH}")
        torch.save(model.state_dict(), MODEL_PATH)

    # --------------------------------------------------------
    # DETECTION: RUN MODEL ON A SINGLE TEST IMAGE
    # --------------------------------------------------------
    # Example: test on test_dataset[0]
    detect_single_image(model, test_dataset, device, index=0)


# ------------------------------------------------------------
# RUN PROGRAM
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
