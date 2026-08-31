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
#   loss.backward()
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
# ✅ Dynamic in training
# ✅ Full learning
# ✅ Classic CNN
#

class StaticInitLearnableCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=True)
        self.fc    = nn.Linear(32 * 32 * 32, num_classes)

        # Static initialization (starting point)
        self._init_conv1_static()
        self._init_conv2_static()

    # ----------------------------------------------------------
    # STATIC INITIALIZATION: LAYER 1
    # ----------------------------------------------------------
    def _init_conv1_static(self):
        with torch.no_grad():
            w = self.conv1.weight

            sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,2]])
            sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]])
            laplace = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]])
            sharpen = torch.tensor([[0,-1,0],[-1,5,-1],[0,-1,0]])
            avg = (1/9) * torch.ones((3,3))
            identity = torch.zeros((3,3)); identity[1,1] = 1

            kernels = [sobel_x, sobel_y, laplace, sharpen, avg, identity]

            def rgb(k): return k.repeat(3,1,1)

            for i in range(16):
                w[i].copy_(rgb(kernels[i % len(kernels)]))

    # ----------------------------------------------------------
    # STATIC INITIALIZATION: LAYER 2
    # ----------------------------------------------------------
    def _init_conv2_static(self):
        with torch.no_grad():
            w = self.conv2.weight

            edge_h = torch.tensor([[-1,-1,-1],[2,2,2],[-1,-1,-1]])
            edge_v = torch.tensor([[-1,2,-1],[-1,2,-1],[-1,2,-1]])
            emboss = torch.tensor([[-2,-1,0],[-1,1,1],[0,1,2]])
            avg = (1/9) * torch.ones((3,3))

            kernels = [edge_h, edge_v, emboss, avg]

            def full(k): return k.repeat(16,1,1)

            for i in range(32):
                w[i].copy_(full(kernels[i % len(kernels)]))

    # ----------------------------------------------------------
    # FORWARD PASS
    # ----------------------------------------------------------
    def forward(self, x):
        x = F.relu(self.conv1(x))   # Layer 1 learns
        x = F.relu(self.conv2(x))   # Layer 2 learns
        x = torch.flatten(x, 1)
        x = self.fc(x)              # Classifier learns
        return x
