import torch
import torch.nn.functional as F


def manual_conv2d(x, weight, padding=1):
    # =====================================================================
    # 🧠 WHAT x CONTAINS (INPUT IMAGE TENSOR)
    # =====================================================================
    # x has shape: [B, C, H, W]
    #
    # B = Batch size → how many images are processed at once
    # C = Channels   → e.g. 3 = RGB, 1 = grayscale
    # H = Height     → image height in pixels
    # W = Width      → image width in pixels
    #
    # ✅ NUMERIC EXAMPLE:
    #
    # x.shape = [1,1,4,4]
    #
    # x[0,0] =
    # [  1    2    3    4
    #    5    6    7    8
    #    9   10   11   12
    #   13   14   15   16 ]
    #
    # Each number is one pixel intensity.
    #
    # =====================================================================
    # 🧠 WHAT weight CONTAINS (DYNAMIC FILTERS / KERNELS)
    # =====================================================================
    # weight has shape: [F_out, F_in, KH, KW]
    #
    # F_out = Number of filters (output feature maps)
    # F_in  = Number of channels per filter (must match C)
    # KH    = Kernel height
    # KW    = Kernel width
    #
    # These filters ARE NOT designed by hand.
    # PyTorch initializes them using RANDOM values.
    #
    # Example initialization rule (Kaiming / He initialization):
    #   weights ~ Normal(0, sqrt(2 / fan_in))
    #
    # Example:
    # fan_in = C × KH × KW = 1 × 3 × 3 = 9
    # std = sqrt(2/9)
    #
    # ✅ NUMERIC EXAMPLE:
    #
    # weight.shape = [1,1,3,3]
    #
    # weight[0,0] (ONE FILTER) initially random:
    #
    # [  0.12   -0.34    0.08
    #   -0.21    0.05   -0.11
    #    0.03   -0.15    0.07 ]
    #
    # This is NOT an edge detector yet.
    # It's random noise.
    #
    # During training:
    # W_new = W_old - learning_rate * ∂Loss/∂W
    #
    # The filter gradually transforms into:
    # - edges
    # - corners
    # - curves
    # - textures
    #
    # =====================================================================

    # ---------------------------------------------------------
    # Extract dimensions from x
    # ---------------------------------------------------------
    B, C, H, W = x.shape

    # ---------------------------------------------------------
    # Extract dimensions from weight
    # ---------------------------------------------------------
    F_out, F_in, KH, KW = weight.shape

    # ---------------------------------------------------------
    # Allocate output tensor
    # Shape = [B, F_out, H, W]
    # ---------------------------------------------------------
    out = torch.zeros(B, F_out, H, W)

    # ---------------------------------------------------------
    # Apply zero padding
    # padding = 1 → add a 1-pixel zero border
    #
    # Example padded input:
    #
    # [ 0   0   0   0   0   0
    #   0   1   2   3   4   0
    #   0   5   6   7   8   0
    #   0   9  10  11  12   0
    #   0  13  14  15  16   0
    #   0   0   0   0   0   0 ]
    # ---------------------------------------------------------
    padded = F.pad(x, (padding, padding, padding, padding))

    # ---------------------------------------------------------
    # Perform convolution
    # ---------------------------------------------------------
    for b in range(B):  # loop over images
        for f in range(F_out):  # loop over filters
            for i in range(H):  # vertical position
                for j in range(W):  # horizontal position

                    # -----------------------------------------------------
                    # Extract region from padded image
                    #
                    # region.shape = [C, KH, KW]
                    #
                    # ✅ NUMERIC EXAMPLE (at i=0, j=0):
                    #
                    # region =
                    # [ 0   0   0
                    #   0   1   2
                    #   0   5   6 ]
                    #
                    # -----------------------------------------------------
                    region = padded[b, :, i:i+KH, j:j+KW]

                    # -----------------------------------------------------
                    # Compute convolution by dot product:
                    #
                    # Multiply region with filter and sum.
                    #
                    # ✅ NUMERIC EXAMPLE:
                    #
                    # region =
                    # [ 0   0   0
                    #   0   1   2
                    #   0   5   6 ]
                    #
                    # weight =
                    # [  0.12   -0.34    0.08
                    #   -0.21    0.05   -0.11
                    #    0.03   -0.15    0.07 ]
                    #
                    # Calculation:
                    #
                    # 0*0.12 + 0*(-0.34) + 0*0.08
                    # 0*(-0.21) + 1*0.05 + 2*(-0.11)
                    # 0*0.03 + 5*(-0.15) + 6*0.07
                    #
                    # = 0 + 0 + 0
                    #   + 0.05 - 0.22
                    #   - 0.75 + 0.42
                    #
                    # = -0.50
                    #
                    # -----------------------------------------------------
                    out[b, f, i, j] = torch.sum(region * weight[f])

    # ---------------------------------------------------------
    # FINAL OUTPUT:
    #
    # out[b,f,i,j] = one value from convolution.
    #
    # Example output matrix:
    #
    # [ -0.50   ...
    #     ...
    #     ...
    # ]
    #
    # After training, these values highlight edges and features.
    # ---------------------------------------------------------
    return out
