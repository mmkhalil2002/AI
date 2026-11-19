import os
import cv2
import numpy as np

# Path to image directory
image_dir = r'C:\Users\Public\mkhalil\AI\BSDS500\RAW-BSDS500\data\images\train'

# Load all image paths
image_paths = sorted([
    os.path.join(image_dir, fname)
    for fname in os.listdir(image_dir)
    if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
])

if not image_paths:
    raise ValueError("No images found in the specified directory.")

# === Define Filters (3x3 kernels or functions) ===
filters =[
     
    ("Identity", np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)),
    ("Edge Detection", np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)),
    ("Sharpen", np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)),
    ("Box Blur", np.ones((3, 3), dtype=np.float32) / 9.0),
    ("Gaussian Blur", cv2.getGaussianKernel(3, 1) @ cv2.getGaussianKernel(3, 1).T),

    # Edge Detection Filters
    ("Edge Detection 0°", np.array([1, 1, 1, -2, -2, -2, 1, 1, 1]).reshape(3, 3)),
    ("Edge Detection 45°", np.array([1, -2, 1, 1, -2, 1, 1, -2, 1]).reshape(3, 3)),
    ("Edge Detection 90°", np.array([-2, 1, 1, 1, -2, 1, 1, 1, -2]).reshape(3, 3)),
    ("Edge Detection 135°", np.array([1, 1, -2, -2, 1, 1, 1, -2, -2]).reshape(3, 3)),
    ("Edge Detection 180°", np.array([-1, -1, -1, 2, 2, 2, -1, -1, -1]).reshape(3, 3)),
    ("Edge Detection 225°", np.array([2, -1, -1, -1, 2, -1, -1, -1, 2]).reshape(3, 3)),
    ("Edge Detection 270°", np.array([-1, 2, -1, -1, 2, -1, -1, 2, -1]).reshape(3, 3)),
    ("Edge Detection 315°", np.array([1, -1, -1, -1, 1, 1, -1, -1, 1]).reshape(3, 3)),

    # Corner Detection Filters
    ("Corner Detection 0°", np.array([1, 1, 0, 1, 0, -1, 0, -1, -1]).reshape(3, 3)),
    ("Corner Detection 45°", np.array([1, 0, 1, 0, -1, 1, -1, -1, 0]).reshape(3, 3)),
    ("Corner Detection 90°", np.array([0, 1, 1, -1, 0, 1, -1, -1, 0]).reshape(3, 3)),
    ("Corner Detection 135°", np.array([1, 0, -1, 0, 1, 1, 0, -1, -1]).reshape(3, 3)),
    ("Corner Detection 180°", np.array([1, 1, 0, 1, 0, -1, 0, -1, -1]).reshape(3, 3)),
    ("Corner Detection 225°", np.array([1, 0, 1, 0, -1, 1, -1, -1, 0]).reshape(3, 3)),
    ("Corner Detection 270°", np.array([0, 1, 1, -1, 0, 1, -1, -1, 0]).reshape(3, 3)),
    ("Corner Detection 315°", np.array([1, 0, -1, 0, 1, 1, 0, -1, -1]).reshape(3, 3)),

    # Curve Detection Filters
    ("Curve Detection 0°", np.array([0, 1, 0, -1, 1, -1, 0, -1, 0]).reshape(3, 3)),
    ("Curve Detection 45°", np.array([1, 0, -1, 0, 1, 0, -1, 0, 1]).reshape(3, 3)),
    ("Curve Detection 90°", np.array([0, -1, 0, 1, 1, 1, 0, -1, 0]).reshape(3, 3)),
    ("Curve Detection 135°", np.array([-1, 0, 1, 0, 1, 0, 1, 0, -1]).reshape(3, 3)),
    ("Curve Detection 180°", np.array([0, -1, 0, -1, 1, -1, 0, 1, 0]).reshape(3, 3)),
    ("Curve Detection 225°", np.array([-1, 0, 1, 0, 1, 0, 1, 0, -1]).reshape(3, 3)),
    ("Curve Detection 270°", np.array([0, 1, 0, 1, 1, 1, 0, -1, 0]).reshape(3, 3)),
    ("Curve Detection 315°", np.array([1, 0, -1, 0, 1, 0, -1, 0, 1]).reshape(3, 3)),

    # Line Detection Filters
    ("Line Detection 0°", np.array([1, 1, 1, -2, -2, -2, 1, 1, 1]).reshape(3, 3)),
    ("Line Detection 45°", np.array([-2, 1, 1, 1, -2, 1, 1, 1, -2]).reshape(3, 3)),
    ("Line Detection 90°", np.array([-1, -1, -1, 2, 2, 2, -1, -1, -1]).reshape(3, 3)),
    ("Line Detection 135°", np.array([1, -2, 1, 1, -2, 1, 1, -2, 1]).reshape(3, 3)),
    ("Line Detection 180°", np.array([-1, -1, -1, -2, -2, -2, -1, -1, -1]).reshape(3, 3)),
    ("Line Detection 225°", np.array([1, 1, -2, 1, -2, 1, -2, 1, 1]).reshape(3, 3)),
    ("Line Detection 270°", np.array([-1, 2, -1, -1, 2, -1, -1, 2, -1]).reshape(3, 3)),
    ("Line Detection 315°", np.array([1, -1, 1, -1, -2, -1, 1, -1, 1]).reshape(3, 3)),

    # Sobel Filters (Edge Detection)
    ("Sobel 0°", np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1], dtype=np.float32)),
    ("Sobel 45°", np.array([ 0, -1, -2, 1, 0, -1, 2, 1, 0], dtype=np.float32)),
    ("Sobel 90°", np.array([ 1, 2, 1, 0, 0, 0, -1, -2, -1], dtype=np.float32)),
    ("Sobel 135°", np.array([ 2, 1, 0, 1, 0, -1, 0, -1, -2], dtype=np.float32)),
    ("Sobel 180°", np.array([ 1, 0, -1, 2, 0, -2, 1, 0, -1], dtype=np.float32)),
    ("Sobel 225°", np.array([ 0, 1, 2, -1, 0, 1, -2, -1, 0], dtype=np.float32)),
    ("Sobel 270°", np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1], dtype=np.float32)),
    ("Sobel 315°", np.array([-2, -1, 0, -1, 0, 1, 0, 1, 2], dtype=np.float32)),

   
]


# Index tracking
image_index = 0
filter_index = 0
original_window = "Original Image"
filtered_window = None

# Load and show first image
img = cv2.imread(image_paths[image_index])
height, width, channels = img.shape
print(f"Width: {width}, Height: {height}, Channels: {channels}")
print(f"Showing image: {os.path.basename(image_paths[image_index])}")
cv2.imshow(original_window, img)

while True:
    key = cv2.waitKey(3) & 0xFF

    if key == ord('q'):
        break

    # Move UP in image list
    elif key == ord('U'):
        cv2.destroyWindow(original_window)
        if filtered_window:
            cv2.destroyWindow(filtered_window)
            filtered_window = None
        image_index = (image_index - 1 + len(image_paths)) % len(image_paths)
        img = cv2.imread(image_paths[image_index])
        height, width, channels = img.shape
        print(f"Width: {width}, Height: {height}, Channels: {channels}")
        print(f"Showing image: {os.path.basename(image_paths[image_index])}")
        cv2.imshow(original_window, img)

    # Move DOWN in image list
    elif key == ord('D'):
        cv2.destroyWindow(original_window)
        if filtered_window:
            cv2.destroyWindow(filtered_window)
            filtered_window = None
        image_index = (image_index + 1) % len(image_paths)
        img = cv2.imread(image_paths[image_index])
        height, width, channels = img.shape
        print(f"Width: {width}, Height: {height}, Channels: {channels}")
        print(f"Showing image: {os.path.basename(image_paths[image_index])}")
        cv2.imshow(original_window, img)

    # Move UP in filter list
    elif key == ord('u'):
        filter_index = (filter_index - 1 + len(filters)) % len(filters)
        print(f"Selected Filter: {filters[filter_index][0]}")

    # Move DOWN in filter list
    elif key == ord('d'):
        filter_index = (filter_index + 1) % len(filters)
        print(f"Selected Filter: {filters[filter_index][0]}")

    # Apply filter on color image (no channel split)
    elif key == ord('a'):
        #if filtered_window:
        #    cv2.destroyWindow(filtered_window)

        filter_name, kernel = filters[filter_index]
        print(f"Applying filter: {filter_name}")

        # Apply the filter directly on the full color image
        filtered_img = cv2.filter2D(img, -1, kernel)

        # Show filtered image
        filtered_window = f"Filtered: {filter_name}"
        cv2.imshow(filtered_window, filtered_img)

cv2.destroyAllWindows()
