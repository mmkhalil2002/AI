
import torch
import os
import pyttsx3
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import time
import cv2
import random
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # <-- Add this line

# Define all static filters
"""


custom_data/
├── train/
│   ├── class1/
│   │   └── *.jpg
│   └── class2/
│       └── *.jpg
├── test/
│   ├── class1/
│   │   └── *.jpg
│   └── class2/
│       └── *.jpg
└── test_images/           # <- Optional (used for inference, unlabelled)
    └── *.jpg              # No folder/class structure needed

    
 """

# Global Variables
MODEL_PATH = "../../../"
MODEL_FILENAME = "cifar10_model_file"
DATA_PATH = "../../../data"
TEST_PATH = "../../:/TEST"
IMG_WIDTH, IMG_HEIGHT = 32, 32  # Based on training dataset
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for valid detections
FILTER_WIDTH = 3
FILTER_HEIGHT = 3
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Neural Network Constants
OUT_CHANNELS1 = 16
OUT_CHANNELS2 = 32
FC1_INPUT = OUT_CHANNELS2 * 8 * 8  # 32 filters * 8x8 after pooling
FC1_OUTPUT = 128
NUM_CLASSES = 10
POOL_KERNEL = 2
POOL_STRIDE = 2
LOG_FILE = 'log'
_dynamic_learning = True
_print_label = False
_draw_boundary= False

# Get max CPU and GPU count
MAX_CPU = torch.get_num_threads()
MAX_GPU = torch.cuda.device_count() 


class ObjectDetectionCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, use_static_filters=False):
        super(ObjectDetectionCNN, self).__init__()

        # Set the device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set the flag for static filter usage
        self.use_static_filters = use_static_filters

        # First convolutional layer (Conv1)
        # - Input: 3-channel RGB images
        # - Output: 16 feature maps
        # - Kernel size: 3x3
        # - Padding: 1 (preserves spatial dimensions)
        # - Bias is False as we're using static filters
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)

        if not self.use_static_filters:
            # Freeze conv1 weights if dynamic learning is not used
            for param in self.conv1.parameters():
                param.requires_grad = False

        # Second convolutional layer (Conv2)
        # - Input: 16 feature maps from Conv1
        # - Output: 32 feature maps
        # - Same kernel size and padding as Conv1
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)

        if not self.use_static_filters:
            # Freeze conv2 weights if dynamic learning is not used
            for param in self.conv2.parameters():
                param.requires_grad = False

        # Define the filters (you can load or define them as needed)
        self.filters = self.get_static_filters()  # Static filter method (predefined or user-defined)

        # Initialize static filters for both convolutional layers
        self.initialize_static_filters_conv1(self.conv1, self.filters[:16])  # Assign first 16 filters to Conv1
        self.initialize_static_filters_conv2(self.conv2, self.filters[16:48])  # Assign next 32 filters to Conv2

        # Max pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)

        # Dynamically determine the flattened size after convolutions and pooling
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, IMG_WIDTH, IMG_HEIGHT)  # Simulated CIFAR-10 input
            dummy_output = self.pool(F.relu(self.conv1(dummy_input)))  # After conv1 + pool
            dummy_output = self.pool(F.relu(self.conv2(dummy_output)))  # After conv2 + pool
            self.flattened_size = dummy_output.view(1, -1).size(1)  # Compute flattened feature size

        # Fully connected layers
        # First FC layer (128 neurons), input size determined dynamically
        self.fc1 = nn.Linear(self.flattened_size, 128)

        # Final output layer (10 output classes for CIFAR-10)
        self.fc2 = nn.Linear(128, num_classes)

    def initialize_static_filters_conv1(self, conv_layer, filters):
        """Initialize the weights of conv1 with the first 16 static filters."""
        with open(LOG_FILE, 'a') as f:
            print("Enter initialize_static_filters_conv1", file=f)
            filters_tensor = torch.tensor(filters, dtype=torch.float32).unsqueeze(1)  # Shape: [16, 1, 3, 3]
            filters_tensor = filters_tensor.repeat(1, 3, 1, 1)  # Repeat across 3 input channels
            conv_layer.weight.data = filters_tensor
            print("Exit initialize_static_filters_conv1", file=f)

    def initialize_static_filters_conv2(self, conv_layer, filters):
        """Initialize the weights of conv2 with 32 static filters."""
        with open(LOG_FILE, 'a') as f:
            print("Enter initialize_static_filters_conv2", file=f)
            filters_tensor = torch.tensor(filters, dtype=torch.float32).unsqueeze(1)  # Shape: [32, 1, 3, 3]
            filters_tensor = filters_tensor.repeat(1, 16, 1, 1)  # Repeat across 16 input channels
            conv_layer.weight.data = filters_tensor
            print("Exit initialize_static_filters_conv2", file=f)

    def forward(self, x):
        """Define the forward pass of the network."""
        with open(LOG_FILE, 'a') as f:
            print("Enter forward", file=f)

            # Apply the first convolutional layer with static filters
            x = self.conv1(x)

            # Apply ReLU activation
            x = F.relu(x)

            # Apply first max pooling: reduces spatial dimensions from 32x32 to 16x16
            x = self.pool(x)

            # Apply second convolutional layer (static filters)
            x = self.conv2(x)

            # Apply ReLU activation
            x = F.relu(x)

            # Apply second max pooling: reduces from 16x16 to 8x8
            x = self.pool(x)

            # Print shape before flattening for debug
            print("Shape before flattening:", x.shape, file=f)  # Expected: [batch_size, 32, H, W]

            # Flatten the feature maps into a 1D vector for fully connected layers
            x = x.view(x.size(0), -1)  # Dynamic flattening

            # Apply the first fully connected layer (FC1) with 128 neurons
            x = F.relu(self.fc1(x))

            # Apply the second fully connected layer (FC2) to output 10 classes
            x = self.fc2(x)

            print("Exit forward", file=f)
        return x


    

    def get_static_filters(self):
        f = open(LOG_FILE, 'a')
        print("Enter get_static_filters",file=f)
        filters = [
         
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

            # Sobel Filters (Edge Detection) with reshape
            ("Sobel 0°", np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1], dtype=np.float32).reshape(3, 3)),
            ("Sobel 45°", np.array([ 0, -1, -2, 1, 0, -1, 2, 1, 0], dtype=np.float32).reshape(3, 3)),
            ("Sobel 90°", np.array([ 1,  2,  1,  0,  0,  0, -1, -2, -1], dtype=np.float32).reshape(3, 3)),
            ("Sobel 135°", np.array([ 2,  1,  0,  1,  0, -1,  0, -1, -2], dtype=np.float32).reshape(3, 3)),
            ("Sobel 180°", np.array([ 1,  0, -1,  2,  0, -2,  1,  0, -1], dtype=np.float32).reshape(3, 3)),
            ("Sobel 225°", np.array([ 0,  1,  2, -1,  0,  1, -2, -1,  0], dtype=np.float32).reshape(3, 3)),
            ("Sobel 270°", np.array([-1, -2, -1,  0,  0,  0,  1,  2,  1], dtype=np.float32).reshape(3, 3)),
            ("Sobel 315°", np.array([-2, -1,  0, -1,  0,  1,  0,  1,  2], dtype=np.float32).reshape(3, 3)),
            ]
        print("Exit get_static_filters",file=f)
        return [f[1] for f in filters]  # Only return the kernels, discard names
    
def load_training_data(_batch_size=BATCH_SIZE, _data_dir=MODEL_PATH, _num_workers=MAX_CPU):
    with open(LOG_FILE, 'a') as f:
        print("Enter load_training_data", file=f)

        """
        Function to load a custom training dataset organized in subfolders (ImageFolder format).

        Args:
            _batch_size (int): Number of images per batch.
            _data_dir (str): Root directory where each class has its own subfolder.
            _num_workers (int): Number of parallel processes for loading.

        Returns:
            DataLoader: PyTorch DataLoader for the custom dataset.
        """

        # Define transformations for the images
        transform = transforms.Compose([
            transforms.Resize((32, 32)),              # Resize all images to 32x32
            transforms.ToTensor(),                    # Convert to tensor
            transforms.Normalize((0.5,), (0.5,))      # Normalize to [-1, 1]
        ])

        # Load dataset using ImageFolder
        train_dataset = datasets.ImageFolder(root=_data_dir, transform=transform)

        # Optional: print class-to-index mapping
        print("Detected classes:", train_dataset.class_to_idx, file=f)

        # Create DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=_num_workers
        )

        print("Exit load_training_data", file=f)
        return train_loader




def train_model(model, train_loader, device, epochs, save_dir="trained_model", model_filename="model.pth"):
    with open(LOG_FILE, 'a') as f:
        print("➡️ Entering training function", file=f)

        """
        Train a CNN model, safely initializing weights while preserving static filters.

        Args:
            model (nn.Module): The CNN model to train.
            train_loader (DataLoader): DataLoader for training data.
            device (torch.device): Training device (CPU/GPU).
            epochs (int): Number of training epochs.
            save_dir (str): Directory to save the trained model.
            model_filename (str): Name of the saved model file.
        """

        # Define loss function
        criterion = torch.nn.CrossEntropyLoss()

        # Use Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Learning rate scheduler (decays LR by 0.5 every 10 epochs)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # Optional: Apply He initialization ONLY to trainable layers
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if hasattr(m.weight, 'requires_grad') and not m.weight.requires_grad:
                    return  # Skip static filters
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        model.apply(init_weights)  # Apply safe initialization

        model.train()  # Switch model to training mode

        total_start_time = time.time()

        for epoch in range(epochs):
            running_loss = 0.0
            epoch_start_time = time.time()

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()  # Ensure float32 input

                # Forward pass
                outputs = model(inputs)

                # Ensure output and label dimensions match
                min_batch = min(outputs.shape[0], labels.shape[0])
                outputs = outputs[:min_batch]
                labels = labels[:min_batch]

                # Loss computation
                loss = criterion(outputs, labels)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            scheduler.step()  # Step LR scheduler

            avg_loss = running_loss / len(train_loader)
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Time: {epoch_duration:.2f} sec")

        total_duration = time.time() - total_start_time
        print(f"\n✅ Total training time: {total_duration:.2f} seconds")

        # Save model
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"✅ Created model directory: {save_dir}")

        save_path = os.path.join(save_dir, model_filename)
        try:
            torch.save(model.state_dict(), save_path)
            os.chmod(save_path, 0o666)  # rw-rw-rw-
            print(f"✅ Model saved to: {os.path.abspath(save_path)}")
            print("✅ File permissions set to read/write for all.")
        except PermissionError:
            print("❌ Permission denied while saving model.")

        print("⬅️ Exiting training function", file=f)



def load_model(model_path, model_filename, use_static_filters=True):
    """Load the trained model from the saved file with error handling and static filter support."""
    
    with open(LOG_FILE, 'a') as f:
        print("Enter load_model", file=f)

        print("🔄 Initializing model loading...")  # Print message indicating model loading has started

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose device based on availability
        print(f"🖥️  Using device: {device}")  # Print which device (CPU or GPU) will be used

        # ✅ Initialize the model with or without static filters (must match training setup)
        model = ObjectDetectionCNN(num_classes=NUM_CLASSES, use_static_filters=use_static_filters).to(device)

        # Combine MODEL_PATH and model_filename to get the full model path
        full_model_path = os.path.join(model_path, model_filename)

        # Print the model path being used for loading
        print(f"📂 Looking for model file: {full_model_path}")
        
        # Check if the model file exists at the specified path
        if not os.path.exists(full_model_path):
            print(f"❌ Error: Model file '{full_model_path}' not found!")  # Print error if model file doesn't exist
            raise FileNotFoundError(f"Model file '{full_model_path}' does not exist.")  # Raise exception to stop execution

        try:
            print("📦 Loading model weights...")  # Print message indicating model weight loading

            # Load the trained model weights from the specified file
            model.load_state_dict(torch.load(full_model_path, map_location=device))  # Load weights and map them to the correct device
            model.eval()  # Set the model to evaluation mode

            # 🟢 Success decorative message: This is printed when the model is loaded successfully
            print("\033[1;32;40m\n*******************************************************")
            print("✅ Model loaded successfully and set to evaluation mode!")
            print(f"🔒 Static filters: {'ENABLED' if use_static_filters else 'DISABLED'}")  # Log static filter state
            print("*******************************************************\033[0m")

        except Exception as e:
            print(f"❌ Failed to load the model: {e}")  # Print error message if loading the model fails
            raise  # Raise the exception to stop further execution

        print("Exit load_model", file=f)

        return model, device  # Return the loaded model and the device it's on

def preprocess_frame(frame):
    """Preprocess a frame for the CNN model."""
    with open(LOG_FILE, 'a') as f:
        print("Enter preprocess_frame", file=f)

        # Convert BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision transforms
        frame = Image.fromarray(frame)

        # Apply same transform as in custom data loader
        
        transform = transforms.Compose([
            transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])

        frame = transform(frame)
        frame = frame.unsqueeze(0)  # Add batch dimension: (1, 3, 32, 32)

        print("Exit preprocess_frame", file=f)
        return frame




# Define CIFAR-10 class labels
CIFAR10_CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    ]


# Function to detect objects in a frame
#import torch.nn.functional as F  # Add this at the top of your script if not already imported

def detect_objects(model, device, frame, confidence_threshold):
    """Detect objects in the captured image."""
    with open(LOG_FILE, 'a') as f:
        print("Enter preprocess_frame", file=f)

        # Step 1: Convert torch.Tensor to NumPy if needed
        if isinstance(frame, torch.Tensor):
            frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
        else:
            frame_np = frame

        # Step 2: Ensure correct format: RGB → BGR, uint8, contiguous
        if frame_np.dtype != np.uint8:
            frame_np = frame_np.astype(np.uint8)
        if not frame_np.flags['C_CONTIGUOUS']:
            frame_np = np.ascontiguousarray(frame_np)
        
        # Convert to BGR for OpenCV drawing compatibility
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        # Step 3: Preprocess for model input (e.g., normalization, resizing)
        input_tensor = preprocess_frame(frame_np).to(device)

        # Step 4: Model inference
        with torch.no_grad():
            outputs = model(input_tensor)

        # Step 5: Get class probabilities
        probabilities = F.softmax(outputs[0], dim=0)

        # Step 6: Log probabilities
        print("\n🧠 Confidence Scores for All CIFAR-10 Classes:")
        for idx, prob in enumerate(probabilities):
            print(f"{CIFAR10_CLASSES[idx]:>10}: {prob.item():.4f}")

        detected_objects = []
        frame_height, frame_width = frame_bgr.shape[:2]

        # Step 7: Generate bounding boxes for confident detections
        for idx, confidence in enumerate(probabilities):
            if confidence.item() > confidence_threshold:
                label = CIFAR10_CLASSES[idx]
                slice_width = frame_width // len(probabilities)
                x1 = idx * slice_width
                x2 = x1 + slice_width
                y1 = int(frame_height * 0.25)
                y2 = int(frame_height * 0.75)
                box = (x1, y1, x2, y2)

                detected_objects.append((label, confidence.item(), box))
                print(f"🟢 Detected {label} with confidence: {confidence.item():.4f}")

        print("Exit detect_objects", file=f)
        return detected_objects, frame_bgr




import os
import random
from PIL import Image
import torch
from torchvision import transforms

def get_random_test_image(data_dir):
    """Loads a random image from the custom test directory."""
    with open(LOG_FILE, 'a') as f:
        print("Enter get_random_test_image ", file=f)
        
        # Get all image files from the directory
        image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print("No images found in the directory.", file=f)
            return None, None, None

        # Pick a random image file
        random_image_path = os.path.join(data_dir, random.choice(image_files))
        
        # Open the image using PIL
        image = Image.open(random_image_path)
        
        # Define the transform (resize to 32x32, convert to tensor, normalize)
        transform = transforms.Compose([
            transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),  # Resize the image
            transforms.ToTensor(),                       # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
        
        # Apply the transformation to the image
        image_tensor = transform(image)
        
        # Add batch dimension (1, 3, 32, 32)
        image_tensor = image_tensor.unsqueeze(0)
        
        # If you want a label, you could assign it based on directory name (e.g., folder name as label)
        label = os.path.basename(os.path.dirname(random_image_path))  # Example of using folder name as label
        
        print("Exit get_random_test_image", file=f)
        return image, image_tensor, label

    
    

import cv2
import pyttsx3
import numpy as np
from typing import List, Tuple

# Initialize TTS engine
tts_engine = pyttsx3.init()

# Optional: Set volume and rate
tts_engine.setProperty('volume', 1.0)       # Max volume
tts_engine.setProperty('rate', 150)         # Normal speaking rate

def draw_detected_objects(frame, detected_objects, draw_boundary=True, print_label=True):
    """
    Draws a bounding box and label only for the highest-confidence detected object in the frame.
    Optionally announces the top object using text-to-speech (TTS), and can toggle text/box display.

    Args:
        frame (np.ndarray): The input video/image frame (BGR format).
        detected_objects (List[Tuple[str, float, Tuple[int, int, int, int]]]): Detected objects.
        draw_boundary (bool): Whether to draw a rectangle around the detected object.
        print_label (bool): Whether to print the label text on the frame.

    Returns:
        np.ndarray: Frame with annotations.
    """
    with open(LOG_FILE, 'a') as f:
        print("Enter draw_detected_objects", file=f)
        print(f"draw_boundary={draw_boundary}, print_label={print_label}", file=f)

    # Validate input frame
    if not isinstance(frame, np.ndarray):
        raise ValueError("Frame must be a numpy array.")
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # Sort detections by descending confidence
    detected_objects = sorted(detected_objects, key=lambda x: x[1], reverse=True)

    # Process only the top object if any detected
    if detected_objects:
        top_label, top_confidence, top_box = detected_objects[0]

        # Announce via TTS
        try:
            announcement = f"The detected object is {top_label}"
            tts_engine.say(announcement)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")

        # Unpack bounding box
        x1, y1, x2, y2 = top_box

        # Draw bounding box if enabled
        if draw_boundary:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Prepare label text
        text = f"{top_label}: {top_confidence:.2f}"

        if print_label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.08  # Reduced font size (1/5th of original 0.4)
            thickness = 1

            # Get text size for placement
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x, text_y = x1, max(y1 - 10, text_height + 2)

            # Define semi-transparent background area
            bg_top_left = (text_x, text_y - text_height - 4)
            bg_bottom_right = (text_x + text_width + 4, text_y + baseline)

            # Draw transparent background rectangle
            overlay = frame.copy()
            cv2.rectangle(overlay, bg_top_left, bg_bottom_right, (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # Draw the label text itself
            cv2.putText(frame, text, (text_x + 2, text_y - 2),
                        fontFace=font,
                        fontScale=font_scale,
                        color=(0, 255, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA)

    with open(LOG_FILE, 'a') as f:
        print("Exit draw_detected_objects", file=f)

    return frame










# Global threshold value for converting filter values
filter_threshold = 0.5


# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('volume', 1.0)       # Max volume
tts_engine.setProperty('rate', 150)         # Normal speaking rate


def log_all_learned_filters(model, log_file=LOG_FILE):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("\n==================== LEARNED FILTERS LOG ====================\n")
        f.write(f"Threshold used for binarization: filter_threshold = {filter_threshold:.4f}\n")

        # Collect all layers that are instances of torch.nn.Conv2d
        conv_layers = [layer for name, layer in model.named_modules() if isinstance(layer, torch.nn.Conv2d)]

        for i, conv in enumerate(conv_layers, start=1):
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"                convolutional layer {i} (conv{i})\n")
            f.write("=" * 60 + "\n")

            # Get the filter weights: shape [out_channels, in_channels, kernel_height, kernel_width]
            weights = conv.weight.data.cpu().numpy()

            # Iterate through each output channel (filter)
            for out_ch in range(weights.shape[0]):
                for in_ch in range(weights.shape[1]):
                    f.write(f"\nconv{i} | filter {out_ch} | in_ch {in_ch}:\n")

                    # Original filter
                    kernel = weights[out_ch, in_ch]
                    f.write("original filter:\n")
                    for row in kernel:
                        row_str = ", ".join(f"{v:.4f}" for v in row)
                        f.write(f"    [{row_str}]\n")

                    # Thresholded filter
                    f.write(f"thresholded filter (abs > {filter_threshold:.2f} => ±1, else 0):\n")
                    for row in kernel:
                        thresholded_row = []
                        for v in row:
                            if abs(v) > filter_threshold:
                                thresholded_row.append(1 if v > 0 else -1)
                            else:
                                thresholded_row.append(0)
                        row_str = ", ".join(f"{v:2d}" for v in thresholded_row)
                        f.write(f"    [{row_str}]\n")

        f.write("\n================== END OF FILTER LOG ==================\n")



#  main program 

if __name__ == "__main__":
    os.makedirs(MODEL_PATH, exist_ok=True)
    model_filename = os.path.join(MODEL_PATH, MODEL_FILENAME)

    # Ask for number of epochs if model doesn't exist
    if not os.path.exists(model_filename):
        try:
            num_epochs = int(input("Enter number of epochs (e.g., 10): "))
            if num_epochs <= 0:
                raise ValueError
        except ValueError:
            print("Invalid input, using default of 10 epochs.")
            num_epochs = 10

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load training data
    print("Loading training data...")
    train_loader = load_training_data()  

    # Initialize model
    print("Creating the model...")
    model = ObjectDetectionCNN(num_classes=NUM_CLASSES).to(device)

    # Train if model doesn't exist
    if not os.path.exists(model_filename):
        print(f"Training for {num_epochs} epochs...")
        train_model(model, train_loader, device, num_epochs, MODEL_PATH, MODEL_FILENAME)
        print("Training complete.")

    # Load trained model
    print("Loading model from disk...")
    model, device = load_model(MODEL_PATH, MODEL_FILENAME)
    print("Model loaded.")

    # Get a test image (original RGB image and preprocessed tensor)
    print("Fetching random test image...")
    original_image_np, input_tensor, true_label = get_random_test_image(TEST_PATH)
    input_tensor = input_tensor.to(device)

    # Get confidence threshold from user
    try:
        confidence_threshold = float(input("Enter confidence threshold (0.0 to 1.0): "))
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError
    except ValueError:
        print("Invalid threshold, using default (0.5).")
        confidence_threshold = 0.5

    # Detect objects on the image
    print("Running object detection...")
    detected_objects, original_image_bgr = detect_objects(model, device, original_image_np, confidence_threshold)

    # Show detections in terminal
    print("\nDetected Objects:")
    for label, confidence, box in detected_objects:
        print(f"Label: {label}, Confidence: {confidence:.4f}, Box: {box}")

    # Optionally log learned filters
    log_all_learned_filters(model)

    # Annotate image with detections
    print("Annotating original image...")
    drawn_image_np = draw_detected_objects(original_image_bgr.copy(), detected_objects, draw_boundary=_draw_boundary, print_label=_print_label)

    # Ask user for resize factor
    try:
        resize_factor = float(input("Enter resize factor (e.g., 0.5 for half size): "))
        if resize_factor <= 0:
            raise ValueError
    except ValueError:
        print("Invalid resize factor, using default (1.0).")
        resize_factor = 1.0

    # Resize original image
    original_resized = cv2.resize(original_image_bgr, (
        int(original_image_bgr.shape[1] * resize_factor),
        int(original_image_bgr.shape[0] * resize_factor)
    ))

    # Resize annotated image
    annotated_resized = cv2.resize(drawn_image_np, (
        int(drawn_image_np.shape[1] * resize_factor),
        int(drawn_image_np.shape[0] * resize_factor)
    ))

    # Display both images
    print("Displaying original and annotated images...")
    cv2.imshow("Original Image", original_resized)
    cv2.imshow("Annotated Image", annotated_resized)

    print("Press 'q' or ESC to quit.")
    while True:
        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:  # ESC
            break

    cv2.destroyAllWindows()
