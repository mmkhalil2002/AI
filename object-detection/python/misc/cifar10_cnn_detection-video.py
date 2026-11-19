import cv2
import torch
import time
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Global Variables
MODEL_PATH = "cifar10_trained_model.pth"
DATA_PATH = "./data"
IMG_WIDTH, IMG_HEIGHT = 32, 32  # Based on training dataset
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for valid detections
FILTER_WIDTH = 3
FILTER_HEIGHT = 3
BATCH_SIZE = 64
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

# Get max CPU and GPU count
MAX_CPU = torch.get_num_threads()
MAX_GPU = torch.cuda.device_count()

# Define all static filters
def get_static_filters():
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

    return filters

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import time
import cv2
from PIL import Image
import os

# Constants
NUM_CLASSES = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
IMG_WIDTH = 32
IMG_HEIGHT = 32
DATA_PATH = "./data"
MODEL_PATH = "./model.pth"
CONFIDENCE_THRESHOLD = 0.5
MAX_CPU = 4  # For data loading

# Define CNN Model for Object Detection
class ObjectDetectionCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ObjectDetectionCNN, self).__init__()

        # Set the device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # First convolutional layer (Conv1)
        # - Input: 3-channel RGB images
        # - Output: 16 feature maps
        # - Kernel size: 3x3
        # - Padding: 1 (preserves spatial dimensions)
        # - Bias is False as we're using static filters
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        
        # Second convolutional layer (Conv2)
        # - Input: 16 feature maps from Conv1
        # - Output: 32 feature maps
        # - Same kernel size and padding as Conv1
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        
        # Define the filters (you can load or define them as needed)
        self.filters = self.get_static_filters()  # Assuming you have a method to get filters
        self.initialize_static_filters(self.conv1, self.filters[:16])  # Assign first 16 filters to Conv1
        self.initialize_static_filters(self.conv2, self.filters[16:48])  # Assign next 32 filters to Conv2
        
        # Max pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers (FC1 and FC2)
        # First FC layer (128 neurons)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        
        # Final output layer (10 output classes, for CIFAR-10 classification)
        self.fc2 = nn.Linear(128, 10)

    def initialize_static_filters(self, conv_layer, filters):
        """Initialize the convolutional layer with predefined static filters."""
        weight = torch.zeros_like(conv_layer.weight)  # Initialize the filter weights as zeros
        
        # Load each filter into the conv layer
        for i, filter_matrix in enumerate(filters):
            if filter_matrix.shape != (3, 3):  # Ensure filters are 3x3
                raise ValueError(f"Filter {i} must be of shape 3x3, but got shape: {filter_matrix.shape}")
            
            weight[i] = filter_matrix.clone().detach().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Set the static filters as the weights of the convolutional layer
        conv_layer.weight = nn.Parameter(weight)

    def forward(self, x):
        """Define the forward pass of the network."""
        # x: Input tensor with shape [batch_size, 3, 32, 32]
        
        # Apply the first convolutional layer with static filters
        # Output shape: [batch_size, 16, 32, 32] due to padding
        x = self.conv1(x)
        
        # Apply ReLU activation function to introduce non-linearity
        x = F.relu(x)
        
        # Apply max pooling with a 2x2 window and stride of 2
        # Reduces spatial dimensions from 32x32 to 16x16
        x = self.pool(x)
        
        # Apply the second convolutional layer (learned filters)
        x = self.conv2(x)
        
        # Apply ReLU activation again
        x = F.relu(x)
        
        # Apply second max pooling, reducing spatial dimensions from 16x16 to 8x8
        x = self.pool(x)
        
        # Flatten the feature maps into a 1D vector for fully connected layers
        x = x.view(-1, 32 * 8 * 8)  # Flatten to shape [batch_size, 2048]
        
        # Apply the first fully connected layer (FC1) with 128 neurons
        x = F.relu(self.fc1(x))
        
        # Apply the second fully connected layer (FC2) to output 10 classes
        x = self.fc2(x)
        
        # Return the raw scores for each class (no softmax applied yet)
        return x

    def get_static_filters(self):
        """Return the predefined static filters for edge detection, etc."""
        # Example 3x3 filters (replace with your actual filters)
        # In practice, replace these with real edge detection or other static filters
        return [
            torch.ones(3, 3),  # Example static filter 1
            torch.zeros(3, 3), # Example static filter 2
            # Add your actual filters here
        ]

# Function to train the model
def train_model(epochs=EPOCHS):
    """Train the CNN model on the CIFAR-10 dataset."""
    print("Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data preparation and transformation
    transform = transforms.Compose([
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),  # Resize to match the input size of the model
        transforms.ToTensor()  # Convert image to tensor
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root=DATA_PATH, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=MAX_CPU)

    # Initialize model, loss function, and optimizer
    model = ObjectDetectionCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()  # Loss function for classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Optimizer for training

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero gradients before the backward pass
            
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
            
            running_loss += loss.item()  # Track the total loss

        end_time = time.time()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Time: {end_time - start_time:.4f} sec")
    
    # Save trained model weights
    torch.save(model.state_dict(), MODEL_PATH)
    print("Training completed and model saved.")

# Load a trained model
def load_model():
    """Load the trained model from the saved file."""
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ObjectDetectionCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # Load saved weights
    model.eval()  # Set the model to evaluation mode
    
    print("Model loaded successfully.")
    return model, device

# Preprocess frame to match model input size
def preprocess_frame(frame):
    """Resize and transform the input frame to match the model's expected input."""
    transform = transforms.Compose([
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),  # Resize frame
        transforms.ToTensor()  # Convert frame to tensor
    ])
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert frame to RGB
    return transform(frame).unsqueeze(0)  # Add batch dimension

# Capture image from webcam
def capture_image():
    """Capture an image from the webcam."""
    cap = cv2.VideoCapture(0)  # Open the webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    ret, frame = cap.read()  # Read the frame
    cap.release()  # Release the webcam
    if not ret:
        print("Error: Could not capture frame.")
        return None
    return frame

# Detect objects in the captured image
def detect_objects(model, device, frame):
    """Detect objects in the captured image using the trained model."""
    input_tensor = preprocess_frame(frame).to(device)  # Preprocess frame and send to device
    with torch.no_grad():  # No gradient calculation for inference
        outputs = model(input_tensor)  # Get raw model outputs
    
    detected_objects = []
    frame_height, frame_width = frame.shape[:2]
    
    for idx, confidence in enumerate(outputs[0]):
        if confidence.item() > CONFIDENCE_THRESHOLD:  # If confidence is above threshold
            label = f"Object_{idx}"  # Label the object (replace with actual label if needed)
            slice_width = frame_width // len(outputs[0])  # Compute width of each bounding box
            x1 = idx * slice_width  # Start of bounding box
            x2 = x1 + slice_width  # End of bounding box
            y1 = int(frame_height * 0.25)  # Fixed y-position (for demonstration)
            y2 = int(frame_height * 0.75)  # Fixed y-position (for demonstration)
            box = (x1, y1, x2, y2)  # Bounding box coordinates
            detected_objects.append((label, confidence.item(), box))  # Append detected object
    
    return detected_objects

# Draw bounding boxes on the image
def draw_detected_objects(frame, detected_objects):
    """Draw bounding boxes around detected objects on the image."""
    output_frame = frame.copy()
    for label, confidence, box in detected_objects:
        x1, y1, x2, y2 = box
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)  # Draw rectangle
        cv2.putText(output_frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Draw label
    return output_frame

# Main entry point for training and inference
if __name__ == "__main__":
    # Train model if not already saved
    if not os.path.exists(MODEL_PATH):
        train_model()  # Train the model if not found

    # Load the trained model and use it for inference
    model, device = load_model()  # Load the trained model

    # Capture image and detect objects
    frame = capture_image()  # Capture image from webcam
    if frame is not None:
        detected_objects = detect_objects(model, device, frame)  # Detect objects in the image
        frame_with_bboxes = draw_detected_objects(frame, detected_objects)  # Draw bounding boxes on image

        # Show the frame with detected objects
        cv2.imshow("Detected Objects", frame_with_bboxes)  # Display the image
        cv2.waitKey(0)  # Wait for key press to close window
        cv2.destroyAllWindows()  # Close window
