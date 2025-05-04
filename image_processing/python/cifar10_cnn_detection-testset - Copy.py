import cv2
import torch
import time
import random
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # <-- Add this line


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
        self.initialize_static_filters_conv1(self.conv1, self.filters[:16])  # For Conv1
        self.initialize_static_filters_conv2(self.conv2, self.filters[16:48])  # For Conv2
        
        # Max pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers (FC1 and FC2)
        # First FC layer (128 neurons)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        
        # Final output layer (10 output classes, for CIFAR-10 classification)
        self.fc2 = nn.Linear(128, 10)

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
    
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    def get_trained_loader(batch_size=32, data_dir='./data', num_workers=4):
        """
        Function to get the training data loader for training a CNN.

        Args:
            batch_size (int): The batch size to load data in batches.
            data_dir (str): The directory where the dataset is stored.
            num_workers (int): The number of worker processes to use for loading the data.

        Returns:
            DataLoader: PyTorch DataLoader for the training dataset.
        """
        
        # Define data transformations (you can modify this based on your dataset)
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for pretrained models
        ])
        
        # Load the training dataset (here using CIFAR-10 as an example)
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        # Create the DataLoader for the training dataset
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        return train_loader

        
    def get_static_filters(self):
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
        updated_filters = []

          # For loop to scan and fix
        for filt in filters:
          if isinstance(filt, tuple):
            # If it's a (name, array) tuple, keep only the array part
            updated_filters.append(filt[1])
          elif isinstance(filt, np.ndarray):
             # If it's already an array, keep it directly
            updated_filters.append(filt)

    # Return torch tensors with 3 channels replicated
        return [torch.tensor(f).unsqueeze(0).repeat(3, 1, 1) for f in updated_filters]


    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms



    def initialize_static_filters_conv1(self, conv_layer, filters):
            """
            Initializes conv1 layer (first convolution) with a subset of static filters (first 16 filters).

            Args:
                conv_layer: The first convolution layer (e.g., self.conv1).
                filters: The list or tensor of static filters to initialize with.
            """
            # Extract the first 16 filters for conv1
            filters_conv1 = filters[:16]  # Filters from index 0 to 15 (total 16 filters)
            # 🔵 This means the filters for conv1 will be the first 16 filters

            # Stack the extracted filters into a tensor
            filters_tensor = torch.tensor(np.stack(filters_conv1), dtype=torch.float32)
            # 🔵 Now shape: (16, 3, 3)

            # Repeat each filter across the 3 input channels (since conv1 takes 3 channels as input)
            filters_tensor = filters_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
            # 🔵 Shape after repeat: (16, 3, 3, 3)
            # Explanation:
            #    1 - We add 1 batch dimension (does not modify the batch size, stays as 1).
            #    3 - We repeat the filters across the 3 input channels (RGB channels).
            #    1, 1 - The kernel size remains 3x3.
            # So now you have 16 filters, each of size (3, 3, 3) (3 input channels).

            # Set the conv_layer's weights to the prepared filters
            conv_layer.weight.copy_(filters_tensor)
            # 🔵 Directly assign the filters tensor to conv_layer's weight

            # Freeze the layer to make the filters static (not updated during training)
            conv_layer.weight.requires_grad = False
            # 🔵 No updates from backpropagation.

    def initialize_static_filters_conv2(self, conv_layer, filters):
        """
        Initializes conv2 layer (second convolution) with a subset of static filters (filters from index 16 to 47).

        Args:
            conv_layer: The second convolution layer (e.g., self.conv2).
            filters: The list or tensor of static filters to initialize with.
        """
        # Extract the filters for conv2 from index 16 to 47 (total 32 filters)
        filters_conv2 = filters[16:48]  # Filters between indices 16 to 47 inclusive
        # 🔵 This means the filters for conv2 will be from indices 16 to 47

        # Stack the extracted filters into a tensor
        filters_tensor = torch.tensor(np.stack(filters_conv2), dtype=torch.float32)
        # 🔵 Now shape: (32, 3, 3)

        # Repeat each filter across the 16 input channels (since conv1 outputs 16 channels)
        filters_tensor = filters_tensor.unsqueeze(1).repeat(1, 16, 1, 1)
        # 🔵 Shape after repeat: (32, 16, 3, 3)
        # Explanation:
        #    1 - We're adding 1 batch dimension (we don’t modify this, it’s a batch of filters).
        #    16 - We repeat the filters across 16 input channels (conv1's output).
        #    3, 3 - The kernel size (filter size is 3x3).
        # So now you have 32 filters, each of size (16, 3, 3) (16 input channels).

        # Set the conv_layer's weights to the prepared filters
        conv_layer.weight.copy_(filters_tensor)
        # 🔵 Directly assign the filters tensor to conv_layer's weight

        # Freeze the layer to make the filters static (not updated during training)
        conv_layer.weight.requires_grad = False
        # 🔵 No updates from backpropagation.





   

    # Function to train the CNN model
    def train_model(model, train_loader, device, epochs=10):
        model.train()  # Set model to training mode (important for layers like Dropout, BatchNorm, etc.)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
        criterion = nn.CrossEntropyLoss()  # Loss function for classification

        for epoch in range(epochs):
            start_time = time.time()  # Record start time of epoch
            running_loss = 0.0  # Accumulate loss for this epoch

            # Iterate through batches
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to CPU or GPU

                optimizer.zero_grad()  # Clear previous gradients
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

                running_loss += loss.item()  # Add loss for reporting

            end_time = time.time()  # Record end time of epoch

            # Print the results for this epoch
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Time: {end_time - start_time:.4f} sec")


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

    def preprocess_frame(self, frame):
        # Preprocess a frame (input image) before feeding into the model

        if isinstance(frame, torch.Tensor):
            # If input is a Tensor (like CIFAR-10 image), convert it to numpy format
            frame = frame.permute(1, 2, 0).numpy()  # Change from [C, H, W] to [H, W, C] format
            frame = (frame * 255).astype(np.uint8)  # Convert pixel values from [0,1] float to [0,255] uint8

        # Convert numpy image from BGR (OpenCV format) to RGB and then to PIL Image
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Define transformation pipeline
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Ensure image size matches model input (32x32)
            transforms.ToTensor(),        # Convert PIL image to PyTorch Tensor
            transforms.Normalize(         # Normalize using ImageNet mean and std
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        # Apply transformations and add batch dimension
        input_tensor = transform(frame).unsqueeze(0)  # Shape: [1, 3, 32, 32]
        return input_tensor

    def get_random_test_image():
        """
        Loads a random image from the CIFAR-10 test set.
        Returns: original image (numpy), tensor image (for model), and label
        """
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        testset = CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
        
        # Pick a random sample
        idx = random.randint(0, len(testset) - 1)
        image_tensor, label = testset[idx]

        # Convert tensor to numpy array for drawing (CHW -> HWC)
        image_np = image_tensor.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype('uint8')  # de-normalize if needed

        return image_np, image_tensor.unsqueeze(0), label

    def preprocess_frame(frame):
        """Preprocess a frame (numpy array or tensor) before feeding into the model."""
        if isinstance(frame, torch.Tensor):
            frame = frame.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
            frame = (frame * 255).astype(np.uint8)  # [0,1] -> [0,255]

        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # BGR to RGB

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(frame).unsqueeze(0)  # [1, 3, 32, 32]
        return input_tensor

    def get_random_test_image():
        """Load a random image from CIFAR-10 test set."""
        transform = transforms.Compose([transforms.ToTensor()])
        testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

        idx = random.randint(0, len(testset) - 1)
        image_tensor, label = testset[idx]

        image_np = image_tensor.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype('uint8')

        return image_np, image_tensor.unsqueeze(0), label


    import cv2

    def draw_detected_objects(frame, detected_objects):
        """
        Draw bounding boxes and labels on the frame for each detected object.

        Args:
            frame (numpy.ndarray): The original input image (OpenCV format, BGR).
            detected_objects (list): List of (label, confidence, box) tuples.

        Returns:
            frame_with_boxes (numpy.ndarray): The modified image with bounding boxes and labels.
        """
        # Make a copy of the original frame to draw on
        frame_with_boxes = frame.copy()

        # Loop through each detected object
        for label, confidence, box in detected_objects:
            x1, y1, x2, y2 = box  # Unpack bounding box coordinates

            # Draw a rectangle (bounding box) around the detected object
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Prepare the label text with confidence
            text = f"{label}: {confidence:.2f}"

            # Choose a font and put the label text above the bounding box
            cv2.putText(frame_with_boxes, text, (x1, y1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        thickness=2)

        # Return the modified frame
        return frame_with_boxes


    import torch
    import numpy as np
    import cv2

    # Define CIFAR-10 class labels
    CIFAR10_CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    ]


    # Function to detect objects in a frame
    def detect_objects(model, device, frame, confidence_threshold):
        """Detect objects in the captured image."""
        # If frame is a tensor, convert it to numpy array for visualization and processing
        if isinstance(frame, torch.Tensor):
            frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy
            frame_np = (frame_np * 255).astype(np.uint8)  # Scale tensor values to 0-255 range for image processing
        else:
            frame_np = frame  # If already a numpy array, use it directly

        # Preprocess the frame image for the model input
        input_tensor = preprocess_frame(frame_np).to(device)  # Apply preprocessing and move to device

        # Perform inference without calculating gradients (for efficient prediction)
        with torch.no_grad():  
            outputs = model(input_tensor)  # Get predictions from the model

        detected_objects = []  # List to store detected objects and their information
        frame_height, frame_width = frame_np.shape[:2]  # Get the dimensions of the image frame

        # Loop through the model's outputs (predictions) for each object
        for idx, confidence in enumerate(outputs[0]):  # Iterate through each predicted confidence score
            # If the confidence is above the threshold, consider it a valid detection
            if confidence.item() > confidence_threshold:
                label = CIFAR10_CLASSES[idx]  # Get the class label corresponding to this object (CIFAR-10 class)
                
                # Calculate bounding box coordinates (simplified for demonstration)
                slice_width = frame_width // len(outputs[0])  # Divide the width by the number of predictions
                x1 = idx * slice_width  # Left boundary of the bounding box
                x2 = x1 + slice_width  # Right boundary of the bounding box
                y1 = int(frame_height * 0.25)  # Top boundary of the bounding box (fixed height margin)
                y2 = int(frame_height * 0.75)  # Bottom boundary of the bounding box (fixed height margin)
                box = (x1, y1, x2, y2)  # Bounding box coordinates (x1, y1, x2, y2)

                # Append the detected object with label, confidence, and bounding box info
                detected_objects.append((label, confidence.item(), box))

                # Print the confidence of the detected object for debugging and tracking
                print(f"Detected {label} with confidence: {confidence.item():.4f}")

        # Return the list of detected objects with their details
        return detected_objects  

    import cv2
    import numpy as np
    import os

    # Replace with your actual model path
    MODEL_PATH = "your_model_path.pth"  

    import os

# Main function to capture frames, process, and detect objects
    if __name__ == "__main__":
        # Check if the model exists, if not, train it
        if not os.path.exists(MODEL_PATH):
            # Ask user how many epochs to train
            try:
                num_epochs = int(input("Enter the number of epochs for training (e.g., 10, 20, 50): "))
                if num_epochs <= 0:
                    print("Invalid number of epochs! Using default (10 epochs).")
                    num_epochs = 10
            except ValueError:
                print("Invalid input! Using default (10 epochs).")
                num_epochs = 10
            
            # Initialize device (CPU or GPU)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            # Load training data
            print("Loading training data...")
            train_loader = get_trained_loader()  # <-- Make sure this function is defined!

            # Create the ObjectDetectionCNN model
            print("Creating the model...")
            model = ObjectDetectionCNN(num_classes=NUM_CLASSES).to(device)
            print("Model created.")

            # Train the model
            print(f"Training the model for {num_epochs} epochs...")
            train_model(model, train_loader, device, epochs=num_epochs)
            print("Model training completed.")

        # Load the trained model and device (CPU or GPU)
        print("Loading the trained model...")
        model, device = load_model()
        print("Trained model loaded.")

        # Get a random test image (both numpy and tensor format) and its label
        print("Getting a random test image...")
        original_image_np, input_tensor, true_label = get_random_test_image()
        print("Random test image obtained.")

        # Move the input tensor to the correct device
        input_tensor = input_tensor.to(device)

        # Ask user for confidence threshold
        try:
            confidence_threshold = float(input("Enter confidence threshold (0 to 1, e.g., 0.5): "))
            if not (0 <= confidence_threshold <= 1):
                print("Invalid threshold! Using default (0.5).")
                confidence_threshold = 0.5
        except ValueError:
            print("Invalid input! Using default (0.5).")
            confidence_threshold = 0.5

        # Detect objects in the input tensor
        print(f"Detecting objects with confidence threshold {confidence_threshold}...")
        detected_objects = detect_objects(model, device, input_tensor, confidence_threshold)
        
        # Print detected objects and their confidence
        print("\nDetected Objects:")
        confidence_list = []
        for label, confidence, box in detected_objects:
            print(f"Label: {label}, Confidence: {confidence:.4f}, Bounding Box: {box}")
            confidence_list.append((label, confidence))
        
        # Print summary of all labels with confidences
        confidence_list = sorted(confidence_list, key=lambda x: x[1], reverse=True)
        print("\nSummary of Detected Labels and Confidences:")
        for idx, (label, confidence) in enumerate(confidence_list):
            print(f"{idx+1}. Label: {label} - Confidence: {confidence:.4f}")

        # Draw the detected bounding boxes on a copy of the original image
        print("Drawing detected bounding boxes on the image...")
        drawn_image_np = draw_detected_objects(original_image_np.copy(), detected_objects)

        # Combine original and drawn images side-by-side (horizontal stacking)
        print("Combining original and detected images...")
        combined_image = np.hstack((original_image_np, drawn_image_np))

        # Convert combined image from RGB to BGR for OpenCV display
        combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

        # Ask the user for a resize factor (e.g., 0.5 to shrink, 2.0 to enlarge)
        try:
            resize_factor = float(input("Enter resize factor (e.g., 0.5 for half size, 2.0 for double size): "))
            if resize_factor <= 0:
                print("Invalid factor! Using default (1.0).")
                resize_factor = 1.0
        except ValueError:
            print("Invalid input! Using default (1.0).")
            resize_factor = 1.0

        # Calculate new dimensions based on resize factor
        new_width = int(combined_image.shape[1] * resize_factor)
        new_height = int(combined_image.shape[0] * resize_factor)

        # Resize the combined image
        print(f"Resizing the combined image to {new_width}x{new_height}...")
        resized_combined_image = cv2.resize(combined_image, (new_width, new_height))

        # Display the resized combined image
        print("Displaying the image...")
        cv2.imshow('Original (Left) vs Detected (Right)', resized_combined_image)

        # Wait for key press, allow quitting by pressing 'q'
        print("\nPress 'q' to quit.")
        while True:
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:  # 'q' key or ESC
                print("Exiting the program...")
                break

        cv2.destroyAllWindows()



