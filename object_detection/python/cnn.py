import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

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

    return filters

class StaticFilterCNN(nn.Module):
    def __init__(self):
        super(StaticFilterCNN, self).__init__()
        # Set the device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # First convolutional layer:
        # - Input: 3-channel RGB images
        # - Output: 16 feature maps
        # - Kernel size: 3x3
        # - Padding: 1 (to preserve input dimensions)
        # - Bias is disabled because we use static filters (no learning)

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        # Second convolutional layer:
        # - Input: 16 feature maps from conv1
        # - Output: 32 feature maps
        # - Same kernel size and padding as conv1
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)

        self.filters = get_static_filters()
        self.initialize_static_filters(self.conv1, self.filters[:16])
        self.initialize_static_filters(self.conv2, self.filters[16:48])
        
        # Max pooling layer:
        # - Window size: 2x2
        # - Stride: 2 (downsamples the feature map by half)

        self.pool = nn.MaxPool2d(2, 2)

       # First fully connected (dense) layer:
       # - Input features: 32 feature maps × 8 height × 8 width (after pooling)
       # - Output features: 128 neurons

        self.fc1 = nn.Linear(32 * 8 * 8, 128)

        # Final output layer:
        # - Input features: 128
        # - Output classes: 10 (e.g., for CIFAR-10)
        
        self.fc2 = nn.Linear(128, 10)

    def initialize_static_filters(self, conv_layer, filters):
        weight = torch.zeros_like(conv_layer.weight.data)
        for i, (_, f) in enumerate(filters):
            if i >= conv_layer.out_channels:
                break
            kernel = torch.tensor(f, dtype=torch.float32)
            for c in range(conv_layer.in_channels):
                weight[i, c] = kernel
        conv_layer.weight = torch.nn.Parameter(weight)
        conv_layer.weight.requires_grad = False


    def forward(self, x):
        # x: input tensor with shape [batch_size, 3, 32, 32] from CIFAR-10 (RGB images)

        # Apply the first convolutional layer using static filters (conv1)
        # Output shape: [batch_size, 16, 32, 32] because padding=1 keeps spatial dimensions
        x = self.conv1(x)

        # Apply ReLU non-linearity to introduce non-linear features
        x = F.relu(x)

        # Apply max pooling with a 2x2 window and stride 2
        # Reduces each spatial dimension by half: [batch_size, 16, 16, 16]
        x = self.pool(x)

        # Apply the second convolutional layer (learned filters)
        # Output shape remains [batch_size, 32, 16, 16] because of padding=1
        x = self.conv2(x)

        # Apply ReLU again for non-linearity
        x = F.relu(x)

        # Apply second max pooling: [batch_size, 32, 8, 8]
        x = self.pool(x)

        # Flatten the feature maps into a 1D vector for the fully connected layer
        # Final flattened shape: [batch_size, 32*8*8] = [batch_size, 2048]
        x = x.view(-1, 32 * 8 * 8)

        # Apply the first fully connected layer (fc1) to reduce to 128 features
        x = F.relu(self.fc1(x))  # [batch_size, 128]

        # Apply the second fully connected layer to map to 10 output classes
        x = self.fc2(x)  # [batch_size, 10]
        x = self.softmax(x)  #  calculate the probability for each object

        # Return the final logits (raw scores for each class)
        return x

    # Load the trained CNN model and prepare it for inference
    def load_model():
      print("Loading model...")  # Notify user that model loading has started

      # Select GPU if available, otherwise fallback to CPU
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      # Initialize the model and move it to the selected device
      model = ObjectDetectionCNN(num_classes=NUM_CLASSES).to(device)

    # Load the saved model weights from disk into the model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Set the model to evaluation mode (disables dropout, etc.)
    model.eval()

    print("Model loaded successfully.")  # Confirm successful loading

    # Return the model and device for inference
    return model, device


    def train_model(self, epochs=5):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.to(self.device)
        self.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

    def detect_objects(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        tensor_img = transform(image).unsqueeze(0).to(self.device)
        output = self(tensor_img)
        _, predicted = torch.max(output, 1)
        label = predicted.item()
        return label

    def capture_and_detect(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera could not be opened.")
            return

        print("Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = to_pil_image(torch.tensor(rgb).permute(2, 0, 1))
            label = self.detect_objects(image)
            class_name = CIFAR10.classes[label]
            cv2.putText(frame, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model = StaticFilterCNN()
    model.train_model()
    model.capture_and_detect()
