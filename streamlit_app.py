import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For showing progress bar
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st

# Define Dataset class
import cv2
import numpy as np
import os
from torchvision import transforms

class MRI_Dataset(Dataset):
    def __init__(self, tumor_dir, healthy_dir, image_size=(128, 128), mean=None, std=None):
        self.images = []
        self.labels = []
        self.image_size = image_size
        self.mean = mean
        self.std = std
        img = cv2.imread(os.path.join(tumor_dir, filename))
        if img is not None:
            img = ensure_rgb(img)  # Ensure it's in RGB
            img = cv2.resize(img, self.image_size)

        # Function to ensure RGB
        def ensure_rgb(image):
            if len(image.shape) == 2 or image.shape[2] != 3:  # Grayscale or non-RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to RGB
            return image

        # Load tumor images
        for filename in os.listdir(tumor_dir):
            img = cv2.imread(os.path.join(tumor_dir, filename))
            if img is not None:
                img = ensure_rgb(img)
                img = cv2.resize(img, self.image_size)
                self.images.append(img)
                self.labels.append(1)  # Tumor = 1

        # Load healthy images
        for filename in os.listdir(healthy_dir):
            img = cv2.imread(os.path.join(healthy_dir, filename))
            if img is not None:
                img = ensure_rgb(img)
                img = cv2.resize(img, self.image_size)
                self.images.append(img)
                self.labels.append(0)  # Healthy = 0

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        # Define transform with normalization after converting to [0, 1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor and scale to [0, 1]
            transforms.Normalize(mean=self.mean, std=self.std)  # Normalize
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0  # Normalize to [0, 1]
        label = self.labels[idx]
        image = self.transform(image)  # Apply normalization
        return {'image': image, 'label': torch.tensor(label, dtype=torch.float32)}



# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)  # Binary classification
        
        self.dropout = nn.Dropout(0.5)  # Dropout to prevent overfitting

    def forward(self, x):
        if x.shape[1] == 1:  # If the input has only 1 channel
            x = x.repeat(1, 3, 1, 1)  # Duplicate the channel 3 times to mimic RGB

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(-1, 256 * 8 * 8)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x

# Define image transformations (normalization and resizing)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize like ImageNet
])

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

# Load trained model weights
model.load_state_dict(torch.load('final_model.pth'))
print("Model weights loaded successfully!")
model.eval()

# Streamlit UI
st.title("Brain Tumor Detection")
st.write("Upload MRI images to predict if there is a brain tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and process the uploaded image
    image = Image.open(uploaded_file).convert('RGB')  # Convert to RGB explicitly
    image = transform(image).unsqueeze(0).to(device)  # Apply transformations and add batch dimension

    # Predict
    with torch.no_grad():  # Disable gradients during inference
        outputs = model(image)  # Forward pass
        print("Raw model outputs:", outputs)  # Check the raw model outputs
        probabilities = torch.sigmoid(outputs).squeeze()  # Get probabilities from sigmoid
        print("Probabilities after sigmoid:", probabilities)  # Check the probabilities
        prediction = (probabilities > 0.5).float()  # Classify based on 0.5 threshold
        print("Predicted class:", prediction.item())  # Check the final predicted class

    # Display results
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"Prediction: {'Tumor' if prediction.item() == 1 else 'No Tumor'}")
    st.write(f"Confidence: {probabilities.item() * 100:.2f}%")

    # Show output probability plot
    plt.figure(figsize=(5, 3))
    plt.bar(['No Tumor', 'Tumor'], [1 - probabilities.item(), probabilities.item()])
    plt.title('Prediction Probability')
    plt.ylabel('Probability')
    st.pyplot(plt)

