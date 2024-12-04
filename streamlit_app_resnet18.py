import torch
import numpy as np
import cv2
import os
import streamlit as st
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torch.nn.functional as F

# Define the MRI Dataset class
class MRI_Dataset(Dataset):
    def __init__(self, tumor_dir, healthy_dir, image_size=(224, 224)):
        self.images = []
        self.labels = []
        self.image_size = image_size

        # Load tumor images
        for filename in os.listdir(tumor_dir):
            img = cv2.imread(os.path.join(tumor_dir, filename))
            if img is not None:
                img = cv2.resize(img, self.image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.images.append(img)
                self.labels.append(1)  # Tumor = 1

        # Load healthy images
        for filename in os.listdir(healthy_dir):
            img = cv2.imread(os.path.join(healthy_dir, filename))
            if img is not None:
                img = cv2.resize(img, self.image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.images.append(img)
                self.labels.append(0)  # Healthy = 0

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0  # Normalize
        label = self.labels[idx]
        return {'image': torch.tensor(image).permute(2, 0, 1), 'label': torch.tensor(label, dtype=torch.float32)}

# Load pre-trained ResNet18 model
model = resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 1),  # Binary classification
    nn.Sigmoid()  # Sigmoid for output probabilities
)

# Load model weights
# model.load_state_dict(torch.load('Resnet18_model_epoch_29.pth'))
model.load_state_dict(torch.load('Resnet18False_model_epoch_25.pth'))
model.eval()

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Streamlit app
st.title("Brain Tumor Detection Using MRI Scans")

st.write("""
    Upload an MRI image to predict whether it contains a brain tumor or not.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and process the uploaded image
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Resize and normalize image for ResNet18
    image = cv2.resize(image, (224, 224))  # Resize to match ResNet18 input size
    image = image.astype(np.float32) / 255.0  # Normalize
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension
    
    # Make prediction
    with torch.no_grad():
        output = model(image).squeeze()  # Forward pass through the model
        prediction = (output > 0.5).float()  # 1 if tumor, 0 if no tumor
    
    # Display prediction result
    if prediction == 1:
        st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)
        st.write("Prediction: Tumor detected")
    else:
        st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)
        st.write("Prediction: No tumor detected")
