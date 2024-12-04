import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import numpy as np
import cv2
from model import CNN  # Import your model (ensure the model class is in this file)

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('final_model.pth'))
model.eval()  # Set to evaluation mode

# Transform function (same as used in the training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image():
    # Open file dialog to choose an image
    filepath = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if filepath:
        # Open the image
        img = Image.open(filepath)
        img = img.resize((128, 128))  # Resize image to match model input size
        img_tk = ImageTk.PhotoImage(img)

        # Display the image in the window
        image_label.config(image=img_tk)
        image_label.image = img_tk  # Keep a reference to avoid garbage collection

        # Process the image for the model
        image_array = np.array(img)  # Convert to NumPy array
        image_tensor = transform(image_array).unsqueeze(0)  # Add batch dimension
        return image_tensor
    return None

def predict():
    # Load and preprocess the image
    img_tensor = load_image()
    if img_tensor is not None:
        # Send image to device (if using GPU, make sure to send model and image to the same device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_tensor = img_tensor.to(device)
        model.to(device)

        # Get model prediction
        with torch.no_grad():
            output = model(img_tensor)
            prediction = output.item()  # Get the scalar value from the output tensor

        # Display the result
        result = "Tumor Detected" if prediction > 0.5 else "No Tumor Detected"
        messagebox.showinfo("Prediction", result)

# Create the main window
window = tk.Tk()
window.title("MRI Tumor Detection")

# Create and place GUI elements
open_button = tk.Button(window, text="Open Image", command=predict)
open_button.pack(pady=20)

image_label = tk.Label(window)
image_label.pack(pady=20)

# Run the Tkinter event loop
window.mainloop()
