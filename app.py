import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# -------------------
# Load Model
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture (must match training)
# Example: simple CNN, replace with your architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*54*54, 128)  # adjust dims if needed
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model weights
model = SimpleCNN(num_classes=3)
model.load_state_dict(torch.load("model_3.pth", map_location=device))
model.to(device)
model.eval()

# -------------------
# Preprocessing
# -------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

class_names = ["Normal", "Pneumonia", "COVID-19"]

# -------------------
# Prediction Function
# -------------------
def predict(image):
    image = transform(image).unsqueeze(0).to(device)  # (1,3,224,224)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# -------------------
# Streamlit UI
# -------------------
st.title("🩻 Chest X-ray Classifier")
st.write("Upload a chest X-ray to classify as **Normal**, **Pneumonia**, or **COVID-19**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Classify"):
        label = predict(image)
        st.success(f"Prediction: **{label}**")
