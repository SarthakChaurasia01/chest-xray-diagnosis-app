import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ChestXRayClassifier

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    input_shape = 3
    output_shape = 3
    hidden_units = 10   # 👈 your training value

    model = ChestXRayClassifier(input_shape, output_shape, hidden_units)
    model.load_state_dict(torch.load("model_3.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ----------------------------
# Transforms (same as training)
# ----------------------------
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# Class labels
# ----------------------------
CLASS_NAMES = ["Normal", "Pneumonia", "COVID-19"]

# ----------------------------
# Prediction function
# ----------------------------
def predict(image: Image.Image):
    image = data_transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return CLASS_NAMES[predicted.item()]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🩻 Chest X-Ray Classifier")
st.write("Upload a chest X-ray image, and the model will predict if it's **Normal**, **Pneumonia**, or **COVID-19**.")

uploaded_file = st.file_uploader("📂 Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("🔍 Predict"):
        prediction = predict(image)
        st.success(f"Prediction: **{prediction}**")
