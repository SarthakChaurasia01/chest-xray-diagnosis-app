import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ChestXRayClassifier
import os

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    input_shape = 3
    output_shape = 3
    hidden_units = 10   # 👈 your training value

    model = ChestXRayClassifier(input_shape, output_shape, hidden_units)
    state_dict = torch.load("model_3.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# ----------------------------
# Transforms
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
# Save wrong predictions
# ----------------------------
def save_wrong_feedback(image, filename, predicted_label):
    feedback_dir = "feedback"
    os.makedirs(feedback_dir, exist_ok=True)
    save_path = os.path.join(feedback_dir, f"WRONG_{predicted_label}_{filename}")
    image.save(save_path)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🩻 Chest X-Ray Classifier")
st.write("Upload one or multiple chest X-ray images, and the model will predict if it's **Normal**, **Pneumonia**, or **COVID-19**.")

uploaded_files = st.file_uploader("📂 Upload X-ray Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        if st.button(f"🔍 Predict {uploaded_file.name}"):
            prediction = predict(image)
            st.success(f"Prediction for {uploaded_file.name}: **{prediction}**")

            # Wrong prediction feedback
            if st.button(f"❌ Report Wrong Prediction for {uploaded_file.name}"):
                save_wrong_feedback(image, uploaded_file.name, prediction)
                st.info(f"Your feedback for {uploaded_file.name} has been recorded ✅")
