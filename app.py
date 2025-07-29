
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Chest X-Ray Classifier", layout="centered")
st.title("🫁 Chest X-Ray Image Classification")
st.write("Upload a chest X-ray image and the model will predict whether it's **Normal**, **COVID-19**, or **Pneumonia**.")

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Chest X-Ray Classification Model (1).h5")
    return model

model = load_model()

# Define the class names (ensure these match your training labels)
class_names = ["COVID-19", "Normal", "Pneumonia"]

# Upload image
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", use_column_width=True)

    # Preprocess the image
    img_size = (224, 224)
    image = ImageOps.fit(image, img_size, Image.ANTIALIAS)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
