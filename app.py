import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Chest-X-Ray Classification Model (1).h5")
    return model

model = load_model()

# Class names (edit if your model uses different labels)
CLASS_NAMES = ['Normal', 'Pneumonia', 'COVID-19']

st.set_page_config(page_title="Chest X-Ray Classifier", layout="centered")
st.title("🔬 Chest X-Ray Classification")
st.write("Upload a chest X-ray image and the model will predict if it's Normal, Pneumonia, or COVID-19.")

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_batch)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Output
    st.markdown(f"### 🧠 Prediction: `{predicted_class}`")
    st.markdown(f"Confidence: **{confidence:.2f}%**")

    # Optional: Show raw scores
    st.subheader("Prediction Scores:")
    for i, score in enumerate(prediction):
        st.write(f"{CLASS_NAMES[i]}: {score:.4f}")
