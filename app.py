import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("Chest_Xray_Model.h5")

class_names = ['Normal', 'COVID-19', 'Pneumonia']

def predict(image):
    image = image.resize((224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    prediction = model.predict(image)[0]
    return {class_names[i]: float(prediction[i]) for i in range(len(class_names))}

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Chest X-Ray Disease Classifier",
    description="Upload an X-ray image to detect Normal, COVID-19, or Pneumonia."
)

if __name__ == "__main__":
    interface.launch()
