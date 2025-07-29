import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.layers import InputLayer

# Custom InputLayer to handle batch_shape
class CustomInputLayer(InputLayer):
    @classmethod
    def from_config(cls, config):
        if "batch_shape" in config:
            config["batch_input_shape"] = config.pop("batch_shape")  # Convert to compatible key
        return cls(**config)

# Load the model with custom objects
model = tf.keras.models.load_model("model.h5", custom_objects={"InputLayer": CustomInputLayer})

# Preprocess function
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Prediction function
def predict(image):
    image = preprocess(image)
    predictions = model.predict(image)[0]
    classes = ["Normal", "COVID-19", "Pneumonia"]
    results = {cls: float(predictions[i]) for i, cls in enumerate(classes)}
    return results

# Create interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Chest X-Ray Classifier",
    description="Upload a chest X-ray image. The model will classify it as Normal, COVID-19, or Pneumonia."
)

demo.launch()
