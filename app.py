import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("Chest_X-Ray_Classification_Model.h5")
labels = ["Normal", "COVID-19", "Pneumonia"]

def classify(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    return {labels[i]: float(preds[i]) for i in range(len(labels))}

demo = gr.Interface(fn=classify, inputs=gr.Image(type="pil"), outputs=gr.Label())
demo.launch(server_name="0.0.0.0", server_port=8080)
