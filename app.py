import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


model = load_model("mobilenet_model.h5")

class_labels = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

st.title("Flower Image Classification App")
st.write("Upload a flower image and get the prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def prepare_image(img):
    img = img.resize((224, 224))   # MobileNet/NASNet dono 224x224 expect karte hain
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = prepare_image(img)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    st.subheader("Prediction Result:")
    st.write(f" Flower Type: **{class_labels[class_index]}**")
    st.write(f" Confidence: **{confidence:.4f}**")
