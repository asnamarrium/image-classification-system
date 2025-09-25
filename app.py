import streamlit as st
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os


def load_models():
    mobilenet_model = tf.keras.models.load_model("mobilenet_model.h5")
    nasnet_model = tf.keras.models.load_model("nasnet_model.h5")
    return mobilenet_model, nasnet_model

mobilenet_model, nasnet_model = load_models()
class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


st.title("🌸 Flower Image Classification App")
st.write("Upload an image and classify it using **MobileNet** and **NASNet** models.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    img = image.load_img(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    processed_img = preprocess_image(img)

    mobilenet_pred = mobilenet_model.predict(processed_img)
    nasnet_pred = nasnet_model.predict(processed_img)

    mobilenet_class = class_labels[np.argmax(mobilenet_pred)]
    nasnet_class = class_labels[np.argmax(nasnet_pred)]

    st.subheader("Predictions")
    st.write(f"**MobileNet Prediction:** {mobilenet_class} ({np.max(mobilenet_pred)*100:.2f}%)")
    st.write(f"**NASNet Prediction:** {nasnet_class} ({np.max(nasnet_pred)*100:.2f}%)")

    st.bar_chart(dict(zip(class_labels, mobilenet_pred[0])))
    st.bar_chart(dict(zip(class_labels, nasnet_pred[0])))
