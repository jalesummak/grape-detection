import streamlit as st
import tensorflow as tf
import pickle
from PIL import Image
import numpy as np

st.set_page_config(page_title="German Traffic Sign Recognition", page_icon="🚗")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('traffic_sign.h5')
    with open('labels.pkl', 'rb') as f:
        label_dict = pickle.load(f)
    return model, label_dict

model, label_dict = load_assets()

st.title("🚗 German Traffic Sign Classifier")
st.write("Scan a traffic sign using your camera or upload an image to predict its meaning.")

img_file = st.camera_input("Take a picture of the sign")
uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "png", "jpeg"])

target_file = img_file if img_file is not None else uploaded_file

if target_file:
    image = Image.open(target_file)
    st.image(image, caption='Processed Image', use_container_width=True)
    
    img = image.resize((30, 30))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner('Analyzing...'):
        prediction = model.predict(img_array)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction) * 100

    st.success(f"### Prediction: {label_dict[class_id]}")
    st.write(f"**Confidence Level:** {confidence:.2f}%")
    st.progress(int(confidence))