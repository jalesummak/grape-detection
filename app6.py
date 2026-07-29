import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Sayfa yapılandırması
st.set_page_config(page_title="Üzüm Hastalık Analizi", page_icon="🍇")

@st.cache_resource
def load_assets():
    # Model dosyası: grape_disease_mobilenet_98.keras
    model_path = 'grape_disease_mobilenet_98.keras'
    
    # compile=False: Önceki projede aldığın tensor hatalarını engeller
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Etiketleri oku
    with open('labels.json', 'r', encoding='utf-8') as f:
        labels = json.load(f)
    return model, labels

model, labels = load_assets()

st.title("🍇 Üzüm Yaprağı Teşhis Sistemi")

img_file = st.camera_input("Yaprağın fotoğrafını çekin")
uploaded_file = st.file_uploader("Veya fotoğraf yükleyin", type=["jpg", "png", "jpeg"])

target_file = img_file if img_file is not None else uploaded_file

if target_file:
    image = Image.open(target_file).convert('RGB')
    st.image(image, caption='İşlenen Görüntü', use_container_width=True)
    
    # Boyutlandırma: MobileNetV2 için mutlaka 224 olmalı
    img = image.resize((224, 224)) 
    img_array = np.array(img).astype('float32')
    
    # Normalizasyon: -1 ile 1 arası (MobileNetV2 standardı)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner('Yapay zeka analiz ediyor...'):
        prediction = model.predict(img_array)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction) * 100

    # Etiket ismini sözlük yapısına göre al
    try:
        class_name = labels[str(class_id)] if isinstance(labels, dict) else labels[class_id]
    except:
        class_name = f"Bilinmeyen Sınıf (ID: {class_id})"

    st.success(f"### Tahmin: {class_name}")
    st.info(f"**Doğruluk Oranı:** %{confidence:.2f}")
    st.progress(min(int(confidence), 100))