import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image
import numpy as np

model = load_model('model_1.keras')

class_names = ["Meningioma", "Glioma", "Pituitary"]

def preprocess(img):
    img = img.resize((224, 224))            
    img = np.array(img)
    img = preprocess_input(img)             
    img = np.expand_dims(img, axis=0)
    return img

st.title("Brain Tumor Classification App")
st.write("Upload gambar MRI otak untuk mendeteksi jenis tumor.")

uploaded_file = st.file_uploader("Upload Brain MRI", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image")

    if st.button("Predict"):
        input_tensor = preprocess(img)
        
        st.write("DEBUG SHAPE:", input_tensor.shape)  # DEBUG

        pred = model.predict(input_tensor)
        class_id = np.argmax(pred)
        confidence = float(np.max(pred))

        st.success(f"Prediction: **{class_names[class_id]}**")
        st.info(f"Confidence: {confidence:.4f}")

