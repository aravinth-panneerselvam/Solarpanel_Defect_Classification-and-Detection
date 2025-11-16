import streamlit as st
import numpy as np
import torch
import json
import torchvision.transforms as transforms
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import ultralytics
from ultralytics import YOLO
import tempfile
import os


st.title("Solar Panel Defect Classifier")
st.write("Upload an image to classify the defect type using a trained deep learning model.")

# -------------------------------
# Load Model and class indices
# -------------------------------

@st.cache_resource
def load_trained_model():
    class_model = load_model('vgg16_solarpanel_model.h5',compile=False, safe_mode=False)
    return class_model

class_model = load_trained_model()

# -------------------------------
# Class Names
# -------------------------------
class_names = ['Clean', 'Dusty', 'Bird-Drop', 'Electrical-Damage', 'Physical-Damage', 'Snow-Covered']


# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse mapping (index → label)
class_labels = list(class_indices.keys())

# -------------------------------
# Image Upload
# -------------------------------

uploaded_file = st.file_uploader("📤 Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:

    # Load and show image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image for prediction
    img = img.resize((224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    pred = class_model.predict(img_array)
    # st.write(pred)
    class_idx = np.argmax(pred, axis=1)[0]
    # class_label = list(train_data.class_indices.keys())[class_idx]
    class_label = class_labels[class_idx]

    # Display results
    st.success(f"✅ Predicted Class: **{class_label}**")

#============================================================================
# Object Detection
#============================================================================

model = YOLO(r"E:/01. Academics/05. GUVI/Mini_Project/05. Solar_panel_defect_Detection/models/best.pt")

st.title("Solar Panel Defect Detection")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Read the image bytes ONCE
    image_bytes = uploaded_file.read()

    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Save image correctly to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(image_bytes)
        temp_path = temp.name

    # YOLO prediction
    results = model.predict(
        source=temp_path,
        conf=0.25,
        imgsz=640,
        save=False
    )

    # Plot detections
    result_img = results[0].plot()
    st.image(result_img, caption="Detected Defects", use_container_width=True)

    # Cleanup
    os.remove(temp_path)
