import streamlit as st
import numpy as np
from PIL import Image

# Load TFLite model with fallback imports to support both local dev and Render environment
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError:
            import tensorflow as tf
            tflite = tf.lite

# Load and allocate tensors
interpreter = tflite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Image size used during training
IMAGE_SIZE = 128

# Class labels (update according to your dataset folder names)
class_labels = ['glioma', 'meningioma', 'pituitary', 'notumor']  # Change if needed

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI scan image and the model will predict the tumor type.")

# File upload
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded MRI Scan", use_column_width=True)

    # Convert to model format
    img = Image.open(uploaded_file).convert('RGB')
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader("🔍 Prediction Result")
    st.write(f"### 🧩 Tumor Type: **{predicted_class}**")
    st.write(f"### 📌 Confidence: **{confidence:.2f}%**")

    # Optional: show all class prediction scores
    st.write("---")
    st.write("### 📊 Prediction Scores:")
    for label, score in zip(class_labels, prediction[0]):
        st.write(f"- **{label}** → {score * 100:.2f}%")
