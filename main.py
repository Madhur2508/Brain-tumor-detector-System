import streamlit as st
import numpy as np
from PIL import Image

# Load TFLite model with fallback imports to support both local dev and Render environment
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        import tensorflow as tf
        tflite = tf.lite

# Page config
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

# Load and allocate tensors
interpreter = tflite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Image size used during training
IMAGE_SIZE = 128

# Class labels
class_labels = ['glioma', 'meningioma', 'pituitary', 'notumor']
class_names = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'pituitary': 'Pituitary Tumor',
    'notumor': 'No Tumor'
}

# Simple CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    
    .stApp {
        max-width: 1000px;
        margin: 0 auto;
    }
    
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 20px 0;
    }
    
    .upload-text {
        text-align: center;
        color: #7f8c8d;
        font-size: 14px;
        margin-top: 10px;
    }
    
    .result-box {
        background-color: white;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 20px 0;
        text-align: center;
    }
    
    .prediction-label {
        font-size: 28px;
        font-weight: bold;
        color: #2c3e50;
        margin: 15px 0;
    }
    
    .confidence-value {
        font-size: 48px;
        font-weight: bold;
        color: #3498db;
        margin: 10px 0;
    }
    
    .score-item {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .score-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    
    .score-name {
        font-weight: 600;
        color: #2c3e50;
    }
    
    .score-value {
        font-weight: 600;
        color: #3498db;
    }
    
    .progress-bar {
        background-color: #e9ecef;
        border-radius: 10px;
        height: 10px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background-color: #3498db;
        transition: width 0.5s ease;
    }
    
    .disclaimer {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
    }
    
    .disclaimer-title {
        font-weight: bold;
        color: #856404;
        margin-bottom: 8px;
    }
    
    .disclaimer-text {
        color: #856404;
        font-size: 14px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 Brain Tumor Detection")

# Upload section
st.markdown("### Upload MRI Image")
uploaded_file = st.file_uploader("Choose an MRI scan image", type=["jpg", "jpeg", "png"])
st.markdown('<p class="upload-text">Supported formats: JPG, JPEG, PNG</p>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_file, caption="Uploaded MRI Scan", use_column_width=True)
    
    # Process image
    img = Image.open(uploaded_file).convert('RGB')
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    with st.spinner('Analyzing...'):
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Display result
    st.markdown(f"""
    <div class="result-box">
        <div style="color: #7f8c8d; font-size: 16px; margin-bottom: 10px;">Diagnosis</div>
        <div class="prediction-label">{class_names[predicted_class]}</div>
        <div class="confidence-value">{confidence:.1f}%</div>
        <div style="color: #7f8c8d; font-size: 14px;">Confidence</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed scores
    st.markdown("### Prediction Scores")
    
    for label, score in zip(class_labels, prediction[0]):
        score_percent = score * 100
        st.markdown(f"""
        <div class="score-item">
            <div class="score-header">
                <span class="score-name">{class_names[label]}</span>
                <span class="score-value">{score_percent:.1f}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {score_percent}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <div class="disclaimer-title">⚠️ Medical Disclaimer</div>
        <div class="disclaimer-text">
            This tool is for educational purposes only. Do not use it as a substitute for 
            professional medical advice. Always consult with qualified healthcare providers.
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload an MRI scan image to begin analysis")
    
    st.markdown("### How it works:")
    st.markdown("""
    1. Upload an MRI brain scan image
    2. The AI model analyzes the image
    3. Get instant results with confidence scores
    4. View detailed breakdown of predictions
    """)