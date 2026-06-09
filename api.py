from flask import Flask, request, jsonify, send_file
import numpy as np
from PIL import Image
import io

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

app = Flask(__name__)

# Load and allocate tensors
interpreter = tflite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMAGE_SIZE = 128
# The labels typically output by the model (matches the original main.py logic)
class_labels = ['glioma', 'meningioma', 'pituitary', 'notumor']

@app.route('/')
def home():
    # Serve the frontend HTML
    return send_file('mri_tumor_detection_enhanced.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    try:
        # Load image via PIL
        img = Image.open(file.stream).convert('RGB')
        
        # Resize to match what model expects
        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # Convert to array and normalize (using standard numpy)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction using TFLite interpreter
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Format results
        results = {}
        for label, score in zip(class_labels, prediction):
            results[label] = float(score)
            
        best_class_idx = np.argmax(prediction)
        
        return jsonify({
            'probs': results,
            'best_class': class_labels[best_class_idx],
            'confidence': float(np.max(prediction))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
