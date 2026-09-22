import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify
import json
import os
import threading
import time

app = Flask(__name__)

# ==============================================================================
# CONFIGURATION & GLOBAL STATE
# ==============================================================================
MODEL_PATH = "festival_model.keras"
LABELS_PATH = "festival_model.labels.json"
CONFIDENCE_THRESHOLD = 0.85

# State Variables
current_class = "background"
current_confidence = 0.0
camera = None
model = None
class_names = []

# ==============================================================================
# INITIALIZATION
# ==============================================================================
def load_model_and_labels():
    global model, class_names
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("Warning: Model or labels not found. Ensure you run train_model.py first.")
        return False
        
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABELS_PATH, 'r') as f:
            class_names = json.load(f)
        print(f"Successfully loaded model with {len(class_names)} classes.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Initialize at startup
load_model_and_labels()

# ==============================================================================
# OPENCV & INFERENCE LOGIC
# ==============================================================================
def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        # Reduce resolution for better streaming performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def run_inference(frame):
    """Run model prediction on a single frame."""
    global current_class, current_confidence
    
    if model is None or not class_names:
        return frame
        
    try:
        # Preprocess
        resized = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_arr = np.expand_dims(rgb_frame, axis=0)
        
        # Predict
        predictions = model.predict(input_arr, verbose=0)[0]
        max_idx = int(np.argmax(predictions))
        confidence = float(predictions[max_idx])
        
        # Update State
        if confidence >= CONFIDENCE_THRESHOLD:
            current_class = class_names[max_idx]
        else:
            current_class = "background"
            
        current_confidence = confidence
        
        # Overlay Bounding Box / Text on the frame itself
        text = f"{current_class} ({current_confidence*100:.1f}%)"
        color = (0, 255, 0) if current_class != 'background' else (0, 0, 255)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
    except Exception as e:
        print(f"Inference error: {e}")
        
    return frame

def generate_frames():
    """Generator for MJPEG stream."""
    cam = get_camera()
    frame_skip = 5 # Run inference every 5th frame to avoid lag
    count = 0
    
    while True:
        success, frame = cam.read()
        if not success:
            break
            
        if count % frame_skip == 0:
            frame = run_inference(frame)
        else:
            # Just overlay the last known state without running heavy inference
            text = f"{current_class} ({current_confidence*100:.1f}%)"
            color = (0, 255, 0) if current_class != 'background' else (0, 0, 255)
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
        count += 1
            
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==============================================================================
# FLASK ROUTES
# ==============================================================================
@app.route('/')
def index():
    """Render the main dashboard."""
    # Auto-map audio files based on class names
    audio_mapping = {}
    audio_dir = os.path.join(app.root_path, 'static', 'audio')
    if os.path.exists(audio_dir):
        files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]
        for c in class_names:
            for f in files:
                if c.lower() in f.lower():
                    audio_mapping[c] = f
                    break
                    
    return render_template('index.html', classes=class_names, audio_mapping=audio_mapping)

@app.route('/video_feed')
def video_feed():
    """Route for the MJPEG video stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    """Returns the current detection status."""
    return jsonify({
        'class': current_class,
        'confidence': current_confidence
    })

if __name__ == '__main__':
    print("Starting Flask Dashboard on http://localhost:5000")
    # debug=False and threaded=True is required for reliable OpenCV threading in Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
