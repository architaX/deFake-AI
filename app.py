# Import necessary libraries
from flask import Flask, render_template, request, jsonify  # Flask web framework components
import os  # Operating system interfaces
import cv2  # OpenCV for computer vision tasks
import numpy as np  # Numerical computing
import tensorflow as tf  # Machine learning framework
from werkzeug.utils import secure_filename  # Secure file upload handling
from flask_cors import CORS  # Cross-Origin Resource Sharing support
from facenet_pytorch import MTCNN  # Face detection model
import torch  # PyTorch deep learning framework
import torchaudio  # Audio processing library
from transformers import Wav2Vec2Processor, Wav2Vec2Model  # Pretrained audio models
from torch.nn import Linear  # Neural network linear layer
import torch.nn as nn  # Neural network base class
import json  # JSON data handling
import requests  # HTTP requests library

# Set computation device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Flask application
app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True  # Enable exception propagation
CORS(app)  # Enable Cross-Origin Resource Sharing
app.config['UPLOAD_FOLDER'] = 'uploads'  # Directory for file uploads
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'wav', 'mp3'}  # Supported file types

# Load Sapling.ai API key from environment variables
SAPLING_API_KEY = os.environ.get("SAPLING_API_KEY") 

# Verify API key availability
if not SAPLING_API_KEY:
    print("⚠️ Error: SAPLING_API_KEY not found in environment variables")
else:
    print(f"✅ API Key loaded (first 5 chars): {SAPLING_API_KEY[:5]}...")

# Setup feedback data directory
FEEDBACK_DIR = "feedback_data"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# Global variables for models
image_model = None  # TensorFlow image detection model
video_model = None  # TensorFlow video detection model
audio_processor = None  # Wav2Vec2 audio processor
audio_model = None  # Wav2Vec2 base model
classifier_model = None  # Custom audio classifier

# Image processing parameters
IMG_SIZE = (128, 128)  # Target image dimensions for model input
mtcnn = MTCNN(select_largest=True, device='cpu')  # Face detection model (CPU only)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def optimize_models():
    """Configure TensorFlow performance settings"""
    tf.config.threading.set_intra_op_parallelism_threads(2)  # Intra-operation threads
    tf.config.threading.set_inter_op_parallelism_threads(2)  # Inter-operation threads
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU for TensorFlow

def load_models():
    """Load image and video detection models"""
    global image_model, video_model
    try:
        image_model = tf.keras.models.load_model('simple_deepfake_detector.h5')
        video_model = tf.keras.models.load_model('video_deepfake_detector.h5')
        print("✅ Both models loaded successfully.")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")

def load_audio_model():
    """Load audio processing pipeline including Wav2Vec2 and classifier"""
    global audio_processor, audio_model, classifier_model
    try:
        print("⏳ Loading Wav2Vec2 processor...")
        audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        print("⏳ Loading Wav2Vec2 model...")
        audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        
        print("⏳ Initializing classifier...")
        classifier_model = Wav2Vec2Classifier().to(device)
        
        print("🔍 Looking for classifier weights...")
        if not os.path.exists("wav2vec2_deepfake_classifier.pth"):
            raise FileNotFoundError("Model weights file 'wav2vec2_deepfake_classifier.pth' not found")
            
        print("⏳ Loading classifier weights...")
        state_dict = torch.load("wav2vec2_deepfake_classifier.pth", map_location=device)
        classifier_model.load_state_dict(state_dict)
        classifier_model.eval()
        
        print("✅ Audio pipeline loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Audio model loading failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        # Clean up partial loading
        audio_processor = None
        audio_model = None
        classifier_model = None
        return False
    
class Wav2Vec2Classifier(nn.Module):
    """Custom classifier for audio deepfake detection"""
    def __init__(self):
        super(Wav2Vec2Classifier, self).__init__()
        self.wav2vec2 = audio_model  # Base audio model
        self.fc = Linear(self.wav2vec2.config.hidden_size, 2)  # Binary classification layer

    def forward(self, input_values):
        """Forward pass through the network"""
        with torch.no_grad():
            outputs = self.wav2vec2(input_values).last_hidden_state.mean(dim=1)  # Feature extraction
        return self.fc(outputs)  # Classification result

def detect_ai_text(text):
    """Detect AI-generated text using Sapling API"""
    try:
        if not SAPLING_API_KEY:
            raise ValueError("API key not configured")
        response = requests.post(
            "https://api.sapling.ai/api/v1/aidetect",
            headers={"Content-Type": "application/json", "API-KEY": "QSIW0I996C4CVLVUP3MIH2V9EW2KPLFC"},
            json={"text": text,"key": "QSIW0I996C4CVLVUP3MIH2V9EW2KPLFC"},timeout=10
        )
        response.raise_for_status()
        result = response.json()
        ai_score = result["score"]
        is_ai = ai_score >= 0.7
        confidence = round(ai_score * 100, 2) if is_ai else round((1 - ai_score) * 100, 2)
        return {
            "is_ai": is_ai,
            "confidence": confidence,
            "verdict": "AI-generated" if is_ai else "Human-written"
        }
    except Exception as e:
        return {"error": str(e)}

def detect_image_deepfake(filepath):
    """Detect deepfakes in images"""
    global image_model
    try:
        if image_model is None:
            load_models()

        img = cv2.imread(filepath)
        if img is None:
            return {'error': 'Could not read image file'}

        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        confidence = float(image_model.predict(img)[0][0])
        is_deepfake = confidence < 0.5
        adjusted_confidence = 1 - confidence if is_deepfake else confidence

        return {
            'type': 'image',
            'is_deepfake': is_deepfake,
            'confidence': round(adjusted_confidence * 100,2),
            'verdict': 'Deepfake' if is_deepfake else 'Genuine'
        }
    except Exception as e:
        return {'error': str(e)}

def detect_video_deepfake(filepath):
    """Detect deepfakes in videos by sampling frames"""
    global video_model
    try:
        if video_model is None:
            load_models()

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return {'error': 'Could not open video file'}

        # Process video frames
        results = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_every = max(1, frame_count // 20)  # Sample 20 frames

        for i in range(0, frame_count, sample_every):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Preprocess each frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)

            confidence = float(video_model.predict(img)[0][0])
            results.append(confidence)

        cap.release()

        # Calculate average confidence
        avg_confidence = sum(results) / len(results) if results else 0.5
        is_deepfake = avg_confidence < 0.5

        return {
            'type': 'video',
            'is_deepfake': is_deepfake,
            'confidence': round((1 - avg_confidence if is_deepfake else avg_confidence) * 100, 2),
            'verdict': 'Deepfake' if is_deepfake else 'Genuine'
        }
    except Exception as e:
        return {'error': str(e)}

def detect_audio_deepfake(filepath):
    """Detect deepfakes in audio files"""
    global audio_processor, audio_model, classifier_model

    # Check and load models if needed
    if None in [audio_processor, audio_model, classifier_model]:
        load_audio_model()
        if None in [audio_processor, audio_model, classifier_model]:
            return {'error': 'Audio models failed to load', 'details': 'Check server logs'}

    try:
        if classifier_model is None:
            return {'error': 'Audio classifier model not loaded'}

        # Load and preprocess audio
        speech_array, sample_rate = torchaudio.load(filepath)
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        speech_array = resampler(speech_array).mean(dim=0).squeeze()

        # Process audio for Wav2Vec2
        inputs = audio_processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device).squeeze(0).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = classifier_model(input_values)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][1].item()  # Probability of being real

        is_deepfake = confidence < 0.5
        return {
            'type': 'audio',
            'is_deepfake': is_deepfake,
            'confidence': round((1 - confidence if is_deepfake else confidence) * 100, 2),
            'verdict': 'Deepfake' if is_deepfake else 'Genuine'
        }

    except Exception as e:
        return {'error': f'Audio analysis failed: {str(e)}'}

# Flask API endpoints
@app.route('/detect-text', methods=['POST'])
def detect_text():
    """API endpoint for text detection"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text or len(text) < 20:
            return jsonify({
                "error": "Invalid input",
                "message": "Text must be at least 20 characters"
            }), 400
            
        result = detect_ai_text(text)
        
        return jsonify({
            "success": True,
            "type": "text",
            "is_ai": result.get("is_ai", False),
            "confidence": result.get("confidence", 0),
            "verdict": result.get("verdict", "Unknown")
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
@app.route('/model-status')
def model_status():
    """Endpoint to check model loading status"""
    return jsonify({
        'image_model': image_model is not None,
        'video_model': video_model is not None,
        'audio_processor': audio_processor is not None,
        'audio_model': audio_model is not None, 
        'classifier_model': classifier_model is not None,
        'device': str(device)
    })

@app.route('/test-sapling')
def test_sapling():
    """Test endpoint for Sapling API connectivity"""
    try:
        test_text = "This is an API connectivity test"
        result = detect_ai_text(test_text)
        return jsonify({
            "status": "success",
            "sapling_response": result
        })
    except Exception as e:
        return jsonify({
            "status": "failed", 
            "error": str(e)
        }), 500

@app.route('/')
def index():
    """Main page endpoint"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint for file upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        # Route to appropriate detector based on file type
        ext = filename.rsplit('.', 1)[1].lower()
        if ext in ['wav', 'mp3']:
            result = detect_audio_deepfake(filepath)
        elif ext == 'mp4':
            result = detect_video_deepfake(filepath)
        else:
            result = detect_image_deepfake(filepath)

        os.remove(filepath)  # Clean up uploaded file
        return jsonify(result)

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/feedback', methods=['POST'])
def collect_feedback():
    """Endpoint for collecting user feedback on predictions"""
    data = request.json
    file_path = data.get('file_path')
    correct_label = data.get('correct_label')
    feedback_entry = {"file_path": file_path, "correct_label": correct_label}
    feedback_file = os.path.join(FEEDBACK_DIR, "feedback_log.json")
    
    # Load existing feedback or create new file
    feedback_data = json.load(open(feedback_file)) if os.path.exists(feedback_file) else []
    feedback_data.append(feedback_entry)
    
    # Save updated feedback
    json.dump(feedback_data, open(feedback_file, "w"), indent=4)
    return jsonify({"message": "Feedback received!"})

# Application entry point
if __name__ == "__main__":
    app.run(debug=True)

if __name__ == '__main__':
    optimize_models()  # Configure TensorFlow
    load_models()  # Load image/video models
    load_audio_model()  # Load audio models
    app.run(debug=True)  # Start Flask server
    