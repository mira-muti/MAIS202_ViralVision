"""
Flask API server for ViralVision React frontend.

This server provides REST API endpoints for the React frontend.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import traceback
from pathlib import Path
from werkzeug.utils import secure_filename

# Add backend directory to path for imports
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from predict import predict_video
from prediction_logger import load_predictions

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration (store uploads under backend/data/uploads)
UPLOAD_FOLDER = str(backend_dir / "data" / "uploads")
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
MAX_FILE_SIZE = 300 * 1024 * 1024  # 300MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


def _handle_predict():
    """Predict video engagement."""
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get form data
        title = request.form.get('title', '')
        hashtags = request.form.get('hashtags', '')
        niche = request.form.get('niche', '')
        
        if not niche:
            return jsonify({'error': 'Niche is required'}), 400
        
        # Validate file
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: mp4, mov, avi, mkv'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)
        
        try:
            # Run prediction
            result = predict_video(
                video_path=video_path,
                title=title,
                hashtags=hashtags,
                niche=niche,
                video_filename=filename
            )
            
            # Convert numpy types to native Python types for JSON
            if 'top_positive_features' in result and isinstance(result['top_positive_features'], list):
                result['top_positive_features'] = [
                    {
                        'feature': f.get('feature'),
                        'importance': float(f.get('importance', 0.0))
                    }
                    for f in result['top_positive_features']
                ]
            
            if 'top_negative_features' in result and isinstance(result['top_negative_features'], list):
                result['top_negative_features'] = [
                    {
                        'feature': f.get('feature'),
                        'importance': float(f.get('importance', 0.0))
                    }
                    for f in result['top_negative_features']
                ]
            
            return jsonify(result)
            
        except Exception as pred_error:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Prediction error: {error_trace}", flush=True)
            return jsonify({'error': f'Prediction failed: {str(pred_error)}'}), 500
        
        finally:
            # Clean up uploaded file
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file: {e}")
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"API error: {error_trace}", flush=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Backward-compatible prediction endpoint."""
    return _handle_predict()


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """New analyze endpoint (same behavior as /api/predict)."""
    return _handle_predict()


@app.route('/api/history', methods=['GET'])
def history():
    """Get prediction history."""
    try:
        history_data = load_predictions()
        return jsonify(history_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting ViralVision API Server...")
    print("📡 Backend API: http://localhost:8000")
    print("🎨 Frontend should connect to: http://localhost:5173")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8000, debug=True)
