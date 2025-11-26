"""
Flask API server for ViralVision frontend.
Provides REST API endpoint for video prediction.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from predict import predict_video
from prediction_logger import load_predictions

app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle video prediction request from frontend."""
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # Get form data
        title = request.form.get('title', '')
        hashtags = request.form.get('hashtags', '')
        niche = request.form.get('niche', '')
        
        if not title:
            return jsonify({'error': 'Title is required'}), 400
        if not niche:
            return jsonify({'error': 'Niche is required'}), 400
        
        # Save uploaded file temporarily with original extension
        file_ext = os.path.splitext(video_file.filename)[1] or '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            video_path = tmp_file.name
            video_file.save(video_path)
        
        try:
            # Call prediction function with filename for logging
            result = predict_video(video_path, title, hashtags, niche, video_filename=video_file.filename)
            
            return jsonify(result)
        except Exception as pred_error:
            # Log the actual error for debugging
            import traceback
            error_trace = traceback.format_exc()
            print(f"Prediction error: {error_trace}", flush=True)
            return jsonify({'error': f'Prediction failed: {str(pred_error)}'}), 500
        finally:
            # Clean up temporary file
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception as cleanup_error:
                    print(f"Cleanup error: {cleanup_error}", flush=True)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"API error: {error_trace}", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def history():
    """Get prediction history."""
    try:
        predictions = load_predictions()
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
