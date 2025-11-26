"""
Prediction logging system for storing prediction history.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


LOG_FILE = "data/predictions_log.json"


def log_prediction(video_filename: str, title: str, niche: str, result: Dict) -> None:
    """
    Log a prediction to the history file.
    
    Args:
        video_filename: Name of the uploaded video file
        title: Video title/caption
        niche: Selected niche category
        result: Prediction result dictionary
    """
    log_dir = Path(LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing logs
    predictions = load_predictions()
    
    # Create new entry
    entry = {
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'video_name': video_filename,
        'title': title,
        'niche': niche,
        'predicted_label': result.get('label', 'Unknown'),
        'prob_high': result.get('prob_high', 0.0),
        'prob_low': result.get('prob_low', 0.0),
    }
    
    predictions.append(entry)
    
    # Save to file
    with open(LOG_FILE, 'w') as f:
        json.dump(predictions, f, indent=2)


def load_predictions(limit: Optional[int] = None) -> List[Dict]:
    """
    Load prediction history from log file.
    
    Args:
        limit: Maximum number of predictions to return (most recent first)
        
    Returns:
        List of prediction entries
    """
    if not os.path.exists(LOG_FILE):
        return []
    
    try:
        with open(LOG_FILE, 'r') as f:
            predictions = json.load(f)
        
        # Sort by timestamp (most recent first)
        predictions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        if limit:
            return predictions[:limit]
        
        return predictions
    except Exception:
        return []

