# Debugging the 500 Error

## What's Happening

The frontend is successfully sending the request to the backend (the proxy is working!), but the backend is crashing with a 500 error.

## Quick Fix Steps

### 1. Check if Model Files Exist
```bash
ls -la models/
```
You should see:
- `model.pkl`
- `model_encoder.pkl`

If they're missing, run:
```bash
python3 src/train_model.py
```

### 2. Check the API Server Logs

When you run `python3 api_server.py`, you should see error messages in the terminal. Look for:
- Import errors
- File not found errors
- Prediction errors

### 3. Test the Prediction Function Directly

```bash
python3 -c "
import sys
sys.path.insert(0, 'src')
from predict import predict_video
result = predict_video('test_video.mp4', 'Test Title', '#test', 'music')
print(result)
"
```

### 4. Common Issues

**Issue: Model files missing**
- Solution: Run `python3 src/train_model.py` first

**Issue: MoviePy not installed**
- Solution: `pip install moviepy`

**Issue: Video file format not supported**
- Solution: Make sure the video is MP4 or MOV

**Issue: Niche not in training data**
- Solution: Use one of: music, dance, food, GRWM, OOTD, DIYProjects, summervibes

## How to See the Actual Error

1. Open a terminal
2. Run: `python3 api_server.py`
3. Try uploading a video from the frontend
4. Look at the terminal output - it will show the actual error

The error message will tell you exactly what's wrong!

