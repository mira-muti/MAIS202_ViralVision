#!/bin/bash
# Script: 03_sample_frames.sh
# Owner: Backend Pod – Person 1
# Purpose: Sample frames at 1 fps from first 5 seconds of each video

# Requirements:
# - ffmpeg must be installed
# - metadata.csv must exist with downloaded_path column

echo "=========================================="
echo "Step 3: Sampling Frames (1 fps, 5s)"
echo "=========================================="

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg not found - please install:"
    echo "   macOS: brew install ffmpeg"
    echo "   Linux: sudo apt install ffmpeg"
    exit 1
fi

# Create frames root directory
mkdir -p media/frames

# Check if metadata.csv exists
if [ ! -f "data/raw/metadata.csv" ]; then
    echo "❌ metadata.csv not found - run 01_download_videos.sh first"
    exit 1
fi

# Counter
COUNT=0
ERRORS=0

# Read metadata.csv and sample frames
# Skip header line
tail -n +2 data/raw/metadata.csv | while IFS=, read -r video_id creator url likes comments shares views caption niche path; do
    # Remove quotes from path if present
    path=$(echo "$path" | tr -d '"')
    
    # Check if video file exists
    if [ ! -f "$path" ]; then
        echo "⚠️  Video not found: $path"
        ((ERRORS++))
        continue
    fi
    
    # Create directory for this video's frames
    FRAMES_DIR="media/frames/${video_id}"
    mkdir -p "$FRAMES_DIR"
    
    # Sample frames at 1 fps for first 5 seconds
    ffmpeg -y -i "$path" \
        -t 5 \
        -vf "fps=1" \
        "${FRAMES_DIR}/frame_%03d.jpg" \
        -loglevel error
    
    if [ $? -eq 0 ]; then
        FRAME_COUNT=$(ls -1 "$FRAMES_DIR" | wc -l)
        ((COUNT++))
        echo "✅ [$COUNT] Sampled $FRAME_COUNT frames: ${video_id}/"
    else
        echo "❌ Failed to sample frames from: $path"
        ((ERRORS++))
    fi
done

echo ""
echo "=========================================="
echo "✅ Step 3 Complete"
echo "   Processed: $COUNT videos"
echo "   Errors: $ERRORS"
echo "=========================================="

