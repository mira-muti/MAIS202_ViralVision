#!/bin/bash
# Script: 02_extract_audio.sh
# Owner: Backend Pod – Person 1
# Purpose: Extract first 5 seconds of audio from videos to WAV format

# Requirements:
# - ffmpeg must be installed
# - metadata.csv must exist with downloaded_path column

echo "=========================================="
echo "Step 2: Extracting Audio (first 5s)"
echo "=========================================="

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg not found - please install:"
    echo "   macOS: brew install ffmpeg"
    echo "   Linux: sudo apt install ffmpeg"
    exit 1
fi

# Create audio directory
mkdir -p media/audio

# Check if metadata.csv exists
if [ ! -f "data/raw/metadata.csv" ]; then
    echo "❌ metadata.csv not found - run 01_download_videos.sh first"
    exit 1
fi

# Counter
COUNT=0
ERRORS=0

# Read metadata.csv and extract audio
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
    
    # Extract audio (first 5 seconds, mono, 44.1kHz)
    OUTPUT="media/audio/${video_id}.wav"
    
    ffmpeg -y -i "$path" \
        -t 5 \
        -vn \
        -acodec pcm_s16le \
        -ar 44100 \
        -ac 1 \
        "$OUTPUT" \
        -loglevel error
    
    if [ $? -eq 0 ]; then
        ((COUNT++))
        echo "✅ [$COUNT] Extracted audio: ${video_id}.wav"
    else
        echo "❌ Failed to extract audio from: $path"
        ((ERRORS++))
    fi
done

echo ""
echo "=========================================="
echo "✅ Step 2 Complete"
echo "   Extracted: $COUNT audio files"
echo "   Errors: $ERRORS"
echo "=========================================="

