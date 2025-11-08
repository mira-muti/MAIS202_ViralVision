#!/bin/bash
# Script: 01_download_videos.sh
# Owner: Backend Pod – Person 1
# Purpose: Download TikTok videos using your scraper and create metadata.csv

# Notes:
# - This script is a placeholder for your TikTok scraping logic
# - Replace with your actual scraper implementation
# - Output should be: data/raw/metadata.csv + media/videos/*.mp4

# Expected output format for metadata.csv:
# video_id,creator_handle,video_url,likes,comments,shares,views,caption,niche,downloaded_path

echo "=========================================="
echo "Step 1: Downloading TikTok Videos"
echo "=========================================="

# Create directories
mkdir -p data/raw
mkdir -p media/videos

# TODO: Replace with your actual TikTok scraper
# Example:
# python scripts/tiktok_scraper.py \
#   --niche fitness \
#   --count 60 \
#   --output data/raw/metadata.csv \
#   --download_dir media/videos/

echo "⚠️  PLACEHOLDER: Replace this with your TikTok scraper"
echo ""
echo "Your scraper should:"
echo "  1. Fetch video metadata (likes, views, comments, etc.)"
echo "  2. Download video files to media/videos/"
echo "  3. Create metadata.csv with columns:"
echo "     - video_id, creator_handle, video_url"
echo "     - likes, comments, shares, views"
echo "     - caption, niche, downloaded_path"
echo ""
echo "Example metadata.csv row:"
echo "7239401123,@fitcoach,https://tiktok.com/@...,125034,1032,756,980000,\"3 exercises\",fitness,media/videos/7239401123.mp4"
echo ""
echo "For now, you can create a sample metadata.csv manually for testing."

# Check if metadata.csv exists
if [ -f "data/raw/metadata.csv" ]; then
    echo "✅ metadata.csv found"
    VIDEO_COUNT=$(wc -l < data/raw/metadata.csv)
    echo "📊 Total videos: $VIDEO_COUNT"
else
    echo "❌ metadata.csv not found - please run your scraper first"
    exit 1
fi

echo "=========================================="
echo "✅ Step 1 Complete"
echo "=========================================="

