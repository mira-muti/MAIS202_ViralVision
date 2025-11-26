#!/usr/bin/env python3
"""
Quick test script to debug the API prediction.
Run this to see what error you're getting.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("Testing prediction function...")
print("=" * 50)

try:
    from predict import predict_video
    print("✓ Successfully imported predict_video")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test with a dummy video path (this will fail but show us where)
print("\nNote: This will fail because we don't have a real video,")
print("but it will show us where the error occurs.\n")

try:
    # This will fail, but we'll see the error
    result = predict_video(
        video_path="test.mp4",  # Dummy path
        title="Test Video",
        hashtags="#test",
        niche="music"
    )
    print("✓ Prediction successful!")
    print(result)
except FileNotFoundError as e:
    print(f"✗ File not found (expected): {e}")
    print("\nThis is normal - we're just testing the imports.")
except Exception as e:
    import traceback
    print(f"✗ Error occurred:")
    print(traceback.format_exc())
    print("\nThis is the actual error you need to fix!")

