"""
File: extract_audio_features.py

Purpose:
    Extract FFT-based audio features from video files.
"""


def extract_fft_features(video_path, seconds=5):
    """
    Extract dominant frequency and amplitude from the first N seconds of a video.
    Fully compatible with modern MoviePy (AudioFileClip no longer supports .subclip()).
    """
    try:
        import numpy as np
        from moviepy.editor import VideoFileClip

        # Load video
        video = VideoFileClip(video_path)

        # Ensure the clip is long enough
        clip_duration = min(seconds, video.duration)

        # Extract audio from the first N seconds of the VIDEO subclip
        subclip = video.subclip(0, clip_duration)
        audio = subclip.audio
        if audio is None:
            raise RuntimeError("This video has no audio track.")

        # Convert audio to samples
        audio_fps = audio.fps
        samples = audio.to_soundarray(fps=audio_fps)

        # Convert to mono
        mono = samples.mean(axis=1)

        # FFT
        fft_vals = np.abs(np.fft.rfft(mono))
        fft_freqs = np.fft.rfftfreq(len(mono), d=1.0 / audio_fps)

        # Find peak frequency
        idx = np.argmax(fft_vals)
        max_freq = float(fft_freqs[idx])
        max_amp = float(fft_vals[idx])

        # Cleanup
        video.close()
        subclip.close()
        audio.close()

        return max_freq, max_amp

    except Exception as e:
        raise RuntimeError(f"Error extracting audio features: {str(e)}")

