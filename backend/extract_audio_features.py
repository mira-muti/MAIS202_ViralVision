"""
File: extract_audio_features.py

Purpose:
    Extract FFT-based audio features from video files.
"""


def extract_fft_features(video_path, seconds=60):
    """
    Extract dominant frequency and amplitude from the first N seconds of a video.
    Fully compatible with modern MoviePy (uses audio extraction without subclip).
    Defaults to 60 seconds to capture more audio data for better analysis.
    """
    try:
        import numpy as np
        from moviepy import VideoFileClip

        # Load video
        video = VideoFileClip(video_path)

        # Extract audio from the video
        audio = video.audio
        if audio is None:
            video.close()
            raise RuntimeError("This video has no audio track.")

        # Ensure the clip duration (use up to 'seconds' or full video if shorter)
        clip_duration = min(seconds, video.duration)

        # Get audio samples for the first N seconds
        audio_fps = audio.fps
        # Calculate how many samples we need for the desired duration
        num_samples = int(clip_duration * audio_fps)
        
        # Extract audio samples
        samples = audio.to_soundarray(fps=audio_fps, nbytes=2)
        
        # Limit to the first N seconds
        if len(samples) > num_samples:
            samples = samples[:num_samples]

        # Convert to mono (average across channels if stereo)
        if len(samples.shape) > 1:
            mono = samples.mean(axis=1)
        else:
            mono = samples

        # FFT
        fft_vals = np.abs(np.fft.rfft(mono))
        fft_freqs = np.fft.rfftfreq(len(mono), d=1.0 / audio_fps)

        # Find peak frequency
        idx = np.argmax(fft_vals)
        max_freq = float(fft_freqs[idx])
        max_amp = float(fft_vals[idx])

        # Cleanup
        audio.close()
        video.close()

        return max_freq, max_amp

    except Exception as e:
        raise RuntimeError(f"Error extracting audio features: {str(e)}")

