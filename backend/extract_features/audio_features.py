import numpy as np
import subprocess
import tempfile
from pathlib import Path
import librosa


def extract_audio_features(video_path: str, max_seconds: float = 3.0) -> dict:
    """
    Extract audio features from the FIRST max_seconds of the video.
    Avoids MoviePy subclip() — uses ffmpeg directly.
    """

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Temporary WAV file for the first N seconds
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp_wav = tmp.name

        # ---------------------------
        # Cut first N seconds with ffmpeg
        # ---------------------------
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-t", str(max_seconds),
            "-ac", "1",
            "-ar", "44100",
            tmp_wav
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error extracting audio: {e.stderr.decode()}")

        # ---------------------------
        # Load audio and compute features
        # ---------------------------
        y, sr = librosa.load(tmp_wav, sr=44100)

        # Basic safety check
        if len(y) == 0:
            raise RuntimeError("Empty audio segment")

        rms = float(np.mean(librosa.feature.rms(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        
        # Tempo estimation
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo)
        except Exception:
            tempo = 0.0  # Default if tempo detection fails

        # FFT features
        fft = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1/sr)
        max_idx = np.argmax(fft)

        return {
            "rms_energy": rms,
            "zcr": zcr,
            "spectral_centroid": spec_centroid,
            "spectral_rolloff": spec_rolloff,
            "tempo": tempo,
            "fft_max_freq": float(freqs[max_idx]),
            "fft_max_amp": float(fft[max_idx]),
        }
