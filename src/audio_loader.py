"""
Audio Loader Module
Handles loading and preprocessing of audio files in various formats (WAV, MP3, FLAC).
"""

import os
import io
import tempfile
import numpy as np
import librosa
import soundfile as sf

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac"}
DEFAULT_SR = 22050  # Default sample rate for analysis
MAX_DURATION = 600  # Maximum duration in seconds (10 minutes)


class AudioLoadError(Exception):
    """Custom exception for audio loading errors."""
    pass


class AudioData:
    """Container for loaded audio data and metadata."""

    def __init__(self, y: np.ndarray, sr: int, filename: str, duration: float, channels: int):
        self.y = y                  # Audio time series (mono)
        self.sr = sr                # Sample rate
        self.filename = filename    # Original filename
        self.duration = duration    # Duration in seconds
        self.channels = channels    # Original number of channels
        self.samples = len(y)       # Total number of samples

    def __repr__(self):
        return (
            f"AudioData(filename='{self.filename}', "
            f"duration={self.duration:.2f}s, "
            f"sr={self.sr}Hz, "
            f"samples={self.samples})"
        )

    def to_dict(self):
        """Return metadata as dictionary (without raw audio data)."""
        return {
            "filename": self.filename,
            "duration": round(self.duration, 2),
            "sample_rate": self.sr,
            "samples": self.samples,
            "channels": self.channels,
        }


def validate_format(filename: str) -> bool:
    """Check if the file format is supported."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_FORMATS


def load_audio_file(file_path: str, sr: int = DEFAULT_SR) -> AudioData:
    """
    Load an audio file from disk.

    Args:
        file_path: Path to the audio file.
        sr: Target sample rate for resampling.

    Returns:
        AudioData object containing the loaded audio.

    Raises:
        AudioLoadError: If the file cannot be loaded.
    """
    if not os.path.exists(file_path):
        raise AudioLoadError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)
    if not validate_format(filename):
        ext = os.path.splitext(filename)[1]
        raise AudioLoadError(
            f"Unsupported format: '{ext}'. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    try:
        # Get original channel info
        info = sf.info(file_path)
        original_channels = info.channels

        # Load with librosa (automatically converts to mono and resamples)
        y, loaded_sr = librosa.load(file_path, sr=sr, mono=True)

        duration = librosa.get_duration(y=y, sr=loaded_sr)

        if duration > MAX_DURATION:
            raise AudioLoadError(
                f"Audio file too long ({duration:.0f}s). Maximum allowed: {MAX_DURATION}s."
            )

        return AudioData(
            y=y,
            sr=loaded_sr,
            filename=filename,
            duration=duration,
            channels=original_channels,
        )

    except AudioLoadError:
        raise
    except Exception as e:
        raise AudioLoadError(f"Failed to load audio file: {str(e)}")


def load_audio_bytes(file_bytes: bytes, filename: str, sr: int = DEFAULT_SR) -> AudioData:
    """
    Load audio from raw bytes (e.g., from an upload).

    Args:
        file_bytes: Raw file bytes.
        filename: Original filename (used for format detection).
        sr: Target sample rate.

    Returns:
        AudioData object.

    Raises:
        AudioLoadError: If the file cannot be loaded.
    """
    if not validate_format(filename):
        ext = os.path.splitext(filename)[1]
        raise AudioLoadError(
            f"Unsupported format: '{ext}'. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    try:
        # Write bytes to a temporary file for librosa to read
        ext = os.path.splitext(filename)[1].lower()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            audio_data = load_audio_file(tmp_path, sr=sr)
            # Override the filename with the original name
            audio_data.filename = filename
            return audio_data
        finally:
            os.unlink(tmp_path)

    except AudioLoadError:
        raise
    except Exception as e:
        raise AudioLoadError(f"Failed to load audio from bytes: {str(e)}")


def get_audio_segment(audio: AudioData, start: float, end: float) -> AudioData:
    """
    Extract a segment from the audio.

    Args:
        audio: Source AudioData.
        start: Start time in seconds.
        end: End time in seconds.

    Returns:
        New AudioData with the segment.
    """
    start_sample = int(start * audio.sr)
    end_sample = int(end * audio.sr)

    start_sample = max(0, start_sample)
    end_sample = min(len(audio.y), end_sample)

    segment = audio.y[start_sample:end_sample]
    duration = len(segment) / audio.sr

    return AudioData(
        y=segment,
        sr=audio.sr,
        filename=audio.filename,
        duration=duration,
        channels=1,
    )
