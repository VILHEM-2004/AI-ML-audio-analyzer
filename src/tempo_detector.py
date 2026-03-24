"""
Tempo & Key Detection Module
Analyzes rhythm and tonal properties of audio to estimate BPM and musical key.
"""

import numpy as np
import librosa
from typing import Dict, Any, Tuple, Optional

from .audio_loader import AudioData


# Musical key mappings
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Kessler key profiles for major and minor keys
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


class TempoDetector:
    """Detects tempo (BPM) from audio data."""

    def __init__(self, hop_length: int = 512):
        self.hop_length = hop_length

    def detect_tempo(self, audio: AudioData) -> Dict[str, Any]:
        """
        Detect the tempo of the audio.

        Args:
            audio: AudioData object.

        Returns:
            Dictionary with tempo information.
        """
        # Use librosa's beat tracker
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio.y, sr=audio.sr, hop_length=self.hop_length
        )

        # Convert beat frames to timestamps
        beat_times = librosa.frames_to_time(
            beat_frames, sr=audio.sr, hop_length=self.hop_length
        )

        # Get onset envelope for visualization
        onset_env = librosa.onset.onset_strength(
            y=audio.y, sr=audio.sr, hop_length=self.hop_length
        )

        # Compute tempo via autocorrelation for alternative estimate
        tempo_ac = librosa.feature.tempo(
            onset_envelope=onset_env, sr=audio.sr, hop_length=self.hop_length
        )

        # Handle both scalar and array returns from librosa
        primary_tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        alt_tempo = float(tempo_ac[0]) if len(tempo_ac) > 0 else primary_tempo

        # Compute beat regularity (how consistent the beats are)
        if len(beat_times) > 1:
            intervals = np.diff(beat_times)
            regularity = 1.0 - (float(np.std(intervals)) / float(np.mean(intervals)))
            regularity = max(0.0, min(1.0, regularity))
        else:
            regularity = 0.0

        # Determine tempo confidence
        if abs(primary_tempo - alt_tempo) < 5:
            confidence = "High"
        elif abs(primary_tempo - alt_tempo) < 15:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "bpm": round(primary_tempo, 1),
            "bpm_alternative": round(alt_tempo, 1),
            "beat_count": len(beat_times),
            "beat_times": beat_times.tolist(),
            "beat_regularity": round(regularity, 3),
            "confidence": confidence,
            "onset_envelope": onset_env.tolist(),
        }


class KeyDetector:
    """Detects the musical key of audio using chroma analysis."""

    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def detect_key(self, audio: AudioData) -> Dict[str, Any]:
        """
        Detect the musical key using the Krumhansl-Schmuckler algorithm.

        Args:
            audio: AudioData object.

        Returns:
            Dictionary with key detection results.
        """
        # Compute chroma features
        chroma = librosa.feature.chroma_cqt(
            y=audio.y, sr=audio.sr, hop_length=self.hop_length
        )

        # Get mean chroma vector
        chroma_mean = np.mean(chroma, axis=1)

        # Correlate with all major and minor key profiles
        major_correlations = []
        minor_correlations = []

        for i in range(12):
            # Rotate the profile to match each root note
            major_rotated = np.roll(MAJOR_PROFILE, i)
            minor_rotated = np.roll(MINOR_PROFILE, i)

            major_corr = float(np.corrcoef(chroma_mean, major_rotated)[0, 1])
            minor_corr = float(np.corrcoef(chroma_mean, minor_rotated)[0, 1])

            major_correlations.append(major_corr)
            minor_correlations.append(minor_corr)

        # Find best match
        best_major_idx = int(np.argmax(major_correlations))
        best_minor_idx = int(np.argmax(minor_correlations))

        best_major_corr = major_correlations[best_major_idx]
        best_minor_corr = minor_correlations[best_minor_idx]

        # Determine if major or minor
        if best_major_corr > best_minor_corr:
            detected_key = PITCH_CLASSES[best_major_idx]
            detected_mode = "Major"
            correlation = best_major_corr
        else:
            detected_key = PITCH_CLASSES[best_minor_idx]
            detected_mode = "Minor"
            correlation = best_minor_corr

        # Determine confidence
        if correlation > 0.8:
            confidence = "High"
        elif correlation > 0.5:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Build all key correlations for visualization
        all_correlations = {}
        for i in range(12):
            all_correlations[f"{PITCH_CLASSES[i]} Major"] = round(major_correlations[i], 4)
            all_correlations[f"{PITCH_CLASSES[i]} Minor"] = round(minor_correlations[i], 4)

        return {
            "key": detected_key,
            "mode": detected_mode,
            "key_name": f"{detected_key} {detected_mode}",
            "correlation": round(correlation, 4),
            "confidence": confidence,
            "chroma_profile": {
                PITCH_CLASSES[i]: round(float(chroma_mean[i]), 4) for i in range(12)
            },
            "all_correlations": all_correlations,
        }


def analyze_rhythm_and_key(audio: AudioData) -> Dict[str, Any]:
    """
    Convenience function to run both tempo and key detection.

    Args:
        audio: AudioData object.

    Returns:
        Combined dictionary with tempo and key results.
    """
    tempo_detector = TempoDetector()
    key_detector = KeyDetector()

    tempo_result = tempo_detector.detect_tempo(audio)
    key_result = key_detector.detect_key(audio)

    return {
        "tempo": tempo_result,
        "key": key_result,
    }
