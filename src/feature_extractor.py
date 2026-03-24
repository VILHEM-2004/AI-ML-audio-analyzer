"""
Feature Extraction Module
Extracts musical features from audio data including spectral, tonal, and rhythmic features.
"""

import numpy as np
import librosa
from typing import Dict, Any, Optional

from .audio_loader import AudioData


class FeatureExtractor:
    """Extracts comprehensive audio features for analysis."""

    def __init__(self, n_fft: int = 2048, hop_length: int = 512, n_mfcc: int = 13):
        """
        Args:
            n_fft: FFT window size.
            hop_length: Hop length for STFT.
            n_mfcc: Number of MFCC coefficients.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc

    def extract_all(self, audio: AudioData) -> Dict[str, Any]:
        """
        Extract all available features from the audio.

        Args:
            audio: AudioData object.

        Returns:
            Dictionary containing all extracted features.
        """
        features = {}

        # Spectral features
        features["spectral_centroid"] = self._spectral_centroid(audio)
        features["spectral_bandwidth"] = self._spectral_bandwidth(audio)
        features["spectral_rolloff"] = self._spectral_rolloff(audio)
        features["spectral_contrast"] = self._spectral_contrast(audio)

        # Tonal features
        features["chroma"] = self._chroma_features(audio)
        features["tonnetz"] = self._tonnetz(audio)

        # Rhythm features
        features["zero_crossing_rate"] = self._zero_crossing_rate(audio)
        features["rms_energy"] = self._rms_energy(audio)

        # Cepstral features
        features["mfcc"] = self._mfcc(audio)

        # Compute summary statistics
        features["summary"] = self._compute_summary(features)

        return features

    def _spectral_centroid(self, audio: AudioData) -> np.ndarray:
        """Compute spectral centroid (brightness of sound)."""
        centroid = librosa.feature.spectral_centroid(
            y=audio.y, sr=audio.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        return centroid[0]

    def _spectral_bandwidth(self, audio: AudioData) -> np.ndarray:
        """Compute spectral bandwidth."""
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio.y, sr=audio.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        return bandwidth[0]

    def _spectral_rolloff(self, audio: AudioData) -> np.ndarray:
        """Compute spectral rolloff frequency."""
        rolloff = librosa.feature.spectral_rolloff(
            y=audio.y, sr=audio.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        return rolloff[0]

    def _spectral_contrast(self, audio: AudioData) -> np.ndarray:
        """Compute spectral contrast."""
        contrast = librosa.feature.spectral_contrast(
            y=audio.y, sr=audio.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        return contrast

    def _chroma_features(self, audio: AudioData) -> np.ndarray:
        """Compute chroma features (pitch class distribution)."""
        chroma = librosa.feature.chroma_stft(
            y=audio.y, sr=audio.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        return chroma

    def _tonnetz(self, audio: AudioData) -> np.ndarray:
        """Compute tonal centroid features (tonnetz)."""
        harmonic = librosa.effects.harmonic(audio.y)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=audio.sr)
        return tonnetz

    def _zero_crossing_rate(self, audio: AudioData) -> np.ndarray:
        """Compute zero crossing rate."""
        zcr = librosa.feature.zero_crossing_rate(
            audio.y, frame_length=self.n_fft, hop_length=self.hop_length
        )
        return zcr[0]

    def _rms_energy(self, audio: AudioData) -> np.ndarray:
        """Compute RMS energy."""
        rms = librosa.feature.rms(
            y=audio.y, frame_length=self.n_fft, hop_length=self.hop_length
        )
        return rms[0]

    def _mfcc(self, audio: AudioData) -> np.ndarray:
        """Compute Mel-frequency cepstral coefficients."""
        mfcc = librosa.feature.mfcc(
            y=audio.y, sr=audio.sr, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        return mfcc

    def _compute_summary(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics for all features."""
        summary = {}

        # Scalar summaries for 1D features
        for key in ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
                     "zero_crossing_rate", "rms_energy"]:
            if key in features and isinstance(features[key], np.ndarray):
                arr = features[key]
                summary[key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "median": float(np.median(arr)),
                }

        # Energy level classification
        if "rms_energy" in features:
            mean_rms = float(np.mean(features["rms_energy"]))
            if mean_rms > 0.1:
                summary["energy_level"] = "High"
            elif mean_rms > 0.03:
                summary["energy_level"] = "Medium"
            else:
                summary["energy_level"] = "Low"

        # MFCC summary
        if "mfcc" in features:
            mfcc = features["mfcc"]
            summary["mfcc"] = {
                "mean": [float(x) for x in np.mean(mfcc, axis=1)],
                "std": [float(x) for x in np.std(mfcc, axis=1)],
            }

        # Chroma summary (pitch class distribution)
        if "chroma" in features:
            chroma = features["chroma"]
            pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F',
                             'F#', 'G', 'G#', 'A', 'A#', 'B']
            chroma_means = np.mean(chroma, axis=1)
            summary["chroma_profile"] = {
                pitch_classes[i]: float(chroma_means[i]) for i in range(12)
            }

        return summary

    def get_feature_vector(self, audio: AudioData) -> np.ndarray:
        """
        Get a flat feature vector suitable for ML models.

        Returns:
            1D numpy array of feature values.
        """
        features = self.extract_all(audio)
        vector = []

        # Add mean and std of spectral features
        for key in ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
                     "zero_crossing_rate", "rms_energy"]:
            arr = features[key]
            vector.extend([np.mean(arr), np.std(arr)])

        # Add MFCC means and stds
        mfcc = features["mfcc"]
        vector.extend(np.mean(mfcc, axis=1))
        vector.extend(np.std(mfcc, axis=1))

        # Add chroma means
        chroma = features["chroma"]
        vector.extend(np.mean(chroma, axis=1))

        # Add spectral contrast means
        contrast = features["spectral_contrast"]
        vector.extend(np.mean(contrast, axis=1))

        return np.array(vector, dtype=np.float32)
