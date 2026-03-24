"""
Genre Classification Module
Uses machine learning to classify audio into musical genres.
Supports both a simple heuristic classifier and a trained ML model.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
import os
import pickle

from .audio_loader import AudioData
from .feature_extractor import FeatureExtractor

# Genre labels
GENRES = ["classical", "electronic", "hip-hop", "jazz", "pop", "rock", "r&b", "metal"]


class HeuristicGenreClassifier:
    """
    Rule-based genre classifier using audio features.
    This is used when no trained model is available.
    """

    def classify(self, features: Dict[str, Any], tempo_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify genre using heuristic rules based on features.

        Args:
            features: Extracted audio features.
            tempo_info: Tempo detection results.

        Returns:
            Genre classification results.
        """
        summary = features.get("summary", {})
        bpm = tempo_info.get("bpm", 120)
        beat_regularity = tempo_info.get("beat_regularity", 0.5)

        # Get feature values
        spectral_centroid_mean = summary.get("spectral_centroid", {}).get("mean", 2000)
        rms_mean = summary.get("rms_energy", {}).get("mean", 0.05)
        zcr_mean = summary.get("zero_crossing_rate", {}).get("mean", 0.1)
        energy_level = summary.get("energy_level", "Medium")

        # Score each genre based on feature patterns
        scores = {}

        # Classical: lower energy, moderate centroid, lower ZCR
        scores["classical"] = (
            (1.0 if energy_level == "Low" else 0.3 if energy_level == "Medium" else 0.1) * 0.3
            + (1.0 if spectral_centroid_mean < 2000 else 0.4) * 0.3
            + (1.0 if bpm < 120 else 0.5) * 0.2
            + (1.0 if beat_regularity < 0.6 else 0.4) * 0.2
        )

        # Electronic: high energy, high regularity, specific BPM range
        scores["electronic"] = (
            (1.0 if energy_level == "High" else 0.5 if energy_level == "Medium" else 0.2) * 0.25
            + (1.0 if 120 <= bpm <= 150 else 0.5 if 100 <= bpm <= 160 else 0.2) * 0.25
            + (1.0 if beat_regularity > 0.8 else 0.5 if beat_regularity > 0.6 else 0.2) * 0.3
            + (1.0 if spectral_centroid_mean > 3000 else 0.5) * 0.2
        )

        # Hip-hop: moderate energy, specific BPM, higher bass
        scores["hip-hop"] = (
            (1.0 if energy_level in ["Medium", "High"] else 0.3) * 0.25
            + (1.0 if 80 <= bpm <= 110 else 0.4) * 0.3
            + (1.0 if spectral_centroid_mean < 2500 else 0.4) * 0.25
            + (1.0 if beat_regularity > 0.7 else 0.4) * 0.2
        )

        # Jazz: moderate energy, varying tempo, complex rhythm
        scores["jazz"] = (
            (1.0 if energy_level == "Medium" else 0.4) * 0.2
            + (1.0 if 90 <= bpm <= 160 else 0.4) * 0.2
            + (1.0 if beat_regularity < 0.6 else 0.3) * 0.3
            + (1.0 if spectral_centroid_mean > 2000 else 0.5) * 0.3
        )

        # Pop: moderate everything, high regularity
        scores["pop"] = (
            (1.0 if energy_level == "Medium" else 0.5) * 0.25
            + (1.0 if 100 <= bpm <= 135 else 0.4) * 0.25
            + (1.0 if beat_regularity > 0.7 else 0.4) * 0.25
            + (1.0 if 1500 < spectral_centroid_mean < 4000 else 0.4) * 0.25
        )

        # Rock: high energy, high ZCR, moderate-high BPM
        scores["rock"] = (
            (1.0 if energy_level == "High" else 0.4 if energy_level == "Medium" else 0.2) * 0.3
            + (1.0 if 110 <= bpm <= 160 else 0.4) * 0.2
            + (1.0 if zcr_mean > 0.1 else 0.5) * 0.3
            + (1.0 if rms_mean > 0.08 else 0.4) * 0.2
        )

        # R&B: moderate-low energy, slower tempo
        scores["r&b"] = (
            (1.0 if energy_level in ["Low", "Medium"] else 0.3) * 0.3
            + (1.0 if 60 <= bpm <= 100 else 0.4) * 0.3
            + (1.0 if beat_regularity > 0.6 else 0.4) * 0.2
            + (1.0 if spectral_centroid_mean < 3000 else 0.4) * 0.2
        )

        # Metal: very high energy, high ZCR, fast
        scores["metal"] = (
            (1.0 if energy_level == "High" else 0.2) * 0.35
            + (1.0 if bpm > 140 else 0.3) * 0.2
            + (1.0 if zcr_mean > 0.15 else 0.4) * 0.25
            + (1.0 if rms_mean > 0.12 else 0.4) * 0.2
        )

        # Normalize scores to probabilities
        total = sum(scores.values())
        if total > 0:
            probabilities = {k: round(v / total, 4) for k, v in scores.items()}
        else:
            probabilities = {k: round(1.0 / len(scores), 4) for k in scores}

        # Sort by probability
        sorted_genres = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_genre = sorted_genres[0][0]
        top_probability = sorted_genres[0][1]

        # Confidence based on margin
        if len(sorted_genres) > 1:
            margin = sorted_genres[0][1] - sorted_genres[1][1]
            if margin > 0.1:
                confidence = "High"
            elif margin > 0.04:
                confidence = "Medium"
            else:
                confidence = "Low"
        else:
            confidence = "Low"

        return {
            "predicted_genre": top_genre,
            "confidence": confidence,
            "probabilities": dict(sorted_genres),
            "top_3": [
                {"genre": g, "probability": round(p, 4)}
                for g, p in sorted_genres[:3]
            ],
            "method": "heuristic",
        }


class MLGenreClassifier:
    """
    Machine learning-based genre classifier using Random Forest.
    Can be trained on labeled data or loaded from a saved model.
    """

    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False

    def train(self, audio_files: List[AudioData], labels: List[str]):
        """
        Train the classifier on labeled audio data.

        Args:
            audio_files: List of AudioData objects.
            labels: Corresponding genre labels.
        """
        # Extract feature vectors
        feature_vectors = []
        for audio in audio_files:
            vec = self.feature_extractor.get_feature_vector(audio)
            feature_vectors.append(vec)

        X = np.array(feature_vectors)
        y = np.array(labels)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, audio: AudioData) -> Dict[str, Any]:
        """
        Predict genre for a single audio file.

        Args:
            audio: AudioData object.

        Returns:
            Classification results.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Train or load a model first.")

        vec = self.feature_extractor.get_feature_vector(audio)
        X = self.scaler.transform(vec.reshape(1, -1))

        predicted = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        class_labels = self.model.classes_
        probabilities = {
            label: round(float(prob), 4)
            for label, prob in zip(class_labels, proba)
        }

        sorted_genres = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        return {
            "predicted_genre": predicted,
            "confidence": "High" if sorted_genres[0][1] > 0.6 else "Medium" if sorted_genres[0][1] > 0.3 else "Low",
            "probabilities": dict(sorted_genres),
            "top_3": [
                {"genre": g, "probability": round(p, 4)}
                for g, p in sorted_genres[:3]
            ],
            "method": "ml_random_forest",
        }

    def save_model(self, path: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("No trained model to save.")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, path: str):
        """Load a trained model from disk."""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.is_trained = True


def classify_genre(audio: AudioData, features: Dict[str, Any],
                   tempo_info: Dict[str, Any],
                   model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Classify the genre of an audio file.

    Uses ML model if available, falls back to heuristic classifier.

    Args:
        audio: AudioData object.
        features: Extracted features.
        tempo_info: Tempo detection results.
        model_path: Optional path to a trained model.

    Returns:
        Genre classification results.
    """
    # Try ML classifier first
    if model_path and os.path.exists(model_path):
        try:
            ml_classifier = MLGenreClassifier()
            ml_classifier.load_model(model_path)
            return ml_classifier.predict(audio)
        except Exception:
            pass  # Fall back to heuristic

    # Use heuristic classifier
    heuristic = HeuristicGenreClassifier()
    return heuristic.classify(features, tempo_info)
