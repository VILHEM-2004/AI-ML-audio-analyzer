"""
Visualization Module
Generates waveform, spectrogram, and other audio visualizations.
Outputs are returned as base64-encoded PNG images for web display.
"""

import io
import base64
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Any, Optional

from .audio_loader import AudioData


# Color scheme for visualizations
COLORS = {
    "bg": "#0a0a1a",
    "fg": "#e0e0ff",
    "accent1": "#6c63ff",
    "accent2": "#00d4ff",
    "accent3": "#ff6b9d",
    "grid": "#1a1a3a",
    "waveform": "#6c63ff",
    "gradient_start": "#6c63ff",
    "gradient_end": "#00d4ff",
}


def _setup_figure(title: str, figsize: tuple = (14, 5)) -> tuple:
    """Create a styled figure with dark theme."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_title(title, color=COLORS["fg"], fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors=COLORS["fg"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.xaxis.label.set_color(COLORS["fg"])
    ax.yaxis.label.set_color(COLORS["fg"])
    return fig, ax


def _fig_to_base64(fig: Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=120, pad_inches=0.3)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_waveform(audio: AudioData) -> str:
    """
    Generate a waveform visualization.

    Args:
        audio: AudioData object.

    Returns:
        Base64-encoded PNG image string.
    """
    fig, ax = _setup_figure("Waveform", figsize=(14, 4))

    times = np.linspace(0, audio.duration, len(audio.y))

    # Create gradient effect by plotting in segments
    n_segments = 200
    segment_len = len(audio.y) // n_segments

    for i in range(n_segments):
        start = i * segment_len
        end = min((i + 1) * segment_len + 1, len(audio.y))
        ratio = i / n_segments

        # Interpolate colors
        r = int(108 + (0 - 108) * ratio)
        g = int(99 + (212 - 99) * ratio)
        b = int(255 + (255 - 255) * ratio)
        color = f"#{r:02x}{g:02x}{b:02x}"

        ax.fill_between(
            times[start:end], audio.y[start:end], 0,
            alpha=0.6, color=color, linewidth=0
        )
        ax.plot(times[start:end], audio.y[start:end],
                color=color, linewidth=0.3, alpha=0.8)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, audio.duration)
    ax.grid(True, alpha=0.15, color=COLORS["grid"])

    return _fig_to_base64(fig)


def generate_spectrogram(audio: AudioData, n_fft: int = 2048,
                          hop_length: int = 512) -> str:
    """
    Generate a spectrogram visualization.

    Args:
        audio: AudioData object.
        n_fft: FFT window size.
        hop_length: Hop length.

    Returns:
        Base64-encoded PNG image string.
    """
    fig, ax = _setup_figure("Spectrogram", figsize=(14, 5))

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(
        y=audio.y, sr=audio.sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=128
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(
        S_dB, sr=audio.sr, hop_length=hop_length,
        x_axis="time", y_axis="mel", ax=ax,
        cmap="magma"
    )

    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=COLORS["fg"])
    cbar.ax.yaxis.label.set_color(COLORS["fg"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS["fg"])

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")

    return _fig_to_base64(fig)


def generate_chromagram(audio: AudioData, hop_length: int = 512) -> str:
    """
    Generate a chromagram visualization showing pitch class distribution over time.

    Returns:
        Base64-encoded PNG image string.
    """
    fig, ax = _setup_figure("Chromagram", figsize=(14, 4))

    chroma = librosa.feature.chroma_cqt(
        y=audio.y, sr=audio.sr, hop_length=hop_length
    )

    img = librosa.display.specshow(
        chroma, sr=audio.sr, hop_length=hop_length,
        x_axis="time", y_axis="chroma", ax=ax,
        cmap="cool"
    )

    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=COLORS["fg"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS["fg"])

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Pitch Class")

    return _fig_to_base64(fig)


def generate_spectral_features_plot(features: Dict[str, Any],
                                      audio: AudioData) -> str:
    """
    Generate a multi-panel plot of spectral features.

    Returns:
        Base64-encoded PNG image string.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.patch.set_facecolor(COLORS["bg"])

    feature_configs = [
        ("spectral_centroid", "Spectral Centroid (Hz)", COLORS["accent1"]),
        ("spectral_bandwidth", "Spectral Bandwidth (Hz)", COLORS["accent2"]),
        ("rms_energy", "RMS Energy", COLORS["accent3"]),
    ]

    for ax, (key, label, color) in zip(axes, feature_configs):
        ax.set_facecolor(COLORS["bg"])
        data = features.get(key)

        if data is not None:
            frames = np.arange(len(data))
            times = librosa.frames_to_time(
                frames, sr=audio.sr, hop_length=512
            )
            ax.plot(times, data, color=color, linewidth=0.8, alpha=0.9)
            ax.fill_between(times, data, alpha=0.2, color=color)
            ax.set_ylabel(label, color=COLORS["fg"], fontsize=10)

        ax.tick_params(colors=COLORS["fg"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])
        ax.grid(True, alpha=0.15, color=COLORS["grid"])

    axes[0].set_title("Spectral Features", color=COLORS["fg"],
                       fontsize=14, fontweight="bold", pad=15)
    axes[-1].set_xlabel("Time (seconds)", color=COLORS["fg"])

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_mfcc_plot(features: Dict[str, Any], audio: AudioData) -> str:
    """
    Generate MFCC visualization.

    Returns:
        Base64-encoded PNG image string.
    """
    fig, ax = _setup_figure("MFCC (Mel-Frequency Cepstral Coefficients)", figsize=(14, 5))

    mfcc = features.get("mfcc")
    if mfcc is not None:
        img = librosa.display.specshow(
            mfcc, sr=audio.sr, hop_length=512,
            x_axis="time", ax=ax, cmap="viridis"
        )
        cbar = fig.colorbar(img, ax=ax, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color=COLORS["fg"])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS["fg"])

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("MFCC Coefficient")

    return _fig_to_base64(fig)


def generate_all_visualizations(audio: AudioData,
                                 features: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Generate all available visualizations.

    Args:
        audio: AudioData object.
        features: Pre-extracted features (optional, will extract if not provided).

    Returns:
        Dictionary mapping visualization names to base64 PNG strings.
    """
    from .feature_extractor import FeatureExtractor

    if features is None:
        extractor = FeatureExtractor()
        features = extractor.extract_all(audio)

    visualizations = {}

    visualizations["waveform"] = generate_waveform(audio)
    visualizations["spectrogram"] = generate_spectrogram(audio)
    visualizations["chromagram"] = generate_chromagram(audio)
    visualizations["spectral_features"] = generate_spectral_features_plot(features, audio)
    visualizations["mfcc"] = generate_mfcc_plot(features, audio)

    return visualizations
