<div align="center">

# 🎵 AI Audio Analyzer

**A comprehensive, high-performance tool for extracting deep musical insights from audio files.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![librosa](https://img.shields.io/badge/librosa-0.10+-orange?style=for-the-badge)](https://librosa.org/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)


Upload audio files (WAV, MP3, FLAC) and get an exhaustive analysis including tempo detection, musical key estimation, genre classification, and beautiful visualizations.

</div>

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🥁 **Tempo Detection** | Highly accurate BPM estimation with beat tracking and confidence scoring |
| 🎹 **Key Detection** | Musical key estimation employing the Krumhansl-Schmuckler algorithm |
| 🎸 **Genre Classification** | Robust genre classification mapping songs across 8 distinct categories |
| 📊 **Spectrogram** | Detailed Mel-frequency spectrogram with heatmap visualization |
| 🌊 **Waveform** | Dynamic amplitude waveform display for visual volume mapping |
| 🎼 **Chromagram** | Precise pitch class distribution over time |
| 📈 **Feature Extraction** | Deep analysis yielding Spectral centroid, Bandwidth, MFCC, ZCR, RMS energy, and more |
| ⚡ **RESTful API** | Extremely fast built-in endpoints powered by FastAPI |
| 🎨 **Premium Interface**| A beautifully animated, responsive dark-themed web dashboard |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed on your system:
- **Python** 3.9 or higher
- **pip** (Python package installer)

### Quick Installation

1. **Download the repository:**
   ```bash
   git clone https://github.com/VILHEM-2004/AI-ML-audio-analyzer

   then unzip the file and open that folder and in search bar type cmd and then step 2
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   
   source venv/bin/activate   # On macOS/Linux
   
   venv\Scripts\activate    # On Windows
   ```

3. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Launching the App

Start the RESTful server directly from your terminal:
```bash
python api/server.py
```
*The web dashboard will be instantly available at **http://localhost:8000**.*

---

## 📖 How to Use

### Using the Web Dashboard

1. Launch your browser and navigate to `http://localhost:8000`.
2. Drag and drop any supported audio file (**WAV, MP3, FLAC**) into the upload designated area.
3. Click on the **"Analyze Audio"** button.
4. Dive into the results featuring tempo, key, genre metrics, and extensive visual charts.

### Interacting with the API

The backend offers fully decoupled API endpoints for your own scripts and applications.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/analyze` | `POST` | Execute a complete audio analysis sweeping all features |
| `/api/tempo` | `POST` | Isolate and return only the Tempo (BPM) detection |
| `/api/key` | `POST` | Run isolated Musical key detection |
| `/api/features` | `POST` | Extract numerical audio features only |
| `/api/health` | `GET` | Health check to verify service uptime |
| `/docs` | `GET` | Interactive API documentation (Swagger UI) |

#### Example Usage

**Full Analysis:**
```bash
curl -X POST -F "file=@song.mp3" http://localhost:8000/api/analyze
```

**Tempo Detection Only:**
```bash
curl -X POST -F "file=@song.mp3" http://localhost:8000/api/tempo
```

**(Response Format)**
```json
{
  "file_info": {
    "filename": "song.mp3",
    "duration": 213.45,
    "sample_rate": 22050,
    "channels": 2
  },
  "tempo_summary": {
    "bpm": 128.0,
    "confidence": "High"
  },
  "key": {
    "key_name": "A Minor",
    "confidence": "High"
  }
}
```

---

## 🏗️ Project Architecture

```text
ai-audio-analyzer/
├── README.md                 # Project documentation
├── LICENSE                   # MIT License
├── requirements.txt          # Minimum Python dependencies
│
├── data/                     # Data stores (Audio/Models)
│
├── src/                      # Analysis engine modules
│   ├── audio_loader.py       # Seamless audio loading and streaming operations
│   ├── feature_extractor.py  # Extraction module for acoustic characteristics
│   ├── tempo_detector.py     # BPM scaling and key algorithms
│   ├── genre_classifier.py   # Genre clustering via heuristic checks and ML integrations
│   └── visualizer.py         # Advanced chart image generators
│
├── api/                      # Backend interface
│   └── server.py             # FastAPI entry point
│
└── frontend/                 # Client interface
    ├── index.html            # Main markup
    ├── styles.css            # Dark mode stylesheet and design tokens
    └── app.js                # UI interactivity and State Management
```

---

## 🔬 Extracted Acoustic Features

This analyzer peels back the layers of your audio to expose the following variables:

- **Spectral Centroid:** The center of mass of the spectrum. Measures "Brightness".
- **Spectral Bandwidth:** Extent of the width of the frequency band.
- **Spectral Rolloff:** The exact coordinate where 85% of standard concentrated energy drops off.
- **Spectral Contrast:** Differences between peaks and valleys representing distinct timbres.
- **Chroma Features:** Granular classification highlighting the 12 different pitch classes.
- **MFCC:** Timbral features widely utilized in speech and audio recognition.
- **Zero Crossing Rate:** Rate at which the signal shifts standard signs.
- **RMS Energy:** Fundamental loudness calculation based on the Root Mean Square.
- **Tonnetz:** Distinct tonal centroid mappings.

---

## 👨‍💻 Tech Stack

- **Core & Logic:** `Python 3.9+`, `librosa`, `NumPy`, `SciPy`
- **Machine Learning Integration:** `scikit-learn`
- **Charting & Visuals:** `Matplotlib`
- **Backend & Serving:** `FastAPI`, `Uvicorn`
- **Client Side:** `HTML5`, `CSS3`, `Vanilla JavaScript`

---

## 🔮 Roadmap

- [ ] Interactive Chord progression detection
- [ ] Vocal isolation & multi-stem detachment
- [ ] Neural net powered recommendation integrations
- [ ] Extended TensorFlow / PyTorch modules integration
- [ ] Live microphone and real-time buffer analysis
- [ ] Command Line Interface (CLI) functionality
- [ ] Batch processing directory tools

---

## 📄 License

This code and its resources fall under the standard open-source **[MIT License](LICENSE)**. See the `LICENSE` file for more details.

---
<div align="center">
  <p><strong> Made by VILHEM</strong></p>
</div>
