"""
FastAPI Server for AI Audio Analyzer
Provides REST API endpoints for audio analysis.
"""

import os
import sys
import json
import time
import traceback
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_loader import load_audio_bytes, AudioLoadError, AudioData
from src.feature_extractor import FeatureExtractor
from src.tempo_detector import TempoDetector, KeyDetector, analyze_rhythm_and_key
from src.genre_classifier import classify_genre
from src.visualizer import generate_all_visualizations

# Create FastAPI app
app = FastAPI(
    title="AI Audio Analyzer",
    description="Open-source AI tool for analyzing audio files and extracting musical information.",
    version="1.0.0",
    license_info={"name": "MIT License"},
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Create data directory
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(data_dir, exist_ok=True)

# Initialize components
feature_extractor = FeatureExtractor()
tempo_detector = TempoDetector()
key_detector = KeyDetector()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend dashboard."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>AI Audio Analyzer API</h1><p>Frontend not found. Visit /docs for API documentation.</p>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/api/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    include_visualizations: bool = Query(True, description="Include visualization images"),
    include_features: bool = Query(True, description="Include detailed features"),
    include_genre: bool = Query(True, description="Include genre classification"),
):
    """
    Analyze an uploaded audio file.

    Accepts WAV, MP3, and FLAC files.

    Returns comprehensive analysis including:
    - File metadata
    - Tempo (BPM) detection
    - Musical key detection
    - Audio feature extraction
    - Genre classification
    - Visualizations (waveform, spectrogram, etc.)
    """
    start_time = time.time()

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Read file bytes
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Load audio
    try:
        audio = load_audio_bytes(file_bytes, file.filename)
    except AudioLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio loading error: {str(e)}")

    # Build response
    result = {
        "file_info": audio.to_dict(),
        "analysis_time": 0,
    }

    try:
        # Tempo and key detection
        rhythm_key = analyze_rhythm_and_key(audio)
        result["tempo"] = rhythm_key["tempo"]
        result["key"] = rhythm_key["key"]

        # Remove large arrays from tempo response for JSON
        tempo_response = {k: v for k, v in result["tempo"].items()
                         if k not in ["onset_envelope", "beat_times"]}
        tempo_response["beat_count"] = result["tempo"]["beat_count"]
        result["tempo_summary"] = tempo_response

    except Exception as e:
        result["tempo"] = {"error": str(e)}
        result["key"] = {"error": str(e)}
        result["tempo_summary"] = {"error": str(e)}

    try:
        # Feature extraction
        features = feature_extractor.extract_all(audio)

        if include_features:
            # Send only summary (full arrays are too large for JSON)
            result["features"] = features.get("summary", {})
        else:
            result["features"] = {"note": "Detailed features not requested"}

    except Exception as e:
        features = None
        result["features"] = {"error": str(e)}

    try:
        # Genre classification
        if include_genre and features:
            tempo_info = result.get("tempo", {})
            genre_result = classify_genre(audio, features, tempo_info)
            result["genre"] = genre_result
        else:
            result["genre"] = {"note": "Genre classification not requested"}

    except Exception as e:
        result["genre"] = {"error": str(e)}

    try:
        # Visualizations
        if include_visualizations:
            visualizations = generate_all_visualizations(audio, features)
            result["visualizations"] = visualizations
        else:
            result["visualizations"] = {"note": "Visualizations not requested"}

    except Exception as e:
        result["visualizations"] = {"error": str(e)}

    # Total analysis time
    result["analysis_time"] = round(time.time() - start_time, 2)

    return JSONResponse(content=result)


@app.post("/api/tempo")
async def detect_tempo(file: UploadFile = File(...)):
    """Detect tempo (BPM) only."""
    try:
        file_bytes = await file.read()
        audio = load_audio_bytes(file_bytes, file.filename)
        result = tempo_detector.detect_tempo(audio)
        # Remove large arrays
        return {k: v for k, v in result.items() if k not in ["onset_envelope", "beat_times"]}
    except AudioLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/key")
async def detect_key(file: UploadFile = File(...)):
    """Detect musical key only."""
    try:
        file_bytes = await file.read()
        audio = load_audio_bytes(file_bytes, file.filename)
        result = key_detector.detect_key(audio)
        return result
    except AudioLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/features")
async def extract_features(file: UploadFile = File(...)):
    """Extract audio features only."""
    try:
        file_bytes = await file.read()
        audio = load_audio_bytes(file_bytes, file.filename)
        features = feature_extractor.extract_all(audio)
        return features.get("summary", {})
    except AudioLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
