"""
DeepfakeGuard API (v2).

A small web service around the V2 image model.
- GET  /               simple upload page for people
- POST /predict/image  JSON answer for programs
- GET  /health         is the server up and the model loaded?
- GET  /docs           interactive API documentation

Run from the project root:
    python -m uvicorn api.main:app
Then open http://127.0.0.1:8000
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Same quality limits the video analyser already uses.
MIN_BLUR_SCORE = 35.0
MIN_FACE_AREA_RATIO = 0.02

# Where the model comes from: the local models/ folder if it exists,
# otherwise it is downloaded once from Hugging Face.
MODEL_REPO = os.getenv("MODEL_REPO", "Samson5827/deepfakeguard-v2")
MODEL_FILENAME = "saved_model_image_classifier_v2.keras"
LABELS_FILENAME = "labels_v2.json"
LOCAL_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "image_classifier_v2"
STATIC_DIR = Path(__file__).resolve().parent / "static"


class ImagePrediction(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    face_detected: bool
    face_box: Optional[List[int]] = None
    blur_score: float
    face_area_ratio: float
    warnings: List[str]


def resolve_model_files() -> Tuple[Path, Path, str]:
    """Return (model path, labels path, where they came from)."""
    local_model = LOCAL_MODEL_DIR / MODEL_FILENAME
    local_labels = LOCAL_MODEL_DIR / LABELS_FILENAME
    if local_model.exists() and local_labels.exists():
        return local_model, local_labels, "local"

    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    labels_path = hf_hub_download(repo_id=MODEL_REPO, filename=LABELS_FILENAME)
    return Path(model_path), Path(labels_path), f"huggingface:{MODEL_REPO}"


def load_model():
    """Load the V2 image model. Imported here so tests can swap in a fake model
    without needing TensorFlow or the real weights."""
    from src.ml.deepfake_inference import DeepfakeInference

    model_path, labels_path, source = resolve_model_files()
    app.state.model_source = source
    return DeepfakeInference(model_path=str(model_path), labels_path=str(labels_path))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the server starts, not on every request.
    app.state.model_source = None
    try:
        app.state.model = load_model()
        app.state.model_error = None
    except Exception as exc:  # missing weights, no internet, bad labels file, etc.
        app.state.model = None
        app.state.model_error = str(exc)
    yield


app = FastAPI(
    title="DeepfakeGuard API",
    description="Checks whether the face in an image looks real or fake. "
    "Research and non-commercial use only: a probability, not proof.",
    version="2.1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """The upload page people see in the browser."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """Quick check that the server is running and the model loaded."""
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
        "model_source": app.state.model_source,
        "model_error": app.state.model_error,
    }


@app.post("/predict/image", response_model=ImagePrediction)
def predict_image(file: UploadFile = File(...)) -> ImagePrediction:
    """Upload one image (JPG, PNG or WEBP) and get real/fake with a confidence score."""
    if app.state.model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {app.state.model_error}")

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Please upload a JPG, PNG or WEBP image.")

    data = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB.")

    image_bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="This file could not be read as an image.")

    result = app.state.model.predict_from_array(image_bgr, input_color="bgr")

    warnings = []
    if not result["face_detected"]:
        warnings.append("No face found, so the whole image was checked. Result is less reliable.")
    elif result["face_area_ratio"] < MIN_FACE_AREA_RATIO:
        warnings.append("The face is very small in the image. Result is less reliable.")
    if result["blur_score"] < MIN_BLUR_SCORE:
        warnings.append("The image is blurry. Result is less reliable.")

    box = result.get("face_bbox")
    return ImagePrediction(
        label=str(result["predicted_label"]),
        confidence=round(float(result["confidence"]), 4),
        probabilities={str(k): round(float(v), 4) for k, v in result["probabilities"].items()},
        face_detected=bool(result["face_detected"]),
        face_box=[int(v) for v in box] if box is not None else None,
        blur_score=round(float(result["blur_score"]), 2),
        face_area_ratio=round(float(result["face_area_ratio"]), 4),
        warnings=warnings,
    )
