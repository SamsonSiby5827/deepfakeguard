"""
DeepfakeGuard API (v2).

A small web service around the existing V2 image model.
Other programs send an image to /predict/image and get a JSON answer back.

Run from the project root:
    python -m uvicorn api.main:app
Then open http://127.0.0.1:8000/docs to try it in the browser.
"""

from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Same quality limits the video analyser already uses.
MIN_BLUR_SCORE = 35.0
MIN_FACE_AREA_RATIO = 0.02


class ImagePrediction(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    face_detected: bool
    face_box: Optional[List[int]] = None
    blur_score: float
    face_area_ratio: float
    warnings: List[str]


def load_model():
    """Load the V2 image model. Imported here so tests can swap in a fake model
    without needing TensorFlow or the real weights."""
    from src.ml.deepfake_inference import DeepfakeInference

    return DeepfakeInference()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the server starts, not on every request.
    try:
        app.state.model = load_model()
        app.state.model_error = None
    except Exception as exc:  # missing weights, bad labels file, etc.
        app.state.model = None
        app.state.model_error = str(exc)
    yield


app = FastAPI(
    title="DeepfakeGuard API",
    description="Checks whether the face in an image looks real or fake.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Quick check that the server is running and the model loaded."""
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
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
