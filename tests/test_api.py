"""
Automated checks for the DeepfakeGuard API.

These tests use a FAKE model, so they run in seconds and do not need
TensorFlow or the real model weights (which are not in this repository).
They check the API's own logic: file checks, error messages, warnings
and the shape of the JSON answer.

Run from the project root:
    python -m pytest
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main


class FakeModel:
    """Stands in for DeepfakeInference and returns a fixed, chosen answer."""

    def __init__(self, face_detected=True, blur_score=120.0, face_area_ratio=0.2):
        self.face_detected = face_detected
        self.blur_score = blur_score
        self.face_area_ratio = face_area_ratio

    def predict_from_array(self, image_array, input_color="bgr"):
        return {
            "predicted_label": "fake",
            "confidence": np.float32(0.87),
            "probabilities": {"fake": np.float32(0.87), "real": np.float32(0.13)},
            "face_resized_bgr": np.zeros((224, 224, 3), dtype=np.uint8),
            "face_detected": self.face_detected,
            "face_bbox": (np.int32(10), np.int32(20), np.int32(90), np.int32(90)) if self.face_detected else None,
            "original_image_shape": image_array.shape,
            "blur_score": self.blur_score,
            "face_area_ratio": self.face_area_ratio,
        }


def make_png(width=64, height=64):
    ok, buffer = cv2.imencode(".png", np.full((height, width, 3), 127, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def client_with(monkeypatch, model):
    monkeypatch.setattr(api_main, "load_model", lambda: model)
    return TestClient(api_main.app)


def upload(client, data, filename="face.png", content_type="image/png"):
    return client.post("/predict/image", files={"file": (filename, data, content_type)})


def test_health_reports_model_loaded(monkeypatch):
    with client_with(monkeypatch, FakeModel()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_clean_json(monkeypatch):
    with client_with(monkeypatch, FakeModel()) as client:
        response = upload(client, make_png())
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "fake"
    assert body["confidence"] == pytest.approx(0.87, abs=1e-3)
    assert body["face_box"] == [10, 20, 90, 90]
    assert body["warnings"] == []


def test_rejects_non_image_file_type(monkeypatch):
    with client_with(monkeypatch, FakeModel()) as client:
        response = upload(client, b"hello", filename="notes.txt", content_type="text/plain")
    assert response.status_code == 415


def test_rejects_corrupt_image(monkeypatch):
    with client_with(monkeypatch, FakeModel()) as client:
        response = upload(client, b"this is not really a png")
    assert response.status_code == 400


def test_rejects_image_over_size_limit(monkeypatch):
    monkeypatch.setattr(api_main, "MAX_IMAGE_BYTES", 100)
    with client_with(monkeypatch, FakeModel()) as client:
        response = upload(client, make_png(200, 200))
    assert response.status_code == 413


def test_warns_when_no_face_found(monkeypatch):
    with client_with(monkeypatch, FakeModel(face_detected=False, face_area_ratio=0.0)) as client:
        body = upload(client, make_png()).json()
    assert body["face_box"] is None
    assert any("No face found" in w for w in body["warnings"])


def test_warns_when_blurry_and_face_tiny(monkeypatch):
    with client_with(monkeypatch, FakeModel(blur_score=10.0, face_area_ratio=0.01)) as client:
        body = upload(client, make_png()).json()
    assert any("blurry" in w for w in body["warnings"])
    assert any("very small" in w for w in body["warnings"])


def test_returns_503_when_model_failed_to_load(monkeypatch):
    def broken_loader():
        raise FileNotFoundError("Model file not found")

    monkeypatch.setattr(api_main, "load_model", broken_loader)
    with TestClient(api_main.app) as client:
        assert client.get("/health").json()["model_loaded"] is False
        response = upload(client, make_png())
    assert response.status_code == 503
