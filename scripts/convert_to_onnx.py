"""
Convert the DeepfakeGuard V2 Keras model to ONNX so it can run inside a web browser,
then check that both versions give the same answers.

Usage (from the project root):
    python scripts/convert_to_onnx.py
    python scripts/convert_to_onnx.py --model path/to/model.keras --out path/to/model.onnx
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
import onnxruntime as ort
from tensorflow import keras

DEFAULT_MODEL = Path("models/image_classifier_v2/saved_model_image_classifier_v2.keras")
DEFAULT_OUT = Path("models/image_classifier_v2/deepfakeguard_v2.onnx")
TEST_FOLDER = Path("test_images")
MAX_DIFFERENCE = 1e-3  # biggest allowed gap between Keras and ONNX probabilities


AUGMENTATION_TYPES = (
    "RandomFlip", "RandomRotation", "RandomZoom", "RandomTranslation", "RandomContrast",
    "RandomBrightness", "RandomCrop", "RandomHeight", "RandomWidth",
)


def is_augmentation(layer) -> bool:
    """Training-only layers (random flips, rotations, ...) that do nothing at prediction time."""
    if type(layer).__name__ in AUGMENTATION_TYPES or "augmentation" in layer.name:
        return True
    if isinstance(layer, keras.Sequential):
        return len(layer.layers) > 0 and all(is_augmentation(sub) for sub in layer.layers)
    return False


def inference_model(model):
    """Rebuild the model without the data-augmentation layers.

    They are switched off when predicting anyway, but they still leave random-number
    operations in the exported graph, which ONNX cannot run.
    """
    skipped = [layer.name for layer in model.layers if is_augmentation(layer)]
    if not skipped:
        return model
    inputs = keras.Input(shape=model.input_shape[1:], name="input")
    x = inputs
    for layer in model.layers[1:]:
        if is_augmentation(layer):
            continue
        x = layer(x, training=False)
    print(f"Removed training-only layers: {', '.join(skipped)}")
    return keras.Model(inputs, x, name="deepfakeguard_v2_inference")


def load_test_batch(n_random: int = 8) -> np.ndarray:
    """Real face images from test_images/ if available, plus random images."""
    batch = []
    if TEST_FOLDER.exists():
        import cv2

        for path in sorted(TEST_FOLDER.rglob("*.jpg"))[:8]:
            img = cv2.imread(str(path))
            if img is None:
                continue
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            batch.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32))
    rng = np.random.default_rng(0)
    batch.extend(rng.uniform(0, 255, size=(n_random, 224, 224, 3)).astype(np.float32))
    return np.stack(batch)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    model = keras.models.load_model(args.model)
    export_model = inference_model(model)

    # Keras 3 -> TensorFlow SavedModel -> ONNX (the most reliable route)
    with tempfile.TemporaryDirectory() as tmp:
        saved_model_dir = Path(tmp) / "saved_model"
        print("Exporting SavedModel ...")
        export_model.export(str(saved_model_dir), verbose=False)
        print("Converting to ONNX ...")
        result = subprocess.run(
            [sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(saved_model_dir),
             "--output", str(args.out), "--opset", "17"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(result.stderr[-2000:])
            print("FAILED: conversion error (see above).")
            return 1
    size_mb = args.out.stat().st_size / 1e6
    print(f"Saved {args.out} ({size_mb:.1f} MB)")

    print("Checking the ONNX model gives the same answers as Keras ...")
    batch = load_test_batch()
    keras_probs = model.predict(batch, verbose=0).reshape(-1)
    session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_probs = session.run(None, {input_name: batch})[0].reshape(-1)
    biggest_gap = float(np.max(np.abs(keras_probs - onnx_probs)))
    same_labels = bool(np.all((keras_probs >= 0.5) == (onnx_probs >= 0.5)))

    print(f"Images compared: {len(batch)}")
    print(f"Biggest probability difference: {biggest_gap:.6f}")
    print(f"Same real/fake label on every image: {same_labels}")
    if biggest_gap > MAX_DIFFERENCE or not same_labels:
        print("FAILED: the ONNX model does not match closely enough. Do not use it.")
        return 1
    print("PASSED: the ONNX model matches the Keras model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
