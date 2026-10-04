import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = BASE_DIR / "models" / "image_classifier_v2" / "saved_model_image_classifier_v2.keras"
DEFAULT_LABELS_PATH = BASE_DIR / "models" / "image_classifier_v2" / "labels_v2.json"

IMG_SIZE = (224, 224)


class DeepfakeInference:
    """
    Deepfake image inference engine for the V2 face-focused model.

    Supports:
    - prediction from an image file path
    - prediction from an in-memory OpenCV / NumPy image array
    """

    def __init__(
        self,
        model_path: str = None,
        labels_path: str = None,
        img_size: Tuple[int, int] = (224, 224),
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.labels_path = Path(labels_path) if labels_path else DEFAULT_LABELS_PATH
        self.img_size = img_size

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        self.model = keras.models.load_model(self.model_path)
        self.label_info = self._load_labels(self.labels_path)

        self.class_names = self.label_info["class_names"]
        self.index_to_label = {
            int(k): v for k, v in self.label_info["index_to_label"].items()
        }

        if 0 not in self.index_to_label or 1 not in self.index_to_label:
            raise ValueError("labels_v2.json must contain index_to_label entries for 0 and 1.")

        self.face_cascade = self._load_face_cascade()

    @staticmethod
    def _load_labels(labels_path: Path) -> Dict[str, Any]:
        with open(labels_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_face_cascade() -> cv2.CascadeClassifier:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            raise FileNotFoundError(f"OpenCV Haar cascade not found: {cascade_path}")

        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            raise RuntimeError("Failed to load OpenCV Haar cascade classifier.")
        return cascade

    @staticmethod
    def _blur_score(image_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _face_area_ratio(face_bbox: Optional[Tuple[int, int, int, int]], image_shape: Tuple[int, ...]) -> float:
        if face_bbox is None:
            return 0.0

        _, _, w, h = face_bbox
        image_h, image_w = image_shape[:2]
        image_area = float(image_w * image_h)

        if image_area <= 0:
            return 0.0

        return float((w * h) / image_area)

    def _extract_face_crop(
        self,
        image_bgr: np.ndarray,
        margin_ratio: float = 0.18,
    ) -> Tuple[np.ndarray, bool, Optional[Tuple[int, int, int, int]]]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Input image is empty or invalid.")

        if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel BGR image.")

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        if len(faces) == 0:
            return image_bgr.copy(), False, None

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image_bgr.shape[1], x + w + margin_x)
        y2 = min(image_bgr.shape[0], y + h + margin_y)

        face_crop = image_bgr[y1:y2, x1:x2].copy()
        return face_crop, True, (x1, y1, x2 - x1, y2 - y1)

    def _prepare_face_for_model(
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool, Optional[Tuple[int, int, int, int]], Tuple[int, ...], float, float]:
        face_crop_bgr, face_detected, face_bbox = self._extract_face_crop(image_bgr)

        face_resized_bgr = cv2.resize(face_crop_bgr, self.img_size, interpolation=cv2.INTER_AREA)
        face_resized_rgb = cv2.cvtColor(face_resized_bgr, cv2.COLOR_BGR2RGB)

        x = face_resized_rgb.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        original_image_shape = image_bgr.shape
        blur_score = self._blur_score(face_resized_bgr)
        face_area_ratio = self._face_area_ratio(face_bbox, original_image_shape)

        return x, face_resized_bgr, face_detected, face_bbox, original_image_shape, blur_score, face_area_ratio

    def preprocess_image(self, image_path: str) -> Dict[str, Any]:
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Image file not found: {image_path_obj}")

        image_bgr = cv2.imread(str(image_path_obj))
        if image_bgr is None:
            raise ValueError(f"Failed to read image file: {image_path_obj}")

        x, face_resized_bgr, face_detected, face_bbox, original_image_shape, blur_score, face_area_ratio = self._prepare_face_for_model(image_bgr)

        return {
            "input_tensor": x,
            "face_resized_bgr": face_resized_bgr,
            "face_detected": face_detected,
            "face_bbox": face_bbox,
            "original_image_shape": original_image_shape,
            "blur_score": blur_score,
            "face_area_ratio": face_area_ratio,
        }

    def preprocess_array(self, image_array: np.ndarray, input_color: str = "bgr") -> Dict[str, Any]:
        if image_array is None or image_array.size == 0:
            raise ValueError("Input image array is empty or invalid.")

        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError("Input image array must be a 3-channel image.")

        if input_color.lower() == "rgb":
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        elif input_color.lower() == "bgr":
            image_bgr = image_array.copy()
        else:
            raise ValueError("input_color must be either 'bgr' or 'rgb'.")

        x, face_resized_bgr, face_detected, face_bbox, original_image_shape, blur_score, face_area_ratio = self._prepare_face_for_model(image_bgr)

        return {
            "input_tensor": x,
            "face_resized_bgr": face_resized_bgr,
            "face_detected": face_detected,
            "face_bbox": face_bbox,
            "original_image_shape": original_image_shape,
            "blur_score": blur_score,
            "face_area_ratio": face_area_ratio,
        }

    def _predict_from_tensor(
        self,
        x: np.ndarray,
        face_resized_bgr: np.ndarray,
        face_detected: bool,
        face_bbox: Optional[Tuple[int, int, int, int]],
        original_image_shape: Tuple[int, ...],
        blur_score: float,
        face_area_ratio: float,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        prob_class1 = float(self.model.predict(x, verbose=0)[0][0])
        prob_class0 = 1.0 - prob_class1

        if prob_class1 >= 0.5:
            pred_index = 1
            confidence = prob_class1
        else:
            pred_index = 0
            confidence = prob_class0

        predicted_label = self.index_to_label[pred_index]
        probabilities = {
            self.index_to_label[0]: prob_class0,
            self.index_to_label[1]: prob_class1,
        }

        result = {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "face_resized_bgr": face_resized_bgr,
            "face_detected": face_detected,
            "face_bbox": face_bbox,
            "original_image_shape": original_image_shape,
            "blur_score": blur_score,
            "face_area_ratio": face_area_ratio,
        }

        if image_path is not None:
            result["image_path"] = str(image_path)

        return result

    def predict(self, image_path: str) -> Dict[str, Any]:
        processed = self.preprocess_image(image_path)
        return self._predict_from_tensor(
            x=processed["input_tensor"],
            face_resized_bgr=processed["face_resized_bgr"],
            face_detected=processed["face_detected"],
            face_bbox=processed["face_bbox"],
            original_image_shape=processed["original_image_shape"],
            blur_score=processed["blur_score"],
            face_area_ratio=processed["face_area_ratio"],
            image_path=image_path,
        )

    def predict_from_array(self, image_array: np.ndarray, input_color: str = "bgr") -> Dict[str, Any]:
        processed = self.preprocess_array(image_array=image_array, input_color=input_color)
        return self._predict_from_tensor(
            x=processed["input_tensor"],
            face_resized_bgr=processed["face_resized_bgr"],
            face_detected=processed["face_detected"],
            face_bbox=processed["face_bbox"],
            original_image_shape=processed["original_image_shape"],
            blur_score=processed["blur_score"],
            face_area_ratio=processed["face_area_ratio"],
            image_path=None,
        )


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/ml/deepfake_inference.py path/to/image.jpg")
        raise SystemExit(1)

    image_path = sys.argv[1]

    try:
        inference = DeepfakeInference(img_size=IMG_SIZE)
        result = inference.predict(image_path)

        print("\nDeepfake Inference Result")
        print("-------------------------")
        print(f"Image: {result.get('image_path', image_path)}")
        print(f"Predicted label: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Face detected: {result['face_detected']}")
        print(f"Face bbox: {result['face_bbox']}")
        print(f"Blur score: {result['blur_score']:.2f}")
        print(f"Face area ratio: {result['face_area_ratio']:.4f}")

        print("\nClass probabilities:")
        for label, prob in result["probabilities"].items():
            print(f"  {label}: {prob:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()