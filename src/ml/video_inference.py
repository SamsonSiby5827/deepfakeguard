from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.ml.deepfake_inference import DeepfakeInference


class VideoDeepfakeAnalyzer:
    """
    Video deepfake analyzer that reuses the existing V2 image model.

    Block 1 improvements:
    - better frame selection
    - blur filtering
    - face-only valid frame analysis
    - improved aggregation based on:
        * average fake probability on valid frames
        * fake frame ratio
    """

    def __init__(
        self,
        inference_engine: DeepfakeInference,
        preview_root: str | Path,
        max_frames: int = 24,
        max_preview_frames: int = 6,
        min_blur_score: float = 35.0,
        min_face_area_ratio: float = 0.02,
        avg_fake_threshold: float = 0.55,
        fake_ratio_threshold: float = 0.60,
    ) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames must be greater than 0.")
        if max_preview_frames <= 0:
            raise ValueError("max_preview_frames must be greater than 0.")

        self.inference_engine = inference_engine
        self.preview_root = Path(preview_root)
        self.max_frames = max_frames
        self.max_preview_frames = max_preview_frames
        self.min_blur_score = min_blur_score
        self.min_face_area_ratio = min_face_area_ratio
        self.avg_fake_threshold = avg_fake_threshold
        self.fake_ratio_threshold = fake_ratio_threshold

        self.preview_root.mkdir(parents=True, exist_ok=True)

    def _compute_sample_indices(self, total_frames: int) -> List[int]:
        """
        Compute evenly spaced frame indices across the full video.
        """
        if total_frames <= 0:
            return []

        sample_count = min(self.max_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, num=sample_count, dtype=int).tolist()

        unique_indices: List[int] = []
        seen = set()
        for idx in indices:
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)

        return unique_indices

    @staticmethod
    def _blur_score(frame_bgr: np.ndarray) -> float:
        """
        Compute blur score using variance of Laplacian.
        Higher = sharper.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _face_area_ratio(face_bbox: Optional[tuple], frame_shape: tuple) -> float:
        """
        Compute face area ratio relative to the original frame area.
        """
        if face_bbox is None:
            return 0.0

        _, _, w, h = face_bbox
        frame_h, frame_w = frame_shape[:2]
        frame_area = float(frame_w * frame_h)

        if frame_area <= 0:
            return 0.0

        return float((w * h) / frame_area)

    def _save_preview_frame(
        self,
        frame_bgr: np.ndarray,
        output_dir: Path,
        frame_index: int,
    ) -> Path:
        """
        Save an analyzed preview frame image to disk.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(output_path), frame_bgr)
        return output_path

    def analyze_video(self, video_path: str | Path) -> Dict[str, Any]:
        """
        Analyze a video by sampling frames and aggregating valid frame-level predictions.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        total_frames_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps_raw = cap.get(cv2.CAP_PROP_FPS)

        total_frames = int(total_frames_raw) if total_frames_raw and total_frames_raw > 0 else 0
        fps = float(fps_raw) if fps_raw and fps_raw > 0 else 0.0
        duration_seconds = (total_frames / fps) if total_frames > 0 and fps > 0 else None

        sample_indices = self._compute_sample_indices(total_frames)
        if not sample_indices:
            cap.release()
            raise ValueError("Unable to determine valid frames for video analysis.")

        preview_dir = self.preview_root / video_path.stem
        preview_dir.mkdir(parents=True, exist_ok=True)

        analyzed_frames: List[Dict[str, Any]] = []
        valid_frames: List[Dict[str, Any]] = []
        preview_frames: List[Dict[str, Any]] = []

        rejected_no_face = 0
        rejected_blurry = 0
        rejected_small_face = 0
        rejected_failed_read = 0
        rejected_other = 0

        for frame_index in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame_bgr = cap.read()

            if not success or frame_bgr is None:
                rejected_failed_read += 1
                continue

            blur_score = self._blur_score(frame_bgr)

            try:
                result = self.inference_engine.predict_from_array(frame_bgr, input_color="bgr")
            except Exception:
                rejected_other += 1
                continue

            fake_prob = float(result["probabilities"].get("fake", 0.0))
            real_prob = float(result["probabilities"].get("real", 1.0 - fake_prob))
            predicted_label = result["predicted_label"]
            face_detected = bool(result.get("face_detected", False))
            face_bbox = result.get("face_bbox")
            face_ratio = self._face_area_ratio(face_bbox, frame_bgr.shape)

            frame_result = {
                "frame_index": frame_index,
                "predicted_label": predicted_label,
                "confidence": float(result["confidence"]),
                "fake_probability": fake_prob,
                "real_probability": real_prob,
                "face_detected": face_detected,
                "face_bbox": face_bbox,
                "face_area_ratio": face_ratio,
                "blur_score": blur_score,
                "is_valid": False,
                "rejection_reason": None,
            }

            analyzed_frames.append(frame_result)

            if not face_detected:
                frame_result["rejection_reason"] = "no_face"
                rejected_no_face += 1
                continue

            if blur_score < self.min_blur_score:
                frame_result["rejection_reason"] = "blurry"
                rejected_blurry += 1
                continue

            if face_ratio < self.min_face_area_ratio:
                frame_result["rejection_reason"] = "small_face"
                rejected_small_face += 1
                continue

            frame_result["is_valid"] = True
            valid_frames.append(frame_result)

            if len(preview_frames) < self.max_preview_frames:
                saved_path = self._save_preview_frame(
                    frame_bgr=result["face_resized_bgr"],
                    output_dir=preview_dir,
                    frame_index=frame_index,
                )
                preview_frames.append(
                    {
                        "frame_index": frame_index,
                        "predicted_label": predicted_label,
                        "confidence": float(result["confidence"]),
                        "fake_probability": fake_prob,
                        "real_probability": real_prob,
                        "saved_path": str(saved_path),
                        "saved_filename": saved_path.name,
                    }
                )

        cap.release()

        if not analyzed_frames:
            raise ValueError("No frames could be analyzed from the uploaded video.")

        if not valid_frames:
            raise ValueError(
                "No valid frames remained after face and quality filtering. "
                "Try a clearer video with a visible face."
            )

        avg_fake_probability = float(np.mean([item["fake_probability"] for item in valid_frames]))
        avg_real_probability = 1.0 - avg_fake_probability

        fake_frame_count = sum(1 for item in valid_frames if item["predicted_label"] == "fake")
        real_frame_count = sum(1 for item in valid_frames if item["predicted_label"] == "real")

        valid_frames_count = len(valid_frames)
        fake_frame_ratio = float(fake_frame_count / valid_frames_count)

        is_fake = (
            avg_fake_probability >= self.avg_fake_threshold
            or fake_frame_ratio >= self.fake_ratio_threshold
        )

        if is_fake:
            final_label = "fake"
            final_confidence = max(avg_fake_probability, fake_frame_ratio)
        else:
            final_label = "real"
            final_confidence = avg_real_probability

        final_confidence = max(0.5, min(float(final_confidence), 1.0))

        return {
            "video_path": str(video_path),
            "filename": video_path.name,
            "total_frames": total_frames,
            "fps": fps,
            "duration_seconds": duration_seconds,
            "sampled_frame_indices": sample_indices,
            "analyzed_frames_count": len(analyzed_frames),
            "valid_frames_count": valid_frames_count,
            "fake_frame_count": fake_frame_count,
            "real_frame_count": real_frame_count,
            "fake_frame_ratio": fake_frame_ratio,
            "average_fake_probability": avg_fake_probability,
            "average_real_probability": avg_real_probability,
            "predicted_label": final_label,
            "confidence": final_confidence,
            "frame_results": analyzed_frames,
            "valid_frame_results": valid_frames,
            "preview_frames": preview_frames,
            "preview_dir": str(preview_dir),
            "rejected_no_face_count": rejected_no_face,
            "rejected_blurry_count": rejected_blurry,
            "rejected_small_face_count": rejected_small_face,
            "rejected_failed_read_count": rejected_failed_read,
            "rejected_other_count": rejected_other,
            "rejected_total_count": (
                rejected_no_face
                + rejected_blurry
                + rejected_small_face
                + rejected_failed_read
                + rejected_other
            ),
        }