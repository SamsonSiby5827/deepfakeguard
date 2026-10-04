from pathlib import Path
import shutil
import uuid
from datetime import datetime, timezone

import cv2
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.ml.deepfake_inference import DeepfakeInference
from src.ml.video_inference import VideoDeepfakeAnalyzer
from src.utils.hashing import compute_sha256
from src.blockchain.web3_client import BlockchainClient


BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
VIDEO_UPLOAD_FOLDER = BASE_DIR / "static" / "videos"
DEBUG_FACE_FOLDER = BASE_DIR / "static" / "debug_faces"
VIDEO_FRAME_FOLDER = BASE_DIR / "static" / "video_frames"

CLASSIFIED_FOLDER = BASE_DIR / "static" / "classified"
CLASSIFIED_FAKE_FOLDER = CLASSIFIED_FOLDER / "fake"
CLASSIFIED_REAL_FOLDER = CLASSIFIED_FOLDER / "real"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

MODEL_PATH = BASE_DIR / "models" / "image_classifier_v2" / "saved_model_image_classifier_v2.keras"
LABELS_PATH = BASE_DIR / "models" / "image_classifier_v2" / "labels_v2.json"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["VIDEO_UPLOAD_FOLDER"] = str(VIDEO_UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
VIDEO_UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
DEBUG_FACE_FOLDER.mkdir(parents=True, exist_ok=True)
VIDEO_FRAME_FOLDER.mkdir(parents=True, exist_ok=True)
CLASSIFIED_FAKE_FOLDER.mkdir(parents=True, exist_ok=True)
CLASSIFIED_REAL_FOLDER.mkdir(parents=True, exist_ok=True)

inference_engine = DeepfakeInference(
    model_path=str(MODEL_PATH),
    labels_path=str(LABELS_PATH),
    img_size=(224, 224),
)

video_analyzer = VideoDeepfakeAnalyzer(
    inference_engine=inference_engine,
    preview_root=VIDEO_FRAME_FOLDER,
    max_frames=24,
    max_preview_frames=6,
    min_blur_score=35.0,
    min_face_area_ratio=0.02,
    avg_fake_threshold=0.55,
    fake_ratio_threshold=0.60,
)

try:
    blockchain_client = BlockchainClient()
except Exception as e:
    print("Blockchain client startup error:", e)
    blockchain_client = None


# In-memory caches for local demo/prototype
BATCH_VIDEO_CACHE: dict[str, list[dict]] = {}
BATCH_IMAGE_CACHE: dict[str, list[dict]] = {}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_video_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def confidence_level(confidence_percent: float) -> str:
    if confidence_percent >= 90:
        return "Very High"
    if confidence_percent >= 80:
        return "High"
    if confidence_percent >= 70:
        return "Moderate"
    if confidence_percent >= 60:
        return "Low"
    return "Very Low"


def confidence_level_class(confidence_percent: float) -> str:
    if confidence_percent >= 90:
        return "confidence-very-high"
    if confidence_percent >= 80:
        return "confidence-high"
    if confidence_percent >= 70:
        return "confidence-moderate"
    if confidence_percent >= 60:
        return "confidence-low"
    return "confidence-very-low"


def display_video_confidence(confidence_percent: float) -> float:
    return round(min(confidence_percent, 99.0), 2)


def format_timestamp_utc(timestamp: int | None) -> str:
    if not timestamp:
        return "N/A"
    dt = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_duration(duration_seconds: float | None) -> str:
    if duration_seconds is None:
        return "N/A"

    total_seconds = int(round(duration_seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def save_uploaded_file(file_storage, destination_folder: Path) -> tuple[Path, str]:
    original_name = secure_filename(file_storage.filename)
    if not original_name:
        raise ValueError("Invalid uploaded filename.")

    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    file_path = destination_folder / unique_name
    file_storage.save(file_path)
    return file_path, original_name


def summarize_batch_results(results: list[dict]) -> tuple[int, int, int, int, int]:
    total_videos = len(results)
    success_count = sum(1 for item in results if item.get("analysis_success"))
    failed_count = total_videos - success_count
    fake_video_count = sum(
        1 for item in results
        if item.get("analysis_success") and item.get("predicted_label") == "fake"
    )
    real_video_count = sum(
        1 for item in results
        if item.get("analysis_success") and item.get("predicted_label") == "real"
    )
    return total_videos, success_count, failed_count, fake_video_count, real_video_count


def summarize_batch_image_results(results: list[dict]) -> tuple[int, int, int]:
    total_images = len(results)
    fake_image_count = sum(1 for item in results if item.get("predicted_label") == "fake")
    real_image_count = sum(1 for item in results if item.get("predicted_label") == "real")
    return total_images, fake_image_count, real_image_count


def create_batch_cache(results: list[dict]) -> str:
    batch_id = uuid.uuid4().hex
    BATCH_VIDEO_CACHE[batch_id] = results
    return batch_id


def create_image_batch_cache(results: list[dict]) -> str:
    batch_id = uuid.uuid4().hex
    BATCH_IMAGE_CACHE[batch_id] = results
    return batch_id


def render_cached_batch_page(batch_id: str, batch_notice: str | None = None):
    results = BATCH_VIDEO_CACHE.get(batch_id)
    if results is None:
        return render_template("index.html", error="Batch results are no longer available. Please run the analysis again.")

    total_videos, success_count, failed_count, fake_video_count, real_video_count = summarize_batch_results(results)

    return render_template(
        "batch_video_results.html",
        results=results,
        batch_id=batch_id,
        total_videos=total_videos,
        success_count=success_count,
        failed_count=failed_count,
        fake_video_count=fake_video_count,
        real_video_count=real_video_count,
        batch_notice=batch_notice,
    )


def render_cached_image_batch_page(batch_id: str, batch_notice: str | None = None):
    results = BATCH_IMAGE_CACHE.get(batch_id)
    if results is None:
        return render_template("index.html", error="Batch image results are no longer available. Please run the analysis again.")

    total_images, fake_image_count, real_image_count = summarize_batch_image_results(results)

    return render_template(
        "batch_results.html",
        results=results,
        batch_id=batch_id,
        total_images=total_images,
        fake_image_count=fake_image_count,
        real_image_count=real_image_count,
        batch_notice=batch_notice,
    )


def update_cached_batch_video_registration(
    batch_id: str,
    video_internal_filename: str,
    record: dict | None,
    tx_hash: str | None = None,
):
    results = BATCH_VIDEO_CACHE.get(batch_id)
    if not results:
        return

    for item in results:
        if not item.get("analysis_success"):
            continue

        if item.get("video_internal_filename") == video_internal_filename:
            item["blockchain_registered"] = True
            item["blockchain_status"] = "Registered on blockchain"
            item["can_register_video"] = False

            if record:
                item["blockchain_record_formatted"] = {
                    "registered_by": record["registered_by"],
                    "registered_at": format_timestamp_utc(record["timestamp"]),
                    "registered_at_raw": record["timestamp"],
                }
            break


def update_cached_batch_image_registration(
    batch_id: str,
    image_filename: str,
    record: dict | None,
):
    results = BATCH_IMAGE_CACHE.get(batch_id)
    if not results:
        return

    for item in results:
        if item.get("filename") == image_filename:
            item["blockchain_registered"] = True
            item["blockchain_status"] = "Registered on blockchain"
            item["can_register"] = False

            if record:
                item["blockchain_record_formatted"] = {
                    "registered_by": record["registered_by"],
                    "registered_at": format_timestamp_utc(record["timestamp"]),
                    "registered_at_raw": record["timestamp"],
                }
            break


def build_image_quality_context(result: dict) -> dict:
    blur_score = round(float(result.get("blur_score", 0.0)), 2)
    face_area_ratio = float(result.get("face_area_ratio", 0.0))
    face_area_percent = round(face_area_ratio * 100, 2)
    face_detected = bool(result.get("face_detected", False))

    quality_status = "Good quality"
    quality_status_class = "quality-good"
    quality_warning = None
    result_reliability = "Standard"

    if not face_detected:
        quality_status = "No clear face detected"
        quality_status_class = "quality-danger"
        quality_warning = "No clear face was detected in the uploaded image. The prediction was still produced, but it may be less reliable."
        result_reliability = "Low"
    elif face_area_ratio < 0.02:
        quality_status = "Low quality / small face"
        quality_status_class = "quality-warning"
        quality_warning = "The detected face is quite small in the image. The prediction may be less reliable."
        result_reliability = "Caution"
    elif blur_score < 35.0:
        quality_status = "Low quality / blurry image"
        quality_status_class = "quality-warning"
        quality_warning = "The detected face appears blurry. The prediction may be less reliable."
        result_reliability = "Caution"

    return {
        "face_detected": face_detected,
        "blur_score": blur_score,
        "face_area_percent": face_area_percent,
        "quality_status": quality_status,
        "quality_status_class": quality_status_class,
        "quality_warning": quality_warning,
        "result_reliability": result_reliability,
    }


def build_image_result_context(file_path: Path) -> dict:
    result = inference_engine.predict(str(file_path))

    predicted_label = result["predicted_label"].lower()
    confidence = round(result["confidence"] * 100, 2)
    confidence_band = confidence_level(confidence)
    confidence_band_class = confidence_level_class(confidence)

    probabilities = {
        label: round(prob * 100, 2)
        for label, prob in result["probabilities"].items()
    }

    file_hash = compute_sha256(str(file_path))
    image_url = f"static/uploads/{file_path.name}"

    debug_face_filename = f"debug_face_{file_path.stem}.jpg"
    debug_face_path = DEBUG_FACE_FOLDER / debug_face_filename

    face_resized_bgr = result["face_resized_bgr"]
    cv2.imwrite(str(debug_face_path), face_resized_bgr)

    debug_face_url = f"static/debug_faces/{debug_face_filename}"

    if predicted_label == "fake":
        classified_destination = CLASSIFIED_FAKE_FOLDER / file_path.name
    else:
        classified_destination = CLASSIFIED_REAL_FOLDER / file_path.name

    shutil.copy2(file_path, classified_destination)

    blockchain_status = "Blockchain not configured"
    blockchain_registered = False
    blockchain_record = None
    blockchain_error = None
    blockchain_record_formatted = None

    if blockchain_client and blockchain_client.is_configured:
        try:
            blockchain_registered = blockchain_client.is_hash_registered(file_hash)

            if blockchain_registered:
                blockchain_record = blockchain_client.get_file_record(file_hash)
                blockchain_status = "Registered on blockchain"
                blockchain_record_formatted = {
                    "registered_by": blockchain_record["registered_by"],
                    "registered_at": format_timestamp_utc(blockchain_record["timestamp"]),
                    "registered_at_raw": blockchain_record["timestamp"],
                }
            else:
                blockchain_status = "Not registered on blockchain"

        except Exception as e:
            blockchain_status = "Blockchain check failed"
            blockchain_error = str(e)

    quality_context = build_image_quality_context(result)

    return {
        "filename": file_path.name,
        "original_filename": file_path.name.split("_", 1)[1] if "_" in file_path.name else file_path.name,
        "image_url": image_url,
        "debug_face_url": debug_face_url,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "confidence_band": confidence_band,
        "confidence_band_class": confidence_band_class,
        "probabilities": probabilities,
        "file_hash": file_hash,
        "blockchain_status": blockchain_status,
        "blockchain_registered": blockchain_registered,
        "blockchain_record": blockchain_record,
        "blockchain_record_formatted": blockchain_record_formatted,
        "blockchain_error": blockchain_error,
        "can_register": bool(blockchain_client and blockchain_client.is_configured and not blockchain_registered),
        "tx_hash": None,
        **quality_context,
    }


def build_video_result_context(video_path: Path, original_filename: str) -> dict:
    analysis = video_analyzer.analyze_video(video_path)

    raw_confidence_percent = round(analysis["confidence"] * 100, 2)
    display_confidence_percent = display_video_confidence(raw_confidence_percent)

    avg_fake_percent = round(analysis["average_fake_probability"] * 100, 2)
    avg_real_percent = round(analysis["average_real_probability"] * 100, 2)
    fake_ratio_percent = round(analysis["fake_frame_ratio"] * 100, 2)

    predicted_label = analysis["predicted_label"].lower()
    confidence_band = confidence_level(raw_confidence_percent)
    confidence_band_class = confidence_level_class(raw_confidence_percent)

    preview_frames = []
    for item in analysis["preview_frames"]:
        saved_path = Path(item["saved_path"])
        preview_frames.append(
            {
                "frame_index": item["frame_index"],
                "predicted_label": item["predicted_label"].lower(),
                "confidence_percent": display_video_confidence(round(item["confidence"] * 100, 2)),
                "fake_probability_percent": round(item["fake_probability"] * 100, 2),
                "real_probability_percent": round(item["real_probability"] * 100, 2),
                "url": f"static/video_frames/{video_path.stem}/{saved_path.name}",
            }
        )

    video_hash = compute_sha256(str(video_path))

    blockchain_status = "Blockchain not configured"
    blockchain_registered = False
    blockchain_record = None
    blockchain_error = None
    blockchain_record_formatted = None

    if blockchain_client and blockchain_client.is_configured:
        try:
            blockchain_registered = blockchain_client.is_hash_registered(video_hash)

            if blockchain_registered:
                blockchain_record = blockchain_client.get_file_record(video_hash)
                blockchain_status = "Registered on blockchain"
                blockchain_record_formatted = {
                    "registered_by": blockchain_record["registered_by"],
                    "registered_at": format_timestamp_utc(blockchain_record["timestamp"]),
                    "registered_at_raw": blockchain_record["timestamp"],
                }
            else:
                blockchain_status = "Not registered on blockchain"

        except Exception as e:
            blockchain_status = "Blockchain check failed"
            blockchain_error = str(e)

    return {
        "video_internal_filename": video_path.name,
        "video_original_filename": original_filename,
        "video_url": f"static/videos/{video_path.name}",
        "video_hash": video_hash,
        "predicted_label": predicted_label,
        "confidence": display_video_confidence(raw_confidence_percent),
        "raw_confidence": raw_confidence_percent,
        "confidence_band": confidence_band,
        "confidence_band_class": confidence_band_class,
        "average_fake_probability": avg_fake_percent,
        "average_real_probability": avg_real_percent,
        "fake_frame_ratio_percent": fake_ratio_percent,
        "total_frames": analysis["total_frames"],
        "fps": round(analysis["fps"], 2) if analysis["fps"] else "N/A",
        "duration": format_duration(analysis["duration_seconds"]),
        "analyzed_frames_count": analysis["analyzed_frames_count"],
        "valid_frames_count": analysis["valid_frames_count"],
        "fake_frame_count": analysis["fake_frame_count"],
        "real_frame_count": analysis["real_frame_count"],
        "rejected_total_count": analysis["rejected_total_count"],
        "rejected_no_face_count": analysis["rejected_no_face_count"],
        "rejected_blurry_count": analysis["rejected_blurry_count"],
        "rejected_small_face_count": analysis["rejected_small_face_count"],
        "rejected_failed_read_count": analysis["rejected_failed_read_count"],
        "rejected_other_count": analysis["rejected_other_count"],
        "preview_frames": preview_frames,
        "blockchain_status": blockchain_status,
        "blockchain_registered": blockchain_registered,
        "blockchain_record": blockchain_record,
        "blockchain_record_formatted": blockchain_record_formatted,
        "blockchain_error": blockchain_error,
        "can_register_video": bool(blockchain_client and blockchain_client.is_configured and not blockchain_registered),
        "tx_hash": None,
    }


def build_batch_video_card_context(video_path: Path, original_filename: str) -> dict:
    full_context = build_video_result_context(video_path, original_filename)

    return {
        "analysis_success": True,
        "video_internal_filename": full_context["video_internal_filename"],
        "video_original_filename": full_context["video_original_filename"],
        "predicted_label": full_context["predicted_label"],
        "confidence": full_context["confidence"],
        "raw_confidence": full_context["raw_confidence"],
        "confidence_band": full_context["confidence_band"],
        "confidence_band_class": full_context["confidence_band_class"],
        "average_fake_probability": full_context["average_fake_probability"],
        "average_real_probability": full_context["average_real_probability"],
        "fake_frame_ratio_percent": full_context["fake_frame_ratio_percent"],
        "duration": full_context["duration"],
        "fps": full_context["fps"],
        "total_frames": full_context["total_frames"],
        "analyzed_frames_count": full_context["analyzed_frames_count"],
        "valid_frames_count": full_context["valid_frames_count"],
        "fake_frame_count": full_context["fake_frame_count"],
        "real_frame_count": full_context["real_frame_count"],
        "rejected_total_count": full_context["rejected_total_count"],
        "video_hash": full_context["video_hash"],
        "blockchain_status": full_context["blockchain_status"],
        "blockchain_registered": full_context["blockchain_registered"],
        "blockchain_record_formatted": full_context["blockchain_record_formatted"],
        "blockchain_error": full_context["blockchain_error"],
        "can_register_video": full_context["can_register_video"],
        "preview_frames": full_context["preview_frames"][:3],
    }


def build_failed_batch_video_card_context(video_path: Path, original_filename: str, error_message: str) -> dict:
    video_hash = None

    try:
        video_hash = compute_sha256(str(video_path))
    except Exception:
        video_hash = None

    return {
        "analysis_success": False,
        "video_internal_filename": video_path.name,
        "video_original_filename": original_filename,
        "error_message": error_message,
        "video_hash": video_hash,
        "blockchain_status": "Analysis failed",
        "blockchain_registered": False,
        "blockchain_record_formatted": None,
        "blockchain_error": None,
        "can_register_video": False,
        "preview_frames": [],
    }


def build_batch_video_results_from_saved_items(saved_video_items: list[tuple[Path, str]]) -> list[dict]:
    batch_results = []

    for saved_video_path, original_filename in saved_video_items:
        try:
            card = build_batch_video_card_context(saved_video_path, original_filename)
            batch_results.append(card)
        except Exception as e:
            batch_results.append(
                build_failed_batch_video_card_context(
                    saved_video_path,
                    original_filename,
                    str(e),
                )
            )

    return batch_results


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_files = request.files.getlist("image")

    valid_files = [
        f for f in uploaded_files
        if f and f.filename and allowed_file(f.filename)
    ]

    if not uploaded_files or all(not f.filename for f in uploaded_files):
        return render_template("index.html", error="Please choose at least one image file.")

    if not valid_files:
        return render_template("index.html", error="Only PNG, JPG, and JPEG files are allowed for image analysis.")

    try:
        saved_paths = [save_uploaded_file(f, UPLOAD_FOLDER)[0] for f in valid_files]
        results = [build_image_result_context(path) for path in saved_paths]

        if len(results) == 1:
            return render_template(
                "result.html",
                **results[0],
                has_batch_context=False,
                batch_id="",
                batch_needs_refresh=False,
            )

        batch_id = create_image_batch_cache(results)
        return render_cached_image_batch_page(batch_id)

    except Exception as e:
        return render_template("index.html", error=f"Image prediction failed: {e}")


@app.route("/predict_video", methods=["POST"])
def predict_video():
    uploaded_videos = request.files.getlist("video")

    valid_videos = [
        f for f in uploaded_videos
        if f and f.filename and allowed_video_file(f.filename)
    ]

    if not uploaded_videos or all(not f.filename for f in uploaded_videos):
        return render_template("index.html", error="Please choose at least one video file.")

    if not valid_videos:
        return render_template(
            "index.html",
            error="Only MP4, AVI, MOV, MKV, and WEBM files are allowed for video analysis.",
        )

    try:
        saved_video_items = [save_uploaded_file(f, VIDEO_UPLOAD_FOLDER) for f in valid_videos]

        if len(saved_video_items) == 1:
            saved_video_path, original_filename = saved_video_items[0]
            try:
                context = build_video_result_context(saved_video_path, original_filename)
                return render_template(
                    "video_result.html",
                    **context,
                    has_batch_context=False,
                    batch_id="",
                    batch_needs_refresh=False,
                )
            except Exception as e:
                return render_template("index.html", error=f"Video analysis failed: {e}")

        batch_results = build_batch_video_results_from_saved_items(saved_video_items)
        batch_id = create_batch_cache(batch_results)

        return render_cached_batch_page(batch_id)

    except Exception as e:
        return render_template("index.html", error=f"Video analysis failed: {e}")


@app.route("/view_image_result", methods=["POST"])
def view_image_result():
    filename = request.form.get("filename", "").strip()
    batch_id = request.form.get("batch_id", "").strip()

    if not filename:
        return render_template("index.html", error="Missing image filename.")

    safe_filename = Path(filename).name
    file_path = UPLOAD_FOLDER / safe_filename

    if not file_path.exists():
        return render_template("index.html", error="Image file not found.")

    try:
        context = build_image_result_context(file_path)
        return render_template(
            "result.html",
            **context,
            has_batch_context=bool(batch_id),
            batch_id=batch_id,
            batch_needs_refresh=False,
        )
    except Exception as e:
        return render_template("index.html", error=f"Unable to load image result: {e}")


@app.route("/return_to_image_batch_results", methods=["POST"])
def return_to_image_batch_results():
    batch_id = request.form.get("batch_id", "").strip()
    if not batch_id:
        return render_template("index.html", error="Image batch context not found.")

    return render_cached_image_batch_page(
        batch_id,
        batch_notice="Batch results refreshed with latest blockchain status.",
    )


@app.route("/view_video_result", methods=["POST"])
def view_video_result():
    filename = request.form.get("filename", "").strip()
    original_filename = request.form.get("original_filename", "").strip()
    batch_id = request.form.get("batch_id", "").strip()

    if not filename:
        return render_template("index.html", error="Missing video filename.")

    safe_filename = Path(filename).name
    video_path = VIDEO_UPLOAD_FOLDER / safe_filename

    if not video_path.exists():
        return render_template("index.html", error="Video file not found.")

    try:
        context = build_video_result_context(video_path, original_filename or safe_filename)
        return render_template(
            "video_result.html",
            **context,
            has_batch_context=bool(batch_id),
            batch_id=batch_id,
            batch_needs_refresh=False,
        )
    except Exception as e:
        return render_template("index.html", error=f"Unable to load video result: {e}")


@app.route("/return_to_batch_results", methods=["POST"])
def return_to_batch_results():
    batch_id = request.form.get("batch_id", "").strip()
    if not batch_id:
        return render_template("index.html", error="Batch context not found.")

    return render_cached_batch_page(
        batch_id,
        batch_notice="Batch results refreshed with latest blockchain status.",
    )


@app.route("/register_hash", methods=["POST"])
def register_hash():
    filename = request.form.get("filename", "").strip()
    batch_id = request.form.get("batch_id", "").strip()
    return_to_batch = request.form.get("return_to_batch", "").strip() == "1"

    if not filename:
        return render_template("index.html", error="Missing filename for blockchain registration.")

    safe_filename = Path(filename).name
    file_path = UPLOAD_FOLDER / safe_filename

    if not file_path.exists():
        return render_template("index.html", error="Uploaded file not found for blockchain registration.")

    if not blockchain_client or not blockchain_client.is_configured:
        return render_template("index.html", error="Blockchain is not configured yet.")

    try:
        context = build_image_result_context(file_path)
        registration_result = blockchain_client.register_file_hash(context["file_hash"])

        if registration_result["success"]:
            context["blockchain_registered"] = True
            context["blockchain_status"] = "Registered on blockchain"
            context["blockchain_record"] = registration_result.get("record")
            context["tx_hash"] = registration_result.get("tx_hash")
            context["can_register"] = False

            record = registration_result.get("record")
            if record:
                formatted_record = {
                    "registered_by": record["registered_by"],
                    "registered_at": format_timestamp_utc(record["timestamp"]),
                    "registered_at_raw": record["timestamp"],
                }
                context["blockchain_record_formatted"] = formatted_record

                if batch_id:
                    update_cached_batch_image_registration(
                        batch_id=batch_id,
                        image_filename=safe_filename,
                        record=record,
                    )
        else:
            refreshed_context = build_image_result_context(file_path)
            context = refreshed_context

        if return_to_batch and batch_id:
            return render_cached_image_batch_page(
                batch_id,
                batch_notice=f'Blockchain registration updated for "{context["original_filename"]}".',
            )

        return render_template(
            "result.html",
            **context,
            has_batch_context=bool(batch_id),
            batch_id=batch_id,
            batch_needs_refresh=bool(batch_id),
        )

    except Exception as e:
        return render_template("index.html", error=f"Blockchain registration failed: {e}")


@app.route("/register_video_hash", methods=["POST"])
def register_video_hash():
    filename = request.form.get("filename", "").strip()
    original_filename = request.form.get("original_filename", "").strip()
    batch_id = request.form.get("batch_id", "").strip()
    return_to_batch = request.form.get("return_to_batch", "").strip() == "1"

    if not filename:
        return render_template("index.html", error="Missing video filename for blockchain registration.")

    safe_filename = Path(filename).name
    video_path = VIDEO_UPLOAD_FOLDER / safe_filename

    if not video_path.exists():
        return render_template("index.html", error="Uploaded video not found for blockchain registration.")

    if not blockchain_client or not blockchain_client.is_configured:
        return render_template("index.html", error="Blockchain is not configured yet.")

    try:
        context = build_video_result_context(video_path, original_filename or safe_filename)
        registration_result = blockchain_client.register_file_hash(context["video_hash"])

        if registration_result["success"]:
            context["blockchain_registered"] = True
            context["blockchain_status"] = "Registered on blockchain"
            context["blockchain_record"] = registration_result.get("record")
            context["tx_hash"] = registration_result.get("tx_hash")
            context["can_register_video"] = False

            record = registration_result.get("record")
            if record:
                formatted_record = {
                    "registered_by": record["registered_by"],
                    "registered_at": format_timestamp_utc(record["timestamp"]),
                    "registered_at_raw": record["timestamp"],
                }
                context["blockchain_record_formatted"] = formatted_record

                if batch_id:
                    update_cached_batch_video_registration(
                        batch_id=batch_id,
                        video_internal_filename=safe_filename,
                        record=record,
                        tx_hash=registration_result.get("tx_hash"),
                    )
        else:
            refreshed_context = build_video_result_context(video_path, original_filename or safe_filename)
            context = refreshed_context

        if return_to_batch and batch_id:
            return render_cached_batch_page(
                batch_id,
                batch_notice=f'Blockchain registration updated for "{original_filename or safe_filename}".',
            )

        return render_template(
            "video_result.html",
            **context,
            has_batch_context=bool(batch_id),
            batch_id=batch_id,
            batch_needs_refresh=bool(batch_id),
        )

    except Exception as e:
        return render_template("index.html", error=f"Video blockchain registration failed: {e}")


if __name__ == "__main__":
    app.run(debug=True)