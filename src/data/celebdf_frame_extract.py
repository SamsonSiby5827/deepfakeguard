from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


VALID_SPLITS = ("train", "val", "test")
VALID_LABELS = ("real", "fake")


@dataclass
class VideoRow:
    dataset: str
    source_group: str
    label: str
    filename: str
    extension: str
    absolute_path: str
    split: str


@dataclass
class SavedImageRow:
    dataset: str
    source_group: str
    label: str
    split: str
    source_video_filename: str
    source_video_path: str
    frame_index: int
    saved_image_filename: str
    saved_image_path: str
    image_width: int
    image_height: int
    face_x: int
    face_y: int
    face_w: int
    face_h: int


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_video_rows(csv_path: Path) -> List[VideoRow]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing manifest CSV: {csv_path}")

    rows: List[VideoRow] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "dataset",
            "source_group",
            "label",
            "filename",
            "extension",
            "absolute_path",
            "split",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV missing required columns: {csv_path}")

        for row in reader:
            rows.append(
                VideoRow(
                    dataset=row["dataset"],
                    source_group=row["source_group"],
                    label=row["label"],
                    filename=row["filename"],
                    extension=row["extension"],
                    absolute_path=row["absolute_path"],
                    split=row["split"],
                )
            )
    return rows


def load_face_cascade() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        raise FileNotFoundError(f"OpenCV Haar cascade not found: {cascade_path}")

    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError("Failed to load OpenCV Haar cascade classifier.")
    return cascade


def detect_largest_face(
    image_bgr,
    face_cascade: cv2.CascadeClassifier,
    min_w: int,
    min_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_w, min_h),
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return int(x), int(y), int(w), int(h)


def crop_face_with_margin(
    image_bgr,
    face_box: Tuple[int, int, int, int],
    margin_ratio: float,
):
    x, y, w, h = face_box
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image_bgr.shape[1], x + w + margin_x)
    y2 = min(image_bgr.shape[0], y + h + margin_y)

    crop = image_bgr[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2 - x1, y2 - y1)


def save_image_manifest(rows: List[SavedImageRow], output_csv: Path) -> None:
    ensure_dir(output_csv.parent)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "source_group",
                "label",
                "split",
                "source_video_filename",
                "source_video_path",
                "frame_index",
                "saved_image_filename",
                "saved_image_path",
                "image_width",
                "image_height",
                "face_x",
                "face_y",
                "face_w",
                "face_h",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row.dataset,
                    "source_group": row.source_group,
                    "label": row.label,
                    "split": row.split,
                    "source_video_filename": row.source_video_filename,
                    "source_video_path": row.source_video_path,
                    "frame_index": row.frame_index,
                    "saved_image_filename": row.saved_image_filename,
                    "saved_image_path": row.saved_image_path,
                    "image_width": row.image_width,
                    "image_height": row.image_height,
                    "face_x": row.face_x,
                    "face_y": row.face_y,
                    "face_w": row.face_w,
                    "face_h": row.face_h,
                }
            )


def summarize_saved_rows(rows: List[SavedImageRow]) -> Dict:
    summary = {
        "total_images": len(rows),
        "by_split": {},
        "by_label": {},
        "by_split_and_label": {},
        "source_groups": {},
    }

    for split in VALID_SPLITS:
        summary["by_split"][split] = sum(1 for r in rows if r.split == split)

    for label in VALID_LABELS:
        summary["by_label"][label] = sum(1 for r in rows if r.label == label)

    for split in VALID_SPLITS:
        summary["by_split_and_label"][split] = {}
        for label in VALID_LABELS:
            summary["by_split_and_label"][split][label] = sum(
                1 for r in rows if r.split == split and r.label == label
            )

    groups = sorted({r.source_group for r in rows})
    for group in groups:
        summary["source_groups"][group] = sum(1 for r in rows if r.source_group == group)

    return summary


def extract_from_video(
    video_row: VideoRow,
    output_root: Path,
    debug_root: Path,
    face_cascade: cv2.CascadeClassifier,
    sample_every_n_frames: int,
    max_saved_faces_per_video: int,
    face_margin_ratio: float,
    min_face_w: int,
    min_face_h: int,
    output_size: Tuple[int, int],
    save_debug_images: bool,
) -> Tuple[List[SavedImageRow], Dict[str, int]]:
    video_path = Path(video_row.absolute_path)
    if not video_path.exists():
        return [], {
            "videos_processed": 0,
            "videos_failed": 1,
            "frames_checked": 0,
            "frames_with_face": 0,
            "images_saved": 0,
        }

    split_dir = output_root / video_row.split / video_row.label
    ensure_dir(split_dir)

    debug_dir = debug_root / video_row.split / video_row.label / video_path.stem
    if save_debug_images:
        ensure_dir(debug_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], {
            "videos_processed": 0,
            "videos_failed": 1,
            "frames_checked": 0,
            "frames_with_face": 0,
            "images_saved": 0,
        }

    saved_rows: List[SavedImageRow] = []
    frame_index = 0
    checked_frames = 0
    frames_with_face = 0
    saved_count = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_index % sample_every_n_frames != 0:
                frame_index += 1
                continue

            checked_frames += 1

            face_box = detect_largest_face(
                image_bgr=frame_bgr,
                face_cascade=face_cascade,
                min_w=min_face_w,
                min_h=min_face_h,
            )

            if face_box is not None:
                frames_with_face += 1

                face_crop, expanded_box = crop_face_with_margin(
                    image_bgr=frame_bgr,
                    face_box=face_box,
                    margin_ratio=face_margin_ratio,
                )

                resized_face = cv2.resize(face_crop, output_size, interpolation=cv2.INTER_AREA)

                saved_filename = f"{video_path.stem}_frame_{frame_index:06d}.jpg"
                saved_path = split_dir / saved_filename

                write_ok = cv2.imwrite(str(saved_path), resized_face)
                if not write_ok:
                    frame_index += 1
                    continue

                if save_debug_images:
                    debug_path = debug_dir / saved_filename
                    cv2.imwrite(str(debug_path), resized_face)

                x, y, w, h = expanded_box
                saved_rows.append(
                    SavedImageRow(
                        dataset=video_row.dataset,
                        source_group=video_row.source_group,
                        label=video_row.label,
                        split=video_row.split,
                        source_video_filename=video_row.filename,
                        source_video_path=video_row.absolute_path,
                        frame_index=frame_index,
                        saved_image_filename=saved_filename,
                        saved_image_path=str(saved_path.resolve()),
                        image_width=output_size[0],
                        image_height=output_size[1],
                        face_x=x,
                        face_y=y,
                        face_w=w,
                        face_h=h,
                    )
                )

                saved_count += 1
                if saved_count >= max_saved_faces_per_video:
                    break

            frame_index += 1
    finally:
        cap.release()

    return saved_rows, {
        "videos_processed": 1,
        "videos_failed": 0,
        "frames_checked": checked_frames,
        "frames_with_face": frames_with_face,
        "images_saved": saved_count,
    }


def process_split_csv(
    csv_path: Path,
    output_root: Path,
    debug_root: Path,
    face_cascade: cv2.CascadeClassifier,
    sample_every_n_frames: int,
    max_saved_faces_per_video: int,
    face_margin_ratio: float,
    min_face_w: int,
    min_face_h: int,
    output_size: Tuple[int, int],
    save_debug_images: bool,
) -> Tuple[List[SavedImageRow], Dict[str, int]]:
    video_rows = load_video_rows(csv_path)

    all_saved_rows: List[SavedImageRow] = []
    totals = {
        "videos_processed": 0,
        "videos_failed": 0,
        "frames_checked": 0,
        "frames_with_face": 0,
        "images_saved": 0,
    }

    for video_row in video_rows:
        saved_rows, stats = extract_from_video(
            video_row=video_row,
            output_root=output_root,
            debug_root=debug_root,
            face_cascade=face_cascade,
            sample_every_n_frames=sample_every_n_frames,
            max_saved_faces_per_video=max_saved_faces_per_video,
            face_margin_ratio=face_margin_ratio,
            min_face_w=min_face_w,
            min_face_h=min_face_h,
            output_size=output_size,
            save_debug_images=save_debug_images,
        )

        all_saved_rows.extend(saved_rows)

        for key in totals:
            totals[key] += stats[key]

    return all_saved_rows, totals


def extract_celebdf_images(config_path: str | Path) -> Dict:
    config_path = Path(config_path)
    config = load_json(config_path)

    input_manifest_dir = Path(config["input_manifest_dir"])
    output_root = Path(config["output_root"])
    debug_root = Path(config["debug_root"])

    ensure_dir(output_root)
    ensure_dir(debug_root)

    train_csv = input_manifest_dir / "celeb_df_v2_train.csv"
    val_csv = input_manifest_dir / "celeb_df_v2_val.csv"
    test_csv = input_manifest_dir / "celeb_df_v2_test.csv"

    face_cascade = load_face_cascade()

    sample_every_n_frames = int(config["sample_every_n_frames"])
    max_saved_faces_per_video = int(config["max_saved_faces_per_video"])
    face_margin_ratio = float(config["face_margin_ratio"])
    min_face_w = int(config["min_detected_face_width"])
    min_face_h = int(config["min_detected_face_height"])
    output_size = tuple(config["output_size"])
    save_debug_images = bool(config["save_debug_images"])

    all_rows: List[SavedImageRow] = []
    split_stats: Dict[str, Dict[str, int]] = {}

    for split_name, csv_path in (
        ("train", train_csv),
        ("val", val_csv),
        ("test", test_csv),
    ):
        rows, stats = process_split_csv(
            csv_path=csv_path,
            output_root=output_root,
            debug_root=debug_root,
            face_cascade=face_cascade,
            sample_every_n_frames=sample_every_n_frames,
            max_saved_faces_per_video=max_saved_faces_per_video,
            face_margin_ratio=face_margin_ratio,
            min_face_w=min_face_w,
            min_face_h=min_face_h,
            output_size=output_size,
            save_debug_images=save_debug_images,
        )
        all_rows.extend(rows)
        split_stats[split_name] = stats

    manifests_dir = output_root / "manifests"
    ensure_dir(manifests_dir)

    save_image_manifest(all_rows, manifests_dir / "celeb_df_v2_images_all.csv")
    save_image_manifest([r for r in all_rows if r.split == "train"], manifests_dir / "celeb_df_v2_images_train.csv")
    save_image_manifest([r for r in all_rows if r.split == "val"], manifests_dir / "celeb_df_v2_images_val.csv")
    save_image_manifest([r for r in all_rows if r.split == "test"], manifests_dir / "celeb_df_v2_images_test.csv")

    summary = {
        "dataset_name": "celeb_df_v2",
        "input_manifest_dir": str(input_manifest_dir.resolve()),
        "output_root": str(output_root.resolve()),
        "debug_root": str(debug_root.resolve()),
        "sample_every_n_frames": sample_every_n_frames,
        "max_saved_faces_per_video": max_saved_faces_per_video,
        "face_margin_ratio": face_margin_ratio,
        "min_detected_face_width": min_face_w,
        "min_detected_face_height": min_face_h,
        "output_size": list(output_size),
        "save_debug_images": save_debug_images,
        "split_stats": split_stats,
        "image_summary": summarize_saved_rows(all_rows),
    }

    summary_path = manifests_dir / "celeb_df_v2_image_extraction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary