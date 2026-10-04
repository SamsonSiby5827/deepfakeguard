from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


VALID_LABELS = ("real", "fake")
VALID_SPLITS = ("train", "val", "test")


@dataclass
class VideoRecord:
    dataset: str
    source_group: str
    label: str
    filename: str
    extension: str
    absolute_path: str
    split: str = ""

    def to_row(self) -> Dict[str, str]:
        return {
            "dataset": self.dataset,
            "source_group": self.source_group,
            "label": self.label,
            "filename": self.filename,
            "extension": self.extension,
            "absolute_path": self.absolute_path,
            "split": self.split,
        }


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_video_files(folder: Path, allowed_exts: set[str]) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_exts
    ]
    return sorted(files, key=lambda p: p.name.lower())


def build_records_for_folders(
    dataset_name: str,
    source_root: Path,
    folder_names: List[str],
    label: str,
    allowed_exts: set[str],
) -> List[VideoRecord]:
    records: List[VideoRecord] = []

    for folder_name in folder_names:
        folder_path = source_root / folder_name
        files = read_video_files(folder_path, allowed_exts)

        for file_path in files:
            records.append(
                VideoRecord(
                    dataset=dataset_name,
                    source_group=folder_name,
                    label=label,
                    filename=file_path.name,
                    extension=file_path.suffix.lower(),
                    absolute_path=str(file_path.resolve()),
                )
            )

    return records


def assert_unique_paths(records: List[VideoRecord]) -> None:
    seen = set()
    duplicates = []

    for record in records:
        if record.absolute_path in seen:
            duplicates.append(record.absolute_path)
        seen.add(record.absolute_path)

    if duplicates:
        preview = duplicates[:10]
        raise ValueError(f"Duplicate file paths found: {preview}")


def stratified_split(
    records: List[VideoRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> List[VideoRecord]:
    if not records:
        return []

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    by_label: Dict[str, List[VideoRecord]] = {"real": [], "fake": []}
    for record in records:
        if record.label not in by_label:
            raise ValueError(f"Unexpected label: {record.label}")
        by_label[record.label].append(record)

    rng = random.Random(seed)
    split_records: List[VideoRecord] = []

    for label in VALID_LABELS:
        label_records = by_label[label][:]
        rng.shuffle(label_records)

        n = len(label_records)
        train_count = int(round(n * train_ratio))
        val_count = int(round(n * val_ratio))

        if train_count > n:
            train_count = n
        if train_count + val_count > n:
            val_count = n - train_count

        test_count = n - train_count - val_count

        train_items = label_records[:train_count]
        val_items = label_records[train_count:train_count + val_count]
        test_items = label_records[train_count + val_count:]

        if len(train_items) + len(val_items) + len(test_items) != n:
            raise RuntimeError("Split count mismatch during stratified split.")

        for item in train_items:
            item.split = "train"
            split_records.append(item)

        for item in val_items:
            item.split = "val"
            split_records.append(item)

        for item in test_items:
            item.split = "test"
            split_records.append(item)

    return sorted(
        split_records,
        key=lambda r: (r.split, r.label, r.source_group.lower(), r.filename.lower())
    )


def sample_balanced_fake_records(
    real_records: List[VideoRecord],
    fake_records: List[VideoRecord],
    seed: int,
    use_all_real_videos: bool,
    max_real_videos: Optional[int],
    max_fake_videos: Optional[int],
) -> tuple[List[VideoRecord], List[VideoRecord]]:
    rng = random.Random(seed)

    if use_all_real_videos:
        selected_real = real_records[:]
    else:
        if max_real_videos is None:
            raise ValueError("max_real_videos must be set when use_all_real_videos is false.")
        if max_real_videos > len(real_records):
            raise ValueError(
                f"Requested {max_real_videos} real videos, but only {len(real_records)} are available."
            )
        selected_real = real_records[:]
        rng.shuffle(selected_real)
        selected_real = selected_real[:max_real_videos]

    target_fake_count = len(selected_real)

    if max_fake_videos is not None:
        target_fake_count = min(target_fake_count, max_fake_videos)

    if target_fake_count > len(fake_records):
        raise ValueError(
            f"Requested {target_fake_count} fake videos, but only {len(fake_records)} are available."
        )

    selected_fake = fake_records[:]
    rng.shuffle(selected_fake)
    selected_fake = selected_fake[:target_fake_count]

    selected_real = sorted(selected_real, key=lambda r: (r.source_group.lower(), r.filename.lower()))
    selected_fake = sorted(selected_fake, key=lambda r: (r.source_group.lower(), r.filename.lower()))

    return selected_real, selected_fake


def write_csv(records: List[VideoRecord], output_csv: Path) -> None:
    ensure_dir(output_csv.parent)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "source_group",
                "label",
                "filename",
                "extension",
                "absolute_path",
                "split",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())


def count_by(records: List[VideoRecord], key_name: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        key = getattr(record, key_name)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def nested_split_label_counts(records: List[VideoRecord]) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}
    for split in VALID_SPLITS:
        result[split] = {}
        for label in VALID_LABELS:
            result[split][label] = sum(
                1 for r in records if r.split == split and r.label == label
            )
    return result


def build_summary(
    all_real_records: List[VideoRecord],
    all_fake_records: List[VideoRecord],
    selected_real_records: List[VideoRecord],
    selected_fake_records: List[VideoRecord],
    final_records: List[VideoRecord],
    config: Dict,
) -> Dict:
    return {
        "dataset_name": "celeb_df_v2",
        "source_root": config["source_root"],
        "output_root": config["output_root"],
        "random_seed": config["random_seed"],
        "input_counts": {
            "real_total": len(all_real_records),
            "fake_total": len(all_fake_records),
            "real_by_source_group": count_by(all_real_records, "source_group"),
            "fake_by_source_group": count_by(all_fake_records, "source_group"),
        },
        "selected_counts": {
            "real_selected": len(selected_real_records),
            "fake_selected": len(selected_fake_records),
            "real_selected_by_source_group": count_by(selected_real_records, "source_group"),
            "fake_selected_by_source_group": count_by(selected_fake_records, "source_group"),
        },
        "final_split_counts": nested_split_label_counts(final_records),
        "total_selected_records": len(final_records),
        "split_ratios": config["split_ratios"],
    }


def prepare_celebdf_splits(config_path: str | Path) -> Dict:
    config_path = Path(config_path)
    config = load_json(config_path)

    source_root = Path(config["source_root"])
    output_root = Path(config["output_root"])
    manifests_dir = output_root / "manifests"

    ensure_dir(output_root)
    ensure_dir(manifests_dir)

    allowed_exts = {ext.lower() for ext in config["video_extensions"]}
    seed = int(config["random_seed"])

    train_ratio = float(config["split_ratios"]["train"])
    val_ratio = float(config["split_ratios"]["val"])
    test_ratio = float(config["split_ratios"]["test"])

    all_real_records = build_records_for_folders(
        dataset_name="celeb_df_v2",
        source_root=source_root,
        folder_names=config["real_folders"],
        label="real",
        allowed_exts=allowed_exts,
    )

    all_fake_records = build_records_for_folders(
        dataset_name="celeb_df_v2",
        source_root=source_root,
        folder_names=config["fake_folders"],
        label="fake",
        allowed_exts=allowed_exts,
    )

    assert_unique_paths(all_real_records)
    assert_unique_paths(all_fake_records)

    selected_real_records, selected_fake_records = sample_balanced_fake_records(
        real_records=all_real_records,
        fake_records=all_fake_records,
        seed=seed,
        use_all_real_videos=bool(config["use_all_real_videos"]),
        max_real_videos=config["max_real_videos"],
        max_fake_videos=config["max_fake_videos"],
    )

    combined_records = selected_real_records + selected_fake_records
    final_records = stratified_split(
        records=combined_records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    all_selected_csv = manifests_dir / "celeb_df_v2_selected_all.csv"
    train_csv = manifests_dir / "celeb_df_v2_train.csv"
    val_csv = manifests_dir / "celeb_df_v2_val.csv"
    test_csv = manifests_dir / "celeb_df_v2_test.csv"

    write_csv(final_records, all_selected_csv)
    write_csv([r for r in final_records if r.split == "train"], train_csv)
    write_csv([r for r in final_records if r.split == "val"], val_csv)
    write_csv([r for r in final_records if r.split == "test"], test_csv)

    summary = build_summary(
        all_real_records=all_real_records,
        all_fake_records=all_fake_records,
        selected_real_records=selected_real_records,
        selected_fake_records=selected_fake_records,
        final_records=final_records,
        config=config,
    )

    summary_path = manifests_dir / "celeb_df_v2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary