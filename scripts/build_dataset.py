# TODO: loop raw_clips -> pose_extractor -> save .npy + labels.csv
import os
import csv
from src.pose_extractor import extract_keypoints_from_video
from src.utils import save_numpy_array
from src.logger import logging

RAW_CLIPS_DIR = "data/raw_clips"
KEYPOINTS_DIR = "data/keypoints"
LABELS_CSV_PATH = "data/labels.csv"


def build_dataset():
    os.makedirs(KEYPOINTS_DIR, exist_ok=True)
    rows = []

    class_names = sorted(
        d for d in os.listdir(RAW_CLIPS_DIR)
        if os.path.isdir(os.path.join(RAW_CLIPS_DIR, d))
    )
    logging.info(f"Found classes: {class_names}")

    for shot_class in class_names:
        class_dir = os.path.join(RAW_CLIPS_DIR, shot_class)
        clip_files = sorted(f for f in os.listdir(class_dir) if f.endswith(".mp4"))

        for clip_file in clip_files:
            clip_id = os.path.splitext(clip_file)[0]  # e.g. "header_01"
            video_path = os.path.join(class_dir, clip_file)

            try:
                keypoints = extract_keypoints_from_video(video_path)
                save_numpy_array(os.path.join(KEYPOINTS_DIR, f"{clip_id}.npy"), keypoints)
                rows.append({"clip_id": clip_id, "shot_class": shot_class})
                logging.info(f"Processed {clip_id} -> {shot_class}")
            except Exception as e:
                logging.warning(f"Skipped {clip_id} due to error: {e}")

    with open(LABELS_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "shot_class"])
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"Wrote {len(rows)} rows to {LABELS_CSV_PATH}")
    print(f"Done. {len(rows)} clips processed. labels.csv written to {LABELS_CSV_PATH}")


if __name__ == "__main__":
    build_dataset()