import os
import sys
import cv2
import numpy as np
from src.pose_extractor import extract_keypoints_from_video
from src.renderer import draw_skeleton

def check_clips(class_name):
    raw_clips_dir = f"data/raw_clips/{class_name}"
    output_dir = f"data/clip_check_{class_name}"
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(raw_clips_dir)):
        if not filename.endswith(".mp4"):
            continue

        video_path = os.path.join(raw_clips_dir, filename)
        print(f"Checking {filename}...")

        keypoints = extract_keypoints_from_video(video_path)
        mid_idx = len(keypoints) // 2

        cap = cv2.VideoCapture(video_path)
        for i in range(mid_idx + 1):
            ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"  Could not read frame from {filename}")
            continue

        annotated = draw_skeleton(frame, keypoints[mid_idx])
        out_path = os.path.join(output_dir, f"check_{filename}.jpg")
        cv2.imwrite(out_path, annotated)

    print(f"\nDone. Check images in {output_dir}/")


if __name__ == "__main__":
    class_name = sys.argv[1] if len(sys.argv) > 1 else "header"
    check_clips(class_name)