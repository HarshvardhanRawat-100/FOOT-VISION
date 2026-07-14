import sys
import numpy as np
import cv2
from ultralytics import YOLO

from src.exception import FootVisionException
from src.logger import logging

_pose_model = None
_detect_model = None

BALL_CLASS_ID = 32  # "sports ball" in COCO


def get_model():
    global _pose_model
    if _pose_model is None:
        _pose_model = YOLO("yolov8n-pose.pt")
    return _pose_model


def get_detect_model():
    global _detect_model
    if _detect_model is None:
        _detect_model = YOLO("yolov8n.pt")  # standard object detector, includes "sports ball"
    return _detect_model


def find_ball_center(frame):
    model = get_detect_model()
    results = model(frame, verbose=False, classes=[BALL_CLASS_ID])
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return None
    # if multiple "ball-like" detections, take the most confident one
    confs = results[0].boxes.conf.cpu().numpy()
    best = boxes[np.argmax(confs)]
    cx = (best[0] + best[2]) / 2
    cy = (best[1] + best[3]) / 2
    return np.array([cx, cy])


def select_main_person(results, ball_center=None):
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return None

    if ball_center is None:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return int(np.argmax(areas))

    # pick the person whose bounding box CENTER is closest to the ball
    centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    dists = np.linalg.norm(centers - ball_center, axis=1)
    return int(np.argmin(dists))


def extract_keypoints_from_video(video_path: str) -> np.ndarray:
    try:
        model = get_model()
        cap = cv2.VideoCapture(video_path)
        all_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ball_center = find_ball_center(frame)

            results = model(frame, verbose=False)
            if len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints.data) == 0:
                all_frames.append(np.zeros((17, 3), dtype=np.float32))
                continue

            person_idx = select_main_person(results, ball_center)
            if person_idx is None:
                all_frames.append(np.zeros((17, 3), dtype=np.float32))
                continue
            kp = results[0].keypoints.data[person_idx].cpu().numpy()
            all_frames.append(kp)

        cap.release()
        logging.info(f"Extracted keypoints for {len(all_frames)} frames from {video_path}")
        return np.stack(all_frames)

    except Exception as e:
        raise FootVisionException(e, sys)


if __name__ == "__main__":
    keypoints = extract_keypoints_from_video("data/raw_clips/header/header_02.mp4")
    print("Shape:", keypoints.shape)
    print("First frame keypoints:\n", keypoints[0])