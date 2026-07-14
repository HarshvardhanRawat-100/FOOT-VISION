# TODO: draw skeleton overlay on frames
import cv2
import numpy as np

SKELETON_CONNECTIONS = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12), (5, 11), (6, 12),
]
LEFT_JOINTS = {5, 7, 9, 11, 13, 15}
CONF_THRESHOLD = 0.5


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    out = frame.copy()

    for i, (x, y, conf) in enumerate(keypoints):
        if conf < CONF_THRESHOLD:
            continue
        color = (0, 255, 0) if i in LEFT_JOINTS else (255, 0, 0)
        cv2.circle(out, (int(x), int(y)), 4, color, -1)

    for a, b in SKELETON_CONNECTIONS:
        if keypoints[a, 2] < CONF_THRESHOLD or keypoints[b, 2] < CONF_THRESHOLD:
            continue
        pt1 = (int(keypoints[a, 0]), int(keypoints[a, 1]))
        pt2 = (int(keypoints[b, 0]), int(keypoints[b, 1]))
        cv2.line(out, pt1, pt2, (0, 0, 255), 2)

    return out

def render_annotated_video(video_path: str, keypoints: np.ndarray, output_path: str = "annotated_output.mp4"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(keypoints):
            break
        annotated = draw_skeleton(frame, keypoints[frame_idx])
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path

#TESTING

if __name__ == "__main__":
    import cv2
    from src.pose_extractor import extract_keypoints_from_video

    video_path = "data/raw_clips/header/header_02.mp4"
    keypoints = extract_keypoints_from_video(video_path)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 47  # the contact frame we found in geometry.py
    for i in range(frame_idx + 1):
        ret, frame = cap.read()
    cap.release()

    annotated = draw_skeleton(frame, keypoints[frame_idx])
    cv2.imwrite("test_skeleton_output.jpg", annotated)
    print("Saved test_skeleton_output.jpg — open it to check the skeleton overlay")