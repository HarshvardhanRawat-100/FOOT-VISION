# TODO: angle calculations (knee, torso, hip rotation)
import numpy as np

LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10


def calculate_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """Angle at point B, given three (x, y) points."""
    BA = A - B
    BC = C - B
    cosine = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def knee_angle(frame_kp: np.ndarray, side: str = "right") -> float:
    hip, knee, ankle = (
        (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
        if side == "right"
        else (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    )
    return calculate_angle(frame_kp[hip, :2], frame_kp[knee, :2], frame_kp[ankle, :2])


def torso_lean(frame_kp: np.ndarray) -> float:
    shoulder_mid = (frame_kp[LEFT_SHOULDER, :2] + frame_kp[RIGHT_SHOULDER, :2]) / 2
    hip_mid = (frame_kp[LEFT_HIP, :2] + frame_kp[RIGHT_HIP, :2]) / 2
    vertical = hip_mid + np.array([0, -1])
    return calculate_angle(shoulder_mid, hip_mid, vertical)


def detect_contact_frame(sequence: np.ndarray, shot_type: str = "kick", kicking_side: str = "right") -> int:
    if shot_type == "header":
        shoulder_mid_y = (sequence[:, LEFT_SHOULDER, 1] + sequence[:, RIGHT_SHOULDER, 1]) / 2
        shoulder_conf = (sequence[:, LEFT_SHOULDER, 2] + sequence[:, RIGHT_SHOULDER, 2]) / 2
        valid = shoulder_conf > 0.5
        if not valid.any():
            return 0
        candidate_y = np.where(valid, shoulder_mid_y, np.inf)
        return int(np.argmin(candidate_y))  # highest jump point = smallest y
    else:
        ankle_idx = RIGHT_ANKLE if kicking_side == "right" else LEFT_ANKLE
        ankle_y = sequence[:, ankle_idx, 1]
        ankle_conf = sequence[:, ankle_idx, 2]
        valid = ankle_conf > 0.5
        if not valid.any():
            return 0
        candidate_y = np.where(valid, ankle_y, -np.inf)
        return int(np.argmax(candidate_y))
    
# TESTING 

if __name__ == "__main__":
    from src.pose_extractor import extract_keypoints_from_video

    keypoints = extract_keypoints_from_video("data/raw_clips/header/header_02.mp4")
    print("Shoulder confidence across all frames:\n", (keypoints[:, 5, 2] + keypoints[:, 6, 2]) / 2)
    contact_frame = detect_contact_frame(keypoints, shot_type="header")
    print("Detected contact frame:", contact_frame)
    print("Knee angle at contact:", knee_angle(keypoints[contact_frame]))
    print("Torso lean at contact:", torso_lean(keypoints[contact_frame]))