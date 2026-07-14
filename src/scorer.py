# TODO: rule-based technique scoring against rubrics.json
import json
import sys

from src.exception import FootVisionException
from src.logger import logging

RUBRICS_PATH = "config/rubrics.json"


def _score_metric(value: float, ideal_min: float, ideal_max: float) -> float:
    if ideal_min <= value <= ideal_max:
        return 10.0
    span = ideal_max - ideal_min
    dist = ideal_min - value if value < ideal_min else value - ideal_max
    return max(0.0, 10.0 - (dist / span) * 10.0)


def score_technique(shot_class: str, metrics: dict) -> dict:
    try:
        with open(RUBRICS_PATH) as f:
            rubrics = json.load(f)

        if shot_class not in rubrics:
            raise ValueError(f"No rubric defined for shot class '{shot_class}' yet")

        rubric = rubrics[shot_class]
        results, weighted_total, weight_sum = {}, 0.0, 0.0

        for metric_name, spec in rubric.items():
            if metric_name not in metrics:
                continue
            value = metrics[metric_name]
            score = _score_metric(value, spec["ideal_min"], spec["ideal_max"])
            feedback = spec["low_feedback"] if score < 7 else None
            results[metric_name] = {"value": value, "score": round(score, 1), "feedback": feedback}
            weighted_total += score * spec["weight"]
            weight_sum += spec["weight"]

        overall = round(weighted_total / weight_sum, 1) if weight_sum > 0 else None
        return {"overall_score": overall, "metrics": results}

    except Exception as e:
        raise FootVisionException(e, sys)


if __name__ == "__main__":
    from src.pose_extractor import extract_keypoints_from_video
    from src.geometry import knee_angle, torso_lean, detect_contact_frame

    keypoints = extract_keypoints_from_video("data/raw_clips/header/header_02.mp4")
    contact = detect_contact_frame(keypoints, shot_type="header")
    metrics = {
        "knee_bend": knee_angle(keypoints[contact]),
        "torso_lean": torso_lean(keypoints[contact]),
    }
    result = score_technique("header", metrics)
    print(result)