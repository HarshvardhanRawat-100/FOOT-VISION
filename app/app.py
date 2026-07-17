import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import cv2

from src.pose_extractor import extract_keypoints_from_video
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.scorer import score_technique
from src.geometry import knee_angle, torso_lean, detect_contact_frame
from src.renderer import render_annotated_video

pipeline = None


def analyze(video_path):
    global pipeline
    if pipeline is None:
        pipeline = PredictionPipeline()

    keypoints = extract_keypoints_from_video(video_path)
    result = pipeline.predict(keypoints)
    shot_class = result["shot_class"]

    shot_type_for_geometry = "header" if shot_class == "header" else "kick"
    contact_frame_idx = detect_contact_frame(keypoints, shot_type=shot_type_for_geometry)
    contact_kp = keypoints[contact_frame_idx]

    metrics = {
        "knee_bend": knee_angle(contact_kp),
        "torso_lean": torso_lean(contact_kp),
    }

    try:
        scoring = score_technique(shot_class, metrics)
    except Exception:
        scoring = {"overall_score": None, "metrics": {}}

    annotated_video_path = render_annotated_video(video_path, keypoints)

    feedback_lines = [
        f"- {info['feedback']}"
        for info in scoring["metrics"].values()
        if info.get("feedback")
    ] or ["Technique looks solid on the metrics we could measure."]

    summary = (
        f"Predicted shot: {shot_class} ({result['confidence']*100:.1f}% confidence)\n"
        f"Overall technique score: {scoring['overall_score']}/10\n\n"
        + "\n".join(feedback_lines)
    )

    return annotated_video_path, summary


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Video(label="Upload your shot"),
    outputs=[
        gr.Video(label="Skeleton overlay video"),
        gr.Textbox(label="Analysis"),
    ],
    title="FootVision",
    description="Upload a football shot clip (Header , Volley , Instep_drive(Power_Shot),Curl_finesse) to get AI-powered technique feedback.",
)

if __name__ == "__main__":
    demo.launch()