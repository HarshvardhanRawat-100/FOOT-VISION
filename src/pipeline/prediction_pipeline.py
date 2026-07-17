# TODO: single-clip inference pipeline
import sys

import numpy as np
import torch

from src.exception import FootVisionException
from src.logger import logging
from src.utils import load_object
from src.components.model_trainer import ShotClassifierCNN, NUM_CLASSES
from src.components.data_transformation import normalize_sequence, SHOT_CLASSES

MODEL_PATH = "models/best_model.pth"
PREPROCESSOR_PATH = "artifacts/data_transformation/preprocessor.pkl"


class PredictionPipeline:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = ShotClassifierCNN(num_classes=NUM_CLASSES).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()
        self.preprocessor = load_object(PREPROCESSOR_PATH)

    def predict(self, raw_keypoints: np.ndarray):
        try:
            seq = normalize_sequence(raw_keypoints)
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            return {
                "shot_class": SHOT_CLASSES[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probabilities": dict(zip(SHOT_CLASSES, probs.tolist())),
            }
        except Exception as e:
            raise FootVisionException(e, sys)


if __name__ == "__main__":
    from src.pose_extractor import extract_keypoints_from_video

    keypoints = extract_keypoints_from_video("data/raw_clips/header/header_02.mp4")
    pipeline = PredictionPipeline()
    result = pipeline.predict(keypoints)
    print(result))