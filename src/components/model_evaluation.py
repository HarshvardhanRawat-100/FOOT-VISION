# TODO: ModelEvaluation class - confusion matrix, report
import os
import sys
from dataclasses import dataclass

import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.exception import FootVisionException
from src.logger import logging
from src.utils import load_numpy_array, save_json
from src.components.model_trainer import ShotClassifierCNN, NUM_CLASSES
from src.components.data_transformation import SHOT_CLASSES


@dataclass
class ModelEvaluationConfig:
    test_array_path: str = os.path.join("artifacts", "data_transformation", "test.npy")
    test_labels_path: str = os.path.join("artifacts", "data_transformation", "test_labels.npy")
    model_path: str = os.path.join("models", "best_model.pth")
    report_path: str = os.path.join("artifacts", "model_evaluation", "report.json")


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig = ModelEvaluationConfig()):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def initiate_model_evaluation(self):
        logging.info("Starting model evaluation")
        try:
            X_test = torch.tensor(load_numpy_array(self.config.test_array_path), dtype=torch.float32)
            y_test = load_numpy_array(self.config.test_labels_path)

            model = ShotClassifierCNN(num_classes=NUM_CLASSES).to(self.device)
            model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
            model.eval()

            with torch.no_grad():
                preds = model(X_test.to(self.device)).argmax(dim=1).cpu().numpy()

            report = classification_report(y_test, preds, target_names=SHOT_CLASSES, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, preds).tolist()

            save_json(self.config.report_path, {"classification_report": report, "confusion_matrix": cm})
            logging.info(f"Evaluation complete. Report saved to {self.config.report_path}")
            print("Confusion matrix (rows=actual, cols=predicted):", SHOT_CLASSES)
            print(cm)
            return report

        except Exception as e:
            raise FootVisionException(e, sys)


if __name__ == "__main__":
    ModelEvaluation().initiate_model_evaluation()