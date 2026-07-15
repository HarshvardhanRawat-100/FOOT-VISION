# TODO: DataTransformation class - normalize keypoints, build arrays
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.exception import FootVisionException
from src.logger import logging
from src.utils import save_object, save_numpy_array

SEQ_LEN = 60
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

SHOT_CLASSES = ["header", "Volley", "penalty", "free_kick"]  # matches your current labels.csv exactly


@dataclass
class DataTransformationConfig:
    keypoints_dir: str = os.path.join("data", "keypoints")
    preprocessor_obj_path: str = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")
    train_array_path: str = os.path.join("artifacts", "data_transformation", "train.npy")
    test_array_path: str = os.path.join("artifacts", "data_transformation", "test.npy")
    train_labels_path: str = os.path.join("artifacts", "data_transformation", "train_labels.npy")
    test_labels_path: str = os.path.join("artifacts", "data_transformation", "test_labels.npy")


def normalize_sequence(keypoints: np.ndarray) -> np.ndarray:
    xy = keypoints[:, :, :2].astype(np.float32)

    hip_mid = (xy[:, LEFT_HIP, :] + xy[:, RIGHT_HIP, :]) / 2.0
    shoulder_mid = (xy[:, LEFT_SHOULDER, :] + xy[:, RIGHT_SHOULDER, :]) / 2.0
    torso_height = np.linalg.norm(shoulder_mid - hip_mid, axis=1)
    torso_height[torso_height < 1e-3] = 1e-3

    xy = xy - hip_mid[:, None, :]
    xy = xy / torso_height[:, None, None]

    seq = xy.reshape(xy.shape[0], -1)

    if seq.shape[0] >= SEQ_LEN:
        seq = seq[:SEQ_LEN]
    else:
        pad = np.zeros((SEQ_LEN - seq.shape[0], seq.shape[1]), dtype=np.float32)
        seq = np.vstack([seq, pad])

    return seq


class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config

    def _build_split(self, df: pd.DataFrame):
        sequences, labels = [], []
        for _, row in df.iterrows():
            kp_path = os.path.join(self.config.keypoints_dir, f"{row['clip_id']}.npy")
            if not os.path.exists(kp_path):
                logging.warning(f"Missing keypoints for {row['clip_id']}, skipping")
                continue
            raw = np.load(kp_path)
            sequences.append(normalize_sequence(raw))
            labels.append(SHOT_CLASSES.index(row["shot_class"]))
        return np.stack(sequences), np.array(labels, dtype=np.int64)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        logging.info("Starting data transformation")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train, y_train = self._build_split(train_df)
            X_test, y_test = self._build_split(test_df)

            save_numpy_array(self.config.train_array_path, X_train)
            save_numpy_array(self.config.test_array_path, X_test)
            save_numpy_array(self.config.train_labels_path, y_train)
            save_numpy_array(self.config.test_labels_path, y_test)

            save_object(self.config.preprocessor_obj_path, {"seq_len": SEQ_LEN, "classes": SHOT_CLASSES})

            logging.info(f"Transformation complete. X_train={X_train.shape}, X_test={X_test.shape}")
            return (
                self.config.train_array_path,
                self.config.test_array_path,
                self.config.train_labels_path,
                self.config.test_labels_path,
            )

        except Exception as e:
            raise FootVisionException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    train_path, test_path = DataIngestion().initiate_data_ingestion()
    DataTransformation().initiate_data_transformation(train_path, test_path)