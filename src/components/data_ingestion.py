# TODO: DataIngestion class - read labels.csv, train/test split
import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import FootVisionException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("data", "labels.csv")
    train_data_path: str = os.path.join("artifacts", "data_ingestion", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data_ingestion", "test.csv")
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            df = pd.read_csv(self.config.raw_data_path)
            required = {"clip_id", "shot_class"}
            if not required.issubset(df.columns):
                raise ValueError(f"labels.csv must contain columns {required}")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=df["shot_class"],
            )

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logging.info(f"Ingestion complete. train={len(train_df)} rows, test={len(test_df)} rows")
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise FootVisionException(e, sys)


if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()