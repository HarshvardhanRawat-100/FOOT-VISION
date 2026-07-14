# TODO: chain ingestion -> transformation -> trainer -> evaluation
import sys

from src.exception import FootVisionException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


def run_training_pipeline():
    try:
        logging.info("=== TRAINING PIPELINE START ===")

        train_csv, test_csv = DataIngestion().initiate_data_ingestion()

        DataTransformation().initiate_data_transformation(train_csv, test_csv)

        model_path, best_acc = ModelTrainer().initiate_model_training()
        logging.info(f"Model trained. best_val_acc={best_acc:.4f}, saved to {model_path}")

        report = ModelEvaluation().initiate_model_evaluation()
        logging.info("=== TRAINING PIPELINE COMPLETE ===")
        return report

    except Exception as e:
        raise FootVisionException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()