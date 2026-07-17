# TODO: ModelTrainer class - CNN architecture + training loop
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.exception import FootVisionException
from src.logger import logging
from src.utils import load_numpy_array

NUM_CLASSES = 6  # header, Volley , penalty , free kick — update this when you add more classes


class ShotClassifierCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, seq_len=60, in_features=34):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        flat_dim = 128 * (seq_len // 4)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return self.head(x)


@dataclass
class ModelTrainerConfig:
    train_array_path: str = os.path.join("artifacts", "data_transformation", "train.npy")
    test_array_path: str = os.path.join("artifacts", "data_transformation", "test.npy")
    train_labels_path: str = os.path.join("artifacts", "data_transformation", "train_labels.npy")
    test_labels_path: str = os.path.join("artifacts", "data_transformation", "test_labels.npy")
    model_path: str = os.path.join("models", "best_model.pth")
    epochs: int = 40
    batch_size: int = 8
    lr: float = 1e-3


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def initiate_model_training(self):
        logging.info(f"Starting model training on device: {self.device}")
        try:
            X_train = torch.tensor(load_numpy_array(self.config.train_array_path), dtype=torch.float32)
            y_train = torch.tensor(load_numpy_array(self.config.train_labels_path), dtype=torch.long)
            X_test = torch.tensor(load_numpy_array(self.config.test_array_path), dtype=torch.float32)
            y_test = torch.tensor(load_numpy_array(self.config.test_labels_path), dtype=torch.long)

            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.config.batch_size, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=self.config.batch_size)

            model = ShotClassifierCNN().to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
            criterion = nn.CrossEntropyLoss()

            best_acc = 0.0

            for epoch in range(self.config.epochs):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()

                acc = self._evaluate(model, test_loader)
                logging.info(f"Epoch {epoch+1}/{self.config.epochs} - val_acc={acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
                    torch.save(model.state_dict(), self.config.model_path)

            logging.info(f"Training complete. Best val_acc={best_acc:.4f}")
            return self.config.model_path, best_acc

        except Exception as e:
            raise FootVisionException(e, sys)

    def _evaluate(self, model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    ModelTrainer().initiate_model_training()