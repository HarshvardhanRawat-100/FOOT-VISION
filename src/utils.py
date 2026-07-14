# TODO: save/load helpers (pickle, json, numpy)
import os
import sys
import pickle
import json
import numpy as np

from src.exception import FootVisionException
from src.logger import logging


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Saved object to {file_path}")
    except Exception as e:
        raise FootVisionException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise FootVisionException(e, sys)


def save_json(file_path, data: dict):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        raise FootVisionException(e, sys)


def save_numpy_array(file_path, array: np.ndarray):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, array)
    except Exception as e:
        raise FootVisionException(e, sys)


def load_numpy_array(file_path):
    try:
        return np.load(file_path, allow_pickle=True)
    except Exception as e:
        raise FootVisionException(e, sys)