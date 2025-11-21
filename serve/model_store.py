import joblib
import json
from pydantic import BaseModel
from config.train_config import TRAIN_CONFIG


class ModelStore(BaseModel):
    models: dict = {}
    metadata: dict = {}
    schemas: dict = {}

def load_models():
    model_dict = {}
    for key in TRAIN_CONFIG:
        model_path = TRAIN_CONFIG[key]["model_output"]
        try:
            model = joblib.load(model_path)
            model_dict[key] = model
        except Exception as e:
            print(f"No model found found for {key}: {e}")
    return model_dict

def load_metadata():
    metadata_dict = {}
    for key in TRAIN_CONFIG:
        metadata_path = TRAIN_CONFIG[key]["metadata_output"]
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                metadata_dict[key] = metadata
        except Exception as e:
            print(f"No metadata found for {key} : {e}")
    return metadata_dict

