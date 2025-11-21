import joblib
import json
from fastapi import HTTPException
from config.train_config import TRAIN_CONFIG


class ModelStore:
    def __init__(self,
                models: dict = {},
                metadata: dict = {},
                schemas: dict = {}
   ):
       self.models = models
       self.metadata = metadata
       self.schemas = schemas

    def get_model_info(self, dataset: str):
        if dataset not in self.models:
            raise HTTPException(status_code=404, detail=f"Data for {dataset} not found")
        return self.models[dataset], self.metadata[dataset], self.schemas[dataset]

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

