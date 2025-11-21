import joblib
import json
import logging
logging.basicConfig(level=logging.INFO)


class ModelStore:
    def __init__(self, schemas, config):
        self.models = {}
        self.metadata = {}
        self.schemas = schemas
        self.config = config  # store the config for later

    def load_all(self):
        self.load_models()
        self.load_metadata()

    def load_models(self):
        for key, cfg in self.config.items():
            try:
                self.models[key] = joblib.load(cfg["model_output"])
            except FileNotFoundError:
                logging.warning(f"No model file found for {key}")

    def load_metadata(self):
        for key, cfg in self.config.items():
            try:
                with open(cfg["metadata_output"], "r") as f:
                    self.metadata[key] = json.load(f)
            except FileNotFoundError:
                logging.warning(f"No metadata found for {key}")

    def get(self, dataset):
        if dataset not in self.models:
            raise KeyError(f"Model for '{dataset}' not found")
        if dataset not in self.metadata:
            raise KeyError(f"Metadata for '{dataset}' not found")
        if dataset not in self.schemas:
            raise KeyError(f"Schema for '{dataset}' not found")
        return self.models[dataset], self.metadata[dataset], self.schemas[dataset]

