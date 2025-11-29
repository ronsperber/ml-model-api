"""
module containing class to store models
"""
import joblib
from dataclasses import dataclass
import json
import logging
from typing import Any, Dict
logging.basicConfig(level=logging.INFO)

@dataclass
class ModelEntry:
    model: Any
    metadata: Dict
    schema: Any

class ModelStore:
    def __init__(self, schemas: dict, config: dict):
        self.models = {}
        self.metadata = {}
        self.schemas = schemas
        self.config = config  # store the config for later
        # make sure that schema keys and config keys are the same
        schema_keys = set(self.schemas.keys())
        config_keys = set(self.config.keys())
        missing_in_schema = config_keys - schema_keys
        missing_in_config = schema_keys - config_keys
        if missing_in_config:
            raise ValueError(f"Schemas for keys {missing_in_config} have no corresponding config")
        if missing_in_schema:
            raise ValueError(f"Config for keys {missing_in_schema} have no corresponding schema")

    def load_all(self):
        self.load_models()
        self.load_metadata()

    def load_model(self, key: str):
        if key in self.models:
            return
        try:
            filename = self.config[key]["model_output"]
        except KeyError:
                logging.error(f"No path for {key} model found")
                return
        try:
            self.models[key] = joblib.load(filename)
            logging.info(f"Model for {key} loaded")
        except FileNotFoundError:
                logging.warning((f"Couldn't read {filename} for {key}"))

    def load_models(self):
        for key in self.config:
            self.load_model(key)
    
    def load_model_metadata(self, key:str):
        if key in self.metadata: 
            return
        try:
            filename = self.config[key]["metadata_output"]
        except KeyError:
            logging.error(f"No path for metadata for {key} found")
            return
        try:
            with open(filename, "r") as f:
                self.metadata[key] = json.load(f)
                logging.info(f"Metadata for {key} loaded")
        except FileNotFoundError:
            logging.error(f"No metadata found for {key}")

    def load(self,key:str):
        self.load_model(key)
        self.load_model_metadata(key)

    def load_metadata(self):
        for key in self.config:
            self.load_model_metadata(key)

    def get(self, dataset):
        self.load(dataset)
        if dataset not in self.models:
            raise KeyError(f"Model for '{dataset}' not found")
        if dataset not in self.metadata:
            raise KeyError(f"Metadata for '{dataset}' not found")
        if dataset not in self.schemas:
            raise KeyError(f"Schema for '{dataset}' not found")
        return ModelEntry(
            model=self.models[dataset],
            metadata=self.metadata[dataset],
            schema=self.schemas[dataset],
        )