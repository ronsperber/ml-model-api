"""
module containing class to store models
"""

import joblib
from dataclasses import dataclass
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """
    data class to hold model, metadata, and schema
    """

    model: Any
    metadata: Dict
    schema: Any


class ModelStore:
    """
    class to hold all information for models
    """

    def __init__(self, schemas: dict, config: dict):
        """
        Parameters
        ----------
        schemas : dict
            dictionary of schemas
        config : dict
            dictionary of config
        """
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
            raise ValueError(
                f"Schemas for keys {missing_in_config} have no corresponding config"
            )
        if missing_in_schema:
            raise ValueError(
                f"Config for keys {missing_in_schema} have no corresponding schema"
            )

    def load_all(self) -> None:
        self.load_models()
        self.load_metadata()

    def load_model(self, key: str) -> None:
        """
        loads model for key
        Parameters
        ----------
        key : str
            key for model to get
        """
        # when we already have the model loaded do nothing
        if key in self.models:
            return
        # get the filename where the model should be stored
        try:
            filename = self.config[key]["model_output"]
        except KeyError:
            logger.error(f"No path for {key} model found")
            return
        # load the model in
        try:
            self.models[key] = joblib.load(filename)
            logger.info(f"Model for {key} loaded")
        except FileNotFoundError:
            logger.warning((f"Couldn't read {filename} for {key}"))

    def load_models(self) -> None:
        """
        load all models that are listed in the config
        """
        for key in self.config:
            self.load_model(key)

    def load_model_metadata(self, key: str) -> None:
        """
        load the metadata for the model using key
        Parameters
        ----------
        key : str
            key for which model to get
        """
        # When we already have it loaded, don't do anything
        if key in self.metadata:
            return
        # get the path to where the metadata is stored
        try:
            filename = self.config[key]["metadata_output"]
        except KeyError:
            logger.error(f"No path for metadata for {key} found")
            return
        # store the metadata in the path
        try:
            with open(filename, "r") as f:
                self.metadata[key] = json.load(f)
                logger.info(f"Metadata for {key} loaded")
        except FileNotFoundError:
            logger.error(f"No metadata found for {key}")

    def load(self, key: str) -> None:
        """
        load both model and metadata for a particular data set
        Parameters
        ----------
        key : str
            key for which data set to get
        """
        self.load_model(key)
        self.load_model_metadata(key)

    def load_metadata(self) -> None:
        """
        Load metadata for all models in the config
        """
        for key in self.config:
            self.load_model_metadata(key)

    def get(self, dataset: str) -> ModelEntry:
        """
        retrieve information stored for a particular model
        Parameters
        ----------
        dataset : str
            which data set to get information from
        Returns
        -------
        ModelEntry
            contains model, schema, and metadata for that dataset
        """
        # load the metadata and model
        # note, if it's already loaded, this does nothing
        self.load(dataset)
        # check to make sure that the model information is there
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
