# Train and save a model for a given dataset config.
# Usage: python scripts/train_save.py --config_key <dataset_name>
import joblib
import json
import os
import warnings
from typing import List, Type, Any
from pydantic import BaseModel, Field, ConfigDict
from sklearn.model_selection import BaseCrossValidator
import argparse
import time


# create TrainConfig class
# used to validate a valid TRAIN_CONFIGS[key]
class TrainConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dataset_path: str
    index_col: str | None = None
    target_col: str
    model_type: Type[Any]
    model_params: dict = Field(default_factory=dict)
    param_grid: dict = Field(default_factory=dict)
    grid_search_params: dict = Field(default_factory=dict)
    preprocessing_steps: List[Any] = Field(default_factory=list)
    test_size: float = 0.2
    random_state: int = 42
    min_score: float = 0.95
    score_label: str = "accuracy"
    model_output: str
    metadata_output: str
    cv: int | BaseCrossValidator | None = 5
    shuffle: bool = True
    task: str = "classification"


# suppress sklearn UserWarnings (e.g. convergence warnings during grid search)
warnings.simplefilter("ignore", UserWarning)
# get the config for training
from config.train_config import TRAIN_CONFIG
from training.training import train

# set up arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config_key",
    default="iris",  # default value
    help="Key of the training config to use",
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=int,
    default=0,
    help="verbosity of GridSearchCV",
)
args = parser.parse_args()
key = args.config_key
verbosity=args.verbosity
# train the model and get the model and metadata
if key in TRAIN_CONFIG:
    configs = TrainConfig(**TRAIN_CONFIG[key])
else:
    raise ValueError(
        f"{key} is not a valid key. Valid keys are {list(TRAIN_CONFIG.keys())}"
    )
start = time.perf_counter()
train_results = train(configs.model_dump(), verbosity=verbosity)
end = time.perf_counter()
print(f"Training took {end - start:.4f} seconds")
model = train_results["model"]
metadata = train_results["metadata"]
# get the test accuracy
score = metadata["test_score"]
score_label = list(score.keys())[0]
score_val = score[score_label]
print(f"Test {score_label} : {score_val:.4f}")
if score_val < configs.min_score:
    print(f"{score_label} too low. Try different hyperparameters or a different model")
else:
    # if the directories for saving don't exist, create them
    os.makedirs(os.path.dirname(configs.model_output), exist_ok=True)
    os.makedirs(os.path.dirname(configs.metadata_output), exist_ok=True)
    # save model
    joblib.dump(model, configs.model_output)
    print(f"Model saved to {configs.model_output}.")
    # save metadata
    with open(configs.metadata_output, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {configs.metadata_output}")
