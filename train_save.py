import joblib
import json
import os
from typing import Callable, List, Type, Any
from pydantic import BaseModel, Field
import argparse
# get the config for training
from config.train_config import TRAIN_CONFIG
from training.training import train
# set up arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config_key",
    default="iris",  # default value
    help="Key of the training config to use"
)
args = parser.parse_args()
key = args.config_key
# create TrainConfig class 
# used to validate a valid TRAIN_CONFIGS[key]
class TrainConfig(BaseModel):
    dataset_path : str
    index_col : str | None = None
    target_col : str
    model_type : Type[Any]
    model_params :dict = Field(default={})
    param_grid : dict = Field(default={})
    clean_fn : Callable = Field(default=(lambda x: x))
    feature_steps : List[Any] = Field(default=[])
    pipeline_steps : List[Any] = Field(default=[])
    test_size: float = 0.2
    random_state : int = 42
    min_accuracy : float = 0.95
    model_output: str
    metadata_output: str
# train the model and get the model and metadata
if key in TRAIN_CONFIG:
    configs = TrainConfig(**TRAIN_CONFIG[key])
else:
    raise ValueError(f"{key} is not a valid key. Valid keys are {list(TRAIN_CONFIG.keys())}")
train_results = train(configs.model_dump())
model = train_results["model"]
metadata = train_results["metadata"]
# get the test accuracy
acc = metadata["test_acc"]
print(f"Test accuracy : {acc:.4f}")
if acc < configs.min_accuracy:
    print("Accuracy too low. Try different hyperparameters or a different model")
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
   

