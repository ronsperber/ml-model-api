# train_configs.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from feature_steps.iris_test import iris_test_feature_steps

TRAIN_CONFIG = {
    "iris":
    {
        "dataset_path": "data/Iris.csv",
        "index_col": "Id",
        "target_col": "Species",

        "model_type": RandomForestClassifier,
        "model_params": {"random_state": 42},
        "param_grid": {
            "model__n_estimators": [50, 100, 150],
            "model__max_depth": [None, 3, 4],
            "model__min_samples_split": [2, 3, 4]
        },

        "test_size": 0.2,
        "random_state": 42,
        "min_accuracy": 0.95,

        "model_output": "models/iris_model.pkl",
        "metadata_output": "metadata/iris_metadata.json",
    },
    "iris_test":
    {
        "dataset_path": "data/Iris.csv",
        "target_col": "Species",
        "index_col": "Id",
        "model_type": RandomForestClassifier,
        "param_grid": {
            "model__n_estimators" : [20, 30]
        },
        "model_output": "models/sample.pkl",
        "metadata_output": "metadata/sample.json",
        "pipeline_steps" : [("scaler", StandardScaler())],
        "feature_steps" : iris_test_feature_steps
    }
    }