"""
training configs for models
"""

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, fbeta_score
from preprocessing_steps.loan_data import loan_steps

# note: score_label should match the scoring metric set in grid_search_params
TRAIN_CONFIG = {
    "iris": {
        "dataset_path": "data/Iris.csv",
        "index_col": "Id",
        "target_col": "Species",
        "model_type": RandomForestClassifier,
        "model_params": {"random_state": 42},
        "param_grid": {
            "model__n_estimators": [50, 100, 150],
            "model__max_depth": [None, 3, 4],
            "model__min_samples_split": [2, 3, 4],
        },
        "test_size": 0.2,
        "random_state": 42,
        "min_score": 0.95,
        "model_output": "models/iris_model.pkl",
        "metadata_output": "metadata/iris_metadata.json",
    },
    "loan_data": {
        "dataset_path": "data/loan.csv",
        "target_col": "loan_paid_back",
        "index_col": "id",
        "model_type": LGBMClassifier,
        "test_size": 0.2,
        "random_state": 42,
        "min_score": 0.75,
        "score_label": "F2 for loan default",
        "model_params": {"random_state": 42, "class_weight": "balanced", "verbose": -1},
        "param_grid": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [5, 7, -1],
            "model__learning_rate": [0.01, 0.1],
            "model__min_child_samples": [20, 50, 100],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        },
        "grid_search_params": {
            "scoring": make_scorer(fbeta_score, beta=2, pos_label=0)
        },
        "preprocessing_steps": loan_steps,
        "model_output": "models/loan.pkl",
        "metadata_output": "metadata/loan.json",
    },
}
