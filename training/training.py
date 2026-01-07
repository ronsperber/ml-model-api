"""
module to train models on data
"""
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from .utils import get_feature_names_from_fitted_pipeline
def train(configs: dict) -> dict:
    """
    Function to read configs for a model, train and return the model and metadata
    Parameters
    ----------
    configs : dict 
        dict that has all the configs for a particular dataset needed for training
    Returns
    dict
        has two key-value pairs. model and metadata
    """
    # load the data
    df = pd.read_csv(configs["dataset_path"])
    # if an index column specified set the index column to be the index
    if configs["index_col"] is not None:
        df = df.set_index(configs["index_col"])
    # get the feature set
    X = df.drop(columns=[configs["target_col"]])
    y_raw = df[configs["target_col"]]
    # encode the labels numerically
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    # get the preprocessing steps
    preprocessing_steps = configs.get("preprocessing_steps", [])
    # get the model type
    ModelClass = configs["model_type"]
    # get any model parameters
    kwargs = configs.get("model_params", {})
    # create an instance of the model type using the model parameters
    model_cls = ModelClass(**kwargs)
    # get the grid search parameters (e.g. a scoring function)
    grid_search_params = configs.get("grid_search_params", {})
    # create pipeline with preprocessing steps and the model
    pipeline = Pipeline(
    [
        *preprocessing_steps,
        ("model", model_cls)
    ]
    )
    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=configs.get("test_size", 0.2),
        random_state=configs.get("random_state", 42)
    )
    # create the model
    model = GridSearchCV(
        pipeline,
        configs["param_grid"],
        **grid_search_params
    )
    # train model
    print("Training model...")
    model.fit(X_train, y_train)
    # Extract names from best estimator
    best_pipe = model.best_estimator_
    feature_names = get_feature_names_from_fitted_pipeline(best_pipe, X_train)
    print("Training complete.")
    score = model.score(X_test, y_test)
    score_label = configs.get("score_label", "accuracy")
    # get feature importances if the model records them
    feature_importances = []
    model_estimator = model.best_estimator_["model"]
    if hasattr(model_estimator,"feature_importances_"):
        feature_importances = model_estimator.feature_importances_.tolist()
    metadata = {
        "model_type": configs["model_type"].__name__,
        "features" : feature_names,
        "feature_importances" : feature_importances,
        "best_params": model.best_params_,
        "test_score": {score_label: score},
        "classes": list(le.classes_)
    }
    return {
        "model": model,
        "metadata": metadata
    }