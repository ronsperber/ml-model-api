"""
module to train models on data
"""
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from .utils import get_feature_names_from_fitted_pipeline


def train(
        configs: dict,
        verbosity:int = 0,
        refit_full:bool = False,
        ) -> dict:
    """
    Function to read configs for a model, train and return the model and metadata
    Parameters
    ----------
    configs : dict
        dict that has all the configs for a particular dataset needed for training
    verbosity : int
        verbosity to use for grid search
    refit_full : bool
        whether to retrain on the full dataset.
    Returns
    -------
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
    y_raw:pd.Series = df[configs["target_col"]]
    # encode the labels numerically for classification 
    classes = []
    if configs.get("task","classification") == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        classes = list(le.classes_)
    else: # for regression extract the values
        y = y_raw.values
    # get the preprocessing steps
    preprocessing_steps = configs.get("preprocessing_steps", [])
    # get the model type
    model_class = configs["model_type"]
    # get any model parameters
    kwargs = configs.get("model_params", {})
    # create an instance of the model type using the model parameters
    model_cls = model_class(**kwargs)
    # get the grid search parameters (e.g. a scoring function)
    grid_search_params = configs.get("grid_search_params", {})
    cv = configs.get("cv", 5)  # default to standard 5-fold if not specified
    # create pipeline with preprocessing steps and the model
    pipeline = Pipeline([*preprocessing_steps, ("model", model_cls)])
    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=configs.get("test_size", 0.2),
        random_state=configs.get("random_state", 42),
        shuffle=configs.get("shuffle", True),
    )
    # create the model
    model = GridSearchCV(
        pipeline,
        configs["param_grid"],
        verbose=verbosity,
        cv=cv,
        n_jobs=-1,
        **grid_search_params
        )
    # train model
    print("Finding best hyperparameters")
    model.fit(X_train, y_train)
    # Extract names from best estimator
    best_pipe = model.best_estimator_
    best_params = model.best_params_
    feature_names = get_feature_names_from_fitted_pipeline(best_pipe, X_train)
    print("Hyperparameter tuning complete.")
    score = model.score(X_test, y_test)
    score_label = configs.get("score_label", "accuracy")
    # get feature importances if the model records them
    model = model.best_estimator_
    if refit_full:
        print("Training on full data...")
        model.fit(X,y) #type: ignore
        print("Training on full data complete")
    feature_importances = []
    model_estimator = model["model"] # type: ignore
    if hasattr(model_estimator, "feature_importances_"):
        feature_importances = model_estimator.feature_importances_.tolist()
    metadata = {
        "model_type": configs["model_type"].__name__,
        "features": feature_names,
        "feature_importances": feature_importances,
        "best_params": best_params,
        "test_score": {score_label: score},
        "classes": classes,
        "task": configs.get("task", "classification")
    }
    return {"model": model, "metadata": metadata}
