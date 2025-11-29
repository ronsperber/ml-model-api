from typing import TypedDict, List, Any
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


class ModelMetadata(TypedDict):
    model_type: str
    features: List[str]
    feature_importances: List[float]
    best_params: dict
    test_acc: float
    classes: List[str]

def apply_preprocessing_for_names(
        preproc: Pipeline,
        X_train: pd.DataFrame
        ):
    """
    Replay the preprocessing pipeline (without the model)
    and recover actual column names. Safe as long as each
    step either returns a DataFrame or supports feature_names_out().
    """
    Xt = X_train.copy()

    for name, step in preproc.steps:
        Xt = step.transform(Xt)

        # If returned ndarray → wrap as DataFrame
        if not isinstance(Xt, pd.DataFrame):
            if hasattr(step, "get_feature_names_out"):
                cols = step.get_feature_names_out()
            else:
                # generic fallback for this step
                cols = [f"{name}_{i}" for i in range(Xt.shape[1])]
            Xt = pd.DataFrame(Xt, columns=cols)

    return Xt.columns.tolist()

def get_feature_names_from_fitted_pipeline(
        fitted_pipeline: Pipeline,
        X_train: pd.DataFrame
):
    """
    Extract final feature names from a fitted pipeline.
    Handles:
    - transformers that support get_feature_names_out()
    - arbitrary custom DataFrame-based transforms
    - fallbacks when structure is lost
    """

    # Everything except the model
    preproc = fitted_pipeline[:-1]

    # ---- 1. Best case: final step supports feature names ----
    _, last_step = preproc.steps[-1]
    if hasattr(last_step, "get_feature_names_out"):
        try:
            return list(last_step.get_feature_names_out())
        except Exception:
            pass  # fall through to the next method

    # ---- 2. Replay the preprocessing pipeline ----
    try:
        return apply_preprocessing_for_names(preproc, X_train)
    except Exception:
        pass

    # ---- 3. Final fallback: generic names ----
    try:
        Xt = preproc.transform(X_train)
        return [f"feature_{i}" for i in range(Xt.shape[1])]
    except Exception:
        return []  # nothing we can do


def get_feature_importances(metadata: ModelMetadata) -> pd.DataFrame:
    """
    create a dataframe of feature and feature importance from the model
    Parameters
    ----------
    metadata : ModelMetadata
        dictionary of metadata produced from training
    Returns
    -------
    feat_df : pd.DataFrame
        dataframe with features and feature importances
    """
    feat_df = pd.DataFrame(
        {
            "feature" : metadata["features"],
            "importance" : metadata["feature_importances"]
        }
    )
    return feat_df

def predict_labels(
        model: BaseEstimator | GridSearchCV,
        metadata: ModelMetadata,
        X : pd.DataFrame
          ) -> list[Any]:
    """
    Return predicted labels for a dataset using the model and metadata.

    Parameters
    ----------
    model : BaseEstimator | GridSearchCV
        Trained model used to make predictions.
    metadata : ModelMetadata
        Metadata for the model that must contain a "classes" key.
    X : pd.DataFrame
        Data to predict outcomes for.

    Returns
    -------
    labels : list[Any]
        List of predicted labels corresponding to the original classes.
    """
    if "classes" not in metadata:
        raise ValueError("'classes' missing from metadata")
    y_pred = model.predict(X)
    labels = [metadata["classes"][i] for i in y_pred]
    return labels

def labels_with_df(
        model : BaseEstimator | GridSearchCV,
        metadata: ModelMetadata,
        X : pd.DataFrame,
        pred_col : str = "predicted"
) -> pd.DataFrame:
    """
    Return a DataFrame containing the original data plus predicted labels.

    Parameters
    ----------
    model : BaseEstimator | GridSearchCV
        Trained model used to make predictions.
    metadata : ModelMetadata
        Metadata for the model that must contain a "classes" key.
    X : pd.DataFrame
        Data used to get predictions.
    pred_col : str, default="predicted"
        Column name to store predicted labels.

    Returns
    -------
    pd.DataFrame
        DataFrame of X with predicted labels added as a new column.
    """
    labels = predict_labels(model, metadata, X)
    return X.assign(**{pred_col:labels})