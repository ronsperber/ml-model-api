from typing import TypedDict, List
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV


class ModelMetadata(TypedDict):
    model_type: str
    features: List[str]
    feature_importances: List[float]
    best_params: dict
    test_acc: float
    classes: List[str]

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
          ) -> list[str]:
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
    labels : list[str]
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