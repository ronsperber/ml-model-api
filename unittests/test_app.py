"""
Unit tests for ml-model-api
"""
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from app import app
from serve.model_store import ModelStore
from config.schema import schemas
from config.train_config import TRAIN_CONFIG
from training.utils import (
    get_feature_names_from_fitted_pipeline,
    get_feature_importances,
    predict_labels,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Existing test
# ---------------------------------------------------------------------------

def test_column_transformer():
    df = pd.DataFrame({"num1": [1, 2], "cat1": ["A", "B"]})
    ct = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["cat1"]),
            ("num", "passthrough", ["num1"]),
        ]
    )
    out = ct.fit_transform(df)
    assert out.shape == (2, 3)


# ---------------------------------------------------------------------------
# ModelStore tests
# ---------------------------------------------------------------------------

def test_model_store_schema_config_mismatch():
    """ModelStore should raise ValueError if schema and config keys don't match"""
    mismatched_schemas = {"iris": schemas["iris"], "extra": schemas["iris"]}
    with pytest.raises(ValueError):
        ModelStore(mismatched_schemas, TRAIN_CONFIG)


def test_model_store_get_invalid_dataset():
    """ModelStore.get should raise KeyError for unknown dataset"""
    store = ModelStore(schemas, TRAIN_CONFIG)
    with pytest.raises(KeyError):
        store.get("nonexistent_dataset")


def test_model_store_get_iris():
    """ModelStore.get should return a ModelEntry with model, metadata, and schema"""
    store = ModelStore(schemas, TRAIN_CONFIG)
    entry = store.get("iris")
    assert entry.model is not None
    assert entry.metadata is not None
    assert entry.schema is not None


def test_model_store_lazy_loading():
    """ModelStore should only load a model once"""
    store = ModelStore(schemas, TRAIN_CONFIG)
    store.get("iris")
    # second call should use cached version
    entry = store.get("iris")
    assert entry.model is not None


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

def test_predict_iris_valid():
    """POST /predict with valid iris features should return 200 with label and probs"""
    response = client.post(
        "/predict?dataset=iris",
        json={
            "SepalLengthCm": 5.1,
            "SepalWidthCm": 3.5,
            "PetalLengthCm": 1.4,
            "PetalWidthCm": 0.2,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_label" in data
    assert "predicted_probs" in data
    assert isinstance(data["predicted_probs"], dict)


def test_predict_invalid_dataset():
    """POST /predict with unknown dataset should return 404"""
    response = client.post(
        "/predict?dataset=nonexistent",
        json={"feature": 1.0},
    )
    assert response.status_code == 404


def test_predict_invalid_schema():
    """POST /predict with wrong fields should return 400"""
    response = client.post(
        "/predict?dataset=iris",
        json={"wrong_field": 1.0},
    )
    assert response.status_code == 400


def test_predict_batch_valid():
    """POST /predict_batch with valid iris features should return 200 with responses"""
    response = client.post(
        "/predict_batch?dataset=iris",
        json={
            "items": [
                {
                    "SepalLengthCm": 5.1,
                    "SepalWidthCm": 3.5,
                    "PetalLengthCm": 1.4,
                    "PetalWidthCm": 0.2,
                },
                {
                    "SepalLengthCm": 6.7,
                    "SepalWidthCm": 3.0,
                    "PetalLengthCm": 5.2,
                    "PetalWidthCm": 2.3,
                },
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert len(data["response"]) == 2
    for item in data["response"]:
        assert "predicted_label" in item
        assert "predicted_probs" in item


def test_predict_batch_invalid_payload():
    """POST /predict_batch with missing items key should return 400"""
    response = client.post(
        "/predict_batch?dataset=iris",
        json={"wrong_key": []},
    )
    assert response.status_code == 400


def test_predict_batch_invalid_row():
    """POST /predict_batch with an invalid row should return 400"""
    response = client.post(
        "/predict_batch?dataset=iris",
        json={"items": [{"wrong_field": 1.0}]},
    )
    assert response.status_code == 400


def test_get_metadata_valid():
    """GET /metadata with valid dataset should return 200 with metadata fields"""
    response = client.get("/metadata?dataset=iris")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "features" in data
    assert "classes" in data
    assert "test_score" in data


def test_get_metadata_invalid_dataset():
    """GET /metadata with unknown dataset should return 404"""
    response = client.get("/metadata?dataset=nonexistent")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Training utility tests
# ---------------------------------------------------------------------------

def test_get_feature_names_no_preprocessing():
    """get_feature_names_from_fitted_pipeline with no preprocessing returns column names"""
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y = np.array([0, 1, 0])
    pipeline = Pipeline([("model", RandomForestClassifier(n_estimators=5, random_state=42))])
    pipeline.fit(X, y)
    names = get_feature_names_from_fitted_pipeline(pipeline, X)
    assert names == ["a", "b"]


def test_get_feature_names_with_preprocessing():
    """get_feature_names_from_fitted_pipeline with scaler returns correct names"""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = np.array([0, 1, 0])
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=5, random_state=42))
    ])
    pipeline.fit(X, y)
    names = get_feature_names_from_fitted_pipeline(pipeline, X)
    assert set(names) == {"a", "b"}


def test_get_feature_importances():
    """get_feature_importances should return a dataframe with feature and importance columns"""
    metadata = {
        "features": ["a", "b", "c"],
        "feature_importances": [0.5, 0.3, 0.2],
        "model_type": "RandomForestClassifier",
        "best_params": {},
        "test_score": {"accuracy": 0.95},
        "classes": ["cat", "dog"],
    }
    df = get_feature_importances(metadata)
    assert list(df.columns) == ["feature", "importance"]
    assert len(df) == 3


def test_predict_labels():
    """predict_labels should return correct class labels"""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = np.array([0, 1, 0])
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    metadata = {"classes": ["cat", "dog"]}
    labels = predict_labels(model, metadata, X)
    assert all(l in ["cat", "dog"] for l in labels)
    assert len(labels) == 3