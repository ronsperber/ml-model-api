# ml-model-api

A full-stack machine learning serving system built with FastAPI and Streamlit. Train models via a configurable pipeline, serve predictions through a validated REST API, and explore results through an interactive frontend — all from a single, extensible codebase.

---

## Overview

Most ML projects stop at the notebook. This project bridges the gap between training and deployment with a clean, modular architecture:

- **Training pipeline** — configurable GridSearchCV-based training with preprocessing, evaluation, and artifact saving
- **Model serving** — FastAPI backend with schema-validated single and batch prediction endpoints
- **Interactive frontend** — Streamlit app with auto-generated input forms, probability visualization, and feature importance display
- **Model store** — lazy-loading abstraction that manages models, metadata, and schemas across multiple datasets

The serving layer is model-agnostic: any object with a `predict` method works. Scikit-learn is used for the training pipeline and preprocessing, but the API and `ModelStore` place no constraints on model type.

---

## Architecture

```
ml-model-api/
├── config/
│   ├── schema.py               # Pydantic schemas for each dataset (input validation)
│   └── train_config.py         # Training configs (model type, params, paths, grid search)
├── preprocessing_steps/
│   └── loan_data.py            # Custom preprocessing transformers
├── scripts/
│   └── train_save.py           # Training entrypoint (CLI)
├── serve/
│   └── model_store.py          # ModelStore: lazy loads models, metadata, and schemas
├── training/
│   ├── training.py             # Core train() function
│   └── utils.py                # Pipeline utilities
├── unittests/
│   └── pipeline_test.py        # Unit tests
├── data/                       # Sample datasets (included for reproducibility)
├── app.py                      # FastAPI app with /predict, /predict_batch, /metadata
└── st_app.py                   # Streamlit frontend
```

---

## Quickstart

### Install dependencies

```bash
pip install -e .
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

### Environment setup

Copy `.env.example` to `.env`. The only setting is the API host, which defaults to `http://127.0.0.1:8000` if not set:

```bash
cp .env.example .env
```

### Data

Sample datasets (Iris, loan) are included in `data/` so you can run the project immediately after cloning. To add your own dataset see [Adding a New Dataset](#adding-a-new-dataset).

### Train a model

```bash
python scripts/train_save.py --config_key iris
```

This will:
- Load the dataset specified in `TRAIN_CONFIG`
- Run GridSearchCV over the defined parameter grid
- Evaluate on a held-out test set
- Save the model and metadata only if the test score meets the `min_score` threshold

Available config keys out of the box: `iris`, `loan`

### Start the API

```bash
uvicorn app:app --reload
```

### Launch the Streamlit app

In a separate terminal with the API already running:

```bash
streamlit run st_app.py
```

![Streamlit app screenshot](assets/streamlit_demo.png)

---

## API Endpoints

### `POST /predict`

Single-sample prediction with schema validation.

```bash
curl -X POST "http://localhost:8000/predict?dataset=iris" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

**Response:**
```json
{
  "predicted_label": "setosa",
  "predicted_probs": {
    "setosa": 0.9821,
    "versicolor": 0.0134,
    "virginica": 0.0045
  }
}
```

### `POST /predict_batch`

Batch prediction from a list of samples.

```bash
curl -X POST "http://localhost:8000/predict_batch?dataset=iris" \
  -H "Content-Type: application/json" \
  -d '{"items": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]}'
```

**Response:**
```json
{
  "response": [
    {
      "predicted_label": "setosa",
      "predicted_probs": {
        "setosa": 0.9821,
        "versicolor": 0.0134,
        "virginica": 0.0045
      }
    }
  ]
}
```

### `GET /metadata`

Returns model metadata including model type, features, best hyperparameters, test score, and feature importances.

```bash
curl "http://localhost:8000/metadata?dataset=iris"
```

---

## Adding a New Dataset

The system is designed so that adding a new dataset requires changes in exactly two places:

**1. Define a Pydantic schema** in `config/schema.py`:

```python
class MyDataSchema(BaseModel):
    feature_one: float
    feature_two: float
    feature_three: int
```

**2. Add a training config** in `config/train_config.py`:

```python
"my_dataset": {
    "dataset_path": "data/my_dataset.csv",
    "target_col": "label",
    "index_col": None,
    "model_type": RandomForestClassifier,
    "model_params": {},
    "param_grid": {"model__n_estimators": [50, 100], "model__max_depth": [3, 5]},
    "grid_search_params": {"scoring": "accuracy", "cv": 5},
    "preprocessing_steps": [("scaler", StandardScaler())],
    "min_score": 0.90,
    "score_label": "accuracy",
    "model_output": "models/my_dataset_model.pkl",
    "metadata_output": "models/my_dataset_metadata.json",
}
```

Then train:

```bash
python scripts/train_save.py --config_key my_dataset
```

The API and Streamlit frontend will automatically pick up the new dataset — no other changes needed.

---

## Streamlit Frontend

The Streamlit app connects to the running FastAPI backend and provides:

- **Single prediction mode** — input form auto-generated from the dataset's Pydantic schema, with predicted class and per-class probabilities displayed
- **Batch prediction mode** — upload a CSV, validate columns against the schema, run predictions, and view results with probability columns appended
- **Feature importances** — sidebar display of feature importances from the trained model metadata

The input form is fully dynamic — adding a new dataset with a new schema requires no changes to the frontend code.

---

## Training Pipeline Details

The `train()` function builds a scikit-learn `Pipeline` with configurable preprocessing steps and a model, wraps it in `GridSearchCV`, and returns the best estimator along with metadata. Key features:

- **Min-score gate** — models are only saved if test score meets the configured threshold
- **Flexible preprocessing** — any scikit-learn-compatible transformers can be added as pipeline steps
- **Rich metadata** — saved alongside each model: best hyperparameters, test score, feature names, feature importances, and class labels
- **Label encoding** — target labels are encoded automatically; original class names are preserved in metadata for human-readable predictions

The serving layer is intentionally decoupled from the training framework. Any model with a `predict` method (and optionally `predict_proba`) can be loaded into the `ModelStore` and served through the API.

---

## Running Tests

```bash
pytest unittests/
```

---

## Configuration Reference

| Key | Description |
|-----|-------------|
| `dataset_path` | Path to training CSV |
| `target_col` | Name of target column |
| `index_col` | Index column name, or `None` |
| `model_type` | Scikit-learn estimator class |
| `model_params` | Constructor kwargs for the model |
| `param_grid` | GridSearchCV parameter grid |
| `grid_search_params` | Additional GridSearchCV kwargs (e.g. `scoring`, `cv`) |
| `preprocessing_steps` | List of `(name, transformer)` tuples for the pipeline |
| `test_size` | Train/test split ratio (default `0.2`) |
| `random_state` | Random seed (default `42`) |
| `min_score` | Minimum test score required to save the model |
| `score_label` | Label for the score metric in metadata (default `"accuracy"`) |
| `model_output` | Path to save the trained model |
| `metadata_output` | Path to save model metadata |

---

## Requirements

- Python 3.10+
- FastAPI
- Uvicorn
- Streamlit
- Scikit-learn
- Pandas
- Pydantic
- Joblib
- python-dotenv

See `requirements.txt` for pinned versions.

---

## Future Work

- Live data ingestion from public APIs (NOAA, USGS) with on-demand retraining via `/retrain` endpoint
- Support for regression targets in addition to classification
- MLflow integration for experiment tracking
- Docker deployment configuration
