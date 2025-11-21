from fastapi import FastAPI, Query, Body, HTTPException
import pandas as pd
import joblib
import json
from config.schema import schemas
from config.train_config import TRAIN_CONFIG
app = FastAPI()
models = {}
metadata_vals = {}
model_schemas = {}
def get_info(modelkey: str):
    if modelkey in models:
        return {
            "model" : models[modelkey],
            "metadata" : metadata_vals[modelkey],
            "schema" : model_schemas[modelkey]
            }
    MODEL_PATH = TRAIN_CONFIG[modelkey]["model_output"]
    METADATA_PATH = TRAIN_CONFIG[modelkey]["metadata_output"]
    model =joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    models[modelkey] = model
    metadata_vals[modelkey] = metadata
    model_schemas[modelkey] = schemas[modelkey]
    return {
        "model" : model,
        "metadata": metadata,
        "schema": schemas[modelkey]
    }

@app.post("/predict/")
def predict(
    features: dict = Body(...),
    dataset = Query("iris", description="Dataset chosen"),
    ):
    model_info = get_info(dataset)
    model = model_info["model"]
    metadata = model_info["metadata"]
    schema = model_info["schema"]
    try:
        validated = schema(**features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    classes = metadata["classes"]
    # Convert input to DataFrame (single row)
    X = pd.DataFrame([validated.model_dump()])
    # Run prediction
    y_pred = model.predict(X)
    pred_label = classes[y_pred[0]]
    return {"predicted_species": pred_label}
