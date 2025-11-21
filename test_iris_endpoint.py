from fastapi import FastAPI, Query, Body, HTTPException
import pandas as pd
from config.schema import schemas
from serve.model_store import ModelStore, load_models, load_metadata
app = FastAPI()
model_store = ModelStore()
model_store.models = load_models()
model_store.metadata = load_metadata()
model_store.schemas = schemas


@app.post("/predict/")
def predict(
    features: dict = Body(...),
    dataset = Query("iris", description="Dataset chosen"),
    ):
    model = model_store.models[dataset]
    metadata = model_store.metadata[dataset]
    schema = model_store.schemas[dataset]
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
    return {"predicted_label": pred_label}
