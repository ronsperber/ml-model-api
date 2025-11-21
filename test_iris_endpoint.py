from fastapi import FastAPI, Query, Body, HTTPException
import pandas as pd
from config.schema import schemas
from config.train_config import TRAIN_CONFIG
from serve.model_store import ModelStore
model_store = ModelStore(schemas, TRAIN_CONFIG)
model_store.load_all()
app = FastAPI()



@app.post("/predict/")
def predict(
    features: dict = Body(...),
    dataset = Query("iris", description="Dataset chosen"),
    ):
    try :
        model, metadata, schema = model_store.get(dataset)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
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
