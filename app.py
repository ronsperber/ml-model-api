import logging
from fastapi import FastAPI, Query, Body, HTTPException
import pandas as pd
from config.schema import schemas
from config.train_config import TRAIN_CONFIG
from serve.model_store import ModelStore
model_store = ModelStore(schemas, TRAIN_CONFIG)
model_store.load_all()

# Just configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler("app.log")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)
app = FastAPI()
logger.info("Starting endpoints")
logger.info("Reading in info for models")
model_store = ModelStore(schemas, TRAIN_CONFIG)
model_store.load_all()


@app.post("/predict/")
def predict(
    features: dict = Body(...),
    dataset = Query("iris", description="Dataset chosen"),
    ):
    try :
        model, metadata, schema = model_store.get(dataset)
    except KeyError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    try:
        validated = schema(**features)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))
    classes = metadata["classes"]
    # Convert input to DataFrame (single row)
    X = pd.DataFrame([validated.model_dump()])
    # Run prediction
    logger.info(f"Model for {dataset} making prediction")
    y_pred_proba = list(model.predict_proba(X))[0]
    y_pred = int(y_pred_proba.argmax())
    pred_label = classes[y_pred]
    probs = {k:v for (k,v) in list(zip(classes, y_pred_proba.tolist()))}
    logger.info(f"Model for {dataset} prediction returned")
    return {"predicted_label": pred_label, "predicted_probs": probs}
