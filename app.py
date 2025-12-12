import logging
from fastapi import FastAPI, Query, Body, HTTPException
import pandas as pd
from config.schema import schemas
from config.train_config import TRAIN_CONFIG
from serve.model_store import ModelStore
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
model_store = ModelStore(schemas, TRAIN_CONFIG)

@app.post("/predict")
def predict(
    features:dict = Body(...),
    dataset:str = Query("iris", description="Dataset chosen")
    ):
    """
    Endpoint to predict for a single sample
    Parameters
    ----------
        features : dict
            values of all features expected by the model
        dataset : str (default : 'Iris')
            values for the features to be given to the model
    Returns
    -------
        dict
            label predict and probabilities predict
    """
    try :
        # get the model, schema, and metadata for the dataset
        entry = model_store.get(dataset)
        model = entry.model
        schema = entry.schema
        metadata = entry.metadata
    except KeyError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    try:
        # validate the schema
        validated = schema(**features)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))
    classes = metadata["classes"]
    # Convert input to DataFrame (single row)
    X = pd.DataFrame([validated.model_dump()])
    # Run prediction
    logger.info(f"Model for {dataset} making prediction")
    # if we can predict probalities do that
    if hasattr(model, "predict_proba"):
        y_pred_proba = list(model.predict_proba(X))[0]
        # determine which is the predicted label
        y_pred = int(y_pred_proba.argmax())
        # convert the probability array to list 
        y_pred_proba_list = y_pred_proba.tolist()
    else:
        # if no predict_proba exists, use the predict method to
        # get the prediction and use empty list for probabilities
        y_pred = model.predict(X)[0]
        y_pred_proba_list = []
    pred_label = classes[y_pred]
    probs = dict(zip(classes, y_pred_proba_list))
    logger.info(f"Model for {dataset} prediction returned")
    return {"predicted_label": pred_label, "predicted_probs": probs}

@app.post("/predict_batch")
def predict_batch(
    payload: dict = Body(...),
    dataset: str = Query("iris", description="Dataset chosen")
):
    """
    Endpoint to get predictions for a batch 
    Parameters
    ----------
    payload : dict
        dictionary with one key "items"
        value is a list of samples to predicted
    dataset: str (default 'Iris')
        name of dataset being used
    Returns
    -------
    dict
        one key : response
        value is list of dicts that has predicted label, probabilities
    """
    try :
        # get model , schema, and metadata for dataset
        entry= model_store.get(dataset)
        model = entry.model
        schema = entry.schema
        metadata = entry.metadata
    except KeyError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    # validate payload
    if "items" not in payload or not isinstance(payload["items"], list):
        raise HTTPException(status_code=400, detail="Expected {'items': [...] }")
    validated_rows = []
    # validate each row and append to a list of validated rows
    for i, row in enumerate(payload["items"]):
        try:
            validated = schema(**row)
            validated_rows.append(validated.model_dump())
        except Exception as e:
            logger.error(f"Validation failed on row {i}: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid row {i}: {e}")
    # convert to dataframe
    X = pd.DataFrame(validated_rows)
    logger.info(f"Model for {dataset} making prediction")
    # get the classes 
    classes = metadata["classes"]
    # if we can predict probabilities do that
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
        # determine which is the predicted label
        y_pred = y_proba.argmax(axis=-1)
    else:
        # if no predict_proba exists, use the predict method to
        # get the prediction and use empty list for probabilities
        y_pred = model.predict(X)
        y_proba = None
    response = []
    for i, pred_idx in enumerate(y_pred):
        probs = y_proba[i] if y_proba is not None else []
        obj = {
            "predicted_label": classes[pred_idx],
            "predicted_probs": dict(zip(classes, probs))
        }
        response.append(obj)
    logger.info(f"Model for {dataset} prediction returned")
    return {"response": response}


@app.get("/metadata")
def get_metadata(
    dataset: str = Query("iris", description="Dataset chosen")
):
    """
    endpoint for getting metadata
    Parameters
    ----------
    dataset : str (default : 'Iris')
        label for dataset being used
    """
    # get metadata for dataset
    try:
       entry = model_store.get(dataset)
       metadata = entry.metadata
    except KeyError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))

    return metadata