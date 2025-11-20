from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import json

# Load the trained model and metadata once at startup
MODEL_PATH = "models/iris_model.pkl"
METADATA_PATH = "metadata/iris_metadata.json"

model = joblib.load(MODEL_PATH)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)
classes = metadata["classes"]

app = FastAPI()

# Define input schema for prediction
class IrisFeatures(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

@app.post("/predict/")
def predict(features: IrisFeatures):
    # Convert input to DataFrame (single row)
    X = pd.DataFrame([features.model_dump()])
    
    # Run prediction
    y_pred = model.predict(X)
    pred_label = classes[y_pred[0]]
    
    return {"predicted_species": pred_label}
