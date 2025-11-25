import json
import requests
import streamlit as st
from typing import Type
from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from serve.model_store import ModelStore
from config.schema import schemas
from config.train_config import TRAIN_CONFIG
st.title("Model Predictions")
model_name = st.sidebar.selectbox(label="Model name",options=list(schemas.keys()))
schema = schemas[model_name]
config = TRAIN_CONFIG[model_name]
endpoint = "http://127.0.0.1:8000/predict"
payload ={}


def render_schema_form(schema: Type[BaseModel], form_title: str = "Input Form") -> dict:
    inputs = {}
    st.header(form_title)

    for field_name, field in schema.model_fields.items():
        field_type = field.annotation
        # Use default only if defined, otherwise fallback
        if field.default != PydanticUndefined:
            default = field.default
        else:
            default = None

        if field_type in (int, float):
            value = st.number_input(field_name, value=default if default is not None else 0.0)
        elif field_type is bool:
            value = st.checkbox(field_name, value=default if default is not None else False)
        else:
            value = st.text_input(field_name, value=default if default is not None else "")

        inputs[field_name] = value

    return inputs

payload = render_schema_form(schema, form_title = f"Input form for {model_name}")
predict = st.button("Get prediction")
if predict: 
    result = requests.post(endpoint, params={"dataset":model_name},json=payload)
    response = result.json()
    st.write(f"Predicted class: {response["predicted_label"]}")
    st.write("Class Probabilities")
    for label, prob in response["predicted_probs"].items():
        st.write(f"Label: {label} , Probability {prob:.4f}")