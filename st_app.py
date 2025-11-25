import requests
import streamlit as st
from typing import Type
from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from config.schema import schemas
from config.train_config import TRAIN_CONFIG
st.title("Model Predictions")
# Get the model being used to make a prediction
model_name = st.sidebar.selectbox(label="Model name",options=list(schemas.keys()))
# Get the mode (single prediction or batch mode)
mode = st.sidebar.selectbox(label="Predict mode", options=["Predict single", "Predict batch (csv)"])
# get the schema and config for the model being used
schema = schemas[model_name]
config = TRAIN_CONFIG[model_name]
# create the endpoint string based on single vs batch
endpoint = "http://127.0.0.1:8000/predict"
if mode == "Predict batch (csv)":
    endpoint += "_batch"

def render_schema_form(schema: Type[BaseModel], form_title: str = "Input Form") -> dict:
    """
    Get inputs for single entry prediction
    Parameters
    schema : Type[BaseModel]
        the schema for the model being used
    form_title : str, default is 'Input Form'
        title for the form"""
    inputs = {}
    st.header(form_title)
    # get inputs for each field needed for the model
    for field_name, field in schema.model_fields.items():
        field_type = field.annotation
        # Use default only if defined, otherwise fallback
        if field.default != PydanticUndefined:
            default = field.default
        else:
            default = None
        # get inputs for the model with different methods depending on type
        if field_type in (int, float):
            value = st.number_input(field_name, value=default if default is not None else 0.0)
        elif field_type is bool:
            value = st.checkbox(field_name, value=default if default is not None else False)
        else:
            value = st.text_input(field_name, value=default if default is not None else "")

        inputs[field_name] = value

    return inputs
if mode == "Predict single": #if we are only doing a single prediction
    # get the data to predict on
    payload = render_schema_form(schema, form_title = f"Input form for {model_name}")
    predict = st.button("Get prediction")
    if predict:
        # get the model prediction
        result = requests.post(endpoint, params={"dataset": model_name}, json=payload)
        response = result.json()
        # get the probabilities and label
        probs = response["predicted_probs"]
        pred_label = response["predicted_label"]
        # display the result
        st.markdown("###  Predicted Class")
        st.markdown(f"""
        <div style="
            padding: 10px;
            border-radius: 8px;
            background-color: #e0f7fa;
            font-size: 20px;
        ">
            <b>{pred_label}</b>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### Class Probabilities")
        cols = st.columns(len(probs))
        probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
        for (label, prob), col in zip(probs.items(), cols): 
            highlight = "border: 3px solid #00bcd4;" if label == pred_label else ""
            col.markdown(f"""
            <div style="
                padding: 12px;
                border-radius: 8px;
                background: #f4f4f4;
                margin-bottom: 10px;
                {highlight};
            ">
                <b>{label}</b><br>
                Probability: <b>{prob:.4f}</b>
            </div>
            """, unsafe_allow_html=True)
else: # for batch predictions
    st.subheader("Upload CSV for batch prediction")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is not None:
        import pandas as pd
        # read in the data uploaded as a dataframe
        df = pd.read_csv(uploaded)
        # Validate column names
        required_fields = schema.model_fields.keys()
        missing = [c for c in required_fields if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            st.success("CSV accepted! Ready to send to API.")
            if st.button("Run Batch Prediction"):
                # only send required fields
                df_req = df[list(required_fields)]
                # convert to list of dicts
                df_dict = (df_req.to_dict(orient="records"))
                # put payload in form expected
                payload = {"items" : df_dict}
                # get result from this CSV
                result = requests.post(
                    endpoint,
                    params={"dataset": model_name},
                    json=payload
                )
                # save response
                response = result.json()["response"]
                # Assemble table for display
                # first get all the labels
                pred_labels = [r["predicted_label"] for r in response]
                df_out = df.copy()
                df_out["prediction"] = pred_labels
                # then get the probabilities for each prediction
                prob_dicts =[r["predicted_probs"] for r in response]
                probs_df = pd.DataFrame(prob_dicts)
                probs_df = probs_df.rename(columns={c:f"prob_{c}" for c in probs_df.columns})
                # combine input data, prediction and probabilities
                df_out = pd.concat([df_out, probs_df], axis=1)
                st.subheader("Batch Results")
                st.dataframe(df_out.style.highlight_max(axis=1, subset=probs_df.columns),
                             width='content')

