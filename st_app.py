import requests
import streamlit as st
from typing import Type, get_origin, get_args
from enum import Enum
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from pydantic_core import PydanticUndefined
import pandas as pd
from config.schema import schemas
from training.utils import get_feature_importances

load_dotenv()
API_HOST = os.getenv("API_HOST", "http://127.0.0.1:8000")
st.title("Model Predictions")
# Get the model being used to make a prediction
model_name = st.sidebar.selectbox(label="Model name", options=list(schemas.keys()))
# Get the mode (single prediction or batch mode)
mode = st.sidebar.selectbox(
    label="Predict mode", options=["Predict single", "Predict batch (csv)"]
)
# get the schema and config for the model being used
schema = schemas[model_name]
# create the endpoint string based on single vs batch
predict_endpoint = f"{API_HOST}/predict"
metadata_endpoint = f"{API_HOST}/metadata"
predict_batch_endpoint = f"{API_HOST}/predict_batch"
st.caption(f"API host: {API_HOST}")


@st.cache_data(show_spinner=False)
def fetch_metadata(model_name: str) -> dict:
    """
    get metadata for a model
    Parameters
    ----------
    model_name : str
        name of model to fetch metadata for
    Returns
    -------
    dict
        metadata for requested model
    """
    return requests.get(metadata_endpoint, params={"dataset": model_name}).json()


def render_schema_form(schema: Type[BaseModel], form_title: str = "Input Form") -> dict:
    """
    Get inputs for single entry prediction
    Parameters
    ----------
    schema : Type[BaseModel]
        the schema for the model being used
    form_title : str, default is 'Input Form'
        title for the form
    Returns
    -------
    dict
        inputs as a dict
    """
    inputs = {}
    st.header(form_title)
    # get inputs for each field needed for the model
    for field_name, field in schema.model_fields.items():
        field_type = field.annotation
        # Handle Optional[T]
        origin = get_origin(field_type)
        if origin is not None:
            args = get_args(field_type)
            if len(args) == 2 and type(None) in args:
                field_type = args[0]
        # Use default only if defined, otherwise fallback
        if field.default != PydanticUndefined:
            default = field.default
        else:
            default = None
        # Normalize Enum default to its value
        if isinstance(default, Enum):
            default = default.value

        if isinstance(field_type, type) and issubclass(field_type, Enum):
            options = [e.value for e in field_type]
            # choose default index if possible
            index = options.index(default) if default in options else 0
            value = st.selectbox(field_name, options=options, index=index)
        # get inputs for the model with different methods depending on type
        elif field_type is float:
            value = st.number_input(
                field_name, value=default if default is not None else 0.0
            )
        elif field_type is int:
            value = st.number_input(
                field_name,
                value=default if default is not None else 0,
                step=1,
                format="%d",
            )
        elif field_type is bool:
            value = st.checkbox(
                field_name,
                value=default if default is not None else False,
            )
        else:
            value = st.text_input(
                field_name, value=default if default is not None else ""
            )

        inputs[field_name] = value

    return inputs


if mode == "Predict single":  # if we are only doing a single prediction
    # get the data to predict on
    payload = render_schema_form(schema, form_title=f"Input form for {model_name}")
    predict = st.button("Get prediction")
    if predict:
        # get the model prediction
        result = requests.post(
            predict_endpoint, params={"dataset": model_name}, json=payload
        )
        response = result.json()
        # get the probabilities and label
        probs = response["predicted_probs"]
        pred_label = response["predicted_label"]
        # display the result
        st.markdown("###  Predicted Class")
        st.markdown(
            f"""
        <div style="
            padding: 10px;
            border-radius: 8px;
            background-color: #e0f7fa;
            font-size: 20px;
        ">
            <b>{pred_label}</b>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown("### Class Probabilities")
        cols = st.columns(len(probs))
        probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
        for (label, prob), col in zip(probs.items(), cols):
            highlight = "border: 3px solid #00bcd4;" if label == pred_label else ""
            col.markdown(
                f"""
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
            """,
                unsafe_allow_html=True,
            )
else:  # for batch predictions
    st.subheader("Upload CSV for batch prediction")
    uploaded = st.file_uploader(
        "Upload a CSV file", type=["csv"], key=f"uploader_{model_name}"
    )
    if uploaded is not None:
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
                df_dict = df_req.to_dict(orient="records")
                # put payload in form expected
                payload = {"items": df_dict}
                # get result from this CSV
                result = requests.post(
                    predict_batch_endpoint, params={"dataset": model_name}, json=payload
                )
                # save response
                response = result.json()["response"]
                # Assemble table for display
                # first get all the labels
                pred_labels = [r["predicted_label"] for r in response]
                df_out = df.copy()
                df_out["prediction"] = pred_labels
                # then get the probabilities for each prediction
                prob_dicts = [r["predicted_probs"] for r in response]
                probs_df = pd.DataFrame(prob_dicts)
                probs_df = probs_df.rename(
                    columns={c: f"prob_{c}" for c in probs_df.columns}
                )
                # combine input data, prediction and probabilities
                MAX_STYLED_CELLS = 262144
                df_out = pd.concat([df_out, probs_df], axis=1)
                st.subheader("Batch Results")
                if df_out.shape[0] * df_out.shape[1] <= MAX_STYLED_CELLS:
                    st.dataframe(
                        df_out.style.highlight_max(axis=1, subset=probs_df.columns),
                        use_container_width=True,
                    )
                else:
                    st.caption("Probability highlighting disabled for large datasets.")
                    st.dataframe(df_out, use_container_width=True)

# if requested show the feature importances for the model
get_feature_imp = st.sidebar.button("See feature importances")
if get_feature_imp:
    # get metadata and turn feature importances into a dataframe
    metadata = fetch_metadata(model_name)
    feature_df = get_feature_importances(metadata)
    feature_df["feature"] = feature_df["feature"].str.replace(r"^.*?__", "", regex=True)
    feature_df = feature_df.sort_values("importance", ascending=False).set_index(
        "feature"
    )
    styled = feature_df.style.format("{:.4f}")
    st.sidebar.dataframe(styled, use_container_width=True)
