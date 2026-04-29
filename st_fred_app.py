import requests
import math
import streamlit as st
from dotenv import load_dotenv
import os
from config.schema import schemas
from config.train_config import TRAIN_CONFIG
from training.utils import get_feature_importances
from loaders.fred_data import get_processed_fred
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
load_dotenv()
def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, "") or os.environ.get(key, default)
    except Exception:
        return os.environ.get(key, default)

API_HOST = get_secret("API_HOST", "http://127.0.0.1:8000")
st.title("Model Predictions for FRED Unemployment rate data")
MODEL_NAME = "fred_data"
schema = schemas[MODEL_NAME]
config = TRAIN_CONFIG[MODEL_NAME]
predict_batch_endpoint = f"{API_HOST}/predict_batch"
metadata_endpoint = f"{API_HOST}/metadata"
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
metadata = fetch_metadata(MODEL_NAME)
training_cutoff_str = metadata["training_cutoff"]  
training_cutoff = datetime.strptime(training_cutoff_str, "%m/%d/%Y")
last_data_date = training_cutoff + relativedelta(months=1)  
next_pred_date = training_cutoff + relativedelta(months=2)  
schema = schemas[MODEL_NAME]
index_col= config.get("index_col")
target_col = config.get("target_col")
if "dataset" not in st.session_state:
    st.session_state["dataset"] = get_processed_fred(keep_last=True)
if st.button("Check for updates"):
    st.session_state.pop("dataset")
if "dataset" in st.session_state:
    X = st.session_state["dataset"].set_index(
        index_col).drop(columns=target_col)
    required_fields = schema.model_fields.keys()
    missing = [c for c in required_fields if c not in X.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        df_req = X[list(required_fields)]
        records = df_req.to_dict(orient="records")
        records = [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()} for row in records]
        payload = {"items": records}
        result = requests.post(
            predict_batch_endpoint, params={"dataset": MODEL_NAME} ,json=payload
            )
        response = result.json()['response']
        y_pred = np.array([list(r.values())[0] for r in response])
        df = st.session_state["dataset"].set_index(index_col)
        next_pred = float(y_pred[-1])
        df["predicted"] = pd.Series(y_pred, index=df.index).shift(1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["UNRATE"], name="True UNRATE", mode="lines"))
        fig.add_trace(go.Scatter(x=df.index, y=df["predicted"], name="Predicted UNRATE", mode="lines", line=dict(dash="dash")))
        cutoff_str = last_data_date.strftime("%#m/%#d/%Y")
        fig.add_vline(x=cutoff_str, line_dash="dot", line_color="red")
        fig.add_annotation(x=cutoff_str, y=1, yref="paper", text="Training cutoff", showarrow=False)
        fig.update_layout(title="UNRATE: True vs Predicted", xaxis_title="Date", yaxis_title="Unemployment Rate (%)")
        st.plotly_chart(fig, width="stretch")
        st.write(f"Predicted UNRATE for {next_pred_date.strftime('%B %Y')}: {next_pred:.2f}%")
get_feature_imp = st.sidebar.button("See feature importances")
if get_feature_imp:
    # get metadata and turn feature importances into a dataframe)
    feature_df = get_feature_importances(metadata)
    feature_df["feature"] = feature_df["feature"].str.replace(r"^.*?__", "", regex=True)
    feature_df = feature_df.sort_values("importance", ascending=False).head(20).set_index(
        "feature"
    )
    styled = feature_df.style.format("{:.4f}")
    st.sidebar.dataframe(styled, width='stretch')

with st.expander("Model details"):
    st.write(f"Training cutoff: {last_data_date.strftime('%B %Y')}")
    st.write(f"Test MAE: {abs(metadata['test_score']['Neg MAE']):.3f}")
    st.write(f"Model type: {metadata['model_type']}")
    st.write("Note: Model was trained on all historical data including COVID period, which explains the near-perfect fit on the chart.")
