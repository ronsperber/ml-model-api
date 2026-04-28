"""
schema for various models
"""
from pydantic import create_model
from .iris_schema import IrisFeatures
from .loan_schema import LoanFeatures
from .fred_schema import get_fred_fields
FredFeatures = create_model("FredFeatures", **get_fred_fields())


schemas = {"iris": IrisFeatures, "loan_data": LoanFeatures, "fred_data":FredFeatures}
