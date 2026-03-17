"""
schema for various models
"""
from .iris_schema import IrisFeatures
from .loan_schema import LoanFeatures

schemas = {
    "iris" : IrisFeatures,
    "loan_data": LoanFeatures
}