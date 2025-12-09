"""
schema for various models
"""
from pydantic import BaseModel, Field

class IrisFeatures(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

class LoanFeatures(BaseModel):
    annual_income: float
    debt_to_income_ratio: float
    credit_score: int
    loan_amount: float
    interest_rate: float
    gender: str = Field(default="Male")
    marital_status: str = Field(default="Married")
    education_level: str = Field(default="High School")
    employment_status: str = Field(default="Employed")
    loan_purpose: str = Field(default="Other")
    grade_subgrade: str

schemas = {
    "iris" : IrisFeatures,
    "iris_test": IrisFeatures,
    "loan_data": LoanFeatures
}