from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from category_encoders import TargetEncoder
import pandas as pd
import numpy as np

def add_credit_inc_prod(df):
    """
    add column with product of credit score and income
    """
    return df.assign(credit_income_product = df["credit_score"] * df["annual_income"])


def add_dti_increase(df):
    """
    add increase to DTI (loan amount / income)
    """
    df = df.copy()
    inc = df["annual_income"].replace(0, np.nan)
    df["dti_increase"] = df["loan_amount"] / inc
    return df


def dropcols(
        df: pd.DataFrame,
        columns : str | list = []
 ):
    """
    drop columns from a list
    """
    return df.drop(columns=columns)

# convert to transformers for pipeline
add_credit_inc_prod_transform = FunctionTransformer(add_credit_inc_prod, validate=False)
dti_increase_transform = FunctionTransformer(add_dti_increase, validate=False)
# we drop loan purpose, gender and marital status as they don't correlate to the target
drop_columns_transform = FunctionTransformer(dropcols, kw_args={"columns" : ["loan_purpose", "gender", "marital_status"]}, validate=False)
# create ordinal transformations of grade_subgrade and education
# first list the grade_subgrade combinations in order
fine_grades = ['A1',
 'A2',
 'A3',
 'A4',
 'A5',
 'B1',
 'B2',
 'B3',
 'B4',
 'B5',
 'C1',
 'C2',
 'C3',
 'C4',
 'C5',
 'D1',
 'D2',
 'D3',
 'D4',
 'D5',
 'E1',
 'E2',
 'E3',
 'E4',
 'E5',
 'F1',
 'F2',
 'F3',
 'F4',
 'F5']
# list of education levels in order
education_levels = ["Other", "High School", "Bachelor's", "Master's", "PhD"]
# create the ordinal encoder
ordinal_cols = ["education_level", "grade_subgrade"]
ordinal_categories = [education_levels, fine_grades]
ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)
# create encoder that does ordinal encoding on education level and grade_subgrade
# and uses target encoding for the employment status
encoder = ColumnTransformer(
    transformers=[
        ("ord", ordinal_encoder, ordinal_cols),
        ("target_enc", TargetEncoder(), ["employment_status"]), 
    ],
    remainder="passthrough" 
)
loan_steps = [
    ("drop", drop_columns_transform),
    ("credit_inc_prod", add_credit_inc_prod_transform),
    ("dti_inc", dti_increase_transform),
    ("encoding", encoder)
    ]
