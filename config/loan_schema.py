"""
schema for loan data
"""
from enum import Enum
from pydantic import BaseModel, Field

# create enum classes for each of the categorical features

# create a way to make the input case insensitive
class CaseInsensitiveEnum(str, Enum):
    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None

class Gender(CaseInsensitiveEnum):
    Male = "Male"
    Female = "Female"
    Other = "Other"


class MaritalStatus(CaseInsensitiveEnum):
    Married = "Married"
    Single = "Single"
    Divorced = "Divorced"
    Widowed = "Widowed"

class EducationLevel(CaseInsensitiveEnum):
    Other = "Other"
    HighSchool = "High School"
    Bachelor = "Bachelor's"
    Masters = "Master's"
    Doctorate = "PhD"

class EmploymentStatus(CaseInsensitiveEnum):
    SelfEmployed = "Self-Employed"
    Employed = "Employed"
    Unemployed = "Unemployed"
    Retired = "Retired"
    Student = "Student"

class GradeSubgrade(CaseInsensitiveEnum):
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"
    A5 = "A5"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    B4 = "B4"
    B5 = "B5"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"
    C5 = "C5"
    D1 = "D1"
    D2 = "D2"
    D3 = "D3"
    D4 = "D4"
    D5 = "D5"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"
    E5 = "E5"
    F1 = "F1"
    F2 = "F2"
    F3 = "F3"
    F4 = "F4"
    F5 = "F5"

class LoanFeatures(BaseModel):
    annual_income: float
    debt_to_income_ratio: float
    credit_score: int
    loan_amount: float
    interest_rate: float
    gender: Gender = Gender.Male
    marital_status: MaritalStatus = MaritalStatus.Married
    education_level: EducationLevel = EducationLevel.HighSchool
    employment_status: EmploymentStatus = EmploymentStatus.Employed
    loan_purpose: str = Field(default="Other")
    grade_subgrade: GradeSubgrade