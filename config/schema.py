# schema for various models
from pydantic import BaseModel

class IrisFeatures(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

schemas = {
    "iris" : IrisFeatures
}