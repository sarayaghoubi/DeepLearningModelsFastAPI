from pydantic import BaseModel
from typing import List


class InputDT(BaseModel):
    features: List[float]


class Output(BaseModel):
    bull: float
    bear: float
