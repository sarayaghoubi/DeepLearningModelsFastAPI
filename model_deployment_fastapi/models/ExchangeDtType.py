from pydantic import BaseModel
from numpy import ndarray
from typing import Optional,List


class InputDT(BaseModel):
    features: List[float]


class Output(BaseModel):
    bull: float
    bear: float
