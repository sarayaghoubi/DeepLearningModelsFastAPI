from pydantic import BaseModel
import numpy as np


class InputDT(BaseModel):
    features: np.ndarray


class Output(BaseModel):
    bull: float
    bear: float
