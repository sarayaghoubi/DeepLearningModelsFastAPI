from pydantic import BaseModel


class InputDT(BaseModel):
    features:list[float]