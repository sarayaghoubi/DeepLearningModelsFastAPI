from fastapi import APIRouter, Depends
from starlette.requests import Request

from model_deployment_fastapi.core import security
from model_deployment_fastapi.models.ExchangeDtType import InputDT, Output
from model_deployment_fastapi.services.predictor import predictor

router = APIRouter()


@router.post("/predict", response_model=Output, name="predict")
def post_predict(
        request: Request,
        authenticated: bool = Depends(security.validate_request),
        block_data: InputDT = None
) -> Output:
    model: predictor = request.app.state.model
    prediction: Output = model.predict(block_data)

    return prediction
