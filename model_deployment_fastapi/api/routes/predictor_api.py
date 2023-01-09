from fastapi import APIRouter
from starlette.requests import Request
from DeepLearningModelsFastAPI.model_deployment_fastapi.models.ExchangeDtType import InputDT, Output
from DeepLearningModelsFastAPI.model_deployment_fastapi.services.predictor import Predictor

router = APIRouter()


@router.post("/predict", response_model=Output, name="predict")
def post_predict(
        request: Request,
        block_data: InputDT = None
) -> Output:
    model: Predictor = request.app.state.model
    prediction: Output = model.predict(block_data)

    return prediction
