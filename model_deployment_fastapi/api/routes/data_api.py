
from fastapi import APIRouter

from model_deployment_fastapi.models.ExchangeDtType import InputDT

router = APIRouter()


@router.get("/heartbeat", response_model=InputDT, name="heartbeat")
def get_data() -> InputDT:
    heartbeat = InputDT(is_alive=True)
    return heartbeat
