
from fastapi import APIRouter
from DeepLearningModelsFastAPI.model_deployment_fastapi.models.ExchangeDtType import status
router = APIRouter()


@router.get("/status", response_model=status, name="heartbeat")
def get_data() -> status:
    st = status(condition='ok')
    return st
