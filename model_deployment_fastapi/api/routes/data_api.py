
from fastapi import APIRouter
from DeepLearningModelsFastAPI.model_deployment_fastapi.models.ExchangeDtType import Status
router = APIRouter()


@router.get("/alive")
def get_data() -> Status:
    st = Status(condition='connection is established')
    return st
