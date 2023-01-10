from fastapi import APIRouter
from . import data_api
from . import predictor_api

api_router = APIRouter()
api_router.include_router(data_api.router, tags=["send_data"], prefix="/data")
api_router.include_router(predictor_api.router, tags=[
                          "prediction"], prefix="/model")
