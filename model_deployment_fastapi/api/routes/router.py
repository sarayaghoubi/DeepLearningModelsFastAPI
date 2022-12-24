

from fastapi import APIRouter

from model_deployment_fastapi.api.routes import data_api, predictor_api

api_router = APIRouter()
api_router.include_router(data_api.router, tags=["send_data"], prefix="/data")
api_router.include_router(predictor_api.router, tags=[
                          "prediction"], prefix="/model")
