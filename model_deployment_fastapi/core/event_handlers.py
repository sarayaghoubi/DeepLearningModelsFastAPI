from typing import Callable
from fastapi import FastAPI
from loguru import logger
from DeepLearningModelsFastAPI.model_deployment_fastapi.core.config import DEFAULT_MODEL_PATH
from DeepLearningModelsFastAPI.model_deployment_fastapi.services.predictor import predictor


def _startup_model(app: FastAPI) -> None:
    model_instance = predictor()
    app.state.model = model_instance


def _shutdown_model(app: FastAPI) -> None:
    app.state.model = None


def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        logger.info("Running app start handler.")
        _startup_model(app)
    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        logger.info("Running app shutdown handler.")
        _shutdown_model(app)
    return shutdown
