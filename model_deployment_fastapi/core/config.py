import os
from dotenv import load_dotenv
load_dotenv("../../.env")
APP_VERSION = "0.0.1"
APP_NAME = "Market Prediction Example"
API_PREFIX = "/api"

# config = Config()

API_KEY: str = os.environ.get("API_KEY")
IS_DEBUG: bool = os.environ.get("IS_DEBUG")

DEFAULT_MODEL_PATH: str = os.environ.get("DEFAULT_MODEL_PATH")
