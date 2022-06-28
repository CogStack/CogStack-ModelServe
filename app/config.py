import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    BASE_MODEL_FILE: str = "model.zip"
    BASE_MODEL_FULL_PATH: str = ""
    CODE_TYPE: str = "snomed"
    DEVICE: str = "cpu"
    INCLUDE_SPAN_TEXT: str = "false"
    CONCAT_SIMILAR_ENTITIES: str = "true"
    MLFLOW_TRACKING_URI: str = f'file:{os.path.join(os.path.abspath(os.path.dirname(__file__)), "mlruns")}'
    REDEPLOY_TRAINED_MODEL: str = "false"
    SKIP_SAVE_MODEL: str = "false"

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), "envs", ".env")
        env_file_encoding = "utf-8"
